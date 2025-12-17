from app.models import FactCheckResult, VerdictTypeEnum
from app.ai.threads.thread_utils import ThreadPoolManager, OperationType
from app.observability.logger import get_logger, PipelineStep
from app.ai.log_utils import log_adjudication_output


def _chose_fact_checking_result(
    original_result: FactCheckResult,
    manager: ThreadPoolManager,
    pipeline_id: str
) -> FactCheckResult:
    """
    determines if the fact check result returned to the user will be the one from
    the regular step or the fallback from adjudication with search.

    This is done by checking if the fallback result has found reliable fact-checking sources 
    for claims the main model does not have or if the fallback model and the main model disagree 
    on the number of false veredicts, favoring the one with more false veredicts to be conservative with fact-checking
    
    args:
        original_result: the fact check result from normal adjudication
        manager: thread pool manager to wait for adjudication_with_search job
        pipeline_id: pipeline identifier to filter jobs

    returns:
        either the original result or the adjudication_with_search result
    """
    logger = get_logger(__name__, PipelineStep.ADJUDICATION)
    try:

        total_num_claims = sum(
            1
            for result in original_result.results
            for verdict in result.claim_verdicts
        )

        insufficient_old = sum(
            1 if verdict.verdict == "Fontes insuficientes para verificar" else 0
            for result in original_result.results
            for verdict in result.claim_verdicts
        )

        number_of_original_false_claims = sum(
            1 if (verdict.verdict == VerdictTypeEnum.FALSO) or  (verdict.verdict == VerdictTypeEnum.FORA_DE_CONTEXTO) else 0
            for result in original_result.results
            for verdict in result.claim_verdicts
        )

        # All claims from the main fact-checking were already verified to be false, there is no condition where the fallback would be preffered
        if total_num_claims > 0 and total_num_claims == number_of_original_false_claims:
            return original_result

        # wait for adjudication_with_search job to complete (20 second timeout)
        job_id, search_result = manager.wait_next_completed(
            operation_type=OperationType.ADJUDICATION_WITH_SEARCH,
            timeout=20.0,
            raise_on_error=False,  # don't raise on error, we'll check the result
            pipeline_id=pipeline_id
        )

        # check if result is valid
        if isinstance(search_result, Exception):
            logger.warning(
                f"adjudication_with_search job failed: {type(search_result).__name__}: {search_result}"
            )
            # if original adjudication also failed, raise error
            if len(original_result.results) == 0:
                logger.error("both normal adjudication and fallback failed - raising error")
                raise RuntimeError(
                    f"adjudication failed and fallback also failed. "
                    f"normal adjudication returned no results and adjudication_with_search failed: {search_result}"
                ) from search_result
            logger.info("using original insufficient sources result")
            return original_result
        elif search_result is None or not isinstance(search_result, FactCheckResult):
            logger.warning(
                f"adjudication_with_search returned invalid result: {type(search_result)}"
            )
            # if original adjudication also failed, raise error
            if len(original_result.results) == 0:
                logger.error("both normal adjudication and fallback failed - raising error")
                raise RuntimeError(
                    f"adjudication failed and fallback returned invalid result. "
                    f"normal adjudication returned no results and adjudication_with_search returned: {type(search_result)}"
                )
            logger.info("using original insufficient sources result")
            return original_result
        elif len(search_result.results) == 0:
            logger.warning("adjudication_with_search returned empty results")
            # if original adjudication also failed, raise error
            if len(original_result.results) == 0:
                logger.error("both normal adjudication and fallback failed - raising error")
                raise RuntimeError(
                    "adjudication failed and fallback returned empty results. "
                    "both normal adjudication and adjudication_with_search returned no results."
                )
            logger.info("using original insufficient sources result")
            return original_result
        else:
            # both results are valid - compare


            number_of_fallback_false_claims = sum(
                1 if (verdict.verdict == VerdictTypeEnum.FALSO) or (verdict.verdict == VerdictTypeEnum.FORA_DE_CONTEXTO) else 0
                for result in search_result.results
                for verdict in result.claim_verdicts
            )

            logger.info(
                f"comparing results - original false claims: {number_of_original_false_claims}, "
                f"fallback false claims: {number_of_fallback_false_claims}"
            )

            if number_of_fallback_false_claims >= number_of_original_false_claims:
                logger.info(
                    f"[FALLBACK] using adjudication_with_search result (more false claims): "
                    f"{len(search_result.results)} results, "
                    f"{sum(len(r.claim_verdicts) for r in search_result.results)} verdicts"
                )

                # log both outputs for comparison
                logger.info("[ORIGINAL OUTPUT - Insufficient Sources]")
                log_adjudication_output(original_result)
                logger.info("[FALLBACK OUTPUT - Adjudication with Search]")
                log_adjudication_output(search_result)

                return search_result
            else:
                logger.info(
                    f"[ORIGINAL] using original result (fallback has fewer or equal false claims): "
                    f"{len(original_result.results)} results, "
                    f"{sum(len(r.claim_verdicts) for r in original_result.results)} verdicts"
                )

                # log both outputs for comparison
                logger.info("[ORIGINAL OUTPUT - Selected]")
                log_adjudication_output(original_result)
                logger.info("[FALLBACK OUTPUT - Not Selected]")
                log_adjudication_output(search_result)

                return original_result

    except TimeoutError:
        logger.warning(
            "adjudication_with_search job did not complete within 20 seconds"
        )
        # if original adjudication failed (empty results), we can't return empty result
        if len(original_result.results) == 0:
            logger.error("both normal adjudication and fallback failed - raising error")
            raise RuntimeError(
                "adjudication failed and fallback did not complete in time. "
                "normal adjudication returned no results and adjudication_with_search timed out after 20s."
            )
        logger.info("using original insufficient sources result")
        return original_result
    except Exception as e:
        logger.error(
            f"error while waiting for adjudication_with_search: {type(e).__name__}: {e}",
            exc_info=True
        )

        # if this is already a RuntimeError we raised earlier, just re-raise it
        # (don't double-wrap our own error messages)
        if isinstance(e, RuntimeError):
            raise

        # for other exceptions, check if we need to raise or return original
        if len(original_result.results) == 0:
            logger.error("both normal adjudication and fallback failed - raising error")
            raise RuntimeError(
                f"adjudication failed and fallback also failed. "
                f"normal adjudication returned no results and adjudication_with_search error: {e}"
            ) from e
        logger.info("using original insufficient sources result")
        return original_result