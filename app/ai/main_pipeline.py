"""
Main fact-checking pipeline orchestration.

This module coordinates the full fact-checking flow:
1. Link context expansion - extract and expand URLs from original text
2. Claim extraction - extract claims from all data sources
3. Evidence retrieval - gather supporting/refuting evidence
4. Adjudication - make final verdicts

Architecture:
- Async-first design for efficient IO operations
- Type-safe with Pydantic models throughout
- Stateless functions with explicit dependencies
- Dependency injection for pipeline steps (enables testing and customization)
- Parallel execution using ThreadPoolManager for IO-bound operations
"""

from typing import List

from app.models import (
    DataSource,
    ClaimExtractionOutput,
    PipelineConfig,
    EvidenceRetrievalResult,
    ClaimExtractionInput,
    AdjudicationInput,
    DataSourceWithClaims,
    EnrichedClaim,
    FactCheckResult,
)
from app.ai.pipeline.steps import PipelineSteps
from app.ai.threads.thread_utils import ThreadPoolManager, OperationType
from app.ai.async_code import fire_and_forget_streaming_pipeline
from app.ai.pipeline.claim_extractor import extract_claims
from app.observability.analytics import AnalyticsCollector
from app.observability.logger import get_logger, PipelineStep
from app.ai.log_utils import log_adjudication_input, log_adjudication_output


def build_adjudication_input(
    claim_outputs: List[ClaimExtractionOutput],
    evidence_result: EvidenceRetrievalResult,
) -> AdjudicationInput:
    """
    build adjudication input by grouping enriched claims with their original data sources.

    this function reconstructs the data lineage by:
    1. taking each data source from claim extraction outputs
    2. finding all enriched claims that were extracted from that source
    3. grouping them together into DataSourceWithClaims objects

    args:
        claim_outputs: list of claim extraction outputs (each has a data source + extracted claims)
        evidence_result: evidence retrieval result mapping claim IDs to enriched claims

    returns:
        AdjudicationInput ready for the adjudication step

    example:
        >>> claim_outputs = [...]  # from claim extraction step
        >>> evidence_result = EvidenceRetrievalResult(claim_evidence_map={...})
        >>> adj_input = build_adjudication_input(claim_outputs, evidence_result)
        >>> print(len(adj_input.sources_with_claims))
        2  # number of data sources
    """
    sources_with_claims: List[DataSourceWithClaims] = []

    for output in claim_outputs:
        data_source = output.data_source
        enriched_claims_for_this_source: List[EnrichedClaim] = []

        # for each claim extracted from this data source
        for extracted_claim in output.claims:
            claim_id = extracted_claim.id

            # find the corresponding enriched claim in evidence result
            if claim_id in evidence_result.claim_evidence_map:
                enriched_claim = evidence_result.claim_evidence_map[claim_id]
                enriched_claims_for_this_source.append(enriched_claim)

        # create DataSourceWithClaims object
        source_with_claims = DataSourceWithClaims(
            data_source=data_source,
            enriched_claims=enriched_claims_for_this_source
        )
        sources_with_claims.append(source_with_claims)

    return AdjudicationInput(
        sources_with_claims=sources_with_claims,
        additional_context=None
    )

def _chose_fact_checking_result(
    original_result: FactCheckResult,
    manager: ThreadPoolManager,
    pipeline_id: str
) -> FactCheckResult:
    """
    determines if the fact check result returned to the user will be the one from
    the regular step or the fallback from adjudication with search.

    if all verdicts in the original result are "Fontes insuficientes para verificar",
    waits for the adjudication_with_search job to complete and returns that result
    if it's valid.

    args:
        original_result: the fact check result from normal adjudication
        manager: thread pool manager to wait for adjudication_with_search job
        pipeline_id: pipeline identifier to filter jobs

    returns:
        either the original result or the adjudication_with_search result
    """
    logger = get_logger(__name__, PipelineStep.ADJUDICATION)

    # check if normal adjudication failed (empty results) or all verdicts are "Fontes insuficientes"
    if len(original_result.results) == 0:
        logger.info("normal adjudication failed (no results) - checking adjudication_with_search fallback")
    else:
        all_insufficient = all(
            verdict.verdict == "Fontes insuficientes para verificar"
            for result in original_result.results
            for verdict in result.claim_verdicts
        )

        if not all_insufficient:
            # at least one verdict has sufficient sources, use original result
            return original_result

        logger.info("all verdicts are 'Fontes insuficientes para verificar' - checking adjudication_with_search fallback")

    try:
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
            # valid result - use it as fallback
            logger.info(
                f"[FALLBACK] using adjudication_with_search result: "
                f"{len(search_result.results)} results, "
                f"{sum(len(r.claim_verdicts) for r in search_result.results)} verdicts"
            )

            # log both outputs for comparison
            logger.info("[ORIGINAL OUTPUT - Insufficient Sources]")
            log_adjudication_output(original_result)

            logger.info("[FALLBACK OUTPUT - Adjudication with Search]")
            log_adjudication_output(search_result)

            # use search result as final result
            return search_result

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

async def run_fact_check_pipeline(
    data_sources: List[DataSource],
    config: PipelineConfig,
    steps: PipelineSteps,
    analytics: AnalyticsCollector,
    message_id: str
) -> FactCheckResult:
    """
    run the complete fact-checking pipeline on a list of data sources.

    pipeline steps:
    1. identify original_text sources and extract links
    2. expand links to create new link_context data sources
    3. extract claims from all data sources (original + expanded)
    4. gather evidence for each claim (citations, fact-check APIs, web search)
    5. adjudicate claims and make final verdicts based on evidence

    args:
        data_sources: list of data sources to fact-check
        config: pipeline configuration with timeout and LLM settings (required)
        steps: pipeline steps implementation. If None, uses DefaultPipelineSteps.
        analytics: analytics collector for tracking metrics
        message_id: unique identifier for this request, used for pipeline isolation

    returns:
        FactCheckResult with final verdicts for all claims, grouped by data source

    example:
        >>> from app.models import DataSource
        >>> from app.config.default import get_default_pipeline_config
        >>> from app.ai.pipeline.steps import DefaultPipelineSteps
        >>> sources = [
        ...     DataSource(
        ...         id="msg-001",
        ...         source_type="original_text",
        ...         original_text="Check this claim about vaccines"
        ...     )
        ... ]
        >>> config = get_default_pipeline_config()
        >>> steps = DefaultPipelineSteps()
        >>> result = await run_fact_check_pipeline(sources, config, steps, analytics, "msg-001")
        >>> print(result.results[0].claim_verdicts[0].verdict)
        "Falso"
    """

    # get logger for main pipeline orchestration
    pipeline_logger = get_logger(__name__, PipelineStep.SYSTEM)
    pipeline_logger.info(f"[{message_id}] pipeline isolation enabled with pipeline_id={message_id}")

    # initialize thread pool manager
    manager = ThreadPoolManager.get_instance(max_workers=25)
    manager.initialize()

    try:
        # step 1 & 2 & 3: fire-and-forget claim extraction + link expansion + evidence gathering

        # create wrapper function that binds the config for regular claim extraction
        def extract_claims_with_config(
            extraction_input: ClaimExtractionInput
        ) -> ClaimExtractionOutput:
            """calls extract_claims with bound config"""
            return extract_claims(
                extraction_input=extraction_input,
                llm_config=config.claim_extraction_llm_config
            )

        # create wrapper function for link expansion that binds the config
        def expand_links_with_config(
            sources: List[DataSource]
        ) -> List[DataSource]:
            """calls steps.expand_links_from_sources with bound config"""
            pipeline_logger.info(f"expand_links_with_config wrapper called with {len(sources)} sources")
            result = steps.expand_links_from_sources(sources, config)
            pipeline_logger.info(f"expand_links_with_config completed: {len(result) if result else 0} sources expanded")
            return result

        # get evidence gatherers from pipeline steps
        evidence_gatherers = steps.get_evidence_gatherers()
        pipeline_logger.info(
            f"using {len(evidence_gatherers)} evidence gatherers: "
            f"{', '.join(g.source_name for g in evidence_gatherers)}"
        )

        # run fire-and-forget streaming pipeline (sync call)
        # enable adjudication_with_search to fire in parallel with normal evidence gathering
        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            data_sources,
            extract_claims_with_config,
            evidence_gatherers,
            analytics,
            link_expansion_fn=expand_links_with_config,
            manager=manager,
            pipeline_steps=steps,
            enable_adjudication_with_search=True,
            pipeline_id=message_id,
        )

        if not any(claim_out.has_valid_claims() for claim_out in claim_outputs):
            # no valid claims found, use fallback
            no_claims_fallaback = await steps.handle_no_claims_fallback(data_sources,config) # TODO: Pretty sure the data sources here DO NOT include the expanded links so fix that later
            return FactCheckResult(
                results= [],
                sources_with_claims = [],
                overall_summary=no_claims_fallaback.explanation
            )

        # build final evidence retrieval result
        result = EvidenceRetrievalResult(claim_evidence_map=enriched_claims)
        analytics.populate_claims_from_evidence(result)

        # build adjudication input by grouping enriched claims with data sources
        adjudication_input = build_adjudication_input(claim_outputs, result)

        # log adjudication input details
        log_adjudication_input(adjudication_input, config.adjudication_llm_config)

        # adjudicate claims
        adjudication_logger = get_logger(__name__, PipelineStep.ADJUDICATION)
        adjudication_logger.debug("calling steps.adjudicate_claims...")
        try:
            fact_check_result = steps.adjudicate_claims(
                adjudication_input=adjudication_input,
                llm_config=config.adjudication_llm_config
            )
            adjudication_logger.debug("steps.adjudicate_claims completed successfully")
        except Exception as e:
            adjudication_logger.error(f"exception in adjudicate_claims: {type(e).__name__}")
            adjudication_logger.error(f"error message: {str(e)}")
            adjudication_logger.error("traceback:", exc_info=True)

            # create empty result to allow fallback to adjudication_with_search
            adjudication_logger.warning(
                "normal adjudication failed - creating empty result to trigger adjudication_with_search fallback"
            )
            fact_check_result = FactCheckResult(
                results=[],
                overall_summary="",
                sources_with_claims=adjudication_input.sources_with_claims
            )

        # log adjudication output (only if we have results)
        if fact_check_result.results:
            log_adjudication_output(fact_check_result)

        # choose final result: use adjudication_with_search fallback if normal adjudication failed/insufficient
        fact_check_result = _chose_fact_checking_result(fact_check_result, manager, message_id)

        # summary with prefix
        pipeline_logger.set_prefix("[SUMMARY]")
        pipeline_logger.info(f"{'=' * 80}")

        total_claims = sum(len(output.claims) for output in claim_outputs)
        total_verdicts = sum(len(r.claim_verdicts) for r in fact_check_result.results)

        pipeline_logger.info(f"total claim extraction outputs: {len(claim_outputs)}")
        pipeline_logger.info(f"total claims extracted: {total_claims}")
        pipeline_logger.info(f"total enriched claims: {len(enriched_claims)}")
        pipeline_logger.info(
            f"evidence gathering results: {len(result.claim_evidence_map)} claims with evidence"
        )
        pipeline_logger.info(f"final verdicts: {total_verdicts} verdicts")
        pipeline_logger.clear_prefix()

        analytics.populate_from_adjudication(fact_check_result)
        return fact_check_result

    except Exception as e:
        # log pipeline failure
        pipeline_logger.error(f"pipeline failed: {type(e).__name__}: {str(e)}")
        pipeline_logger.error("full traceback:", exc_info=True)
        raise
    finally:
        # cleanup completed jobs for this pipeline (non-blocking background cleanup)
        pipeline_logger.info(f"[{message_id}] starting background cleanup of completed jobs")
        manager.clear_completed_jobs_async(pipeline_id=message_id)