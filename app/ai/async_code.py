"""
parallel execution utilities for fact-checking pipeline.

provides utilities to run pipeline steps in parallel using ThreadPoolManager,
with support for streaming results and progress tracking.
"""

import logging
from typing import List, Callable, TypeVar, Dict, Any, Optional
from app.observability.analytics import AnalyticsCollector

from app.ai.threads.thread_utils import (
    ThreadPoolManager,
    OperationType,
    map_threaded_async,
)
from app.models import (
    DataSource,
    ClaimExtractionInput,
    ClaimExtractionOutput,
    ExtractedClaim,
    EvidenceRetrievalInput,
    EnrichedClaim,
    EvidenceRetrievalResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

async def parallel_claim_extraction(
    data_sources: List[DataSource],
    extract_fn: Callable[[ClaimExtractionInput], ClaimExtractionOutput],
    manager: Optional[ThreadPoolManager] = None,
) -> List[ClaimExtractionOutput]:
    """
    extract claims from all data sources in parallel.

    args:
        data_sources: list of data sources to extract claims from
        extract_fn: function that extracts claims from a single source
        manager: thread pool manager (uses singleton if None)

    returns:
        list of claim extraction outputs in same order as input sources

    example:
        >>> async def extract_claims(input_data):
        ...     # your claim extraction logic
        ...     return ClaimExtractionOutput(...)
        >>>
        >>> results = await parallel_claim_extraction(
        ...     data_sources=sources,
        ...     extract_fn=extract_claims
        ... )
    """
    if manager is None:
        manager = ThreadPoolManager.get_instance()

    logger.info(f"starting parallel claim extraction for {len(data_sources)} sources")

    # create extraction inputs
    extraction_inputs = [
        ClaimExtractionInput(data_source=source) for source in data_sources
    ]

    # run extractions in parallel
    results = await map_threaded_async(
        items=extraction_inputs,
        func=extract_fn,
        operation_type=OperationType.CLAIMS_EXTRACTION,
        manager=manager,
    )

    logger.info(
        f"claim extraction completed: {len(results)} outputs, "
        f"{sum(len(r.claims) for r in results)} total claims"
    )

    return results


def collect_evidence_results(
    manager: ThreadPoolManager,
    evidence_jobs_submitted: int,
    claim_id_to_claim: Dict[str, ExtractedClaim],
) -> Dict[str, List[Any]]:
    """
    wait for all evidence gathering jobs to complete and collect results.

    waits for evidence jobs to finish and groups citations by claim id.

    args:
        manager: thread pool manager
        evidence_jobs_submitted: total number of evidence jobs submitted
        claim_id_to_claim: dict of claims by id (to initialize citation storage)

    returns:
        dict mapping claim_id to list of citations
    """
    # initialize citations dict for all claims
    claim_citations: Dict[str, List[Any]] = {
        claim_id: [] for claim_id in claim_id_to_claim
    }

    evidence_completed = 0

    while evidence_completed < evidence_jobs_submitted:
        try:
            # wait for next completed evidence gathering (10 second timeout)
            _job_id, result = manager.wait_next_completed(
                operation_type=OperationType.LINK_EVIDENCE_RETRIEVER,
                timeout=10.0,
                raise_on_error=False,  # don't raise on individual failures
            )

            evidence_completed += 1

            # result is a tuple: (claim_id, citations)
            if isinstance(result, tuple) and len(result) == 2:
                claim_id, citations = result

                # add citations to the appropriate claim
                if claim_id in claim_citations and isinstance(citations, list):
                    claim_citations[claim_id].extend(citations)

                    logger.info(
                        f"evidence gathering completed ({evidence_completed}/"
                        f"{evidence_jobs_submitted}): {len(citations)} citations "
                        f"for claim {claim_id}"
                    )
                else:
                    logger.warning(
                        f"received citations for unknown claim {claim_id}"
                    )
            elif isinstance(result, Exception):
                logger.error(f"evidence gathering job failed: {result}")
            else:
                logger.warning(f"unexpected result format: {type(result)}")

        except TimeoutError:
            logger.debug("no evidence gathering completed in last 10s, retrying...")
            continue
        except Exception as e:
            logger.error(f"evidence gathering job failed: {e}", exc_info=True)
            evidence_completed += 1

    return claim_citations


def fire_evidence_jobs_for_claim(
    claim: ExtractedClaim,
    evidence_gatherers: List[Any],
    manager: ThreadPoolManager,
    claim_id_to_claim: Dict[str, ExtractedClaim],
    evidence_jobs_by_claim: Dict[str, List[str]],
) -> int:
    """
    fire individual evidence gathering jobs for a single claim.

    creates one job per evidence gatherer and submits them to the thread pool.

    args:
        claim: the claim to gather evidence for
        evidence_gatherers: list of evidence gatherers to use
        manager: thread pool manager
        claim_id_to_claim: dict to track claims by id (updated in place)
        evidence_jobs_by_claim: dict to track gatherers per claim (updated in place)

    returns:
        number of jobs submitted
    """
    # track claim for later enrichment
    claim_id_to_claim[claim.id] = claim
    evidence_jobs_by_claim[claim.id] = []

    jobs_submitted = 0

    # fire one job per gatherer (max parallelism)
    for gatherer in evidence_gatherers:
        gatherer_name = (
            gatherer.source_name
            if hasattr(gatherer, 'source_name')
            else type(gatherer).__name__
        )

        # create sync wrapper that returns (claim_id, citations) tuple
        def gather_with_gatherer(
            gatherer_instance=gatherer,
            claim_instance=claim,
            c_id=claim.id
        ):
            citations = gatherer_instance.gather_sync(claim_instance)
            return (c_id, citations)  # return tuple for tracking

        manager.submit(
            OperationType.LINK_EVIDENCE_RETRIEVER,
            gather_with_gatherer,
        )
        jobs_submitted += 1
        evidence_jobs_by_claim[claim.id].append(gatherer_name)

        logger.info(
            f"fired {gatherer_name} evidence job for claim: {claim.text[:50]}..."
        )

    return jobs_submitted


def fire_and_forget_streaming_pipeline(
    data_sources: List[DataSource],
    extract_fn: Callable[[ClaimExtractionInput], ClaimExtractionOutput],
    evidence_gatherers: List[Any],
    analytics: AnalyticsCollector,
    link_expansion_fn: Optional[Callable[[List[DataSource]], List[DataSource]]] = None,
    manager: Optional[ThreadPoolManager] = None,
) -> tuple[List[ClaimExtractionOutput], Dict[str, EnrichedClaim]]:
    """
    extract claims and gather evidence using fire-and-forget streaming pattern.

    workflow:
    1. fire claim extraction jobs for original data sources (don't wait)
    2. if link_expansion_fn provided, fire link expansion job (don't wait)
    3. loop while claim extraction or link expansion jobs are pending:
       - if claim extraction completes → fire evidence jobs for each claim
       - if link expansion completes → fire claim extraction jobs for each expanded source
    4. wait for all evidence gathering jobs to complete
    5. group citations by claim id and build enriched claims

    args:
        data_sources: list of original data sources to extract claims from
        extract_fn: function that extracts claims from a single source
        evidence_gatherers: list of evidence gatherers (e.g., WebSearchGatherer, GoogleFactCheckGatherer)
        link_expansion_fn: optional function that expands links from data sources,
                          returns list of new DataSource objects
        manager: thread pool manager (uses singleton if None)

    returns:
        tuple of (claim_outputs, enriched_claims_map)

    example:
        >>> claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
        ...     data_sources=sources,
        ...     extract_fn=extract_claims,
        ...     evidence_gatherers=[WebSearchGatherer(), GoogleFactCheckGatherer()],
        ...     link_expansion_fn=expand_links
        ... )
    """
    if manager is None:
        manager = ThreadPoolManager.get_instance()

    logger.info(f"starting fire-and-forget pipeline for {len(data_sources)} sources")

    # step 1: fire claim extraction jobs for original data sources (don't wait)
    claim_extraction_jobs_submitted = 0
    for source in data_sources:
        extraction_input = ClaimExtractionInput(data_source=source)
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            extract_fn,
            extraction_input,
        )
        claim_extraction_jobs_submitted += 1

    # step 2: fire link expansion pipeline job if provided (don't wait)
    link_expansion_pending = False
    if link_expansion_fn is not None:
        manager.submit(
            OperationType.LINK_EXPANSION_PIPELINE,
            link_expansion_fn,
            data_sources,
        )
        link_expansion_pending = True
        logger.info("fired link expansion pipeline job")

    logger.info(f"fired {claim_extraction_jobs_submitted} claim extraction jobs")

    # track results
    claim_outputs: List[ClaimExtractionOutput] = []
    claim_id_to_claim: Dict[str, ExtractedClaim] = {}

    # track evidence jobs: map (claim_id, gatherer_name) -> job
    evidence_jobs_submitted = 0
    evidence_jobs_by_claim: Dict[str, List[str]] = {}  # claim_id -> list of gatherer names

    # step 3: loop while claim extraction or link expansion jobs are pending
    claim_extractions_completed = 0

    while claim_extractions_completed < claim_extraction_jobs_submitted or link_expansion_pending:
        # check for completed claim extraction
        try:
            _job_id, output = manager.wait_next_completed(
                operation_type=OperationType.CLAIMS_EXTRACTION,
                timeout=0.1,  # non-blocking check
                raise_on_error=True,
            )

            claim_extractions_completed += 1
            claim_outputs.append(output)

            logger.info(
                f"claim extraction completed ({claim_extractions_completed}/"
                f"{claim_extraction_jobs_submitted}): {len(output.claims)} claims "
                f"from source {output.data_source.id}"
            )

            # immediately fire INDIVIDUAL evidence gathering jobs for each claim
            for claim in output.claims:
                jobs_fired = fire_evidence_jobs_for_claim(
                    claim=claim,
                    evidence_gatherers=evidence_gatherers,
                    manager=manager,
                    claim_id_to_claim=claim_id_to_claim,
                    evidence_jobs_by_claim=evidence_jobs_by_claim,
                )
                evidence_jobs_submitted += jobs_fired

        except TimeoutError:
            # no claim extraction completed, check link expansion
            pass
        except Exception as e:
            logger.error(f"claim extraction job failed: {e}", exc_info=True)
            claim_extractions_completed += 1

        # check for completed link expansion pipeline
        if link_expansion_pending:
            try:
                _job_id, expanded_sources = manager.wait_next_completed(
                    operation_type=OperationType.LINK_EXPANSION_PIPELINE,
                    timeout=0.1,  # non-blocking check
                    raise_on_error=True,
                )

                link_expansion_pending = False

                # handle None or non-list results
                if expanded_sources is None:
                    logger.warning("link expansion returned None - no sources expanded")
                    expanded_sources = []
                elif not isinstance(expanded_sources, list):
                    logger.error(f"link expansion returned unexpected type: {type(expanded_sources)}")
                    expanded_sources = []

                logger.info(f"link expansion pipeline completed: {len(expanded_sources)} sources expanded")
                #add new link data sources to Analytics
                analytics.populate_from_data_sources(expanded_sources)


                # fire claim extraction jobs for each expanded source
                for source in expanded_sources:
                    extraction_input = ClaimExtractionInput(data_source=source)
                    manager.submit(
                        OperationType.CLAIMS_EXTRACTION,
                        extract_fn,
                        extraction_input,
                    )
                    claim_extraction_jobs_submitted += 1
                    logger.info(f"fired claim extraction for expanded source: {source.id}")

            except TimeoutError:
                # no link expansion completed yet
                pass
            except Exception as e:
                logger.error(f"link expansion pipeline job failed: {e}", exc_info=True)
                link_expansion_pending = False

    logger.info(
        f"all claim extractions completed. waiting for {evidence_jobs_submitted} "
        f"evidence gathering jobs"
    )

    # step 3: wait for all evidence gathering jobs and group by claim
    claim_citations = collect_evidence_results(
        manager, evidence_jobs_submitted, claim_id_to_claim
    )

    # step 4: build enriched claims from grouped citations
    enriched_claims: Dict[str, EnrichedClaim] = {}
    for claim_id, claim in claim_id_to_claim.items():
        enriched_claims[claim_id] = EnrichedClaim(
            id=claim.id,
            text=claim.text,
            source=claim.source,
            entities=claim.entities,
            llm_comment=claim.llm_comment,
            citations=claim_citations[claim_id],
        )

    logger.info(
        f"fire-and-forget pipeline completed: {len(claim_outputs)} outputs, "
        f"{len(enriched_claims)} enriched claims, "
        f"{sum(len(c) for c in claim_citations.values())} total citations"
    )

    return claim_outputs, enriched_claims