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
from app.ai.threads.thread_utils import ThreadPoolManager
from app.ai.async_code import fire_and_forget_streaming_pipeline
from app.ai.pipeline.claim_extractor import extract_claims
from app.ai.pipeline.judgement import adjudicate_claims
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

async def run_fact_check_pipeline(
    data_sources: List[DataSource],
    config: PipelineConfig,
    steps: PipelineSteps,
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
        >>> result = await run_fact_check_pipeline(sources, config, steps)
        >>> print(result.results[0].claim_verdicts[0].verdict)
        "Falso"
    """

    # get logger for main pipeline orchestration
    pipeline_logger = get_logger(__name__, PipelineStep.SYSTEM)

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
            return steps.expand_links_from_sources(sources, config)

        # get evidence gatherers from pipeline steps
        evidence_gatherers = steps.get_evidence_gatherers()
        pipeline_logger.info(
            f"using {len(evidence_gatherers)} evidence gatherers: "
            f"{', '.join(g.source_name for g in evidence_gatherers)}"
        )

        # run fire-and-forget streaming pipeline (sync call)
        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            data_sources,
            extract_claims_with_config,
            evidence_gatherers,
            link_expansion_fn=expand_links_with_config,
            manager=manager,
        )

        # build final evidence retrieval result
        result = EvidenceRetrievalResult(claim_evidence_map=enriched_claims)

        # build adjudication input by grouping enriched claims with data sources
        adjudication_input = build_adjudication_input(claim_outputs, result)

        # log adjudication input details
        log_adjudication_input(adjudication_input, config.adjudication_llm_config)

        # adjudicate claims
        adjudication_logger = get_logger(__name__, PipelineStep.ADJUDICATION)
        adjudication_logger.debug("calling adjudicate_claims...")
        try:
            fact_check_result = adjudicate_claims(
                adjudication_input=adjudication_input,
                llm_config=config.adjudication_llm_config
            )
            adjudication_logger.debug("adjudicate_claims completed successfully")
        except Exception as e:
            adjudication_logger.error(f"exception in adjudicate_claims: {type(e).__name__}")
            adjudication_logger.error(f"error message: {str(e)}")
            adjudication_logger.error("traceback:", exc_info=True)
            raise

        # log adjudication output
        log_adjudication_output(fact_check_result)

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

        return fact_check_result

    finally:
        # cleanup thread pool
        pipeline_logger.info("shutting down thread pool")
        manager.shutdown(wait=True)
