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
)
from app.ai.pipeline.steps import PipelineSteps
from app.ai.threads.thread_utils import ThreadPoolManager
from app.ai.async_code import fire_and_forget_streaming_pipeline
from app.ai.pipeline.claim_extractor import extract_claims
from app.ai.context.web import WebSearchGatherer
from app.ai.context.factcheckapi import GoogleFactCheckGatherer


async def run_fact_check_pipeline(
    data_sources: List[DataSource],
    config: PipelineConfig,
    steps: PipelineSteps,
) -> List[ClaimExtractionOutput]:
    """
    run the fact-checking pipeline on a list of data sources.

    pipeline steps:
    1. identify original_text sources and extract links
    2. expand links to create new link_context data sources
    3. extract claims from all data sources (original + expanded)
    4. return all extracted claims grouped by source

    args:
        data_sources: list of data sources to fact-check
        config: pipeline configuration with timeout and LLM settings (required)
        steps: pipeline steps implementation. If None, uses DefaultPipelineSteps.

    returns:
        list of claim extraction outputs, one per data source

    example:
        >>> from app.models import DataSource
        >>> from app.config.default import get_default_pipeline_config
        >>> sources = [
        ...     DataSource(
        ...         id="msg-001",
        ...         source_type="original_text",
        ...         original_text="Check this: https://example.com"
        ...     )
        ... ]
        >>> config = get_default_pipeline_config()
        >>> results = await run_fact_check_pipeline(sources, config)
    """

    print("=" * 80)
    print("FACT-CHECK PIPELINE STARTING (FIRE-AND-FORGET MODE)")
    print("=" * 80)

    # initialize thread pool manager
    manager = ThreadPoolManager.get_instance(max_workers=25)
    manager.initialize()

    try:
        # step 1: link context expansion (synchronous for now)
        print(f"\n{'=' * 80}")
        print("LINK EXPANSION PHASE (SYNCHRONOUS)")
        print("=" * 80)

        expanded_link_sources = await steps.expand_data_sources_with_links(
            data_sources, config
        )

        # combine original sources with expanded link sources
        all_data_sources = list[DataSource](data_sources) + expanded_link_sources

        print(f"\n[PIPELINE] Total data sources to process: {len(all_data_sources)}")
        print(f"  Original sources: {len(data_sources)}")
        print(f"  Expanded link sources: {len(expanded_link_sources)}")
        for i, source in enumerate(all_data_sources, 1):
            print(f"  {i}. {source.source_type} (id: {source.id})")

        # step 2 & 3: fire-and-forget claim extraction + evidence gathering
        print(f"\n{'=' * 80}")
        print("STREAMING CLAIM EXTRACTION + EVIDENCE GATHERING")
        print("=" * 80)
        print("Pattern: fire all claim extraction jobs,")
        print("then as each completes, immediately fire evidence gathering")
        print("=" * 80)

        # create wrapper function that binds the config
        def extract_claims_with_config(
            extraction_input: ClaimExtractionInput
        ) -> ClaimExtractionOutput:
            """calls extract_claims with bound config"""
            return extract_claims(
                extraction_input=extraction_input,
                llm_config=config.claim_extraction_llm_config
            )

        # create evidence gatherers
        evidence_gatherers = [
            WebSearchGatherer(max_results=5),
            GoogleFactCheckGatherer(),
        ]

        # run fire-and-forget streaming pipeline (sync call)
        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            all_data_sources,
            extract_claims_with_config,
            evidence_gatherers,
            manager,
        )

        # build final evidence retrieval result
        result = EvidenceRetrievalResult(claim_evidence_map=enriched_claims)

        # summary
        print(f"\n{'=' * 80}")
        print("PIPELINE SUMMARY")
        print("=" * 80)

        total_claims = sum(len(output.claims) for output in claim_outputs)
        print(f"Total data sources processed: {len(all_data_sources)}")
        print(f"Total claims extracted: {total_claims}")
        print(f"Total enriched claims: {len(enriched_claims)}")
        print(f"Evidence gathering results: {len(result.claim_evidence_map)} claims with evidence")

        return claim_outputs

    finally:
        # cleanup thread pool
        manager.shutdown(wait=True)
