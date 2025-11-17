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
"""

from app.models.commondata import DataSource


from typing import List
from app.models import (
    DataSource,
    ClaimExtractionOutput,
    PipelineConfig,
    EvidenceRetrievalInput
)
from app.ai.pipeline.steps import PipelineSteps


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
    print("FACT-CHECK PIPELINE STARTING")
    print("=" * 80)

    # step 1: identify original_text sources and expand their links
    expanded_link_sources = await steps.expand_data_sources_with_links(data_sources, config)

    # combine original sources with expanded link sources
    all_data_sources = list[DataSource](data_sources) + expanded_link_sources

    print(f"\n[PIPELINE] Total data sources to process: {len(all_data_sources)}")
    print(f"  Original sources: {len(data_sources)}")
    print(f"  Expanded link sources: {len(expanded_link_sources)}")
    for i, source in enumerate(all_data_sources, 1):
        print(f"  {i}. {source.source_type} (id: {source.id})")
    
    # step 2: extract claims from all data sources
    print(f"\n{'=' * 80}")
    print("CLAIM EXTRACTION PHASE")
    print("=" * 80)

    claim_outputs = await steps.extract_claims_from_all_sources(
        data_sources=all_data_sources,
        llm_config=config.claim_extraction_llm_config
    )
    
    # summary
    print(f"\n{'=' * 80}")
    print("PIPELINE SUMMARY")
    print("=" * 80)

    total_claims = sum(len(output.claims) for output in claim_outputs)
    print(f"Total data sources processed: {len(all_data_sources)}")
    print(f"Total claims extracted: {total_claims}")

    # step 3: gather evidence for all extracted claims
    print(f"\n{'=' * 80}")
    print("EVIDENCE GATHERING PHASE")
    print("=" * 80)

    # flatten all claims from all sources into a single list
    all_claims = []
    for output in claim_outputs:
        all_claims.extend(output.claims)

    evidence_input = EvidenceRetrievalInput(claims=all_claims)

    # use timeout from config for evidence gathering
    evidence_timeout = config.timeout_config.evidence_retrieval_timeout_per_claim
    result = await steps.gather_evidence(
        retrieval_input=evidence_input,
        timeout=evidence_timeout
    )

    print("Evidence Gathering results: ", result)
    
    return claim_outputs

