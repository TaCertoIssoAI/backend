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
)
from app.ai.pipeline.steps import PipelineSteps
from app.ai.threads.thread_utils import ThreadPoolManager
from app.ai.async_code import fire_and_forget_streaming_pipeline
from app.ai.pipeline.claim_extractor import extract_claims
from app.ai.context.web import WebSearchGatherer
from app.ai.context.factcheckapi import GoogleFactCheckGatherer


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
        # step 1 & 2 & 3: fire-and-forget claim extraction + link expansion + evidence gathering
        print(f"\n{'=' * 80}")
        print("STREAMING CLAIM EXTRACTION + LINK EXPANSION + EVIDENCE GATHERING")
        print("=" * 80)
        print("Pattern: fire claim extraction for original sources,")
        print("fire link expansion job,")
        print("then as each completes, immediately fire evidence gathering")
        print("=" * 80)

        # create wrapper function that binds the config for regular claim extraction
        def extract_claims_with_config(
            extraction_input: ClaimExtractionInput
        ) -> ClaimExtractionOutput:
            """calls extract_claims with bound config"""
            return extract_claims(
                extraction_input=extraction_input,
                llm_config=config.claim_extraction_llm_config
            )

        # create wrapper function for link expansion
        def expand_links_from_sources(
            sources: List[DataSource]
        ) -> List[DataSource]:
            """expands links from sources and returns new DataSource objects"""
            # run link expansion (synchronous function using ThreadPoolManager internally)
            expanded_sources = steps.expand_data_sources_with_links(sources, config)

            # ensure we always return a list
            if expanded_sources is None:
                print("\n[LINK EXPANSION] Warning: expansion returned None")
                return []

            print(f"\n[LINK EXPANSION] Expanded {len(expanded_sources)} link sources")
            for i, source in enumerate(expanded_sources, 1):
                url = source.metadata.get("url", "unknown") if source.metadata else "unknown"
                success = source.metadata.get("success", False) if source.metadata else False
                status = "✓" if success else "✗"
                content_preview = source.original_text[:1000] if source.original_text else "(no content)"
                print(f"  {i}. {status} {source.source_type} (id: {source.id})")
                print(f"     URL: {url}")
                print(f"     Content preview: {content_preview}...")

            return expanded_sources

        # create evidence gatherers
        evidence_gatherers = [
            WebSearchGatherer(max_results=5),
            GoogleFactCheckGatherer(),
        ]

        # run fire-and-forget streaming pipeline (sync call)
        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            data_sources,
            extract_claims_with_config,
            evidence_gatherers,
            link_expansion_fn=expand_links_from_sources,
            manager=manager,
        )

        # build final evidence retrieval result
        result = EvidenceRetrievalResult(claim_evidence_map=enriched_claims)

        # build adjudication input by grouping enriched claims with data sources
        print(f"\n{'=' * 80}")
        print("BUILDING ADJUDICATION INPUT")
        print("=" * 80)

        adjudication_input = build_adjudication_input(claim_outputs, result)

        print(f"Adjudication input created successfully:")
        print(f"  - Total data sources: {len(adjudication_input.sources_with_claims)}")

        for i, ds_with_claims in enumerate(adjudication_input.sources_with_claims, 1):
            ds = ds_with_claims.data_source
            claims = ds_with_claims.enriched_claims
            print(f"\n  {i}. DataSource: {ds.id} ({ds.source_type})")
            print(f"     - Enriched claims: {len(claims)}")

            for j, claim in enumerate(claims, 1):
                citations_count = len(claim.citations)
                print(f"       {j}) Claim ID: {claim.id}")
                print(f"          Text: {claim.text[:80]}...")
                print(f"          Citations: {citations_count}")
                print(f"          Source: {claim.source.source_type} ({claim.source.source_id})")

        # summary
        print(f"\n{'=' * 80}")
        print("PIPELINE SUMMARY")
        print("=" * 80)

        total_claims = sum(len(output.claims) for output in claim_outputs)
        print(f"Total claim extraction outputs: {len(claim_outputs)}")
        print(f"Total claims extracted: {total_claims}")
        print(f"Total enriched claims: {len(enriched_claims)}")
        print(f"Evidence gathering results: {len(result.claim_evidence_map)} claims with evidence")

        return claim_outputs

    finally:
        # cleanup thread pool
        manager.shutdown(wait=True)
