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

        # create wrapper function for link expansion
        def expand_links_from_sources(
            sources: List[DataSource]
        ) -> List[DataSource]:
            """expands links from sources and returns new DataSource objects"""
            # run link expansion (synchronous function using ThreadPoolManager internally)
            expanded_sources = steps.expand_data_sources_with_links(sources, config)

            # ensure we always return a list
            if expanded_sources is None:
                print("\n[LINK EXPANSION] Warning: expansion returned None") #warn
                return []

            print(f"\n[LINK EXPANSION] Expanded {len(expanded_sources)} link sources")  #this is debug 
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

        # step 5: adjudication - make final verdicts
        print(f"\n{'=' * 80}")
        print("STEP 5: ADJUDICATION - MAKING FINAL VERDICTS")
        print("=" * 80)

        # DEBUG: Log adjudication input details
        print(f"\n[DEBUG] Adjudication input validation:")
        print(f"  - sources_with_claims type: {type(adjudication_input.sources_with_claims)}")
        print(f"  - sources_with_claims length: {len(adjudication_input.sources_with_claims)}")

        for i, swc in enumerate(adjudication_input.sources_with_claims):
            print(f"\n  Source {i+1}:")
            print(f"    - data_source.id: {swc.data_source.id}")
            print(f"    - data_source.source_type: {swc.data_source.source_type}")
            print(f"    - enriched_claims length: {len(swc.enriched_claims)}")

            if swc.enriched_claims:
                first_claim = swc.enriched_claims[0]
                print(f"    - first claim ID: {first_claim.id}")
                print(f"    - first claim has 'text' attr: {hasattr(first_claim, 'text')}")
                print(f"    - first claim has 'citations' attr: {hasattr(first_claim, 'citations')}")
                if hasattr(first_claim, 'text'):
                    print(f"    - first claim text preview: {first_claim.text[:50]}...")
                if hasattr(first_claim, 'citations'):
                    print(f"    - first claim citations count: {len(first_claim.citations)}")
                    if first_claim.citations:
                        first_citation = first_claim.citations[0]
                        print(f"    - first citation type: {type(first_citation)}")
                        print(f"    - first citation has 'citation_text': {hasattr(first_citation, 'citation_text')}")
                        print(f"    - first citation has 'date': {hasattr(first_citation, 'date')}")

        print(f"\n[DEBUG] LLM config:")
        print(f"  - model_name: {config.adjudication_llm_config.model_name}")
        print(f"  - temperature: {config.adjudication_llm_config.temperature}")
        print(f"  - timeout: {config.adjudication_llm_config.timeout}")

        print(f"\n[DEBUG] Calling adjudicate_claims...")
        try:
            fact_check_result = adjudicate_claims(
                adjudication_input=adjudication_input,
                llm_config=config.adjudication_llm_config
            )
            print(f"[DEBUG] adjudicate_claims completed successfully")
        except Exception as e:
            print(f"\n[ERROR] Exception in adjudicate_claims:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            import traceback
            print(f"  Traceback:")
            traceback.print_exc()
            raise

        print(f"\nAdjudication completed:")
        print(f"  - Total data source results: {len(fact_check_result.results)}")

        for i, ds_result in enumerate(fact_check_result.results, 1):
            print(f"\n  {i}. DataSource: {ds_result.data_source_id} ({ds_result.source_type})")
            print(f"     - Verdicts: {len(ds_result.claim_verdicts)}")

            for j, verdict in enumerate(ds_result.claim_verdicts, 1):
                print(f"       {j}) Claim: {verdict.claim_text[:60]}...")
                print(f"          Verdict: {verdict.verdict}")
                print(f"          Justification: {verdict.justification[:100]}...")

        if fact_check_result.overall_summary:
            print(f"\n  Overall Summary:")
            print(f"  {fact_check_result.overall_summary}")

        # summary
        print(f"\n{'=' * 80}")
        print("PIPELINE SUMMARY")

        total_claims = sum(len(output.claims) for output in claim_outputs)
        print(f"Total claim extraction outputs: {len(claim_outputs)}")
        print(f"Total claims extracted: {total_claims}")
        print(f"Total enriched claims: {len(enriched_claims)}")
        print(f"Evidence gathering results: {len(result.claim_evidence_map)} claims with evidence")
        print(f"Final verdicts: {sum(len(r.claim_verdicts) for r in fact_check_result.results)} verdicts")

        return fact_check_result

    finally:
        # cleanup thread pool
        manager.shutdown(wait=True)
