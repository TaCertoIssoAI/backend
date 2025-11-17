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
"""
from typing import List
from app.models import (
    DataSource,
    ClaimExtractionInput,
    ClaimExtractionOutput,
    PipelineConfig,
)
from app.ai.pipeline.link_context_expander import expand_link_contexts
from app.ai.pipeline.claim_extractor import extract_claims_async


async def run_fact_check_pipeline(
    data_sources: List[DataSource],
    config: PipelineConfig,
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
    all_data_sources = list(data_sources)  # start with provided sources
    
    for source in data_sources:
        if source.source_type == "original_text":
            print(f"\n[LINK EXPANSION] Processing original_text source: {source.id}")
            print(f"  Text preview: {source.original_text[:100]}...")
            
            # expand link contexts using the existing function
            expanded_sources = await expand_link_contexts(source, config)
            
            if expanded_sources:
                print(f"  Created {len(expanded_sources)} new link_context data source(s):")
                for expanded in expanded_sources:
                    url = expanded.metadata.get("url", "unknown")
                    success = expanded.metadata.get("success", False)
                    status = "✓" if success else "✗"
                    print(f"    {status} {url}")
                
                all_data_sources.extend(expanded_sources)
            else:
                print("  No links found or expanded")
    
    print(f"\n[PIPELINE] Total data sources to process: {len(all_data_sources)}")
    for i, source in enumerate(all_data_sources, 1):
        print(f"  {i}. {source.source_type} (id: {source.id})")
    
    # step 2: extract claims from all data sources
    print(f"\n{'=' * 80}")
    print("CLAIM EXTRACTION PHASE")
    print("=" * 80)
    
    claim_outputs: List[ClaimExtractionOutput] = []
    
    for source in all_data_sources:
        print(f"\n[CLAIM EXTRACTION] Processing {source.source_type} source: {source.id}")
        print(f"  Text preview: {source.original_text[:100]}...")
        
        # create input for claim extractor
        extraction_input = ClaimExtractionInput(data_source=source)
        
        # extract claims
        result = await extract_claims_async(
            extraction_input=extraction_input,
            llm_config=config.claim_extraction_llm_config
        )
        
        claim_outputs.append(result)
        
        # print extracted claims
        if result.claims:
            print(f"  Extracted {len(result.claims)} claim(s):")
            for i, claim in enumerate(result.claims, 1):
                print(f"    {i}. {claim.text}")
                if claim.entities:
                    print(f"       entities: {', '.join(claim.entities)}")
                if claim.llm_comment:
                    print(f"       comment: {claim.llm_comment}")
        else:
            print("  No claims extracted")
    
    # summary
    print(f"\n{'=' * 80}")
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    total_claims = sum(len(output.claims) for output in claim_outputs)
    print(f"Total data sources processed: {len(all_data_sources)}")
    print(f"Total claims extracted: {total_claims}")
    
    return claim_outputs

