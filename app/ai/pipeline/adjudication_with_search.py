"""
This is a fallback/experimental step of the pipeline and it defines a mix of the Evidence Retrieval + Adjundication step
Adjudication with Google Search - Pipeline Step.

This module provides an alternative adjudication step that uses Google GenAI's
grounding with Google Search to fact-check claims in real-time.

Key differences from standard adjudication:
- Uses Google Search grounding instead of pre-gathered citations
- Combines evidence gathering and adjudication in one step
- Single API call with JSON output

Architecture:
- Uses Google GenAI SDK directly (not LangChain)
- Single API call with Google Search grounding that returns JSON
- Parses JSON response into FactCheckResult using Pydantic
- Type-safe with Pydantic models throughout
"""

import os
from typing import List
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.models import (
    ExtractedClaim,
    FactCheckResult,
    LLMAdjudicationOutput,
    DataSource,
    DataSourceWithExtractedClaims,
)

from .prompts import ADJUDICATION_WITH_SEARCH_SYSTEM_PROMPT
from .utils import get_current_date, convert_llm_output_to_data_source_results

# ===== CLIENT INITIALIZATION =====

def _get_genai_client() -> genai.Client:
    """
    get or create Google GenAI client.

    returns:
        configured genai.Client instance
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    return genai.Client(api_key=api_key,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=1
        )
    ))


# ===== PROMPTS =====

def _build_adjudication_prompt(claims: List[ExtractedClaim], current_date: str) -> str:
    """
    build the prompt for adjudication with search.

    args:
        claims: list of extracted claims to fact-check
        current_date: current date in DD-MM-YYYY format

    returns:
        formatted prompt string
    """
    # Format the system prompt with current date
    system_prompt = ADJUDICATION_WITH_SEARCH_SYSTEM_PROMPT.format(current_date=current_date)

    prompt_parts = [system_prompt]
    prompt_parts.append("\n\n## Alegações para Verificar:\n")

    for i, claim in enumerate(claims, 1):
        prompt_parts.append(f"\n**Alegação {i}**:")
        prompt_parts.append(f"- Texto: {claim.text}")
        prompt_parts.append("")

    prompt_parts.append("\nPor favor, verifique cada alegação usando a busca do Google e forneça vereditos fundamentados.")

    return "\n".join(prompt_parts)


# ===== MAIN ADJUDICATION FUNCTION =====

def adjudicate_claims_with_search(
    sources_with_claims: List[DataSourceWithExtractedClaims],
    model: str = "gemini-3-pro-preview"
) -> FactCheckResult:
    """
    adjudicate claims using Google Search grounding in a single API call.

    this is the main entry point for adjudication with search.

    the function uses Google Search grounding to find evidence and returns
    structured JSON output in one call.

    args:
        sources_with_claims: list of DataSourceWithExtractedClaims to fact-check
        model: Google GenAI model to use (default: gemini-3-pro-preview)

    returns:
        FactCheckResult with verdicts for all claims

    raises:
        ValueError: if GOOGLE_API_KEY is not set
        Exception: if API calls fail

    example:
        >>> from app.models import ExtractedClaim, ClaimSource, DataSource, DataSourceWithExtractedClaims
        >>> claim = ExtractedClaim(
        ...     id="claim-1",
        ...     text="A vacina X causa infertilidade",
        ...     source=ClaimSource(source_type="original_text", source_id="msg-1")
        ... )
        >>> data_source = DataSource(id="msg-1", source_type="original_text", original_text="test")
        >>> source_with_claims = DataSourceWithExtractedClaims(data_source=data_source, extracted_claims=[claim])
        >>> result = adjudicate_claims_with_search([source_with_claims])
        >>> print(result.results[0].claim_verdicts[0].verdict)
    """
    print("\n" + "="*80)
    print("[DEBUG] Starting adjudicate_claims_with_search")
    print("="*80)

    # Input validation and logging
    print(f"[DEBUG] Input: {len(sources_with_claims)} sources with claims")
    for idx, source in enumerate(sources_with_claims, 1):
        print(f"  Source {idx}: {source.data_source.id} with {len(source.extracted_claims)} claims")

    client = _get_genai_client()
    print("[DEBUG] GenAI client initialized successfully")

    # Get current date
    current_date = get_current_date()
    print(f"[DEBUG] Current date: {current_date}")

    # Extract all claims from all sources for the prompt
    all_claims = []
    for source_with_claims in sources_with_claims:
        all_claims.extend(source_with_claims.extracted_claims)

    print(f"[DEBUG] Total claims to adjudicate: {len(all_claims)}")
    for idx, claim in enumerate(all_claims, 1):
        print(f"  Claim {idx}: [{claim.id}] {claim.text[:80]}...")

    # Configure Google Search grounding
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1,
        response_mime_type="application/json",
        response_json_schema=LLMAdjudicationOutput.model_json_schema(),
        thinking_config=types.ThinkingConfig(thinking_budget=1)
    )
    print("[DEBUG] Google Search grounding configured")

    # Build the prompt with JSON output instructions and current date
    prompt = _build_adjudication_prompt(all_claims, current_date)
    print(f"[DEBUG] Prompt built, length: {len(prompt)} characters")
    print(f"[DEBUG] Prompt preview (first 200 chars):\n{prompt[:200]}...")

    # Make single API call with Google Search grounding
    print(f"[DEBUG] Calling GenAI API with model: {model}")
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        print("[DEBUG] API call successful")
    except Exception as e:
        print(f"[ERROR] API call failed: {type(e).__name__}: {str(e)}")
        raise

    # Check and print grounding metadata if available
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            print("\n[INFO] Grounding Metadata found in response:")

            # Check for grounding supports
            if hasattr(candidate.grounding_metadata, 'grounding_supports'):
                supports = candidate.grounding_metadata.grounding_supports
                print(f"  - Grounding Supports: {len(supports) if supports else 0} items")
                if supports:
                    for idx, support in enumerate(supports[:3], 1):  # Print first 3
                        print(f"    [{idx}] {support}")

            # Check for grounding chunks
            if hasattr(candidate.grounding_metadata, 'grounding_chunks'):
                chunks = candidate.grounding_metadata.grounding_chunks
                print(f"  - Grounding Chunks: {len(chunks) if chunks else 0} items")
                if chunks:
                    for idx, chunk in enumerate(chunks[:3], 1):  # Print first 3
                        print(f"    [{idx}] {chunk}")
        else:
            print("\n[WARNING] No grounding_metadata found in response candidate")
    else:
        print("\n[WARNING] No candidates found in response")

    # Parse JSON response to LLMAdjudicationOutput
    print("\n[DEBUG] Parsing JSON response...")

    if response.text is None:
        print("[ERROR] response.text is None!")
        print(f"[DEBUG] Response object: {response}")
        print(f"[DEBUG] Response candidates: {response.candidates if hasattr(response, 'candidates') else 'N/A'}")
        raise ValueError("API response text is None - no content returned")

    print(f"[DEBUG] Response text length: {len(response.text)} characters")
    print(f"[DEBUG] Response text preview (first 500 chars):\n{response.text[:500]}")

    try:
        llm_output = LLMAdjudicationOutput.model_validate_json(response.text)
        print(f"[DEBUG] Successfully parsed {len(llm_output.results)} result(s)")

        # Print detailed verdict information
        for idx, result in enumerate(llm_output.results, 1):
            print(f"\n[DEBUG] Result {idx}:")
            print(f"  - data_source_id: {result.data_source_id}")
            print(f"  - Number of verdicts: {len(result.claim_verdicts)}")
            for v_idx, verdict in enumerate(result.claim_verdicts, 1):
                print(f"  - Verdict {v_idx}:")
                print(f"      claim_id: {verdict.claim_id}")
                print(f"      claim_text: {verdict.claim_text[:60]}...")
                print(f"      verdict: {verdict.verdict}")
                print(f"      justification: {verdict.justification[:100]}...")
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        print(f"[ERROR] Full response:\n{response.text}")
        raise

    # Convert LLM output to DataSourceResult using utils
    print("\n[DEBUG] Converting to DataSourceResult...")
    data_source_results = convert_llm_output_to_data_source_results(
        llm_results=llm_output.results,
        sources_with_claims=sources_with_claims
    )

    print(f"[DEBUG] Created {len(data_source_results)} DataSourceResult(s)")
    for idx, ds_result in enumerate(data_source_results, 1):
        print(f"  Result {idx}: {ds_result.data_source_id} - {len(ds_result.claim_verdicts)} verdict(s)")

    # Build final FactCheckResult
    return FactCheckResult(
        results=data_source_results,
        overall_summary=llm_output.overall_summary if llm_output.overall_summary else "",
        sources_with_claims=sources_with_claims
    )


async def adjudicate_claims_with_search_async(
    sources_with_claims: List[DataSourceWithExtractedClaims],
    model: str = "gemini-2.5-flash"
) -> FactCheckResult:
    """
    async version of adjudicate_claims_with_search.

    note: Google GenAI Python SDK doesn't have native async support yet,
    so this uses sync calls in an async wrapper. For true async, consider
    using asyncio.to_thread or similar.

    args:
        sources_with_claims: list of DataSourceWithExtractedClaims to fact-check
        model: Google GenAI model to use

    returns:
        FactCheckResult with verdicts for all claims
    """
    # For now, just call the sync version
    # In production, consider using asyncio.to_thread for true async
    return adjudicate_claims_with_search(sources_with_claims, model)


# ===== HELPER FUNCTIONS =====

def get_claims_from_sources(sources_with_claims: list) -> List[ExtractedClaim]:
    """
    extract all claims from DataSourceWithClaims objects.

    args:
        sources_with_claims: list of DataSourceWithClaims objects

    returns:
        flat list of all ExtractedClaim objects
    """
    all_claims = []
    for source_with_claims in sources_with_claims:
        for enriched_claim in source_with_claims.enriched_claims:
            # EnrichedClaim extends ExtractedClaim, so we can use it directly
            all_claims.append(enriched_claim)

    return all_claims
