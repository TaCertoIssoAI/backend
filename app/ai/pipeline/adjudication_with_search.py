"""
This is a fallback/experimental step of the pipeline and it defines a mix of the Evidence Retrieval + Adjundication step
Adjudication with Web Search - Pipeline Step.

This module provides an alternative adjudication step that uses OpenAI's Responses API
with web search to fact-check claims in real-time.

Key differences from standard adjudication:
- Uses OpenAI web search instead of pre-gathered citations
- Combines evidence gathering and adjudication in one step
- Single API call with structured output

Architecture:
- Uses OpenAI SDK with Responses API
- Single API call with web search tool that returns structured Pydantic output
- Type-safe with Pydantic models throughout
"""

import os
import re
import json
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from app.models import (
    ExtractedClaim,
    FactCheckResult,
    LLMAdjudicationOutput,
    DataSource,
    DataSourceWithExtractedClaims,
)

from .prompts import ADJUDICATION_WITH_SEARCH_SYSTEM_PROMPT
from .utils import get_current_date, convert_llm_output_to_data_source_results

# ===== JSON REPAIR UTILITIES =====

def _repair_json_urls(json_text: str) -> str:
    """
    repair common JSON formatting issues in URLs.

    fixes:
    - unescaped backslashes in URLs
    - unescaped quotes in URLs
    - newlines and tabs in string values
    - malformed URL strings

    args:
        json_text: raw JSON string that may contain malformed URLs

    returns:
        repaired JSON string
    """
    # fix unescaped newlines and tabs in strings
    json_text = json_text.replace('\n', '\\n').replace('\t', '\\t')

    # fix URLs with unescaped backslashes (common issue)
    # find all URL patterns and ensure backslashes are escaped
    url_pattern = r'"url"\s*:\s*"([^"]*)"'

    def fix_url(match):
        url = match.group(1)
        # escape any unescaped backslashes
        url = url.replace('\\', '\\\\')
        # remove any double escaping
        url = url.replace('\\\\\\\\', '\\\\')
        return f'"url": "{url}"'

    json_text = re.sub(url_pattern, fix_url, json_text)

    # fix any remaining unescaped quotes in string values (conservative approach)
    # this is tricky, so we only fix obvious cases

    return json_text


def _parse_with_fallback(raw_response: str) -> LLMAdjudicationOutput:
    """
    parse JSON response with fallback repair logic.

    tries to parse the response as-is first, then applies JSON repair
    if initial parsing fails.

    args:
        raw_response: raw JSON string from the API

    returns:
        parsed LLMAdjudicationOutput

    raises:
        ValueError: if parsing fails even after repair attempts
    """
    # try parsing as-is first
    try:
        parsed_data = json.loads(raw_response)
        return LLMAdjudicationOutput(**parsed_data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[DEBUG] Initial JSON parsing failed: {type(e).__name__}: {str(e)}")
        print("[DEBUG] Attempting JSON repair...")

        # apply repairs
        repaired_json = _repair_json_urls(raw_response)

        try:
            parsed_data = json.loads(repaired_json)
            result = LLMAdjudicationOutput(**parsed_data)
            print("[DEBUG] JSON repair successful!")
            return result
        except (json.JSONDecodeError, ValidationError) as repair_error:
            print(f"[ERROR] JSON repair failed: {type(repair_error).__name__}: {str(repair_error)}")
            print(f"[DEBUG] Original JSON (first 500 chars): {raw_response[:500]}")
            print(f"[DEBUG] Repaired JSON (first 500 chars): {repaired_json[:500]}")
            raise ValueError(
                f"Failed to parse JSON response even after repair. "
                f"Original error: {str(e)}. "
                f"Repair error: {str(repair_error)}"
            ) from repair_error


# ===== CLIENT INITIALIZATION =====

def _get_openai_client() -> OpenAI:
    """
    get or create OpenAI client.

    returns:
        configured OpenAI instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return OpenAI(api_key=api_key)


# ===== PROMPTS =====

def _build_adjudication_prompt(sources_with_claims: List[DataSourceWithExtractedClaims], current_date: str) -> str:
    """
    build the prompt for adjudication with search.

    args:
        sources_with_claims: list of data sources with their extracted claims
        current_date: current date in DD-MM-YYYY format

    returns:
        formatted prompt string
    """
    prompt_parts = []
    prompt_parts.append("\n\n## Alegações para Verificar:\n")

    # group claims by data source
    for source_idx, source_with_claims in enumerate(sources_with_claims, 1):
        data_source_id = source_with_claims.data_source.id
        claims = source_with_claims.extracted_claims

        prompt_parts.append(f"\n### Fonte de Dados {source_idx}: {data_source_id}\n")

        for claim_idx, claim in enumerate(claims, 1):
            prompt_parts.append(f"\n**Alegação {claim_idx}**:")
            prompt_parts.append(f"- ID: {claim.id}")
            prompt_parts.append(f"- Texto: {claim.text}")
            prompt_parts.append("")

    prompt_parts.append("\nPor favor, verifique cada alegação usando a busca do Google e forneça vereditos fundamentados.")
    prompt_parts.append("\nIMPORTANTE: Agrupe os vereditos por fonte de dados. Para cada fonte, retorne UM resultado com todos os vereditos daquela fonte.")

    return "\n".join(prompt_parts)


# ===== MAIN ADJUDICATION FUNCTION =====

def adjudicate_claims_with_search(
    sources_with_claims: List[DataSourceWithExtractedClaims],
    model: str = "gpt-5-nano"
) -> FactCheckResult:
    """
    adjudicate claims using OpenAI web search in a single API call.

    this is the main entry point for adjudication with search.

    the function uses OpenAI Responses API with web search tool to find evidence
    and returns structured Pydantic output in one call.

    args:
        sources_with_claims: list of DataSourceWithExtractedClaims to fact-check
        model: OpenAI model to use (default: gpt-4o-mini)

    returns:
        FactCheckResult with verdicts for all claims

    raises:
        ValueError: if OPENAI_API_KEY is not set
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
    print("[DEBUG] Starting adjudicate_claims_with_search (OpenAI)")
    print("="*80)

    # Input validation and logging
    print(f"[DEBUG] Input: {len(sources_with_claims)} sources with claims")
    for idx, source in enumerate(sources_with_claims, 1):
        print(f"  Source {idx}: {source.data_source.id} with {len(source.extracted_claims)} claims")

    client = _get_openai_client()
    print("[DEBUG] OpenAI client initialized successfully")

    # Get current date
    current_date = get_current_date()
    print(f"[DEBUG] Current date: {current_date}")

    # Count total claims for logging
    total_claims = sum(len(s.extracted_claims) for s in sources_with_claims)
    print(f"[DEBUG] Total claims to adjudicate: {total_claims}")

    for source in sources_with_claims:
        print(f"  Source {source.data_source.id}:")
        for claim in source.extracted_claims:
            print(f"    - [{claim.id}] {claim.text[:80]}...")

    # Build the user message with claims grouped by data source
    user_message = _build_adjudication_prompt(sources_with_claims, current_date)
    print(f"[DEBUG] Prompt built, length: {len(user_message)} characters")
    print(f"[DEBUG] Prompt preview (first 200 chars):\n{user_message[:200]}...")

    # Prepare messages for OpenAI
    messages = [
        {
            "role": "system",
            "content": ADJUDICATION_WITH_SEARCH_SYSTEM_PROMPT.format(current_date=current_date)
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Make single API call with web search and structured output
    print(f"[DEBUG] Calling OpenAI Responses API with model: {model}")

    llm_output = None
    try:
        response = client.responses.parse(
            model=model,
            input=messages,
            tools=[{"type": "web_search"}],
            text_format=LLMAdjudicationOutput,
        )
        print("[DEBUG] API call successful")

        # Parse structured output
        print("\n[DEBUG] Parsing structured output...")

        if not hasattr(response, 'output_parsed') or response.output_parsed is None:
            # fallback: try to get raw output and parse manually
            if hasattr(response, 'output') and response.output:
                print("[DEBUG] output_parsed is None, attempting manual parsing from raw output")
                llm_output = _parse_with_fallback(response.output)
            else:
                print("[ERROR] response.output_parsed is None and no raw output available!")
                print(f"[DEBUG] Response object: {response}")
                raise ValueError("API response output_parsed is None - no content returned")
        else:
            llm_output = response.output_parsed

    except (json.JSONDecodeError, ValidationError) as parse_error:
        # catch JSON parsing errors and attempt repair
        print(f"[ERROR] JSON parsing failed: {type(parse_error).__name__}: {str(parse_error)}")

        # try to get raw response text
        if 'response' in locals() and hasattr(response, 'output') and response.output:
            print("[DEBUG] Attempting to repair malformed JSON from raw output")
            llm_output = _parse_with_fallback(response.output)
        else:
            print("[ERROR] Cannot access raw response for repair")
            raise ValueError(
                f"Failed to parse structured output and cannot access raw response for repair. "
                f"Error: {str(parse_error)}"
            ) from parse_error

    except Exception as e:
        print(f"[ERROR] API call failed: {type(e).__name__}: {str(e)}")
        raise

    if llm_output is None:
        raise ValueError("Failed to obtain parsed output from API response")
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
            print(f"      citations used : {verdict.citations_used}")
    
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
    model: str = "gpt-4o-mini"
) -> FactCheckResult:
    """
    async version of adjudicate_claims_with_search.

    note: uses sync calls in an async wrapper. For true async, consider
    using asyncio.to_thread or similar.

    args:
        sources_with_claims: list of DataSourceWithExtractedClaims to fact-check
        model: OpenAI model to use

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
