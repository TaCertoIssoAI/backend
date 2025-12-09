from datetime import datetime, timezone
from typing import List, Optional, Union

from app.models import (
    DataSourceResult,
    ClaimVerdict,
    DataSourceWithClaims,
    DataSourceWithExtractedClaims,
    LLMDataSourceResult,
    ExtractedClaim,
    EnrichedClaim,
)


# date format for fact-checking context: DD-MM-YYYY
DATE_FORMAT = "%d-%m-%Y"


# ===== DATE UTILITIES =====

def get_current_date() -> str:
    """
    Returns the current date in DD-MM-YYYY format using UTC timezone.

    Returns:
        Formatted date string (e.g., "08-12-2024")
    """
    now = datetime.now(timezone.utc)
    return now.strftime(DATE_FORMAT)


# ===== LLM OUTPUT CONVERSION UTILITIES =====

def get_data_source_with_claims(
    llm_source_result: LLMDataSourceResult,
    sources_with_claims: List[Union[DataSourceWithClaims, DataSourceWithExtractedClaims]],
    result_index: int
) -> Optional[Union[DataSourceWithClaims, DataSourceWithExtractedClaims]]:
    """
    Matches an LLM data source result back to the original input.

    Works with both DataSourceWithClaims (with citations) and
    DataSourceWithExtractedClaims (without citations).

    Uses hybrid matching strategy:
    1. Try to match by data_source_id (if provided by LLM)
    2. Fall back to matching by position/order

    Args:
        llm_source_result: LLM output for one data source
        sources_with_claims: List of original sources with claims (either type)
        result_index: Position of this result in the LLM output list

    Returns:
        Matched DataSourceWithClaims or DataSourceWithExtractedClaims, or None if no match found
    """
    # Create mapping of data_source_id to original source_with_claims
    source_map = {
        source_with_claims.data_source.id: source_with_claims
        for source_with_claims in sources_with_claims
    }

    # Try to match by data_source_id first
    if llm_source_result.data_source_id:
        source_with_claims = source_map.get(llm_source_result.data_source_id)
        if source_with_claims:
            return source_with_claims

        print(f"[WARNING] LLM returned unknown data_source_id: {llm_source_result.data_source_id}")

    # Fallback: match by order (position in list)
    print(f"[INFO] data_source_id missing for result {result_index}, matching by order")
    if result_index < len(sources_with_claims):
        return sources_with_claims[result_index]

    print(f"[WARNING] No source at index {result_index}")
    return None


def get_claim_verdicts(
    llm_source_result: LLMDataSourceResult,
    source_with_claims: Union[DataSourceWithClaims, DataSourceWithExtractedClaims]
) -> List[ClaimVerdict]:
    """
    Converts LLM claim verdicts to ClaimVerdict objects with proper IDs.

    Works with both DataSourceWithClaims (with citations) and
    DataSourceWithExtractedClaims (without citations).

    Uses hybrid matching strategy for claim IDs:
    1. Try to use claim_id from LLM output (if provided and valid)
    2. Fall back to matching by claim_text

    Args:
        llm_source_result: LLM output for one data source
        source_with_claims: Original input for this data source (either type)

    Returns:
        List of ClaimVerdict objects with proper claim_id populated
    """
    # Get claims list regardless of model type
    claims = (
        source_with_claims.enriched_claims
        if isinstance(source_with_claims, DataSourceWithClaims)
        else source_with_claims.extracted_claims
    )

    # Create mappings for claim matching
    claim_id_by_id = {claim.id: claim for claim in claims}
    claim_id_by_text = {claim.text: claim.id for claim in claims}

    # Convert LLM verdicts to ClaimVerdict objects
    claim_verdicts: List[ClaimVerdict] = []
    for llm_verdict in llm_source_result.claim_verdicts:
        # Try to get claim_id: first from LLM output, then from claim_text matching
        if llm_verdict.claim_id and llm_verdict.claim_id in claim_id_by_id:
            # Use claim_id from LLM (most reliable)
            claim_id = llm_verdict.claim_id
        else:
            # Fallback: match by claim_text
            claim_id = claim_id_by_text.get(llm_verdict.claim_text, "unknown")
            if llm_verdict.claim_id:
                print(f"[WARNING] LLM returned unknown claim_id: {llm_verdict.claim_id}, matched by text instead")

        verdict = ClaimVerdict(
            claim_id=claim_id,
            claim_text=llm_verdict.claim_text,
            verdict=llm_verdict.verdict,
            justification=llm_verdict.justification,
            citations_used=llm_verdict.citations_used
        )
        claim_verdicts.append(verdict)

    return claim_verdicts


def convert_llm_output_to_data_source_results(
    llm_results: List[LLMDataSourceResult],
    sources_with_claims: List[Union[DataSourceWithClaims, DataSourceWithExtractedClaims]]
) -> List[DataSourceResult]:
    """
    Converts LLM adjudication output to DataSourceResult objects.

    Works with both DataSourceWithClaims (with citations) and
    DataSourceWithExtractedClaims (without citations).

    This function processes the raw LLM output and matches it back to the original
    input sources, creating properly structured DataSourceResult objects with
    correct IDs and metadata.

    Args:
        llm_results: List of LLM output results (one per data source)
        sources_with_claims: List of original sources with their claims (either type)

    Returns:
        List of DataSourceResult objects ready to be included in FactCheckResult
    """
    data_source_results: List[DataSourceResult] = []

    # Process each LLM result
    for idx, llm_source_result in enumerate(llm_results):
        # Match LLM result to original input data source
        source_with_claims = get_data_source_with_claims(
            llm_source_result=llm_source_result,
            sources_with_claims=sources_with_claims,
            result_index=idx
        )
        if not source_with_claims:
            print(f"[WARNING] No source_with_claims match found for result {idx}")
            continue  # Skip if no match found

        # Convert LLM verdicts to ClaimVerdict objects with proper IDs
        claim_verdicts = get_claim_verdicts(
            llm_source_result=llm_source_result,
            source_with_claims=source_with_claims
        )

        # Create DataSourceResult with info from original input
        source_result = DataSourceResult(
            data_source_id=source_with_claims.data_source.id,
            source_type=source_with_claims.data_source.source_type,
            claim_verdicts=claim_verdicts
        )
        data_source_results.append(source_result)

    return data_source_results