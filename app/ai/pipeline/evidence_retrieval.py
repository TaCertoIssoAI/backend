"""
Evidence Retrieval Step for the Fact-Checking Pipeline.

This module gathers evidence from multiple sources to support or refute claims.
Designed to be composable - new evidence sources can be easily added.

Architecture:
- Receives an EvidenceRetrievalInput containing a list of ExtractedClaims
- For each claim, runs it through multiple evidence gatherers (web search, fact-check APIs, etc.)
- Each gatherer returns zero or more Citations
- Returns an EvidenceRetrievalResult mapping claim IDs to EnrichedClaims with citations

Key Design Principles:
- No LLM calls in this step (pure retrieval)
- Composable architecture via EvidenceGatherer protocol
- Stateless design with explicit state passing
- Type annotations throughout
- Support for both sync and async operations
"""

import asyncio
from typing import List, Dict

from app.models import (
    EvidenceRetrievalInput,
    EvidenceRetrievalResult,
    EnrichedClaim,
    Citation,
)
from app.ai.context import (
    EvidenceGatherer
)
from app.ai.context.web import (
    WebSearchGatherer
)
from app.observability.logger.logger import get_logger

logger = get_logger(__name__)



# ===== MAIN EVIDENCE RETRIEVAL FUNCTIONS =====

async def gather_evidence_async(
    retrieval_input: EvidenceRetrievalInput,
    gatherers: List[EvidenceGatherer] | None = None,
    timeout_per_gatherer: float = 45.0
) -> EvidenceRetrievalResult:
    """
    Main async function to gather evidence for all claims.

    For each claim, runs it through all evidence gatherers and accumulates
    citations. Returns a mapping of claim IDs to enriched claims with evidence.

    Args:
        retrieval_input: Input containing list of claims to gather evidence for
        gatherers: List of evidence gatherers to use. If None, uses default (web search only).

    Returns:
        EvidenceRetrievalResult with claim_evidence_map containing enriched claims

    Example:
        >>> from app.models import EvidenceRetrievalInput, ExtractedClaim, ClaimSource
        >>> claim = ExtractedClaim(
        ...     id="claim-123",
        ...     text="Vaccine X causes infertility",
        ...     source=ClaimSource(source_type="original_text", source_id="msg-001")
        ... )
        >>> input_data = EvidenceRetrievalInput(claims=[claim])
        >>> result = await gather_evidence_async(input_data)
        >>> enriched = result.claim_evidence_map["claim-123"]
        >>> len(enriched.citations) > 0
        True
    """
    # use default gatherers if none provided
    if gatherers is None:
        gatherers = [WebSearchGatherer(max_results=5)]

    # initialize result map (maps claim id to its enriched claim)
    claim_evidence_map: Dict[str, EnrichedClaim] = {}

    # process each claim
    for claim in retrieval_input.claims:
        logger.info(f"processing claim: {claim.text[:80]}...")

        # gather citations from all sources
        all_citations: List[Citation] = []

        for gatherer in gatherers:
            gatherer_name = gatherer.source_name if hasattr(gatherer, 'source_name') else type(gatherer).__name__

            try:
                logger.debug(f"querying {gatherer_name}...")
                citations = await gatherer.gather(claim)

                if citations:
                    logger.info(f"{gatherer_name}: {len(citations)} citation(s) found")
                else:
                    logger.warning(f"{gatherer_name}: no citations found")

                all_citations.extend(citations)

            except asyncio.TimeoutError:
                logger.error(f"{gatherer_name}: TIMEOUT - operation exceeded time limit")
            except Exception as e:
                logger.error(f"{gatherer_name}: ERROR - {type(e).__name__}: {str(e)[:100]}")

        logger.info(f"total citations for this claim: {len(all_citations)}")

        # create enriched claim with citations
        # EnrichedClaim extends ExtractedClaim, so we copy all fields
        enriched_claim = EnrichedClaim(
            id=claim.id,
            text=claim.text,
            source=claim.source,
            llm_comment=claim.llm_comment,
            entities=claim.entities,
            citations=all_citations
        )

        # add to result map
        claim_evidence_map[claim.id] = enriched_claim

    return EvidenceRetrievalResult(claim_evidence_map=claim_evidence_map)


# ===== HELPER FUNCTIONS =====

def deduplicate_citations(citations: List[Citation]) -> List[Citation]:
    """
    Remove duplicate citations based on URL.

    If multiple citations have the same URL, keeps the first one.

    Args:
        citations: List of citations that may contain duplicates

    Returns:
        Deduplicated list of citations
    """
    if not citations:
        return []

    seen_urls: set[str] = set()
    deduplicated: List[Citation] = []

    for citation in citations:
        # normalize URL for comparison
        normalized_url = citation.url.lower().strip()

        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            deduplicated.append(citation)

    return deduplicated


def filter_low_quality_citations(
    citations: List[Citation],
    min_text_length: int = 10
) -> List[Citation]:
    """
    Filter out low-quality citations.

    Removes citations with:
    - Very short citation text
    - Missing critical fields

    Args:
        citations: List of citations to filter
        min_text_length: Minimum length for citation_text

    Returns:
        Filtered list of citations
    """
    if not citations:
        return []

    filtered: List[Citation] = []

    for citation in citations:
        # skip if citation text is too short
        if len(citation.citation_text.strip()) < min_text_length:
            continue

        # skip if critical fields are missing
        if not citation.url or not citation.title:
            continue

        filtered.append(citation)

    return filtered


# ===== CONVENIENCE FUNCTION =====

async def gather_and_filter_evidence(
    retrieval_input: EvidenceRetrievalInput,
    gatherers: List[EvidenceGatherer] | None  = None,
    deduplicate: bool = True,
    filter_quality: bool = True
) -> EvidenceRetrievalResult:
    """
    Gathers evidence and applies quality filters in one call.

    This is the recommended entry point for most use cases.
    Automatically deduplicates and filters low-quality citations.

    Args:
        retrieval_input: Input containing list of claims to gather evidence for
        gatherers: List of evidence gatherers to use. If None, uses default.
        deduplicate: Whether to remove duplicate citations
        filter_quality: Whether to filter out low-quality citations

    Returns:
        EvidenceRetrievalResult with filtered and deduplicated citations
    """
    # gather evidence
    result = await gather_evidence_async(
        retrieval_input=retrieval_input,
        gatherers=gatherers
    )

    # apply filters if requested
    if deduplicate or filter_quality:
        for enriched_claim in result.claim_evidence_map.values():
            citations = enriched_claim.citations

            if deduplicate:
                citations = deduplicate_citations(citations)

            if filter_quality:
                citations = filter_low_quality_citations(citations)

            # update the enriched claim with filtered citations
            enriched_claim.citations = citations

    return result
