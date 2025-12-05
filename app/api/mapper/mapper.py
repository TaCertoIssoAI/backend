"""
mapper functions to convert between API models and internal data models.

this module provides functions to transform API request/response models
into the internal pipeline data structures and vice versa.
"""

import uuid
from typing import List, cast
from datetime import datetime

from app.models.api import Request, ContentType,AnalysisResponse
from app.models.commondata import DataSource
from app.models.factchecking import ClaimSourceType,FactCheckResult
from app.observability.logger.logger import get_logger

from .formating import VEREDICT_SUBSTR,JUSTIFICATION_SUBSTR,CLAIM_SUBSTR, remove_link_like_substrings

logger = get_logger(__name__)


def map_content_type_to_source_type(content_type: ContentType) -> ClaimSourceType:
    """
    map API ContentType to internal ClaimSourceType.
    
    args:
        content_type: the content type from the API request
        
    returns:
        corresponding ClaimSourceType for internal use
    """
    mapping: dict[ContentType, ClaimSourceType] = {
        ContentType.TEXT: cast(ClaimSourceType, "original_text"),
        ContentType.IMAGE: cast(ClaimSourceType, "image"),
        ContentType.AUDIO: cast(ClaimSourceType, "audio_transcript"),
        ContentType.VIDEO: cast(ClaimSourceType, "video_transcript"),
    }
    
    return mapping[content_type]


def request_to_data_sources(
    request: Request,
    locale: str = "pt-BR"
) -> List[DataSource]:
    """
    convert an API Request into a list of DataSource objects.
    
    takes the unified request model from the API and transforms each
    ContentItem into a DataSource ready for the fact-checking pipeline.
    
    args:
        request: the API request containing content items
        locale: language locale for the data sources (default: pt-BR)
        
    returns:
        list of DataSource objects, one per content item
        
    example:
        >>> from app.models.api import Request, ContentItem, ContentType
        >>> request = Request(content=[
        ...     ContentItem(textContent="Neymar voltou ao Santos", type=ContentType.TEXT)
        ... ])
        >>> sources = request_to_data_sources(request)
        >>> len(sources)
        1
        >>> sources[0].source_type
        'original_text'
        >>> sources[0].original_text
        'Neymar voltou ao Santos'
    """
    data_sources: List[DataSource] = []
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    for content_item in request.content:
        # generate unique ID for this data source
        source_id = f"{uuid.uuid4()}"
        
        # map content type to source type
        source_type = map_content_type_to_source_type(content_item.type)
        
        # create DataSource
        data_source = DataSource(
            id=source_id,
            source_type=source_type,
            original_text=content_item.textContent,
            locale=locale,
            timestamp=timestamp,
        )
        
        data_sources.append(data_source)
    
    return data_sources


def fact_check_result_to_response(msg_id: uuid.UUID, result: FactCheckResult)->AnalysisResponse:
        all_verdicts = []
        for ds_result in result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        # build rationale text from verdicts
        if all_verdicts:
            rationale_parts = []

            # add overall summary first if present
            if result.overall_summary:
                rationale_parts.append(f"*Resumo Geral*:\n{result.overall_summary}")

            # add claims section header
            rationale_parts.append("\n\n*Análise por afirmação*:")

            # add each verdict with its justification
            for i, verdict_item in enumerate(all_verdicts, 1):
                rationale_parts.append(f"\n*{CLAIM_SUBSTR} {i}*: {verdict_item.claim_text}")
                rationale_parts.append(f"*{VEREDICT_SUBSTR}* {verdict_item.verdict}")
                rationale_parts.append(f"*{JUSTIFICATION_SUBSTR}* {verdict_item.justification}")

            # check sources_with_claims for debugging
            logger.debug(f"Number of sources_with_claims: {len(result.sources_with_claims)}")
            for idx, swc in enumerate(result.sources_with_claims):
                logger.debug(f"Source {idx}: {swc.data_source.id}, {len(swc.enriched_claims)} claims")
                for claim in swc.enriched_claims:
                    logger.debug(f"  Claim {claim.id}: {len(claim.citations)} citations")

            # build citation lookup map from sources_with_claims
            claim_citations_map = {}
            for source_with_claims in result.sources_with_claims:
                for enriched_claim in source_with_claims.enriched_claims:
                    claim_citations_map[enriched_claim.id] = enriched_claim.citations

            logger.debug(f"Built citation map with {len(claim_citations_map)} entries")
            logger.debug(f"Claim IDs in map: {list(claim_citations_map.keys())}")

            # collect all unique citations from verdicts
            all_citations = []
            citation_urls_seen = set()
            for verdict_item in all_verdicts:
                logger.debug(f"Looking up citations for claim_id: {verdict_item.claim_id}")
                # lookup citations by claim_id
                citations = claim_citations_map.get(verdict_item.claim_id, [])
                logger.debug(f"  Found {len(citations)} citations")
                for citation in citations:
                    # deduplicate by URL
                    if citation.url not in citation_urls_seen:
                        citation_urls_seen.add(citation.url)
                        all_citations.append(citation)

            logger.debug(f"Total unique citations collected: {len(all_citations)}")

            # build main rationale text first
            rationale = "\n".join(rationale_parts)

            # add only citations that were actually referenced in the text
            citation_text = _add_citations_to_final_msg(rationale, all_citations)
            if citation_text:
                rationale = rationale + citation_text
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."


        resp_without_links = remove_link_like_substrings(rationale)
        return AnalysisResponse(
            message_id=str(msg_id),
            rationale=rationale,
            responseWithoutLinks=resp_without_links,
        )

def _add_citations_to_final_msg(rationale_text: str, all_citations: list) -> str:
    """
    extract citation references from text and add only referenced citations.

    finds all [number] patterns in the rationale text and includes only those
    citations in the final sources list.

    args:
        rationale_text: the complete rationale text with [1], [2], etc. references
        all_citations: list of all available citations

    returns:
        formatted string with only referenced sources, or empty string if none
    """
    import re

    if not all_citations:
        return ""

    # find all [number] patterns in the text
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, rationale_text)

    # convert to set of unique numbers and sort
    referenced_numbers = sorted(set(int(num) for num in matches))

    if not referenced_numbers:
        return ""

    # build citation parts for only referenced sources
    citation_parts = ["\n\n*Fontes*:"]

    for num in referenced_numbers:
        # citations are 1-indexed in text, but 0-indexed in list
        idx = num - 1
        if 0 <= idx < len(all_citations):
            citation = all_citations[idx]
            citation_parts.append(f"\n[{num}] {citation.title}")
            citation_parts.append(f"    Fonte: {citation.publisher}")
            citation_parts.append(f"    URL: {citation.url}")
            if citation.date:
                citation_parts.append(f"    Data: {citation.date}")

    return "\n".join(citation_parts)