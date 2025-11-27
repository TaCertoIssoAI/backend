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
            rationale_parts = ["Resultado da verificação:\n"]
            for i, verdict_item in enumerate(all_verdicts, 1):
                rationale_parts.append(f"\n{i}. Alegação: {verdict_item.claim_text}")
                rationale_parts.append(f"Veredito: {verdict_item.verdict}")
                rationale_parts.append(f"Justificativa: {verdict_item.justification}")

            # add overall summary if present
            if result.overall_summary:
                rationale_parts.append(f"\n\nResumo Geral:\n{result.overall_summary}")

            rationale = "\n".join(rationale_parts)
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."

        return AnalysisResponse(
            message_id=str(msg_id),
            rationale=rationale,
            responseWithoutLinks=rationale,
        )