"""
mapper functions to convert between API models and internal data models.

this module provides functions to transform API request/response models
into the internal pipeline data structures and vice versa.
"""

import uuid
from typing import List, cast
from datetime import datetime

from app.models.api import Request, ContentType
from app.models.commondata import DataSource
from app.models.factchecking import ClaimSourceType


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