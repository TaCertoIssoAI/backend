"""
Hybrid link expander for testing without Apify API calls.

uses regex to extract URLs, mocks social media URLs (that would use Apify),
and allows real simple scraping for generic URLs.
"""

import re
from typing import List
from uuid import uuid4

from app.models import DataSource, PipelineConfig
from app.ai.context.web.apify_utils import detectPlatform, PlatformType
from app.ai.pipeline.link_context_expander import expand_link_contexts


# mock dictionary mapping URLs to their content
# this simulates what would be fetched from the web
MOCK_LINK_CONTENT = {
    # social media URLs (would use Apify - always mocked)
    "https://www.facebook.com/post/12345": {
        "title": "Facebook Post About Climate",
        "content": "Breaking: New climate study shows alarming trends. "
                   "Scientists worldwide are calling for immediate action.",
        "success": True
    },
    "https://www.instagram.com/p/abc123": {
        "title": "Instagram Post - Vaccine Information",
        "content": "Educational post about vaccine safety and efficacy. "
                   "Multiple peer-reviewed studies confirm safety profile.",
        "success": True
    },
    "https://twitter.com/user/status/987654": {
        "title": "Tweet About Renewable Energy",
        "content": "Solar energy costs have dropped dramatically. "
                   "Now cheaper than fossil fuels in most markets.",
        "success": True
    },
    "https://x.com/scientist/status/111222": {
        "title": "X Post - Scientific Study",
        "content": "New research published in Nature. "
                   "Groundbreaking findings on climate adaptation.",
        "success": True
    },
    "https://www.tiktok.com/@user/video/555666": {
        "title": "TikTok Video - Fact Check",
        "content": "Educational content debunking common misinformation. "
                   "Sources cited in video description.",
        "success": True
    },
    # generic URLs (would use simple scraping - included for backward compatibility)
    "https://example.com": {
        "title": "Example Domain",
        "content": "This domain is for use in illustrative examples in documents.",
        "success": True
    },
    "https://invalid-url.fake": {
        "title": None,
        "content": None,
        "success": False
    }
}

# default content for URLs not in the mock dictionary
DEFAULT_MOCK_CONTENT = {
    "title": "Mock Page Title",
    "content": "This is mock content for a URL not in the test dictionary. "
               "In a real scenario, this would be fetched from the web.",
    "success": True
}


def extract_urls_from_text(text: str) -> List[str]:
    """
    extract URLs from text using regex.

    args:
        text: text to extract URLs from

    returns:
        list of URLs found in the text
    """
    # regex pattern for URLs (http/https)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

    urls = re.findall(url_pattern, text)
    return urls


def hybrid_expand_link_contexts(
    data_source: DataSource,
    config: PipelineConfig
) -> List[DataSource]:
    """
    hybrid implementation: mocks social media URLs, uses real scraping for generic URLs.

    social media URLs (Facebook, Instagram, Twitter, TikTok) would use Apify API,
    so they are mocked using a dictionary. generic URLs use real simple HTTP scraping
    from the actual link_context_expander implementation.

    args:
        data_source: data source to extract links from
        config: pipeline configuration

    returns:
        list of new 'link_context' data sources (mocked + real)
    """
    if not data_source.original_text:
        return []

    # extract URLs from text
    urls = extract_urls_from_text(data_source.original_text)

    if not urls:
        return []

    expanded_sources: List[DataSource] = []

    # separate URLs into social media (mock) and generic (real scraping)
    social_media_urls = []
    generic_urls = []

    for url in urls:
        platform = detectPlatform(url)
        if platform in [PlatformType.FACEBOOK, PlatformType.INSTAGRAM,
                       PlatformType.TWITTER, PlatformType.TIKTOK]:
            social_media_urls.append(url)
        else:
            generic_urls.append(url)

    # process social media URLs with mocks
    for url in social_media_urls:
        # get mock content for this URL (or use default)
        mock_data = MOCK_LINK_CONTENT.get(url, DEFAULT_MOCK_CONTENT)

        # create metadata
        metadata = {
            "url": url,
            "success": mock_data["success"],
            "title": mock_data["title"],
            "mock": True  # flag to indicate this is mock data
        }

        # create link_context data source
        link_source = DataSource(
            id=f"link-{uuid4().hex[:8]}",
            source_type="link_context",
            original_text=mock_data["content"] if mock_data["success"] else None,
            metadata=metadata,
            parent_source_id=data_source.id
        )

        expanded_sources.append(link_source)

    # process generic URLs with real scraping
    if generic_urls:
        # create temporary data source with only generic URLs
        generic_text = " ".join(generic_urls)
        temp_source = DataSource(
            id=data_source.id,
            source_type=data_source.source_type,
            original_text=generic_text,
            metadata=data_source.metadata,
            parent_source_id=data_source.parent_source_id
        )

        # call real link expansion for generic URLs
        real_expanded = expand_link_contexts(temp_source, config)

        if real_expanded:
            # update parent_source_id to match original
            for source in real_expanded:
                source.parent_source_id = data_source.id
            expanded_sources.extend(real_expanded)

    return expanded_sources
