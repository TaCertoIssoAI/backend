"""
Link Context Expander Step for the Fact-Checking Pipeline.

This module is responsible for extracting links from an 'original_text' DataSource
and transforming each one into a 'link_context' DataSource with expanded content.

Architecture:
- Receives a DataSource of type 'original_text'
- Extracts all URLs from the text
- Expands each link to get its content using web scraping
- Returns a list of new DataSources of type 'link_context'
"""
import re
import uuid
from typing import List

from app.models import DataSource
from app.ai.context.web.apify_utils import scrapeGenericUrl
from app.ai.context.web.models import WebContentResult


def extract_links(text: str) -> List[str]:
    """
    Extract all URLs from text using regex.

    Supports http, https protocols and common URL patterns.
    Returns list of unique URLs found in the text.

    Args:
        text: The text to extract links from

    Returns:
        List of unique URLs found in the text, preserving order

    Example:
        >>> extract_links("Check this out: https://example.com and https://test.com")
        ['https://example.com', 'https://test.com']
    """
    # regex pattern for URLs with http/https
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

    # find all matches
    urls = re.findall(url_pattern, text)
    
    # strip common trailing punctuation that's not part of URLs
    # common punctuation: . , ; : ! ? ) ] } that often follow URLs in text
    trailing_punctuation = '.,:;!?)]}'
    cleaned_urls = []
    for url in urls:
        # strip trailing punctuation
        while url and url[-1] in trailing_punctuation:
            url = url[:-1]
        if url:  # only add non-empty URLs
            cleaned_urls.append(url)

    # remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in cleaned_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


async def expand_link_context(url: str) -> WebContentResult:
    """
    Expand a link and extract its content using web scraping.

    Uses the scrapeGenericUrl function to fetch and parse content from the URL.
    Handles different platforms (social media, generic websites) automatically.

    Args:
        url: The URL to expand and extract content from

    Returns:
        WebContentResult with the scraped content and metadata

    Note:
        This function:
        - Detects platform automatically (Facebook, Instagram, Twitter, TikTok, generic)
        - Tries simple HTTP scraping first for generic sites (no browser, faster)
        - Falls back to Apify actor with browser if simple scraping fails
        - Handles errors (404, timeouts, etc.) gracefully
        - Supports different content types (articles, social media posts, etc.)
        - Processing time is measured by the scraping functions and included in result
    """
    # call the scraping function (processing time is measured internally)
    result_dict = await scrapeGenericUrl(url)

    # parse the result dict into WebContentResult schema
    result = WebContentResult.from_dict(data=result_dict, url=url)

    return result


async def expand_link_contexts(data_source: DataSource) -> List[DataSource]:
    """
    Main async function to expand link contexts from an original_text DataSource.

    Takes a DataSource of type 'original_text', extracts all links from it,
    expands each link to get its content using web scraping, and returns a list
    of new DataSources of type 'link_context'.

    Args:
        data_source: Input DataSource that must be of type 'original_text'

    Returns:
        List of DataSources, one for each link found, with type 'link_context'

    Raises:
        ValueError: If the input DataSource is not of type 'original_text'

    Example:
        >>> from app.models import DataSource
        >>> import asyncio
        >>> original = DataSource(
        ...     id="msg-001",
        ...     source_type="original_text",
        ...     original_text="Check out https://example.com for more info"
        ... )
        >>> expanded = asyncio.run(expand_link_contexts(original))
        >>> len(expanded)
        1
        >>> expanded[0].source_type
        'link_context'
        >>> expanded[0].metadata['url']
        'https://example.com'
    """
    # validate input DataSource type
    if data_source.source_type != "original_text":
        raise ValueError(
            f"expand_link_contexts expects a DataSource of type 'original_text', "
            f"but received type '{data_source.source_type}'"
        )

    # extract links from the text
    links = extract_links(data_source.original_text)

    # if no links found, return empty list
    if not links:
        return []

    # expand each link and create new DataSources
    expanded_sources: List[DataSource] = []

    for url in links:
        # call the scraping function to get expanded content
        web_result = await expand_link_context(url)

        # create metadata dict from web result
        metadata = {
            "success": web_result.success,
            "url": web_result.url,
            "content_length": web_result.content_length,
            "processing_time_ms": web_result.processing_time_ms,
            "parent_source_id": data_source.id,
        }

        # add social media metadata if available
        if web_result.metadata:
            metadata["platform"] = web_result.metadata.platform
            metadata["author"] = web_result.metadata.author
            metadata["timestamp"] = web_result.metadata.timestamp
            metadata["likes"] = web_result.metadata.likes
            metadata["shares"] = web_result.metadata.shares
            metadata["comments"] = web_result.metadata.comments

        # add error to metadata if scraping failed
        if web_result.error:
            metadata["error"] = web_result.error

        # create a new DataSource for this link
        link_source = DataSource(
            id=f"link-{uuid.uuid4()}",
            source_type="link_context",
            original_text=web_result.content if web_result.success else "",
            metadata=metadata,
            locale=data_source.locale,
            timestamp=data_source.timestamp,
        )

        expanded_sources.append(link_source)

    return expanded_sources