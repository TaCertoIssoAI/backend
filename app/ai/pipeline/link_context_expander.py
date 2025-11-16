"""
Link Context Expander Step for the Fact-Checking Pipeline.

This module is responsible for extracting links from an 'original_text' DataSource
and transforming each one into a 'link_context' DataSource with expanded content.

Architecture:
- Receives a DataSource of type 'original_text'
- Extracts all URLs from the text
- Expands each link to get its content (currently mocked, will be implemented later)
- Returns a list of new DataSources of type 'link_context'
"""
import re
import uuid
from typing import List

from app.models import DataSource


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

    # remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def expand_link_context(url: str) -> str:
    """
    Mock function to expand a link and extract its content.

    This is a placeholder that will be replaced with actual implementation
    (web scraping, API calls, etc.) in the future.

    Args:
        url: The URL to expand and extract content from

    Returns:
        Expanded context/content from the URL as a string

    Note:
        Currently returns a mock response. Future implementation will:
        - Fetch the URL content
        - Extract main text content (removing ads, navigation, etc.)
        - Handle errors (404, timeouts, etc.)
        - Support different content types (articles, PDFs, etc.)
    """
    # Mock implementation - will be replaced with actual web scraping
    return f"[MOCK] Expanded content from {url}. This would contain the actual article text, title, and relevant information extracted from the webpage."


def expand_link_contexts(data_source: DataSource) -> List[DataSource]:
    """
    Main function to expand link contexts from an original_text DataSource.

    Takes a DataSource of type 'original_text', extracts all links from it,
    expands each link to get its content, and returns a list of new DataSources
    of type 'link_context'.

    Args:
        data_source: Input DataSource that must be of type 'original_text'

    Returns:
        List of DataSources, one for each link found, with type 'link_context'

    Raises:
        ValueError: If the input DataSource is not of type 'original_text'

    Example:
        >>> from app.models import DataSource
        >>> original = DataSource(
        ...     id="msg-001",
        ...     source_type="original_text",
        ...     original_text="Check out https://example.com for more info"
        ... )
        >>> expanded = expand_link_contexts(original)
        >>> len(expanded)
        1
        >>> expanded[0].source_type
        'link_context'
        >>> expanded[0].metadata['url']
        'https://example.com'
    """
    # Validate input DataSource type
    if data_source.source_type != "original_text":
        raise ValueError(
            f"expand_link_contexts expects a DataSource of type 'original_text', "
            f"but received type '{data_source.source_type}'"
        )

    # Extract links from the text
    links = extract_links(data_source.original_text)

    # If no links found, return empty list
    if not links:
        return []

    # Expand each link and create new DataSources
    expanded_sources: List[DataSource] = []

    for url in links:
        # Call the mock function to get expanded content
        expanded_content = expand_link_context(url)

        # Create a new DataSource for this link
        link_source = DataSource(
            id=f"link-{uuid.uuid4()}",
            source_type="link_context",
            original_text=expanded_content,
            metadata={
                "url": url,
                "parent_source_id": data_source.id,
            },
            locale=data_source.locale,
            timestamp=data_source.timestamp,
        )

        expanded_sources.append(link_source)

    return expanded_sources