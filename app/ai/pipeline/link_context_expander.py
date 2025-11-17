"""
Link Context Expander Step for the Fact-Checking Pipeline.

This module is responsible for extracting links from an 'original_text' DataSource
and transforming each one into a 'link_context' DataSource with expanded content.

Architecture:
- Receives a DataSource of type 'original_text'
- Extracts all URLs from the text
- Expands each link to get its content using web scraping (in parallel using ThreadPool)
- Returns a list of new DataSources of type 'link_context'
- Enforces timeouts from PipelineConfig
"""
import re
import uuid
import asyncio
import logging
from typing import List, Optional
from app.ai.threads.thread_utils import wait_all
from app.models import DataSource, PipelineConfig
from app.ai.context.web.apify_utils import scrapeGenericUrl
from app.ai.context.web.models import WebContentResult
from app.ai.threads.thread_utils import ThreadPoolManager, OperationType

logger = logging.getLogger(__name__)


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


def expand_link_context_sync(url: str, timeout_per_link: float) -> Optional[WebContentResult]:
    """
    Synchronous wrapper for expand_link_context to be used in thread pool.

    This function runs the async expand_link_context in a new event loop,
    making it suitable for execution in worker threads.

    Args:
        url: The URL to expand and extract content from
        timeout_per_link: Timeout in seconds for this link expansion

    Returns:
        WebContentResult if successful, None if timeout or error occurs
    """
    import time
    start_time = time.time()

    logger.info(f"[SYNC] Starting scrape: {url[:80]}...")

    try:
        # create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # run async function with timeout
            result = loop.run_until_complete(
                asyncio.wait_for(
                    expand_link_context(url),
                    timeout=timeout_per_link
                )
            )

            elapsed = time.time() - start_time
            logger.info(
                f"[SYNC] ✅ Success: {url[:60]}... | "
                f"time={elapsed:.2f}s | content={result.content_length} chars | "
                f"success={result.success}"
            )
            return result
        finally:
            loop.close()

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(
            f"[SYNC] ⏱️ TIMEOUT: {url[:60]}... | "
            f"limit={timeout_per_link}s | elapsed={elapsed:.2f}s"
        )
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[SYNC] ❌ ERROR: {url[:60]}... | "
            f"elapsed={elapsed:.2f}s | error={type(e).__name__}: {str(e)[:100]}",
            exc_info=True
        )
        return None


async def expand_link_contexts(
    data_source: DataSource,
    config: PipelineConfig
) -> List[DataSource]:
    """
    Main function to expand link contexts from an original_text DataSource in parallel.

    Takes a DataSource of type 'original_text', extracts all links from it,
    expands each link to get its content using web scraping IN PARALLEL using ThreadPool,
    and returns a list of new DataSources of type 'link_context'.

    Uses ThreadPoolManager with OperationType.LINK_CONTEXT_EXPANDING for priority-based
    parallel execution. Results are collected as they complete for maximum throughput.

    Args:
        data_source: Input DataSource that must be of type 'original_text'
        config: Pipeline configuration with timeout and limit settings

    Returns:
        List of DataSources, one for each successfully expanded link

    Raises:
        ValueError: If the input DataSource is not of type 'original_text'

    Example:
        >>> from app.models import DataSource
        >>> from app.config.default import get_default_pipeline_config
        >>> import asyncio
        >>> original = DataSource(
        ...     id="msg-001",
        ...     source_type="original_text",
        ...     original_text="Check out https://example.com for more info"
        ... )
        >>> config = get_default_pipeline_config()
        >>> expanded = asyncio.run(expand_link_contexts(original, config))
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
        logger.info("no links found in original text")
        return []

    # limit number of links based on config
    original_count = len(links)
    links = links[:config.max_links_to_expand]
    if original_count > len(links):
        logger.info(
            f"limiting link expansion from {original_count} to {len(links)} "
            f"(max_links_to_expand={config.max_links_to_expand})"
        )

    logger.info(f"expanding {len(links)} links in parallel using ThreadPool")

    # get thread pool manager instance
    manager = ThreadPoolManager.get_instance()
    if not manager._initialized:
        manager.initialize()

    # submit all link expansion jobs to thread pool
    timeout_per_link = config.timeout_config.link_content_expander_timeout_per_link

    futures = []
    for url in links:
        future = manager.submit(
            OperationType.LINK_CONTEXT_EXPANDING,
            expand_link_context_sync,
            url,
            timeout_per_link
        )
        futures.append(future)

    # wait for ALL results (simple and clean!)
    try:
        web_results = wait_all(
            futures,
            timeout=config.timeout_config.link_content_expander_timeout_total
        )
    except TimeoutError:
        logger.warning(
            f"total timeout exceeded for link expansion "
            f"(limit: {config.timeout_config.link_content_expander_timeout_total}s)"
        )
        return []

    # process all results
    expanded_sources: List[DataSource] = []
    successful_count = 0
    failed_count = 0

    for web_result in web_results:
        # if result is None, job failed or timed out
        if web_result is None:
            failed_count += 1
            continue

        # create metadata dict from web result
        metadata = {
            "success": web_result.success,
            "url": web_result.url,
            "content_length": web_result.content_length,
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
        successful_count += 1

    logger.info(
        f"link expansion complete: {successful_count} succeeded, {failed_count} failed, "
        f"{len(expanded_sources)} total DataSources created"
    )

    return expanded_sources