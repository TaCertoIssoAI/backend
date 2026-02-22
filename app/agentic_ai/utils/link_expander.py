"""
async link expansion with fire-and-forget task registry.

provides pure-async link scraping using asyncio.gather for concurrency.
uses a module-level dict to store asyncio.Task objects keyed by run_id,
since tasks can't be serialized into LangGraph state.

flow:
  format_input → fire_link_expansion(run_id, urls, ...) → stores Task in registry
  wait_for_async → await_link_expansion(run_id) → pops and awaits Task
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from uuid import uuid4

from app.agentic_ai.config import LINK_SCRAPE_TIMEOUT_PER_URL, MAX_LINKS_TO_EXPAND
from app.agentic_ai.context.web.apify_utils import scrapeGenericUrl
from app.agentic_ai.context.web.models import WebContentResult
from app.models.commondata import DataSource

logger = logging.getLogger(__name__)

# side-channel: run_id → asyncio.Task mapping (not serializable into state)
_pending_link_tasks: dict[str, asyncio.Task] = {}


async def expand_link_context(url: str) -> WebContentResult:
    """expand a link and extract its content using web scraping.

    detects platform automatically, tries simple HTTP first for generic
    sites, falls back to Apify actor with browser if needed.
    """
    result_dict = await scrapeGenericUrl(url)
    return WebContentResult.from_dict(data=result_dict, url=url)


async def _scrape_single_url(
    url: str,
    parent_source_id: str,
    locale: str,
    timestamp: Optional[str],
    timeout: float = LINK_SCRAPE_TIMEOUT_PER_URL,
) -> Optional[DataSource]:
    """scrape a single URL with timeout. returns None on any failure."""
    try:
        result = await asyncio.wait_for(expand_link_context(url), timeout=timeout)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"link scrape failed for {url[:80]}: {type(e).__name__}: {e}")
        return None

    if not result.success or not result.content:
        logger.info(f"link scrape returned no content for {url[:80]}")
        return None

    metadata: dict = {
        "url": url,
        "parent_source_id": parent_source_id,
        "content_length": result.content_length,
        "success": result.success,
    }
    if result.metadata:
        metadata["platform"] = result.metadata.platform
        metadata["author"] = result.metadata.author
        metadata["timestamp"] = result.metadata.timestamp

    return DataSource(
        id=f"link-{uuid4()}",
        source_type="link_context",
        original_text=result.content,
        metadata=metadata,
        locale=locale,
        timestamp=timestamp,
    )


async def expand_all_links(
    urls: list[str],
    parent_source_id: str,
    locale: str,
    timestamp: Optional[str],
) -> list[DataSource]:
    """expand multiple URLs concurrently. used directly for links-only case."""
    limited = urls[:MAX_LINKS_TO_EXPAND]
    if not limited:
        return []

    logger.info(f"expanding {len(limited)} links concurrently")
    tasks = [
        _scrape_single_url(url, parent_source_id, locale, timestamp)
        for url in limited
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # keep only successful DataSource results
    expanded = [r for r in results if isinstance(r, DataSource)]
    logger.info(f"link expansion: {len(expanded)}/{len(limited)} succeeded")
    return expanded


def fire_link_expansion(
    run_id: str,
    urls: list[str],
    parent_source_id: str,
    locale: str,
    timestamp: Optional[str],
) -> int:
    """fire-and-forget: create asyncio.Task, store in registry, return URL count."""
    limited = urls[:MAX_LINKS_TO_EXPAND]
    if not limited:
        return 0

    task = asyncio.create_task(
        expand_all_links(limited, parent_source_id, locale, timestamp)
    )
    _pending_link_tasks[run_id] = task
    logger.info(f"fired link expansion for {len(limited)} URLs (run_id={run_id})")
    return len(limited)


async def await_link_expansion(run_id: str) -> list[DataSource]:
    """pop task from registry and await it. returns [] on missing key or failure."""
    task = _pending_link_tasks.pop(run_id, None)
    if task is None:
        logger.info(f"no pending link task for run_id={run_id}")
        return []

    try:
        return await task
    except Exception as e:
        logger.error(f"link expansion task failed: {type(e).__name__}: {e}")
        return []
