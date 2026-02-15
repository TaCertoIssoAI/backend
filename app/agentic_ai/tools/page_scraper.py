"""
page scraper tool — extracts content from web pages.

reuses scrapeGenericUrl() from app.ai.context.web.apify_utils.
"""

import asyncio
import logging
from uuid import uuid4

from app.models.agenticai import ScrapeTarget, WebScrapeContext, SourceReliability
from app.ai.context.web.apify_utils import scrapeGenericUrl

from app.agentic_ai.config import SCRAPE_TIMEOUT_PER_PAGE

logger = logging.getLogger(__name__)


class PageScraperTool:
    """scrapes web pages and returns structured content."""

    def __init__(self, timeout: float = SCRAPE_TIMEOUT_PER_PAGE):
        self.timeout = timeout

    async def scrape(self, targets: list[ScrapeTarget]) -> list[WebScrapeContext]:
        """scrape all targets concurrently with per-page timeout."""
        tasks = [self._scrape_single(t) for t in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped: list[WebScrapeContext] = []
        for target, result in zip(targets, results):
            if isinstance(result, WebScrapeContext):
                scraped.append(result)
            elif isinstance(result, Exception):
                logger.error(f"scrape error for {target.url}: {result}")
                scraped.append(
                    WebScrapeContext(
                        id=str(uuid4()),
                        url=target.url,
                        parent_id=None,
                        reliability=SourceReliability.POUCO_CONFIAVEL,
                        title=target.title,
                        content="",
                        extraction_status="failed",
                        extraction_tool="error",
                    )
                )
        return scraped

    async def _scrape_single(self, target: ScrapeTarget) -> WebScrapeContext:
        """scrape a single page with timeout."""
        try:
            raw = await asyncio.wait_for(
                scrapeGenericUrl(target.url),
                timeout=self.timeout,
            )

            success = raw.get("success", False)
            content = raw.get("content", "")
            metadata = raw.get("metadata", {})
            extraction_tool = metadata.get("extraction_tool", "unknown")

            return WebScrapeContext(
                id=str(uuid4()),
                url=target.url,
                parent_id=None,
                reliability=SourceReliability.POUCO_CONFIAVEL,
                title=target.title or metadata.get("title", ""),
                content=content,
                extraction_status="success" if success else "failed",
                extraction_tool=extraction_tool,
            )

        except asyncio.TimeoutError:
            logger.warning(f"scrape timeout for {target.url}")
            return WebScrapeContext(
                id=str(uuid4()),
                url=target.url,
                parent_id=None,
                reliability=SourceReliability.POUCO_CONFIAVEL,
                title=target.title,
                content="",
                extraction_status="timeout",
                extraction_tool="timeout",
            )
