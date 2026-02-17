"""tests for page_scraper tool."""

import pytest
from unittest.mock import AsyncMock, patch

from app.agentic_ai.tools.page_scraper import PageScraperTool
from app.models.agenticai import ScrapeTarget, SourceReliability


@pytest.mark.asyncio
async def test_scrape_success():
    mock_result = {
        "success": True,
        "content": "page content here",
        "metadata": {"title": "Page Title", "extraction_tool": "beautifulsoup"},
    }

    with patch("app.agentic_ai.tools.page_scraper.scrapeGenericUrl", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = mock_result
        tool = PageScraperTool(timeout=10.0)
        targets = [ScrapeTarget(url="https://example.com", title="Example")]
        results = await tool.scrape(targets)

        assert len(results) == 1
        assert results[0].extraction_status == "success"
        assert results[0].content == "page content here"
        assert results[0].reliability == SourceReliability.POUCO_CONFIAVEL


@pytest.mark.asyncio
async def test_scrape_failure():
    mock_result = {
        "success": False,
        "content": "",
        "metadata": {},
        "error": "scrape failed",
    }

    with patch("app.agentic_ai.tools.page_scraper.scrapeGenericUrl", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = mock_result
        tool = PageScraperTool(timeout=10.0)
        targets = [ScrapeTarget(url="https://example.com", title="Example")]
        results = await tool.scrape(targets)

        assert len(results) == 1
        assert results[0].extraction_status == "failed"


@pytest.mark.asyncio
async def test_scrape_timeout():
    import asyncio

    async def slow_scrape(url, **kwargs):
        await asyncio.sleep(10)
        return {"success": True, "content": "late", "metadata": {}}

    with patch("app.agentic_ai.tools.page_scraper.scrapeGenericUrl", side_effect=slow_scrape):
        tool = PageScraperTool(timeout=0.1)
        targets = [ScrapeTarget(url="https://slow.com", title="Slow")]
        results = await tool.scrape(targets)

        assert len(results) == 1
        assert results[0].extraction_status == "timeout"


@pytest.mark.asyncio
async def test_scrape_exception_returns_failed():
    with patch("app.agentic_ai.tools.page_scraper.scrapeGenericUrl", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = RuntimeError("unexpected")
        tool = PageScraperTool(timeout=10.0)
        targets = [ScrapeTarget(url="https://broken.com", title="Broken")]
        results = await tool.scrape(targets)

        assert len(results) == 1
        assert results[0].extraction_status == "failed"
