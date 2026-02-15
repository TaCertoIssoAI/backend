"""tests for web_search tool."""

import pytest
from unittest.mock import AsyncMock, patch

from app.agentic_ai.tools.web_search import WebSearchTool, _build_query_with_trusted_domains


def test_build_query_with_trusted_domains_empty():
    with patch("app.agentic_ai.tools.web_search.get_trusted_domains", return_value=[]):
        result = _build_query_with_trusted_domains("test query")
        assert result == "test query"


def test_build_query_with_trusted_domains():
    with patch(
        "app.agentic_ai.tools.web_search.get_trusted_domains",
        return_value=["who.int", "cdc.gov"],
    ):
        result = _build_query_with_trusted_domains("vaccines")
        assert "vaccines" in result
        assert "site:who.int" in result
        assert "site:cdc.gov" in result
        assert "OR" in result


@pytest.mark.asyncio
async def test_search_returns_all_domain_keys():
    mock_items = [
        {"link": "https://test.com/1", "title": "Result 1", "snippet": "s1", "displayLink": "test.com"},
    ]

    with patch("app.agentic_ai.tools.web_search.google_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_items
        tool = WebSearchTool()
        results = await tool.search(["test query"], max_results_per_search=5)

        # should have all domain keys
        assert "geral" in results
        assert "g1" in results
        assert "estadao" in results
        assert "aosfatos" in results
        assert "folha" in results


@pytest.mark.asyncio
async def test_search_handles_empty_results():
    with patch("app.agentic_ai.tools.web_search.google_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        tool = WebSearchTool()
        results = await tool.search(["empty query"])

        for key in results:
            assert results[key] == []


@pytest.mark.asyncio
async def test_search_handles_google_error():
    from app.ai.context.web.google_search import GoogleSearchError

    with patch("app.agentic_ai.tools.web_search.google_search", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = GoogleSearchError("api error")
        tool = WebSearchTool()
        results = await tool.search(["failing query"])

        # should return empty lists, not raise
        for key in results:
            assert results[key] == []
