"""tests for web_search tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agentic_ai.tools.web_search import (
    WebSearchTool,
    _build_query_with_trusted_domains,
    _custom_search,
)


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

    with patch("app.agentic_ai.tools.web_search._custom_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_items
        tool = WebSearchTool()
        results = await tool.search(["test query"], max_results_specific_search=5, max_results_general=5)

        # should have all domain keys
        assert "geral" in results
        assert "especifico" in results


@pytest.mark.asyncio
async def test_search_handles_empty_results():
    with patch("app.agentic_ai.tools.web_search._custom_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        tool = WebSearchTool()
        results = await tool.search(["empty query"])

        for key in results:
            assert results[key] == []


@pytest.mark.asyncio
async def test_search_handles_google_error():
    with patch("app.agentic_ai.tools.web_search._custom_search", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = Exception("api error")
        tool = WebSearchTool()
        results = await tool.search(["failing query"])

        # should return empty lists, not raise
        for key in results:
            assert results[key] == []


@pytest.mark.asyncio
async def test_search_deduplicates_urls_within_domain_key():
    """two queries returning the same URL under the same domain key → only 1 entry."""
    same_item = {"link": "https://dup.com/1", "title": "Dup", "snippet": "s", "displayLink": "dup.com"}

    with patch("app.agentic_ai.tools.web_search._custom_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [same_item]
        tool = WebSearchTool()
        results = await tool.search(["query1", "query2"], max_results_specific_search=5, max_results_general=5)

        # each domain key should have at most 1 entry (the deduped URL)
        for key, entries in results.items():
            urls = [e.url for e in entries]
            assert len(urls) == len(set(urls)), f"duplicate URLs in domain key '{key}': {urls}"


@pytest.mark.asyncio
async def test_search_allows_same_url_across_domain_keys():
    """same URL returned by different domain searches is kept in both domain keys."""
    same_item = {"link": "https://shared.com/article", "title": "Shared", "snippet": "s", "displayLink": "shared.com"}

    with patch("app.agentic_ai.tools.web_search._custom_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [same_item]
        tool = WebSearchTool()
        results = await tool.search(["query1"], max_results_specific_search=5, max_results_general=5)

        # every domain key that returned results should have the URL
        keys_with_results = [k for k, v in results.items() if v]
        assert len(keys_with_results) > 1
        for key in keys_with_results:
            assert results[key][0].url == "https://shared.com/article"


@pytest.mark.asyncio
async def test_custom_search_sends_domains_params():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}

    with patch.dict("os.environ", {"WEB_SERCH_SERVER_URL": "http://127.0.0.1:6050"}):
        with patch("app.agentic_ai.tools.web_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await _custom_search(
                "climate",
                num=5,
                domains=["cdc.gov", "un.org"],
            )

            call_kwargs = mock_client.get.call_args.kwargs
            params = call_kwargs.get("params") or mock_client.get.call_args[1].get("params")

            # verify we send domains params when provided
            assert ("domains", "cdc.gov") in params
            assert ("domains", "un.org") in params


@pytest.mark.asyncio
async def test_search_returns_empty_on_missing_server_url():
    with patch.dict("os.environ", {}, clear=True):
        tool = WebSearchTool()
        results = await tool.search(["test query"])

        for key in results:
            assert results[key] == []
