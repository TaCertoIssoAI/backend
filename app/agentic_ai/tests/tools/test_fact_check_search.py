"""tests for fact_check_search tool."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agentic_ai.tools.fact_check_search import FactCheckSearchTool
from app.models.agenticai import SourceReliability


@pytest.fixture
def tool():
    return FactCheckSearchTool(api_key="test-key", max_results=5, timeout=10.0)


def test_parse_response_with_claims(tool):
    data = {
        "claims": [
            {
                "text": "test claim",
                "claimReview": [
                    {
                        "url": "https://factcheck.org/test",
                        "title": "Fact check title",
                        "publisher": {"name": "FactCheck.org"},
                        "textualRating": "False",
                        "reviewDate": "2025-01-01",
                    }
                ],
            }
        ]
    }
    results = tool._parse_response(data)
    assert len(results) == 1
    assert results[0].reliability == SourceReliability.MUITO_CONFIAVEL
    assert results[0].publisher == "FactCheck.org"
    assert results[0].rating == "Falso"
    assert results[0].claim_text == "test claim"


def test_parse_response_empty(tool):
    assert tool._parse_response({}) == []
    assert tool._parse_response({"claims": []}) == []


def test_parse_response_skips_missing_url(tool):
    data = {
        "claims": [
            {
                "text": "claim",
                "claimReview": [
                    {
                        "url": "",
                        "title": "title",
                        "publisher": {"name": "Pub"},
                        "textualRating": "False",
                    }
                ],
            }
        ]
    }
    assert tool._parse_response(data) == []


def test_parse_response_rating_with_comment(tool):
    data = {
        "claims": [
            {
                "text": "claim",
                "claimReview": [
                    {
                        "url": "https://test.com",
                        "title": "title",
                        "publisher": {"name": "Pub"},
                        "textualRating": "False. This is completely wrong",
                    }
                ],
            }
        ]
    }
    results = tool._parse_response(data)
    assert results[0].rating == "Falso"
    assert results[0].rating_comment == "This is completely wrong"


def test_max_results_limits_output(tool):
    tool.max_results = 2
    data = {
        "claims": [
            {
                "text": f"claim {i}",
                "claimReview": [
                    {
                        "url": f"https://test.com/{i}",
                        "title": f"title {i}",
                        "publisher": {"name": "Pub"},
                        "textualRating": "False",
                    }
                ],
            }
            for i in range(5)
        ]
    }
    results = tool._parse_response(data)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_returns_empty_on_missing_api_key():
    tool = FactCheckSearchTool(api_key="", max_results=5)
    results = await tool.search(["test query"])
    assert results == []


@pytest.mark.asyncio
async def test_search_deduplicates_urls_across_queries():
    """two queries returning the same fact-check URL → only 1 result."""
    api_response = {
        "claims": [
            {
                "text": "claim",
                "claimReview": [
                    {
                        "url": "https://factcheck.org/same",
                        "title": "Same article",
                        "publisher": {"name": "FC"},
                        "textualRating": "False",
                    }
                ],
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = api_response
    mock_response.raise_for_status = MagicMock()

    with patch("app.agentic_ai.tools.fact_check_search.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tool = FactCheckSearchTool(api_key="test-key", max_results=10)
        results = await tool.search(["query1", "query2"])

        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))
        assert len(results) == 1
