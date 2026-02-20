"""
tests for web search fallback behavior (vertex → google → serper).

validates:
- google succeeds → serper never called
- google fails + serper succeeds → results from serper
- google timeout + serper succeeds → results from serper
- both fail → original error propagated
- serper not configured → original error propagated

run with:
    pytest app/ai/context/web/tests/test_google_search_fallback.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from app.ai.context.web.google_search import (
    google_search,
    searchGoogleClaim,
    GoogleSearchError,
    _vertex_search_internal,
)


MOCK_GOOGLE_ITEMS = [
    {"title": "Google Result", "link": "https://google.com/1", "snippet": "from google", "displayLink": "google.com"},
]

MOCK_SERPER_ITEMS = [
    {"title": "Serper Result", "link": "https://serper.com/1", "snippet": "from serper", "displayLink": "serper.com"},
]


# ===== google_search() fallback tests =====

@pytest.mark.asyncio
async def test_google_succeeds_serper_not_called():
    """when google works, serper should never be called."""
    with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
        with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
            mock_google.return_value = MOCK_GOOGLE_ITEMS

            result = await google_search("test query")

            assert result == MOCK_GOOGLE_ITEMS
            mock_google.assert_called_once()
            mock_serper.assert_not_called()


@pytest.mark.asyncio
async def test_google_fails_serper_succeeds():
    """when google raises, fallback to serper and return serper results."""
    with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
        with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
            with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                mock_google.side_effect = GoogleSearchError("quota exceeded")
                mock_serper.return_value = MOCK_SERPER_ITEMS

                result = await google_search("test query")

                assert result == MOCK_SERPER_ITEMS
                mock_serper.assert_called_once()


@pytest.mark.asyncio
async def test_google_timeout_serper_succeeds():
    """when google times out, fallback to serper."""
    with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
        with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
            with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                mock_google.side_effect = httpx.TimeoutException("timed out")
                mock_serper.return_value = MOCK_SERPER_ITEMS

                result = await google_search("test query")

                assert result == MOCK_SERPER_ITEMS


@pytest.mark.asyncio
async def test_both_fail_original_error_raised():
    """when both google and serper fail, original google error is raised."""
    with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
        with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
            with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                mock_google.side_effect = GoogleSearchError("google down")
                mock_serper.side_effect = Exception("serper also down")

                with pytest.raises(GoogleSearchError, match="google down"):
                    await google_search("test query")


@pytest.mark.asyncio
async def test_serper_not_configured_original_error_raised():
    """when serper is not configured, original google error is raised."""
    with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
        with patch("app.ai.context.web.google_search._is_serper_configured", return_value=False):
            mock_google.side_effect = GoogleSearchError("missing keys")

            with pytest.raises(GoogleSearchError, match="missing keys"):
                await google_search("test query")


@pytest.mark.asyncio
async def test_fallback_passes_params_to_serper():
    """verify that relevant params are forwarded to serper_search."""
    with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
        with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
            with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                mock_google.side_effect = GoogleSearchError("fail")
                mock_serper.return_value = []

                await google_search(
                    "test query",
                    num=5,
                    site_search="who.int",
                    site_search_filter="i",
                    date_restrict="d7",
                    language="lang_pt",
                    timeout=30.0,
                )

                mock_serper.assert_called_once_with(
                    query="test query",
                    num=5,
                    site_search="who.int",
                    site_search_filter="i",
                    date_restrict="d7",
                    language="lang_pt",
                    timeout=30.0,
                )


# ===== searchGoogleClaim() fallback tests =====

@pytest.mark.asyncio
async def test_claim_google_succeeds_no_fallback():
    """when google claim search succeeds, no fallback needed."""
    with patch("app.ai.context.web.google_search._searchGoogleClaimInternal", new_callable=AsyncMock) as mock_internal:
        mock_internal.return_value = {
            "success": True,
            "claim": "test",
            "results": [{"title": "r1"}],
            "total_results": 1,
            "metadata": {"api": "google-custom-search"},
            "error": None,
        }

        result = await searchGoogleClaim("test")

        assert result["success"] is True
        assert result["metadata"]["api"] == "google-custom-search"


@pytest.mark.asyncio
async def test_claim_google_fails_serper_succeeds():
    """when google claim search fails, serper fallback returns results."""
    with patch("app.ai.context.web.google_search._searchGoogleClaimInternal", new_callable=AsyncMock) as mock_internal:
        with patch("app.ai.context.web.google_search._searchSerperClaimFallback", new_callable=AsyncMock) as mock_fallback:
            with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                mock_internal.return_value = {
                    "success": False,
                    "claim": "test",
                    "results": [],
                    "total_results": 0,
                    "error": "quota exceeded",
                }
                mock_fallback.return_value = {
                    "success": True,
                    "claim": "test",
                    "results": [{"title": "serper result"}],
                    "total_results": 1,
                    "metadata": {"api": "serper-dev-fallback"},
                    "error": None,
                }

                result = await searchGoogleClaim("test")

                assert result["success"] is True
                assert result["metadata"]["api"] == "serper-dev-fallback"
                mock_fallback.assert_called_once()


@pytest.mark.asyncio
async def test_claim_both_fail_returns_google_error():
    """when both fail, return the original google failure response."""
    with patch("app.ai.context.web.google_search._searchGoogleClaimInternal", new_callable=AsyncMock) as mock_internal:
        with patch("app.ai.context.web.google_search._searchSerperClaimFallback", new_callable=AsyncMock) as mock_fallback:
            with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                google_result = {
                    "success": False,
                    "claim": "test",
                    "results": [],
                    "total_results": 0,
                    "error": "google down",
                }
                mock_internal.return_value = google_result
                mock_fallback.return_value = {
                    "success": False,
                    "claim": "test",
                    "results": [],
                    "total_results": 0,
                    "error": "serper also failed",
                }

                result = await searchGoogleClaim("test")

                # returns original google error when serper also fails
                assert result["success"] is False
                assert result["error"] == "google down"


@pytest.mark.asyncio
async def test_claim_serper_not_configured_returns_google_error():
    """when serper is not configured, return google failure directly."""
    with patch("app.ai.context.web.google_search._searchGoogleClaimInternal", new_callable=AsyncMock) as mock_internal:
        with patch("app.ai.context.web.google_search._is_serper_configured", return_value=False):
            google_result = {
                "success": False,
                "claim": "test",
                "results": [],
                "total_results": 0,
                "error": "missing credentials",
            }
            mock_internal.return_value = google_result

            result = await searchGoogleClaim("test")

            assert result["success"] is False
            assert result["error"] == "missing credentials"


@pytest.mark.asyncio
async def test_vertex_succeeds_google_and_serper_not_called():
    """when vertex works, google and serper should not be called."""
    vertex_items = [{"title": "Vertex Result", "link": "https://vertex.com/1", "snippet": "from vertex", "displayLink": "vertex.com"}]
    with patch("app.ai.context.web.google_search._vertex_search_internal", new_callable=AsyncMock) as mock_vertex:
        with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
            with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
                mock_vertex.return_value = vertex_items

                result = await google_search("test query")

                assert result == vertex_items
                mock_vertex.assert_called_once()
                mock_google.assert_not_called()
                mock_serper.assert_not_called()


@pytest.mark.asyncio
async def test_vertex_fails_google_succeeds_serper_not_called():
    """when vertex fails and google works, serper should not be called."""
    with patch("app.ai.context.web.google_search._vertex_search_internal", new_callable=AsyncMock) as mock_vertex:
        with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
            with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
                mock_vertex.side_effect = Exception("vertex down")
                mock_google.return_value = MOCK_GOOGLE_ITEMS

                result = await google_search("test query")

                assert result == MOCK_GOOGLE_ITEMS
                mock_google.assert_called_once()
                mock_serper.assert_not_called()


@pytest.mark.asyncio
async def test_vertex_and_google_fail_serper_succeeds():
    """when vertex and google fail, fallback to serper."""
    with patch("app.ai.context.web.google_search._vertex_search_internal", new_callable=AsyncMock) as mock_vertex:
        with patch("app.ai.context.web.google_search._google_search_internal", new_callable=AsyncMock) as mock_google:
            with patch("app.ai.context.web.google_search.serper_search", new_callable=AsyncMock) as mock_serper:
                with patch("app.ai.context.web.google_search._is_serper_configured", return_value=True):
                    mock_vertex.side_effect = Exception("vertex down")
                    mock_google.side_effect = GoogleSearchError("quota exceeded")
                    mock_serper.return_value = MOCK_SERPER_ITEMS

                    result = await google_search("test query")

                    assert result == MOCK_SERPER_ITEMS
                    mock_serper.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_internal_passes_domain_filters():
    """site_search include filter should be sent to vertex as allowed domains."""
    with patch("app.ai.context.web.google_search._is_vertex_configured", return_value=True):
        with patch("app.ai.context.web.google_search.vertex_search", new_callable=AsyncMock) as mock_vertex_search:
            mock_vertex_search.return_value = []

            await _vertex_search_internal(
                "test query",
                num=5,
                site_search="g1.globo.com",
                site_search_filter="i",
            )

            mock_vertex_search.assert_called_once_with(
                query="test query",
                num=5,
                allowed_domains=["g1.globo.com"],
            )


@pytest.mark.asyncio
async def test_vertex_internal_ignores_exclude_domain_filter():
    """exclude domain filter should not restrict vertex domains."""
    with patch("app.ai.context.web.google_search._is_vertex_configured", return_value=True):
        with patch("app.ai.context.web.google_search.vertex_search", new_callable=AsyncMock) as mock_vertex_search:
            mock_vertex_search.return_value = []

            await _vertex_search_internal(
                "test query",
                num=5,
                site_search="g1.globo.com",
                site_search_filter="e",
            )

            mock_vertex_search.assert_called_once_with(
                query="test query",
                num=5,
                allowed_domains=None,
            )
