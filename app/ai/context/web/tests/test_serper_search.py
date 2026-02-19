"""
tests for serper.dev search client.

validates:
- parameter mapping (language, date, site_search → query)
- response mapping (organic → items format)
- error handling (missing key, non-200, timeout)
- query building helpers

run with:
    pytest app/ai/context/web/tests/test_serper_search.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from app.ai.context.web.serper_search import (
    serper_search,
    SerperSearchError,
    _is_serper_configured,
    _build_serper_query,
    SERPER_API_URL,
)


# ===== _is_serper_configured =====

def test_is_configured_when_key_set():
    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        assert _is_serper_configured() is True


def test_is_not_configured_when_key_missing():
    with patch.dict("os.environ", {}, clear=True):
        assert _is_serper_configured() is False


def test_is_not_configured_when_key_empty():
    with patch.dict("os.environ", {"SERPER_API_KEY": ""}):
        assert _is_serper_configured() is False


# ===== _build_serper_query =====

def test_build_query_no_site():
    assert _build_serper_query("test query") == "test query"


def test_build_query_no_site_explicit_none():
    assert _build_serper_query("test query", site_search=None) == "test query"


def test_build_query_site_include():
    result = _build_serper_query("vaccines", site_search="who.int", site_search_filter="i")
    assert result == "site:who.int vaccines"


def test_build_query_site_exclude():
    result = _build_serper_query("vaccines", site_search="fake.com", site_search_filter="e")
    assert result == "-site:fake.com vaccines"


def test_build_query_site_default_include():
    """when filter is not 'e', default to include."""
    result = _build_serper_query("vaccines", site_search="who.int")
    assert result == "site:who.int vaccines"


# ===== serper_search — param mapping =====

@pytest.mark.asyncio
async def test_serper_search_basic_params():
    """verify basic query and num are sent correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("test query", num=5)

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["q"] == "test query"
            assert payload["num"] == 5


@pytest.mark.asyncio
async def test_serper_search_date_restrict_mapping():
    """verify date_restrict is mapped to tbs with qdr: prefix."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("query", date_restrict="d7")

            payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
            assert payload["tbs"] == "qdr:d7"


@pytest.mark.asyncio
async def test_serper_search_language_mapping():
    """verify language is mapped to hl and gl."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("query", language="lang_pt")

            payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
            assert payload["hl"] == "pt"
            assert payload["gl"] == "br"


@pytest.mark.asyncio
async def test_serper_search_unknown_language_ignored():
    """unknown language codes should not add hl/gl."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("query", language="lang_xx")

            payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
            assert "hl" not in payload
            assert "gl" not in payload


@pytest.mark.asyncio
async def test_serper_search_num_capped_at_10():
    """num should be capped at 10."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("query", num=20)

            payload = mock_client.post.call_args.kwargs.get("json") or mock_client.post.call_args[1].get("json")
            assert payload["num"] == 10


# ===== serper_search — response mapping =====

@pytest.mark.asyncio
async def test_serper_search_response_mapping():
    """verify organic results are mapped to google cse item format."""
    serper_response = {
        "organic": [
            {
                "title": "Test Title",
                "link": "https://example.com/article",
                "snippet": "A test snippet",
                "domain": "example.com",
                "position": 1,
            },
            {
                "title": "Second Result",
                "link": "https://other.com/page",
                "snippet": "Another snippet",
                "domain": "other.com",
                "position": 2,
            },
        ]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = serper_response

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            items = await serper_search("test")

    assert len(items) == 2

    # verify first item mapping
    assert items[0]["title"] == "Test Title"
    assert items[0]["link"] == "https://example.com/article"
    assert items[0]["snippet"] == "A test snippet"
    assert items[0]["displayLink"] == "example.com"

    # verify second item mapping
    assert items[1]["title"] == "Second Result"
    assert items[1]["link"] == "https://other.com/page"


@pytest.mark.asyncio
async def test_serper_search_empty_organic():
    """empty organic list returns empty items."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            items = await serper_search("test")

    assert items == []


# ===== serper_search — error handling =====

@pytest.mark.asyncio
async def test_serper_search_missing_key():
    """should raise SerperSearchError when key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(SerperSearchError, match="missing SERPER_API_KEY"):
            await serper_search("test")


@pytest.mark.asyncio
async def test_serper_search_non_200():
    """should raise SerperSearchError on non-200 response."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "rate limit exceeded"

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(SerperSearchError, match="429"):
                await serper_search("test")


@pytest.mark.asyncio
async def test_serper_search_timeout():
    """should propagate httpx.TimeoutException."""
    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(httpx.TimeoutException):
                await serper_search("test")


@pytest.mark.asyncio
async def test_serper_search_sends_correct_headers():
    """verify X-API-KEY header is sent."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "my-secret-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("test")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["X-API-KEY"] == "my-secret-key"
            assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_serper_search_posts_to_correct_url():
    """verify request goes to serper api url."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"organic": []}

    with patch.dict("os.environ", {"SERPER_API_KEY": "test-key"}):
        with patch("app.ai.context.web.serper_search.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await serper_search("test")

            call_args = mock_client.post.call_args
            assert call_args[0][0] == SERPER_API_URL
