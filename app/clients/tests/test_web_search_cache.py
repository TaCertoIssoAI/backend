"""
tests for web_search_cache: normalization, key building, serialization,
and cached_custom_search integration with mock redis.
"""

import json
import zlib
from unittest.mock import AsyncMock, patch

import pytest

from app.clients.web_search_cache import (
    normalize_query,
    hash_domains,
    build_cache_key,
    serialize,
    deserialize,
    cached_custom_search,
)


# ── normalize_query ──────────────────────────────────────────────────

class TestNormalizeQuery:
    def test_lowercase_and_strip(self):
        assert normalize_query("  Climate Change  ") == "climate change"

    def test_collapse_whitespace(self):
        assert normalize_query("hello   world\t\nfoo") == "hello world foo"

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_already_normalized(self):
        assert normalize_query("already clean") == "already clean"

    def test_mixed_case_and_spaces(self):
        assert normalize_query("  A  B  c  ") == "a b c"


# ── hash_domains ─────────────────────────────────────────────────────

class TestHashDomains:
    def test_none_returns_nodomain(self):
        assert hash_domains(None) == "nodomain"

    def test_empty_list_returns_nodomain(self):
        assert hash_domains([]) == "nodomain"

    def test_blank_entries_returns_nodomain(self):
        assert hash_domains(["", "  "]) == "nodomain"

    def test_deterministic(self):
        h1 = hash_domains(["a.com", "b.com"])
        h2 = hash_domains(["a.com", "b.com"])
        assert h1 == h2

    def test_order_independent(self):
        h1 = hash_domains(["b.com", "a.com"])
        h2 = hash_domains(["a.com", "b.com"])
        assert h1 == h2

    def test_case_independent(self):
        h1 = hash_domains(["A.COM"])
        h2 = hash_domains(["a.com"])
        assert h1 == h2

    def test_hash_length(self):
        h = hash_domains(["example.com"])
        assert len(h) == 12


# ── build_cache_key ──────────────────────────────────────────────────

class TestBuildCacheKey:
    def test_short_query_inline(self):
        key = build_cache_key("climate change", None)
        assert key == "web_search:v1:climate_change:nodomain"

    def test_long_query_hashed(self):
        long_q = "a " * 60  # > 100 chars
        key = build_cache_key(long_q, None)
        parts = key.split(":")
        assert parts[0] == "web_search"
        assert parts[1] == "v1"
        assert len(parts[2]) == 64  # sha256 hex
        assert parts[3] == "nodomain"

    def test_with_domains(self):
        key = build_cache_key("test", ["a.com", "b.com"])
        assert key.startswith("web_search:v1:test:")
        assert key.split(":")[-1] != "nodomain"

    def test_different_queries_different_keys(self):
        k1 = build_cache_key("query one", None)
        k2 = build_cache_key("query two", None)
        assert k1 != k2

    def test_same_query_different_case_same_key(self):
        k1 = build_cache_key("Hello World", None)
        k2 = build_cache_key("hello world", None)
        assert k1 == k2


# ── serialize / deserialize ──────────────────────────────────────────

class TestSerialization:
    def test_roundtrip(self):
        data = [
            {"title": "Test", "link": "https://example.com", "snippet": "...", "displayLink": "example.com"},
            {"title": "Another", "link": "https://other.com", "snippet": "x", "displayLink": "other.com"},
        ]
        compressed = serialize(data)
        assert isinstance(compressed, bytes)
        result = deserialize(compressed)
        assert result == data

    def test_empty_list_roundtrip(self):
        data = []
        assert deserialize(serialize(data)) == []

    def test_corrupted_data_returns_none(self):
        assert deserialize(b"not valid zlib data") is None

    def test_compressed_is_smaller(self):
        data = [{"title": f"Item {i}", "link": f"https://example.com/{i}", "snippet": "a" * 200, "displayLink": "example.com"} for i in range(10)]
        raw_json = json.dumps(data).encode()
        compressed = serialize(data)
        assert len(compressed) < len(raw_json)

    def test_unicode_roundtrip(self):
        data = [{"title": "Notícia sobre saúde", "link": "https://ex.com", "snippet": "à é ü ñ", "displayLink": "ex.com"}]
        assert deserialize(serialize(data)) == data


# ── cached_custom_search (integration with mock redis) ───────────────

@pytest.fixture
def sample_results():
    return [
        {"title": "Result 1", "link": "https://a.com", "snippet": "snip 1", "displayLink": "a.com"},
        {"title": "Result 2", "link": "https://b.com", "snippet": "snip 2", "displayLink": "b.com"},
    ]


@pytest.fixture
def mock_search_fn(sample_results):
    fn = AsyncMock(return_value=sample_results)
    return fn


class TestCachedCustomSearch:
    @pytest.mark.asyncio
    @patch("app.clients.web_search_cache.safe_get", new_callable=AsyncMock, return_value=None)
    @patch("app.clients.web_search_cache.safe_set", new_callable=AsyncMock, return_value=True)
    async def test_cache_miss_calls_original(self, mock_set, mock_get, mock_search_fn, sample_results):
        result = await cached_custom_search(
            "test query", num=10, domains=None, timeout=15.0,
            original_search_fn=mock_search_fn,
        )
        assert result == sample_results
        mock_search_fn.assert_called_once_with("test query", num=10, domains=None, timeout=15.0)
        mock_set.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.clients.web_search_cache.safe_set", new_callable=AsyncMock)
    @patch("app.clients.web_search_cache.safe_get", new_callable=AsyncMock)
    async def test_cache_hit_skips_original(self, mock_get, mock_set, mock_search_fn, sample_results):
        # simulate cached compressed data
        mock_get.return_value = serialize(sample_results)

        result = await cached_custom_search(
            "test query", num=10, domains=None, timeout=15.0,
            original_search_fn=mock_search_fn,
        )
        assert result == sample_results
        mock_search_fn.assert_not_called()
        mock_set.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.clients.web_search_cache.safe_set", new_callable=AsyncMock, return_value=True)
    @patch("app.clients.web_search_cache.safe_get", new_callable=AsyncMock, return_value=None)
    async def test_redis_unavailable_on_get_falls_through(self, mock_get, mock_set, mock_search_fn, sample_results):
        # safe_get returns None (redis unavailable) — should call original
        result = await cached_custom_search(
            "test query", num=10, domains=None, timeout=15.0,
            original_search_fn=mock_search_fn,
        )
        assert result == sample_results
        mock_search_fn.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.clients.web_search_cache.safe_set", new_callable=AsyncMock, return_value=False)
    @patch("app.clients.web_search_cache.safe_get", new_callable=AsyncMock, return_value=None)
    async def test_redis_error_on_set_still_returns_result(self, mock_get, mock_set, mock_search_fn, sample_results):
        # safe_set returns False (redis error) — result should still be returned
        result = await cached_custom_search(
            "test query", num=10, domains=None, timeout=15.0,
            original_search_fn=mock_search_fn,
        )
        assert result == sample_results

    @pytest.mark.asyncio
    @patch("app.clients.web_search_cache.safe_set", new_callable=AsyncMock)
    @patch("app.clients.web_search_cache.safe_get", new_callable=AsyncMock)
    async def test_corrupted_cache_treated_as_miss(self, mock_get, mock_set, mock_search_fn, sample_results):
        # return corrupted data — should fall through to original
        mock_get.return_value = b"corrupted data"

        result = await cached_custom_search(
            "test query", num=10, domains=None, timeout=15.0,
            original_search_fn=mock_search_fn,
        )
        assert result == sample_results
        mock_search_fn.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.clients.web_search_cache.safe_set", new_callable=AsyncMock, return_value=True)
    @patch("app.clients.web_search_cache.safe_get", new_callable=AsyncMock, return_value=None)
    async def test_empty_results_not_cached(self, mock_get, mock_set):
        empty_fn = AsyncMock(return_value=[])
        result = await cached_custom_search(
            "test", num=10, domains=None, timeout=15.0,
            original_search_fn=empty_fn,
        )
        assert result == []
        mock_set.assert_not_called()
