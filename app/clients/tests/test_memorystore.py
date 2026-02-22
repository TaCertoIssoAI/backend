"""
tests for memorystore: singleton client, safe_get/safe_set,
and circuit breaker behavior.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.clients.memorystore as memorystore
from app.clients.memorystore import (
    get_redis_client,
    safe_get,
    safe_set,
    reset_circuit_breaker,
    reset_client,
    _record_failure,
    _record_success,
    _circuit_is_open,
    _FAILURE_THRESHOLD,
    _RECOVERY_TIMEOUT,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """reset module-level state before and after each test."""
    reset_circuit_breaker()
    reset_client()
    yield
    reset_circuit_breaker()
    reset_client()


# ── get_redis_client ─────────────────────────────────────────────────

class TestGetRedisClient:
    def test_returns_none_when_host_not_set(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        assert get_redis_client() is None

    def test_returns_none_for_blank_host(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "   ")
        assert get_redis_client() is None

    def test_creates_client_when_host_set(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        client = get_redis_client()
        assert client is not None

    def test_singleton_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        c1 = get_redis_client()
        c2 = get_redis_client()
        assert c1 is c2

    def test_custom_port_and_db(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "10.0.0.5")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_DB", "2")
        client = get_redis_client()
        pool = client.connection_pool
        kwargs = pool.connection_kwargs
        assert kwargs["host"] == "10.0.0.5"
        assert kwargs["port"] == 6380
        assert kwargs["db"] == 2

    def test_password_none_when_empty(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)
        client = get_redis_client()
        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["password"] is None

    def test_password_set_when_provided(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "secret123")
        client = get_redis_client()
        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["password"] == "secret123"

    def test_reset_client_allows_new_instance(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        c1 = get_redis_client()
        reset_client()
        monkeypatch.setenv("REDIS_PORT", "6380")
        c2 = get_redis_client()
        assert c1 is not c2


# ── circuit breaker ──────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_circuit_closed_initially(self):
        assert _circuit_is_open() is False

    def test_circuit_stays_closed_under_threshold(self):
        for _ in range(_FAILURE_THRESHOLD - 1):
            _record_failure()
        assert _circuit_is_open() is False

    def test_circuit_opens_at_threshold(self):
        for _ in range(_FAILURE_THRESHOLD):
            _record_failure()
        assert _circuit_is_open() is True

    def test_circuit_closes_after_recovery_timeout(self, monkeypatch):
        for _ in range(_FAILURE_THRESHOLD):
            _record_failure()
        assert _circuit_is_open() is True

        # fast-forward past recovery timeout
        future = time.monotonic() + _RECOVERY_TIMEOUT + 1
        monkeypatch.setattr(time, "monotonic", lambda: future)
        assert _circuit_is_open() is False

    def test_success_resets_circuit(self):
        for _ in range(_FAILURE_THRESHOLD):
            _record_failure()
        assert _circuit_is_open() is True

        _record_success()
        assert _circuit_is_open() is False

    def test_reset_circuit_breaker_clears_state(self):
        for _ in range(_FAILURE_THRESHOLD):
            _record_failure()
        assert _circuit_is_open() is True

        reset_circuit_breaker()
        assert _circuit_is_open() is False

    def test_failures_accumulate_across_calls(self):
        _record_failure()
        _record_failure()
        assert _circuit_is_open() is False
        _record_failure()  # hits threshold
        assert _circuit_is_open() is True

    def test_success_after_partial_failures_resets_count(self):
        _record_failure()
        _record_failure()
        _record_success()
        # counter reset, so 3 more failures needed
        _record_failure()
        _record_failure()
        assert _circuit_is_open() is False


# ── safe_get ─────────────────────────────────────────────────────────

class TestSafeGet:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_host(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        result = await safe_get("some_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_value_on_success(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.return_value = b"cached_data"
        memorystore._redis_client = mock_client

        result = await safe_get("my_key")
        assert result == b"cached_data"
        mock_client.get.assert_called_once_with("my_key")

    @pytest.mark.asyncio
    async def test_returns_none_on_key_not_found(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        memorystore._redis_client = mock_client

        result = await safe_get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("connection refused")
        memorystore._redis_client = mock_client

        result = await safe_get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_records_failure_on_exception(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.side_effect = TimeoutError("timed out")
        memorystore._redis_client = mock_client

        assert memorystore._consecutive_failures == 0
        await safe_get("key")
        assert memorystore._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_records_success_on_hit(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.return_value = b"data"
        memorystore._redis_client = mock_client

        # inject a prior failure
        _record_failure()
        assert memorystore._consecutive_failures == 1

        await safe_get("key")
        assert memorystore._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_skips_call_when_circuit_open(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        memorystore._redis_client = mock_client

        for _ in range(_FAILURE_THRESHOLD):
            _record_failure()

        result = await safe_get("key")
        assert result is None
        mock_client.get.assert_not_called()


# ── safe_set ─────────────────────────────────────────────────────────

class TestSafeSet:
    @pytest.mark.asyncio
    async def test_returns_false_when_no_host(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        result = await safe_set("key", b"val", ex=60)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        memorystore._redis_client = mock_client

        result = await safe_set("key", b"value", ex=300)
        assert result is True
        mock_client.set.assert_called_once_with("key", b"value", ex=300)

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.set.side_effect = ConnectionError("connection refused")
        memorystore._redis_client = mock_client

        result = await safe_set("key", b"val", ex=60)
        assert result is False

    @pytest.mark.asyncio
    async def test_records_failure_on_exception(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.set.side_effect = OSError("broken pipe")
        memorystore._redis_client = mock_client

        await safe_set("key", b"val", ex=60)
        assert memorystore._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_records_success_clears_failures(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        memorystore._redis_client = mock_client

        _record_failure()
        _record_failure()
        assert memorystore._consecutive_failures == 2

        await safe_set("key", b"val", ex=60)
        assert memorystore._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_skips_call_when_circuit_open(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        memorystore._redis_client = mock_client

        for _ in range(_FAILURE_THRESHOLD):
            _record_failure()

        result = await safe_set("key", b"val", ex=60)
        assert result is False
        mock_client.set.assert_not_called()


# ── circuit breaker + safe_get/safe_set integration ──────────────────

class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_three_get_failures_open_circuit(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("refused")
        memorystore._redis_client = mock_client

        for _ in range(_FAILURE_THRESHOLD):
            await safe_get("key")

        # circuit is now open — next call should not reach redis
        mock_client.get.reset_mock()
        result = await safe_get("key")
        assert result is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_three_set_failures_open_circuit(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.set.side_effect = ConnectionError("refused")
        memorystore._redis_client = mock_client

        for _ in range(_FAILURE_THRESHOLD):
            await safe_set("key", b"v", ex=60)

        # circuit open — blocks get too
        mock_client.get = AsyncMock(return_value=b"data")
        result = await safe_get("key")
        assert result is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_allows_probe_after_timeout(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("refused")
        memorystore._redis_client = mock_client

        for _ in range(_FAILURE_THRESHOLD):
            await safe_get("key")
        assert _circuit_is_open() is True

        # fast-forward past recovery
        future = time.monotonic() + _RECOVERY_TIMEOUT + 1
        monkeypatch.setattr(time, "monotonic", lambda: future)

        # circuit allows a probe — redis call happens again
        mock_client.get.side_effect = None
        mock_client.get.return_value = b"recovered"
        result = await safe_get("key")
        assert result == b"recovered"
        # success resets circuit
        assert memorystore._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_mixed_get_set_failures_accumulate(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("refused")
        mock_client.set.side_effect = ConnectionError("refused")
        memorystore._redis_client = mock_client

        await safe_get("k")    # failure 1
        await safe_set("k", b"v", ex=60)  # failure 2
        assert _circuit_is_open() is False
        await safe_get("k")    # failure 3 — opens circuit
        assert _circuit_is_open() is True
