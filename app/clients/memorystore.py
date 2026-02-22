"""
async redis client for GCP Memorystore with circuit breaker.

all public methods return None/False on errors — never raises.
if REDIS_HOST is not set, caching is silently disabled.
"""

import logging
import os
import time
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None

# circuit breaker state
_consecutive_failures: int = 0
_circuit_open_until: float = 0.0
_FAILURE_THRESHOLD: int = 3
_RECOVERY_TIMEOUT: float = 60.0


def _circuit_is_open() -> bool:
    global _circuit_open_until
    if _consecutive_failures < _FAILURE_THRESHOLD:
        return False
    if time.monotonic() >= _circuit_open_until:
        # allow a single probe
        return False
    return True


def _record_success() -> None:
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures = 0
    _circuit_open_until = 0.0


def _record_failure() -> None:
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures += 1
    if _consecutive_failures >= _FAILURE_THRESHOLD:
        _circuit_open_until = time.monotonic() + _RECOVERY_TIMEOUT
        logger.warning(
            "redis circuit breaker OPEN after %d failures, retrying in %.0fs",
            _consecutive_failures,
            _RECOVERY_TIMEOUT,
        )


def get_redis_client() -> Optional[aioredis.Redis]:
    """return a singleton async redis client, or None if not configured."""
    global _redis_client

    host = os.getenv("REDIS_HOST", "").strip()
    if not host:
        return None

    if _redis_client is not None:
        return _redis_client

    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "") or None
    db = int(os.getenv("REDIS_DB", "0"))

    _redis_client = aioredis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
        decode_responses=False,
    )
    logger.info("redis client created for %s:%d db=%d", host, port, db)
    return _redis_client


async def safe_get(key: str) -> Optional[bytes]:
    """get a value from redis. returns None on any error or if disabled."""
    if _circuit_is_open():
        return None

    client = get_redis_client()
    if client is None:
        return None

    try:
        value = await client.get(key)
        _record_success()
        return value
    except Exception as e:
        _record_failure()
        logger.warning("redis GET failed for key=%s: %s", key, e)
        return None


async def safe_set(key: str, value: bytes, ex: int) -> bool:
    """set a value in redis with TTL. returns False on any error or if disabled."""
    if _circuit_is_open():
        return False

    client = get_redis_client()
    if client is None:
        return False

    try:
        await client.set(key, value, ex=ex)
        _record_success()
        return True
    except Exception as e:
        _record_failure()
        logger.warning("redis SET failed for key=%s: %s", key, e)
        return False


def reset_circuit_breaker() -> None:
    """reset circuit breaker state — useful for tests."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures = 0
    _circuit_open_until = 0.0


def reset_client() -> None:
    """reset the singleton client — useful for tests."""
    global _redis_client
    _redis_client = None
