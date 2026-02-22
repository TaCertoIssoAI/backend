"""
caching layer for web search queries using Redis (GCP Memorystore).

normalizes queries, builds deterministic cache keys, and stores results
as zlib-compressed JSON to minimize memory usage.
"""

import hashlib
import json
import logging
import os
import re
import zlib
from typing import Callable, Awaitable, Optional

from app.clients.memorystore import safe_get, safe_set

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_KEY_PREFIX = "web_search:v1"
_MAX_INLINE_QUERY_LEN = 100


def normalize_query(query: str) -> str:
    """lowercase, strip, and collapse whitespace."""
    return _WHITESPACE_RE.sub(" ", query.strip().lower())


def hash_domains(domains: list[str] | None) -> str:
    """deterministic hash for a domain list (order-independent)."""
    if not domains:
        return "nodomain"
    cleaned = sorted(d.strip().lower() for d in domains if d and d.strip())
    if not cleaned:
        return "nodomain"
    raw = ",".join(cleaned)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def build_cache_key(query: str, domains: list[str] | None) -> str:
    """build a deterministic redis key from query and domains."""
    nq = normalize_query(query)
    if len(nq) > _MAX_INLINE_QUERY_LEN:
        query_part = hashlib.sha256(nq.encode()).hexdigest()
    else:
        # replace spaces with underscores for readability
        query_part = nq.replace(" ", "_")
    domain_part = hash_domains(domains)
    return f"{_KEY_PREFIX}:{query_part}:{domain_part}"


def serialize(results: list[dict]) -> bytes:
    """json + zlib compress."""
    raw = json.dumps(results, separators=(",", ":"), ensure_ascii=False)
    return zlib.compress(raw.encode("utf-8"), level=6)


def deserialize(data: bytes) -> Optional[list[dict]]:
    """zlib decompress + json parse. returns None on corruption."""
    try:
        raw = zlib.decompress(data)
        return json.loads(raw)
    except Exception:
        logger.warning("cache deserialization failed, treating as miss")
        return None


def _get_ttl_seconds() -> int:
    """read TTL from env (in minutes), default 60."""
    minutes = int(os.getenv("WEB_SEARCH_CACHE_TTL_MINUTES", "60"))
    return max(minutes, 1) * 60


async def cached_custom_search(
    query: str,
    *,
    num: int,
    domains: list[str] | None,
    timeout: float,
    original_search_fn: Callable[..., Awaitable[list[dict]]],
) -> list[dict]:
    """
    cache-through wrapper for _custom_search().

    on cache hit returns deserialized results directly.
    on miss calls original_search_fn and caches the result.
    any redis error silently falls through to the original function.
    """
    key = build_cache_key(query, domains)

    # try cache
    cached = await safe_get(key)
    if cached is not None:
        results = deserialize(cached)
        if results is not None:
            logger.debug("cache HIT for key=%s (%d results)", key, len(results))
            return results

    # cache miss — call original
    logger.debug("cache MISS for key=%s", key)
    results = await original_search_fn(
        query, num=num, domains=domains, timeout=timeout,
    )

    # best-effort cache store
    if results:
        ttl = _get_ttl_seconds()
        compressed = serialize(results)
        await safe_set(key, compressed, ex=ttl)

    return results
