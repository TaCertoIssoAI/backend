# External Service Clients

Clients for communicating with external services and caching layers.

## Files

| File | Purpose |
|------|---------|
| `analytics_service.py` | fire-and-forget HTTP POST to analytics service. Sends `AnalyticsCollector` data after each pipeline run |
| `web_search_cache.py` | Redis-backed caching for web search results. Normalizes queries, builds deterministic keys, stores zlib-compressed JSON |
| `memorystore.py` | Redis connection wrapper (`safe_get`, `safe_set`) for GCP Memorystore. Falls through silently on connection errors |

## Web Search Cache

`cached_custom_search()` wraps the actual search function:
1. Build cache key from normalized query + hashed domain list
2. Check Redis — on hit, decompress and return
3. On miss, call the original search function, compress result, store with TTL

Key format: `web_search:v1:{query}:{domain_hash}`
TTL: configurable via `WEB_SEARCH_CACHE_TTL_MINUTES` env var (default 60 min)

## Analytics Service

- URL configured via `ANALYTICS_SERVICE_URL` and `ANALYTICS_SERVICE_ENDPOINT` env vars
- Best-effort delivery — exceptions are logged but never raised
- `get_analytics_url_for_fact_check(msg_id)` returns the public URL where the report is hosted

## Note
The `*.json` files in this directory are local analytics payload samples for debugging — not part of the application.
