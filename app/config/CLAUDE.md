# Configuration

Static configuration files and lookup data.

## Files

| File | Purpose |
|------|---------|
| `default.py` | default settings and fallback values |
| `gemini_models.py` | Gemini model name constants and configurations |
| `trusted_domains.py` | loads trusted Brazilian news domains list |
| `trusted_domains.json` | JSON source for trusted domains (g1.globo.com, estadao.com.br, folha.uol.com.br, aosfatos.org, etc.) |

## Trusted Domains

Used by the web search tool to classify source reliability. Domains in this list are tagged as `SourceReliability.MUITO_CONFIAVEL` in search results.

## App Settings

The main settings class lives in `core/config.py` (not here). It reads environment variables via `os.getenv()` with sensible defaults. Access via `get_settings()` which is `@lru_cache`d.

Key env vars:
- `GOOGLE_APPLICATION_CREDENTIALS` — SA JSON path; auth for all Vertex AI / Gemini calls
- `VERTEX_PROJECT_ID` — GCP project for Vertex AI
- `VERTEX_LOCATION` — Vertex region (e.g. `us-central1`)
- `GOOGLE_API_KEY` — Google **Fact Check Tools** API (separate from Gemini auth)
- `GOOGLE_SEARCH_API_KEY` — Custom Search API
- `APIFY_TOKEN` — web scraping
- `ANALYTICS_SERVICE_URL` — telemetry endpoint
- `WEB_SEARCH_CACHE_TTL_MINUTES` — Redis cache TTL
- `TEXT_PROCESSING_TIMEOUT` / `IMAGE_PROCESSING_TIMEOUT` — latency targets
