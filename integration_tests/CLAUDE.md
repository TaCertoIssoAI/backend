# Integration Tests

End-to-end tests that hit the running FastAPI server with real API calls.

## Files

| File | Purpose |
|------|---------|
| `text_only_test.py` | basic text claim verification — sends a claim and validates response structure |
| `text_and_links_test.py` | text with embedded URLs — tests link expansion and scraping |
| `adjudication_with_search_fallback.py` | tests retry logic when initial search returns insufficient sources |
| `prod_load_test.py` | load testing with concurrent requests |

## Running

These tests require:
1. A running server (`uvicorn app.main:app`)
2. Valid API keys in `.env` (Google, Apify, etc.)
3. Network access to external services

They are **not** run in CI — they hit real external APIs and incur costs.

## Environment

Configure via env vars:
- `TEST_BASE_URL` — server URL (default: `http://localhost:8000`)
- Standard `.env` vars for API keys
