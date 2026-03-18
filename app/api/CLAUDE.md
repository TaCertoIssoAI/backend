# API Layer

FastAPI routers, request/response mapping, and input sanitization.

## Structure

```
api/
├── endpoints/         # route handlers
│   ├── text.py        # POST /text — main fact-checking endpoint
│   ├── scraping.py    # web scraping utility endpoints
│   ├── research.py    # research/search utility endpoints
│   └── test.py        # test/debug endpoints
├── mapper/            # data transformation
│   ├── mapper.py      # API models ↔ internal DataSource/FactCheckResult conversion
│   ├── personal_info.py  # PII removal (emails, phones, dates, CPF patterns)
│   └── formating.py   # response text formatting (markdown link cleanup, source section splitting)
└── test/              # tests for mapper logic
```

## Main Endpoint: POST /text

`endpoints/text.py` — handles all content types (text, image descriptions, audio transcripts).

Request flow:
1. `sanitize_request()` — strip PII from input
2. `request_to_data_sources()` — convert `Request` → list of `DataSource`
3. `run_fact_check(data_sources)` — invoke the agentic graph
4. `fact_check_result_to_response()` — build `AnalysisResponse` with citation mapping
5. `sanitize_response()` — strip PII from output
6. fire-and-forget analytics payload via `asyncio.create_task`

## Mapper

`mapper.py` handles two conversions:
- **Request → DataSource**: maps each `ContentItem` to a `DataSource` with appropriate `source_type` (`plain_text`, `image_description`, `audio_transcript`, etc.)
- **FactCheckResult → AnalysisResponse**: maps claims, verdicts, and citations into the API response format. Resolves citation IDs against the collected evidence (fact_check_results, search_results, scraped_pages).

## PII Sanitization

`personal_info.py` uses regex patterns to detect and redact:
- Email addresses
- Phone numbers (Brazilian format)
- CPF numbers
- Date patterns

Applied to both incoming requests and outgoing responses.

## Response Format

`AnalysisResponse` contains:
- `message_id` — unique request identifier
- `rationale` — full analysis text with verdict badges, per-claim discussion, and "Fontes de apoio:" citation section
- `responseWithoutLinks` — rationale text before the sources section (for WhatsApp messages without clickable links)
