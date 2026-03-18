# Data Models

Pydantic models defining data contracts between pipeline stages.

## Files

| File | Purpose |
|------|---------|
| `api.py` | API request/response schemas: `Request`, `ContentItem`, `AnalysisResponse`, enums (`VerdictLabel`, `ContentType`) |
| `commondata.py` | `DataSource` — unified internal representation of any input (text, link, image, audio, video). Has `to_llm_string()` for formatting |
| `factchecking.py` | pipeline output models: `FactCheckResult`, `ClaimVerdict`, `Citation`, `ExtractedClaim`, `EnrichedClaim`, `EvidenceRetrievalResult` |
| `agenticai.py` | agentic pipeline context models: `FactCheckApiContext`, `GoogleSearchContext`, `WebScrapeContext`, `ScrapeTarget`, `SourceReliability` enum, `ContextNodeOutput` |
| `analytics.py` | analytics data models for telemetry collection |
| `config.py` | configuration models (`LLMConfig`, `TimeoutConfig`, etc.) |

## Key Relationships

- `DataSource` is the universal input format — all content types are converted to this
- `FactCheckResult` is the pipeline output — contains per-source results with `ClaimVerdict` lists
- `ClaimVerdict` holds: claim text, verdict string, justification, and `citations_used` (list of `Citation`)
- Evidence context models (`FactCheckApiContext`, `GoogleSearchContext`, `WebScrapeContext`) carry a `reliability` field from `SourceReliability` enum

## Verdict Values

Verdicts are in Portuguese: `"Verdadeiro"`, `"Falso"`, `"Fora de Contexto"`, `"Fontes insuficientes para verificar"`

## Source Reliability Levels

- `MUITO_CONFIAVEL` — highly reliable (fact-check APIs, trusted news domains)
- `POUCO_CONFIAVEL` — less reliable (scraped pages, unknown sources)
- `NEUTRO` — neutral (general web search results)
