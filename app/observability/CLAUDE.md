# Observability

Logging and analytics collection for the fact-checking pipeline.

## Structure

```
observability/
├── logger/
│   ├── logger.py          # logger factory: get_logger(), setup_logging()
│   ├── formatter.py       # PipelineLogAdapter and PipelineLogFormatter
│   ├── pipeline_step.py   # PipelineStep enum (claim_extraction, evidence_retrieval, etc.)
│   └── config.py          # logger configuration from environment
└── analytics/
    └── collector.py       # AnalyticsCollector — gathers per-request telemetry
```

## Logger

`get_logger(name, pipeline_step)` returns a `PipelineLogAdapter` that tags every log entry with its pipeline step.

Key behaviors:
- Initializes on first call (idempotent `setup_logging()`)
- Routes logs to per-step files when file logging is enabled (e.g., `logs/2024-11-23_21-30-45/claim_extraction.log`)
- Rotating file handlers with configurable size limits
- Silences noisy third-party loggers (httpx, httpcore, openai, trafilatura, grpc)
- Configuration via env vars: `LOG_LEVEL`, `LOG_OUTPUT` (STDOUT/FILE/BOTH), `LOG_DIR`

Usage:
```python
from app.observability.logger.logger import get_logger
logger = get_logger(__name__)
logger.info("message here")
```

## Analytics Collector

`AnalyticsCollector` accumulates data through the pipeline lifecycle:
- `populate_from_data_sources()` — records input metadata
- `populate_from_graph_output()` — records claims, verdicts, evidence counts
- `set_final_response()` — captures the response text
- `to_dict()` — serializes for the analytics service

Sent to the external analytics service via `clients/analytics_service.py` as fire-and-forget.
