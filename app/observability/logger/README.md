# centralized logging system

comprehensive logging solution for the fact-checking pipeline with pipeline step context, configurable output destinations, and organized file structure.

## features

- **pipeline step context**: automatic tagging of logs with their originating pipeline step
- **flexible output**: log to stdout, file, or both
- **organized file structure**: separate log files per pipeline step in timestamped session folders
- **message prefixes**: add contextual information (request IDs, batch IDs, etc.) to all logs in a scope
- **configurable log levels**: debug, info, warn, error
- **environment-based configuration**: no code changes needed for different environments
- **rotating file handlers**: automatic log rotation to prevent disk space issues

## quick start

### basic usage

```python
from app.observability.logger import get_logger, PipelineStep, setup_logging

# initialize logging (call once at application startup)
setup_logging()

# get a logger for your module with pipeline step context
logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)

# log messages
logger.debug("detailed debugging information")
logger.info("general information")
logger.warning("warning message")
logger.error("error occurred", exc_info=True)
```

### usage in pipeline steps

```python
# in claim_extractor.py
from app.observability.logger import get_logger, PipelineStep

logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)

def extract_claims(text: str):
    logger.info(f"extracting claims from text of length {len(text)}")
    try:
        claims = perform_extraction(text)
        logger.info(f"extracted {len(claims)} claims")
        return claims
    except Exception as e:
        logger.error(f"claim extraction failed: {e}", exc_info=True)
        raise
```

### request-scoped logging

```python
from app.observability.logger import get_request_logger, PipelineStep

# create a logger with request id for tracking across pipeline steps
logger = get_request_logger(
    __name__,
    PipelineStep.EVIDENCE_RETRIEVAL,
    request_id="req-12345"
)

logger.info("gathering evidence")  # includes request_id in context
```

### using prefixes for contextual logging

```python
from app.observability.logger import get_logger, PipelineStep

logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)

# set prefix for all subsequent logs
logger.set_prefix("[req-123]")
logger.info("processing claim")       # output: [req-123] processing claim
logger.debug("extracting entities")   # output: [req-123] extracting entities

# change prefix dynamically
logger.set_prefix("[batch-5]")
logger.info("batch started")          # output: [batch-5] batch started

# clear prefix to return to normal logging
logger.clear_prefix()
logger.info("completed")              # output: completed
```

## configuration

configure via environment variables:

### log level

```bash
# set minimum log level (DEBUG, INFO, WARN, ERROR)
export LOG_LEVEL=INFO
```

### output destination

```bash
# output to both stdout and file (default)
export LOG_OUTPUT=BOTH

# output to stdout only
export LOG_OUTPUT=STDOUT

# output to file only
export LOG_OUTPUT=FILE
```

### file organization

```bash
# base directory for log files (default: logs)
export LOG_DIR=logs

# organize logs by pipeline step (default: true)
# when true: creates separate files for each pipeline step
# when false: single app.log file for all logs
export LOG_ORGANIZE_BY_STEP=true

# create timestamped session folder (default: true)
# when true: logs/2024-11-23_21-30-45/claim_extraction.log
# when false: logs/claim_extraction.log
export LOG_CREATE_SESSION_FOLDER=true
```

### file rotation

```bash
# maximum size per log file in bytes (default: 10MB)
export LOG_FILE_MAX_BYTES=10485760

# number of backup files to keep (default: 5)
export LOG_FILE_BACKUP_COUNT=5
```

### format customization

```bash
# log message format (default shown)
export LOG_FORMAT="%(asctime)s | %(levelname)-5s | %(pipeline_step)-20s | %(name)s | %(message)s"

# date format (default: %Y-%m-%d %H:%M:%S)
export LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"
```

## file organization

when `LOG_ORGANIZE_BY_STEP=true` and `LOG_CREATE_SESSION_FOLDER=true`:

```
logs/
├── 2024-11-23_21-30-45/        # session timestamp
│   ├── link_expansion.log       # link expansion logs only
│   ├── claim_extraction.log     # claim extraction logs only
│   ├── evidence_retrieval.log   # evidence retrieval logs only
│   ├── adjudication.log         # adjudication logs only
│   ├── api_intake.log           # api intake logs only
│   ├── preprocessing.log        # preprocessing logs only
│   ├── web_scraping.log         # web scraping logs only
│   ├── ocr.log                  # ocr logs only
│   ├── cache.log                # cache logs only
│   ├── system.log               # system logs only
│   └── unknown.log              # logs without pipeline step
└── 2024-11-23_22-15-30/        # next session
    └── ...
```

when `LOG_ORGANIZE_BY_STEP=false`:

```
logs/
├── 2024-11-23_21-30-45/        # session timestamp (if enabled)
│   └── app.log                  # all logs in single file
└── app.log                      # or directly in logs/ if session folder disabled
```

## pipeline steps

available pipeline step contexts:

```python
from app.observability.logger import PipelineStep

PipelineStep.LINK_EXPANSION       # link expansion step
PipelineStep.CLAIM_EXTRACTION     # claim extraction step
PipelineStep.EVIDENCE_RETRIEVAL   # evidence retrieval step
PipelineStep.ADJUDICATION         # final adjudication step
PipelineStep.API_INTAKE           # api request intake
PipelineStep.PREPROCESSING        # preprocessing step
PipelineStep.WEB_SCRAPING         # web scraping operations
PipelineStep.OCR                  # ocr processing
PipelineStep.CACHE                # caching operations
PipelineStep.SYSTEM               # system-level logs
PipelineStep.UNKNOWN              # default for untagged logs
```

## log format

default format:

```
2024-11-23 21:30:45 | INFO  | claim_extraction     | app.ai.pipeline.claim_extractor | extracted 3 claims
│                     │       │                      │                                  │
│                     │       │                      │                                  └─ message
│                     │       │                      └─ module name
│                     │       └─ pipeline step
│                     └─ log level
└─ timestamp
```

## utilities

### get session log directory

```python
from app.observability.logger import get_session_log_dir

# get the current session's log directory
log_dir = get_session_log_dir()
print(f"logs are being written to: {log_dir}")
# output: logs are being written to: logs/2024-11-23_21-30-45
```

## testing

run tests:

```bash
python -m pytest app/observability/logger/test/logger_test.py -v
```

## examples

### basic logging example

see `app/observability/logger/example.py` for a complete working example:

```bash
python app/observability/logger/example.py
```

### prefix functionality example

see `app/observability/logger/prefix_example.py` for prefix usage examples:

```bash
python app/observability/logger/prefix_example.py
```

this demonstrates:
- batch processing with prefixes
- request processing with dynamic prefixes
- operation-scoped prefixes

## integration with fastapi

initialize logging at application startup:

```python
# in app/main.py
from fastapi import FastAPI
from app.observability.logger import setup_logging, get_logger, PipelineStep

# setup logging before creating FastAPI app
setup_logging()

app = FastAPI()
logger = get_logger(__name__, PipelineStep.SYSTEM)

@app.on_event("startup")
async def startup_event():
    logger.info("application starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("application shutting down")
```

## best practices

1. **initialize once**: call `setup_logging()` once at application startup
2. **use pipeline steps**: always provide the appropriate `PipelineStep` when getting a logger
3. **module-level loggers**: create loggers at module level with `__name__`
4. **exception logging**: use `exc_info=True` when logging exceptions
5. **use prefixes for context**: use `set_prefix()` to add request IDs, batch IDs, or operation names
   ```python
   logger.set_prefix(f"[req:{request_id}]")
   try:
       # all logs in this scope will have the prefix
       process_request()
   finally:
       logger.clear_prefix()  # always clear when done
   ```
6. **appropriate levels**:
   - `debug`: detailed diagnostic information
   - `info`: general informational messages
   - `warning`: warning messages for unexpected but handled situations
   - `error`: error messages for serious problems

## troubleshooting

### no logs appearing in files

ensure:
- `LOG_OUTPUT` is set to `FILE` or `BOTH`
- file permissions allow writing to `LOG_DIR`
- `setup_logging()` was called before creating loggers

### logs not organized by pipeline step

check:
- `LOG_ORGANIZE_BY_STEP=true` is set
- you're providing a `PipelineStep` when calling `get_logger()`

### session folder not created

verify:
- `LOG_CREATE_SESSION_FOLDER=true` is set
- parent `LOG_DIR` exists and is writable

## architecture

```
logger/
├── __init__.py              # public api exports
├── config.py                # environment-based configuration
├── pipeline_step.py         # pipeline step enumeration
├── formatter.py             # custom formatter and adapter
├── logger.py                # main logger factory
├── example.py               # usage examples
├── README.md               # this file
└── test/
    ├── __init__.py
    └── logger_test.py      # comprehensive test suite
```

## api reference

### setup_logging()

initialize global logging configuration. call once at application startup.

```python
def setup_logging() -> None
```

### get_logger(name, pipeline_step)

get a logger with pipeline step context.

```python
def get_logger(
    name: str,
    pipeline_step: Optional[PipelineStep] = None
) -> PipelineLogAdapter
```

**parameters:**
- `name`: logger name (typically `__name__`)
- `pipeline_step`: pipeline step context (default: `PipelineStep.UNKNOWN`)

**returns:** logger adapter with pipeline step context

### get_request_logger(name, pipeline_step, request_id)

get a logger with pipeline step and request id context.

```python
def get_request_logger(
    name: str,
    pipeline_step: Optional[PipelineStep] = None,
    request_id: Optional[str] = None
) -> PipelineLogAdapter
```

**parameters:**
- `name`: logger name (typically `__name__`)
- `pipeline_step`: pipeline step context (default: `PipelineStep.UNKNOWN`)
- `request_id`: unique request identifier

**returns:** logger adapter with context

### get_session_log_dir()

get the current session's log directory.

```python
def get_session_log_dir() -> Optional[Path]
```

**returns:** path to session log directory, or `None` if logging not initialized

### logger.set_prefix(prefix)

set a prefix to be prepended to all log messages from this logger.

```python
logger.set_prefix(prefix: str) -> None
```

**parameters:**
- `prefix`: string to prepend to all log messages

**example:**
```python
logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)
logger.set_prefix("[req-123]")
logger.info("processing")  # output: [req-123] processing
```

**use cases:**
- add request IDs to all logs in a request scope
- add batch IDs for batch processing
- add user IDs for user-specific operations
- add operation names for specific code sections

### logger.clear_prefix()

clear the current message prefix, returning to normal logging.

```python
logger.clear_prefix() -> None
```

**example:**
```python
logger.set_prefix("[batch-5]")
logger.info("batch started")     # output: [batch-5] batch started
logger.clear_prefix()
logger.info("all batches done")  # output: all batches done
```
