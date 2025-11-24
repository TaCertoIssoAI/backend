"""
centralized logger factory for the fact-checking pipeline.

provides a unified interface for creating loggers with pipeline step context,
configurable output destinations (stdout, file, or both), and structured formatting.

when file logging is enabled, logs are organized by:
- session timestamp folder (e.g., logs/2024-11-23_21-30-45/)
- pipeline step files (e.g., claim_extraction.log, evidence_retrieval.log)
- general.log for logs without a specific pipeline step
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from app.observability.logger.config import get_logger_config
from app.observability.logger.formatter import PipelineLogAdapter, PipelineLogFormatter
from app.observability.logger.pipeline_step import PipelineStep


# global state
_logging_initialized = False
_session_timestamp: Optional[str] = None
_pipeline_handlers: Dict[str, RotatingFileHandler] = {}


class PipelineStepFilter(logging.Filter):
    """
    filter that only allows logs from a specific pipeline step.

    used to route logs to the correct file based on their pipeline step.
    """

    def __init__(self, pipeline_step: str):
        super().__init__()
        self.pipeline_step = pipeline_step

    def filter(self, record: logging.LogRecord) -> bool:
        """
        check if record matches this filter's pipeline step.

        args:
            record: log record to check

        returns:
            true if record's pipeline step matches this filter
        """
        record_step = getattr(record, "pipeline_step", PipelineStep.UNKNOWN.value)
        return record_step == self.pipeline_step


def _get_session_timestamp() -> str:
    """
    get or create the session timestamp.

    the session timestamp is created once on first logging setup and reused
    for all subsequent logs in the same application run.

    returns:
        timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    global _session_timestamp

    if _session_timestamp is None:
        _session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return _session_timestamp


def _ensure_log_directory(log_dir: Path) -> None:
    """
    ensure the log directory exists.

    args:
        log_dir: path to the log directory
    """
    log_dir.mkdir(parents=True, exist_ok=True)


def _get_log_level(level_str: str) -> int:
    """
    convert log level string to logging constant.

    args:
        level_str: log level string (DEBUG, INFO, WARN, ERROR)

    returns:
        logging level constant
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    return level_map.get(level_str.upper(), logging.INFO)


def _get_log_file_path(pipeline_step: str, config) -> Path:
    """
    get the log file path for a specific pipeline step.

    creates a structured path like:
    - logs/2024-11-23_21-30-45/claim_extraction.log (with session folder)
    - logs/claim_extraction.log (without session folder)

    args:
        pipeline_step: pipeline step name
        config: logger configuration

    returns:
        path to the log file
    """
    base_dir = Path(config.log_dir)

    # add session timestamp folder if enabled
    if config.create_session_folder:
        base_dir = base_dir / _get_session_timestamp()

    # create filename from pipeline step
    filename = f"{pipeline_step}.log"

    return base_dir / filename


def _create_file_handler(
    pipeline_step: str,
    config,
    formatter: PipelineLogFormatter
) -> RotatingFileHandler:
    """
    create a rotating file handler for a specific pipeline step.

    args:
        pipeline_step: pipeline step name
        config: logger configuration
        formatter: log formatter to use

    returns:
        configured rotating file handler
    """
    log_file = _get_log_file_path(pipeline_step, config)
    _ensure_log_directory(log_file.parent)

    handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=config.log_file_max_bytes,
        backupCount=config.log_file_backup_count,
        encoding="utf-8"
    )
    handler.setFormatter(formatter)

    # add filter to only log this pipeline step
    handler.addFilter(PipelineStepFilter(pipeline_step))

    return handler


def _get_or_create_pipeline_handler(
    pipeline_step: str,
    config,
    formatter: PipelineLogFormatter
) -> RotatingFileHandler:
    """
    get existing or create new file handler for a pipeline step.

    maintains a global registry of handlers to avoid creating duplicates.

    args:
        pipeline_step: pipeline step name
        config: logger configuration
        formatter: log formatter to use

    returns:
        file handler for the pipeline step
    """
    global _pipeline_handlers

    if pipeline_step not in _pipeline_handlers:
        _pipeline_handlers[pipeline_step] = _create_file_handler(
            pipeline_step, config, formatter
        )

    return _pipeline_handlers[pipeline_step]


def setup_logging() -> None:
    """
    initialize global logging configuration.

    should be called once at application startup. configures:
    - log level from environment
    - stdout handler (if enabled)
    - file handlers organized by pipeline step (if enabled)

    when file logging is enabled with organize_by_pipeline_step:
    - creates a timestamped session folder (optional)
    - sets up separate log files for each pipeline step
    - creates general.log for untagged logs

    this is idempotent - subsequent calls are no-ops.
    """
    global _logging_initialized

    if _logging_initialized:
        return

    config = get_logger_config()

    # get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_get_log_level(config.log_level))

    # remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # create formatter
    formatter = PipelineLogFormatter(
        fmt=config.log_format,
        datefmt=config.log_date_format
    )

    # add stdout handler
    if config.log_output in ["STDOUT", "BOTH"]:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

    # add file handlers
    if config.log_output in ["FILE", "BOTH"]:
        if config.organize_by_pipeline_step:
            # create handlers for each pipeline step
            for step in PipelineStep:
                handler = _get_or_create_pipeline_handler(
                    step.value, config, formatter
                )
                root_logger.addHandler(handler)
        else:
            # single log file for all logs
            log_file = Path(config.log_dir) / "app.log"
            _ensure_log_directory(log_file.parent)

            handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=config.log_file_max_bytes,
                backupCount=config.log_file_backup_count,
                encoding="utf-8"
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

    _logging_initialized = True


def get_logger(
    name: str,
    pipeline_step: Optional[PipelineStep] = None
) -> PipelineLogAdapter:
    """
    get a logger with pipeline step context.

    creates a logger adapter that automatically includes pipeline step information
    in all log messages. ensures logging is initialized on first call.

    when file logging is enabled with organize_by_pipeline_step, logs will be
    written to a file specific to the pipeline step (e.g., claim_extraction.log).

    args:
        name: logger name (typically __name__ of calling module)
        pipeline_step: pipeline step context (default: UNKNOWN)

    returns:
        logger adapter with pipeline step context

    example:
        >>> from app.observability.logger import get_logger, PipelineStep
        >>> logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)
        >>> logger.info("extracting claims from text")
        # stdout: 2024-11-23 21:00:00 | INFO  | claim_extraction     | app.ai.pipeline.claim_extractor | extracting claims from text
        # file: logs/2024-11-23_21-00-00/claim_extraction.log
    """
    # ensure logging is initialized
    if not _logging_initialized:
        setup_logging()

    # get standard logger
    base_logger = logging.getLogger(name)

    # wrap in adapter with pipeline step
    return PipelineLogAdapter(
        logger=base_logger,
        pipeline_step=pipeline_step or PipelineStep.UNKNOWN
    )


def get_request_logger(
    name: str,
    pipeline_step: Optional[PipelineStep] = None,
    request_id: Optional[str] = None
) -> PipelineLogAdapter:
    """
    get a logger with pipeline step and request id context.

    useful for tracking logs across a single request through multiple pipeline steps.

    args:
        name: logger name (typically __name__ of calling module)
        pipeline_step: pipeline step context (default: UNKNOWN)
        request_id: unique request identifier

    returns:
        logger adapter with pipeline step and request id context

    example:
        >>> logger = get_request_logger(__name__, PipelineStep.EVIDENCE_RETRIEVAL, "req-123")
        >>> logger.info("gathering evidence")
        # logs to: logs/2024-11-23_21-00-00/evidence_retrieval.log
    """
    # ensure logging is initialized
    if not _logging_initialized:
        setup_logging()

    # get standard logger with request_id in name for filtering
    logger_name = f"{name}.{request_id}" if request_id else name
    base_logger = logging.getLogger(logger_name)

    # prepare extra context
    extra = {}
    if request_id:
        extra["request_id"] = request_id

    # wrap in adapter with pipeline step and extra context
    return PipelineLogAdapter(
        logger=base_logger,
        pipeline_step=pipeline_step or PipelineStep.UNKNOWN,
        extra=extra
    )


def get_session_log_dir() -> Optional[Path]:
    """
    get the current session's log directory.

    returns:
        path to the session log directory, or None if logging not initialized

    example:
        >>> from app.observability.logger import get_session_log_dir
        >>> log_dir = get_session_log_dir()
        >>> print(log_dir)
        logs/2024-11-23_21-00-00
    """
    if not _logging_initialized:
        return None

    config = get_logger_config()
    base_dir = Path(config.log_dir)

    if config.create_session_folder:
        return base_dir / _get_session_timestamp()

    return base_dir
