"""
centralized logging system for the fact-checking pipeline.

provides structured logging with pipeline step context, configurable output
destinations (stdout, file, or both), and environment-based configuration.

usage:
    >>> from app.observability.logger import get_logger, PipelineStep, setup_logging, time_profile
    >>>
    >>> # initialize logging (call once at app startup)
    >>> setup_logging()
    >>>
    >>> # get a logger for a specific pipeline step
    >>> logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)
    >>> logger.info("processing claim")
    >>> logger.error("failed to extract claim", exc_info=True)
    >>>
    >>> # use time profiling decorator
    >>> @time_profile(PipelineStep.EVIDENCE_RETRIEVAL)
    >>> async def gather_evidence(claim):
    >>>     # your code here
    >>>     return evidence
    >>> # logs: [TIME PROFILE] gather_evidence completed in 2.34s

configuration:
    set via environment variables:
    - LOG_LEVEL: DEBUG, INFO, WARN, ERROR (default: INFO)
    - LOG_OUTPUT: STDOUT, FILE, BOTH (default: STDOUT)
    - LOG_FILE_PATH: path to log file (default: logs/app.log)
    - LOG_FILE_MAX_BYTES: max size per log file (default: 10485760 = 10MB)
    - LOG_FILE_BACKUP_COUNT: number of backup files to keep (default: 5)
"""

from app.observability.logger.config import LoggerConfig, get_logger_config
from app.observability.logger.logger import (
    get_logger,
    get_request_logger,
    get_session_log_dir,
    setup_logging,
)
from app.observability.logger.pipeline_step import PipelineStep
from app.observability.logger.decorators import time_profile

__all__ = [
    # main logger functions
    "get_logger",
    "get_request_logger",
    "setup_logging",
    "get_session_log_dir",
    # pipeline step enum
    "PipelineStep",
    # configuration
    "LoggerConfig",
    "get_logger_config",
    # decorators
    "time_profile",
]
