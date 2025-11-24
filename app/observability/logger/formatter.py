"""
custom log formatter with pipeline step context.

extends python's standard logging formatter to include pipeline step information
in log messages for better observability.
"""

import logging
from typing import Optional

from app.observability.logger.pipeline_step import PipelineStep


class PipelineLogFormatter(logging.Formatter):
    """
    custom formatter that includes pipeline step in log records.

    adds a 'pipeline_step' attribute to log records, defaulting to 'unknown'
    if not explicitly set.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        format log record with pipeline step context.

        ensures all log records have a pipeline_step attribute, even if not
        explicitly set by the logger.

        args:
            record: log record to format

        returns:
            formatted log message string
        """
        # ensure pipeline_step attribute exists
        if not hasattr(record, "pipeline_step"):
            record.pipeline_step = PipelineStep.UNKNOWN.value

        return super().format(record)


class PipelineLogAdapter(logging.LoggerAdapter):
    """
    logger adapter that automatically includes pipeline step in all log records.

    wraps a standard logger and injects pipeline step context into every log message.
    supports optional message prefix for adding contextual information.
    """

    def __init__(
        self,
        logger: logging.Logger,
        pipeline_step: Optional[PipelineStep] = None,
        extra: Optional[dict] = None
    ):
        """
        initialize adapter with logger and pipeline step.

        args:
            logger: underlying logger instance
            pipeline_step: pipeline step for this logger (default: UNKNOWN)
            extra: additional context to include in all log records
        """
        self.pipeline_step = pipeline_step or PipelineStep.UNKNOWN
        self._prefix: Optional[str] = None

        # merge extra dict with pipeline_step
        if extra is None:
            extra = {}
        extra["pipeline_step"] = self.pipeline_step.value

        super().__init__(logger, extra)

    def set_prefix(self, prefix: str) -> None:
        """
        set a prefix to be prepended to all log messages.

        useful for adding contextual information like request IDs, user IDs,
        or operation names to all logs within a scope.

        args:
            prefix: string to prepend to all log messages

        example:
            >>> logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)
            >>> logger.set_prefix("[req-123]")
            >>> logger.info("processing claim")
            # output: [req-123] processing claim
            >>> logger.clear_prefix()
        """
        self._prefix = prefix

    def clear_prefix(self) -> None:
        """
        clear the current message prefix.

        removes any prefix set by set_prefix(), returning to normal logging.

        example:
            >>> logger.set_prefix("[batch-5]")
            >>> logger.info("batch started")
            # output: [batch-5] batch started
            >>> logger.clear_prefix()
            >>> logger.info("batch completed")
            # output: batch completed
        """
        self._prefix = None

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """
        process log message and inject pipeline step context and prefix.

        args:
            msg: log message
            kwargs: additional keyword arguments

        returns:
            tuple of (message, kwargs) with pipeline_step in extra
        """
        # prepend prefix if set
        if self._prefix:
            msg = f"{self._prefix} {msg}"

        # ensure extra dict exists
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # add pipeline step to extra
        kwargs["extra"]["pipeline_step"] = self.pipeline_step.value

        return msg, kwargs
