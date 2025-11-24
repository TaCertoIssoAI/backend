"""
logging decorators for the fact-checking pipeline.

provides decorators for common logging patterns like timing, error handling, etc.
"""

import functools
import time
from typing import Callable, TypeVar, Any
import asyncio

from app.observability.logger.logger import get_logger
from app.observability.logger.pipeline_step import PipelineStep


F = TypeVar('F', bound=Callable[..., Any])


def time_profile(
    pipeline_step: PipelineStep = PipelineStep.SYSTEM
) -> Callable[[F], F]:
    """
    decorator that logs execution time of a function.

    logs with INFO level and prefix "[TIME PROFILE]" showing the function name
    and time taken to execute.

    works with both sync and async functions.

    args:
        pipeline_step: pipeline step context for the logger (default: SYSTEM)

    example:
        >>> from app.observability.logger import time_profile, PipelineStep
        >>> @time_profile(PipelineStep.CLAIM_EXTRACTION)
        ... def extract_claims(text: str):
        ...     # do work
        ...     return claims
        >>> extract_claims("some text")
        # logs: [TIME PROFILE] extract_claims completed in 1.23s

        >>> @time_profile()  # uses SYSTEM pipeline step
        ... async def fetch_data():
        ...     # do async work
        ...     return data
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__, pipeline_step)

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.set_prefix("[TIME PROFILE]")
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                    return result
                finally:
                    logger.clear_prefix()
            return async_wrapper  # type: ignore

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.set_prefix("[TIME PROFILE]")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                return result
            finally:
                logger.clear_prefix()
        return sync_wrapper  # type: ignore

    return decorator
