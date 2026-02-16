"""
check_edges router — evaluates conditions after the agent stops calling tools.
"""

from __future__ import annotations

import logging

from app.agentic_ai.config import MAX_ITERATIONS, MAX_RETRY_ITERATIONS
from app.agentic_ai.state import ContextAgentState

logger = logging.getLogger(__name__)


def check_edges(state: ContextAgentState) -> str:
    """
    route after context_agent returns with no tool calls.

    uses MAX_RETRY_ITERATIONS when in retry mode (retry_count > 0),
    otherwise uses MAX_ITERATIONS.

    returns:
        "wait_for_async" if there are pending async data sources
        "end" if max iterations reached or no pending async
    """
    iteration_count = state.get("iteration_count", 0)
    pending = state.get("pending_async_count", 0)
    retry_count = state.get("retry_count", 0)
    max_iters = MAX_RETRY_ITERATIONS if retry_count > 0 else MAX_ITERATIONS

    # always collect async results first — never discard pending work
    if pending > 0:
        logger.info(f"{pending} async data sources pending, waiting")
        return "wait_for_async"

    if iteration_count >= max_iters:
        logger.info(f"max iterations ({max_iters}) reached, ending")
        return "end"

    logger.info("no pending async, context gathering complete")
    return "end"
