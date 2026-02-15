"""
check_edges router — evaluates conditions after the agent stops calling tools.
"""

from __future__ import annotations

import logging

from app.agentic_ai.config import MAX_ITERATIONS
from app.agentic_ai.state import ContextAgentState

logger = logging.getLogger(__name__)


def check_edges(state: ContextAgentState) -> str:
    """
    route after context_agent returns with no tool calls.

    returns:
        "wait_for_async" if there are pending async data sources
        "end" if max iterations reached or no pending async
    """
    iteration_count = state.get("iteration_count", 0)
    pending = state.get("pending_async_count", 0)

    # always collect async results first — never discard pending work
    if pending > 0:
        logger.info(f"{pending} async data sources pending, waiting")
        return "wait_for_async"

    if iteration_count >= MAX_ITERATIONS:
        logger.info(f"max iterations ({MAX_ITERATIONS}) reached, ending")
        return "end"

    logger.info("no pending async, context gathering complete")
    return "end"
