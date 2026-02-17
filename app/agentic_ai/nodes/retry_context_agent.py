"""
retry_context_agent node — retries context search with different strategies.

uses a dedicated retry prompt that includes queries to avoid
and adjudication justifications from the previous attempt.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage

from app.agentic_ai.config import MAX_RETRY_ITERATIONS
from app.agentic_ai.prompts.retry_system_prompt import build_retry_system_prompt
from app.agentic_ai.state import ContextAgentState

logger = logging.getLogger(__name__)


def make_retry_context_agent_node(model: Any):
    """factory that returns the retry context agent node function."""

    async def retry_context_agent_node(state: ContextAgentState) -> dict:
        system_prompt = build_retry_system_prompt(
            iteration_count=state.get("iteration_count", 0),
            retry_context=state.get("retry_context", ""),
            max_iterations=MAX_RETRY_ITERATIONS,
        )

        messages = [SystemMessage(content=system_prompt)]
        for msg in state.get("messages", []):
            if not isinstance(msg, SystemMessage):
                messages.append(msg)

        response = await model.ainvoke(messages)
        new_iteration = state.get("iteration_count", 0) + 1

        logger.info(f"retry_context_agent iteration {new_iteration}, "
                     f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}")

        return {
            "messages": [response],
            "iteration_count": new_iteration,
        }

    return retry_context_agent_node
