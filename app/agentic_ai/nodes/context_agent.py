"""
context_agent node — the core LLM agent that calls tools to gather evidence.

builds the system prompt from current state and invokes the LLM.
increments iteration_count on each run.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.agentic_ai.prompts.system_prompt import build_system_prompt
from app.agentic_ai.state import ContextAgentState

logger = logging.getLogger(__name__)


def make_context_agent_node(model: Any):
    """
    factory that returns a context_agent node function.

    the returned function rebuilds the system prompt from state on each call,
    invokes the LLM, and updates iteration_count.
    """

    async def context_agent_node(state: ContextAgentState) -> dict:
        system_prompt = build_system_prompt(
            iteration_count=state.get("iteration_count", 0),
            fact_check_results=state.get("fact_check_results", []),
            search_results=state.get("search_results", {}),
            scraped_pages=state.get("scraped_pages", []),
        )

        # build messages: system + conversation history (skip old system messages)
        messages = [SystemMessage(content=system_prompt)]
        for msg in state.get("messages", []):
            if not isinstance(msg, SystemMessage):
                messages.append(msg)

        response = await model.ainvoke(messages)

        new_iteration = state.get("iteration_count", 0) + 1
        logger.info(f"context_agent iteration {new_iteration}, "
                     f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}")

        return {
            "messages": [response],
            "iteration_count": new_iteration,
        }

    return context_agent_node
