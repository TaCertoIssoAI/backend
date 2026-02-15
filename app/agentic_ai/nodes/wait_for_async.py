"""
wait_for_async node — placeholder for waiting on async data sources.

in the current implementation this is a passthrough that decrements
pending_async_count. a production version would block on an asyncio.Event
or poll a shared queue until async data sources resolve.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.agentic_ai.state import ContextAgentState

logger = logging.getLogger(__name__)


async def wait_for_async_node(state: ContextAgentState) -> dict:
    """
    wait for pending async data sources, then route back to context_agent.

    currently a stub that resets pending_async_count to 0 so the graph
    can complete. when real async data source handling is implemented,
    this node will block until new data arrives and inject it into state.
    """
    pending = state.get("pending_async_count", 0)
    logger.info(f"wait_for_async: {pending} source(s) pending (stub — resolving)")

    return {
        "messages": [
            HumanMessage(
                content="[sistema] Dados assíncronos recebidos. "
                "Reavalie as fontes e decida se mais buscas são necessárias."
            )
        ],
        "pending_async_count": 0,
    }
