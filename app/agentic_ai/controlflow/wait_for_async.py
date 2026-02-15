"""
wait_for_async node — awaits pending link expansion tasks and injects
results as DataSource objects into the graph state.

collects all results at once via await_link_expansion, rebuilds the
formatted_data_sources string with priority headers, and sends a
system notification to the context agent.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.agentic_ai.state import ContextAgentState
from app.agentic_ai.nodes.format_input import _format_data_sources
from app.agentic_ai.utils.link_expander import await_link_expansion

logger = logging.getLogger(__name__)


async def wait_for_async_node(state: ContextAgentState) -> dict:
    """await pending link expansion and inject results into state."""
    run_id = state.get("run_id", "")
    new_sources = await await_link_expansion(run_id)
    successful = [ds for ds in new_sources if ds.original_text]

    # rebuild formatted_data_sources with original + new link sources
    existing_sources = state.get("data_sources", [])
    all_sources = existing_sources + successful
    formatted = _format_data_sources(all_sources)

    # system notification for the context agent
    if successful:
        urls = [ds.metadata.get("url", "?") for ds in successful]
        url_list = "\n".join(f"  - {u}" for u in urls)
        msg = (
            f"[sistema] {len(successful)} link(s) expandidos:\n{url_list}\n"
            "Reavalie o contexto com as novas fontes."
        )
    else:
        msg = "[sistema] Nenhum link expandido. Continue com as fontes disponiveis."

    return {
        "messages": [HumanMessage(content=msg)],
        "pending_async_count": 0,
        "data_sources": successful,  # appended via operator.add reducer
        "formatted_data_sources": formatted,  # last-write-wins (rebuilt from all sources)
    }
