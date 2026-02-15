"""
start node that converts structured DataSource objects into a
formatted string for the context agent prompt.
"""

from __future__ import annotations

from app.agentic_ai.state import ContextAgentState
from app.models.commondata import DataSource


def _format_data_sources(data_sources: list[DataSource]) -> str:
    """format a list of DataSource objects into a single LLM-ready string."""
    if not data_sources:
        return "(nenhum conteudo fornecido)"

    if len(data_sources) == 1:
        return data_sources[0].to_llm_string()

    parts: list[str] = []
    for i, ds in enumerate(data_sources, 1):
        parts.append(f"=== Fonte {i} ===")
        parts.append(ds.to_llm_string())
        parts.append("")
    return "\n".join(parts)


async def format_input_node(state: ContextAgentState) -> dict:
    """start node — converts data_sources into formatted_data_sources string."""
    data_sources = state.get("data_sources", [])
    formatted = _format_data_sources(data_sources)
    return {"formatted_data_sources": formatted}
