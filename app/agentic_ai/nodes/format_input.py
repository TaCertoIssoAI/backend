"""
start node that converts structured DataSource objects into a
formatted string for the context agent prompt.

also handles link extraction and async expansion:
- links-only input → blocks and awaits expansion here
- text + links → fires async expansion, context_agent starts in parallel
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage

from app.agentic_ai.state import ContextAgentState
from app.agentic_ai.utils.link_expander import (
    expand_all_links,
    fire_link_expansion,
)
from app.ai.pipeline.link_context_expander import extract_links
from app.models.commondata import DataSource


def _is_links_only(text: str, urls: list[str]) -> bool:
    """check if original text contains only URLs with no meaningful claim text."""
    remaining = text
    for url in urls:
        remaining = remaining.replace(url, "")
    remaining = remaining.strip(" \t\n\r,;:.!?")
    return len(remaining) == 0


def _format_data_sources(data_sources: list[DataSource]) -> str:
    """format a list of DataSource objects into a single LLM-ready string."""
    if not data_sources:
        return "(nenhum conteudo fornecido)"

    if len(data_sources) == 1:
        return data_sources[0].to_llm_string()

    parts: list[str] = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, ds in enumerate(data_sources):
        label = labels[i] if i < len(labels) else str(i + 1)
        header = f"=== Entrada {label} ==="
        if ds.source_type == "link_context":
            header += " [CONTEUDO EXPANDIDO DE LINK — PRIORIDADE ALTA]"
        parts.append(header)
        parts.append(ds.to_llm_string())
        parts.append("")
    return "\n".join(parts)


async def format_input_node(state: ContextAgentState) -> dict:
    """start node — extracts links, fires async expansion, formats data_sources."""
    data_sources = state.get("data_sources", [])
    run_id = state.get("run_id") or str(uuid.uuid4())

    # extract URLs from original_text sources
    all_urls: list[str] = []
    parent_source_id = ""
    locale = "pt-BR"
    timestamp = None

    for ds in data_sources:
        if ds.source_type == "original_text":
            all_urls.extend(extract_links(ds.original_text))
            parent_source_id = ds.id
            locale = ds.locale
            timestamp = ds.timestamp

    result: dict = {"run_id": run_id}

    if all_urls and _is_links_only(data_sources[0].original_text, all_urls):
        # links-only: block here, await expansion, include results immediately
        expanded = await expand_all_links(
            all_urls, parent_source_id, locale, timestamp
        )
        all_sources = data_sources + expanded
        formatted = _format_data_sources(all_sources)
        result["formatted_data_sources"] = formatted
        result["data_sources"] = expanded  # appended via operator.add reducer
    elif all_urls:
        # normal: fire-and-forget, context_agent starts in parallel
        pending = fire_link_expansion(
            run_id, all_urls, parent_source_id, locale, timestamp
        )
        formatted = _format_data_sources(data_sources)
        result["formatted_data_sources"] = formatted
        if pending > 0:
            result["pending_async_count"] = pending
    else:
        # no links: just format
        formatted = _format_data_sources(data_sources)
        result["formatted_data_sources"] = formatted

    # send data sources as a HumanMessage for the context agent
    result["messages"] = [HumanMessage(
        content=f"Conteudo que precisa de fontes e contexto para verificacao:\n\n{formatted}"
    )]

    return result
