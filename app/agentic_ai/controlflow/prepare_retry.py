"""
prepare_retry node — checks if all verdicts are insufficient and prepares retry state.

after adjudication, this node decides whether to retry context search
with different queries or end the graph.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END
from langchain_core.messages import HumanMessage, RemoveMessage

from app.agentic_ai.config import MAX_RETRY_COUNT
from app.agentic_ai.state import ContextAgentState
from app.models.factchecking import FactCheckResult, VerdictTypeEnum

logger = logging.getLogger(__name__)


def _all_verdicts_insufficient(result: Optional[FactCheckResult]) -> bool:
    """check if every claim verdict is 'Fontes insuficientes para verificar'."""
    if not result or not result.results:
        return False
    all_verdicts = []
    for ds_result in result.results:
        all_verdicts.extend(ds_result.claim_verdicts)
    if not all_verdicts:
        return False
    return all(
        VerdictTypeEnum.FONTES_INSUFICIENTES == cv.verdict
        for cv in all_verdicts
    )


def _extract_used_queries(messages: list) -> list[str]:
    """scan message history for search tool call queries, deduplicated (case-insensitive)."""
    seen_lower: set[str] = set()
    queries: list[str] = []
    for msg in messages:
        if not hasattr(msg, "tool_calls"):
            continue
        for tc in msg.tool_calls:
            if tc["name"] in ("search_web", "search_fact_check_api"):
                for q in tc["args"].get("queries", []):
                    if q.lower() not in seen_lower:
                        seen_lower.add(q.lower())
                        queries.append(q)
    return queries


def _build_retry_context(result: FactCheckResult, used_queries: list[str]) -> str:
    """format adjudication justifications + used queries for the retry prompt."""
    parts = []

    if used_queries:
        parts.append("## Queries ja utilizadas — NAO repita estas buscas\n")
        for i, q in enumerate(used_queries, 1):
            parts.append(f"{i}. {q}")
        parts.append("")

    parts.append("## Resultado do julgamento anterior\n")
    if result.overall_summary:
        parts.append(f"Resumo geral: {result.overall_summary}\n")

    for ds_result in result.results:
        for cv in ds_result.claim_verdicts:
            parts.append(f"Alegacao: \"{cv.claim_text}\"")
            parts.append(f"Veredito: {cv.verdict}")
            parts.append(f"Justificativa: {cv.justification}")
            parts.append("")

    return "\n".join(parts)


async def prepare_retry_node(state: ContextAgentState) -> dict:
    """check adjudication result and prepare state for retry if all verdicts are insufficient."""
    result = state.get("adjudication_result")
    retry_count = state.get("retry_count", 0)

    if not _all_verdicts_insufficient(result) or retry_count >= MAX_RETRY_COUNT:
        logger.info("prepare_retry: no retry needed")
        return {}

    messages = state.get("messages", [])
    used_queries = _extract_used_queries(messages)
    retry_context = _build_retry_context(result, used_queries)

    # clear message history and seed with a HumanMessage so the retry agent
    # starts the same way as the normal context_agent
    removals = [RemoveMessage(id=m.id) for m in messages if m.id]
    formatted_data_sources = state.get("formatted_data_sources", "")
    removals.append(HumanMessage(content=formatted_data_sources))

    logger.info(f"prepare_retry: triggering retry (count={retry_count + 1}), "
                f"clearing {len(removals) - 1} messages, {len(used_queries)} queries to avoid")

    return {
        "messages": removals,
        "retry_count": retry_count + 1,
        "retry_context": retry_context,
        "iteration_count": 0,
        "adjudication_result": None,
    }


def route_after_prepare_retry(state: ContextAgentState) -> str:
    """cleared adjudication_result + retry_count > 0 -> retry agent. otherwise -> END."""
    if state.get("adjudication_result") is None and state.get("retry_count", 0) > 0:
        return "retry_context_agent"
    return END
