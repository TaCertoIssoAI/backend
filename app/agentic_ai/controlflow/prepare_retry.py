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
from app.agentic_ai.prompts.context_formatter import (
    build_source_reference_list,
    filter_cited_references,
)
from app.agentic_ai.state import ContextAgentState
from app.models.agenticai import FactCheckApiContext, GoogleSearchContext, WebScrapeContext
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


def _get_cited_numbers(
    fact_check_results: list[FactCheckApiContext],
    search_results: dict[str, list[GoogleSearchContext]],
    scraped_pages: list[WebScrapeContext],
    adjudication_result: FactCheckResult,
) -> set[int]:
    """find which [N] source numbers were cited in the adjudication output.

    reuses build_source_reference_list (numbering) and filter_cited_references
    (regex extraction) from context_formatter.
    """
    refs = build_source_reference_list(fact_check_results, search_results, scraped_pages)
    texts = [adjudication_result.overall_summary or ""]
    for ds in adjudication_result.results:
        for cv in ds.claim_verdicts:
            texts.append(cv.justification)
    cited_refs = filter_cited_references(refs, *texts)
    return {num for num, _, _ in cited_refs}


# domain iteration order — must match format_context / build_source_reference_list
_SOURCE_DOMAIN_ORDER = ("aosfatos", "g1", "estadao", "folha")


def _filter_to_cited_sources(
    fact_check_results: list[FactCheckApiContext],
    search_results: dict[str, list[GoogleSearchContext]],
    scraped_pages: list[WebScrapeContext],
    cited: set[int],
) -> tuple[
    list[FactCheckApiContext],
    dict[str, list[GoogleSearchContext]],
    list[WebScrapeContext],
]:
    """keep only sources whose [N] number is in cited. returns (fc, search, scraped).

    walks sources in the same order as build_source_reference_list so the
    counter-to-source mapping is consistent.
    """
    counter = 1

    retained_fc = []
    for entry in fact_check_results:
        if counter in cited:
            retained_fc.append(entry)
        counter += 1

    retained_search: dict[str, list[GoogleSearchContext]] = {}
    for domain_key in _SOURCE_DOMAIN_ORDER:
        domain_list: list[GoogleSearchContext] = []
        for entry in search_results.get(domain_key, []):
            if counter in cited:
                domain_list.append(entry)
            counter += 1
        retained_search[domain_key] = domain_list

    geral_list: list[GoogleSearchContext] = []
    for entry in search_results.get("geral", []):
        if counter in cited:
            geral_list.append(entry)
        counter += 1
    retained_search["geral"] = geral_list

    retained_scraped = []
    for entry in scraped_pages:
        if counter in cited:
            retained_scraped.append(entry)
        counter += 1

    return retained_fc, retained_search, retained_scraped


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

    # keep only sources that were cited in the adjudication
    fc = state.get("fact_check_results", [])
    sr = state.get("search_results", {})
    sp = state.get("scraped_pages", [])
    cited = _get_cited_numbers(fc, sr, sp, result)
    retained_fc, retained_search, retained_scraped = _filter_to_cited_sources(fc, sr, sp, cited)

    logger.info(
        f"prepare_retry: triggering retry (count={retry_count + 1}), "
        f"clearing {len(removals) - 1} messages, {len(used_queries)} queries to avoid, "
        f"retaining {len(retained_fc)} fact_check, "
        f"{sum(len(v) for v in retained_search.values())} search, "
        f"{len(retained_scraped)} scraped sources"
    )

    return {
        "messages": removals,
        "retry_count": retry_count + 1,
        "retry_context": retry_context,
        "iteration_count": 0,
        "adjudication_result": None,
        "fact_check_results": retained_fc,
        "search_results": retained_search,
        "scraped_pages": retained_scraped,
    }


def route_after_prepare_retry(state: ContextAgentState) -> str:
    """cleared adjudication_result + retry_count > 0 -> retry agent. otherwise -> END."""
    if state.get("adjudication_result") is None and state.get("retry_count", 0) > 0:
        return "retry_context_agent"
    return END
