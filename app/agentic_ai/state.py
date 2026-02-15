"""
graph state schema for the context agent.

extends LangGraph's MessagesState with typed context fields
and control-flow counters.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from langgraph.graph import MessagesState

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
)
from app.models.commondata import DataSource
from app.models.factchecking import FactCheckResult


def _merge_search_results(
    existing: dict[str, list[GoogleSearchContext]],
    new: dict[str, list[GoogleSearchContext]],
) -> dict[str, list[GoogleSearchContext]]:
    """merge two search_results dicts by extending each domain key's list."""
    merged = dict(existing)
    for key, entries in new.items():
        if key in merged:
            merged[key] = merged[key] + entries
        else:
            merged[key] = list(entries)
    return merged


class ContextAgentState(MessagesState):
    """typed state for the context search loop graph."""

    # accumulated context (append-only via reducers)
    fact_check_results: Annotated[list[FactCheckApiContext], operator.add]
    search_results: Annotated[
        dict[str, list[GoogleSearchContext]], _merge_search_results
    ]
    scraped_pages: Annotated[list[WebScrapeContext], operator.add]

    # control flow
    iteration_count: int
    pending_async_count: int

    # structured input data sources (set at graph entry)
    data_sources: list[DataSource]

    # formatted data sources text (populated by format_input node)
    formatted_data_sources: str

    # adjudication output (set once by the adjudication node)
    adjudication_result: Optional[FactCheckResult]
