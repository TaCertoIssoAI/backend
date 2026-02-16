"""
public API for running the agentic fact-checking graph.

usage from FastAPI or any async caller:

    from app.agentic_ai.run import run_fact_check
    result = await run_fact_check(data_sources)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.commondata import DataSource
from app.models.agenticai import FactCheckApiContext, GoogleSearchContext, WebScrapeContext
from app.models.factchecking import FactCheckResult
from app.agentic_ai.config import (
    DEFAULT_MODEL,
    ADJUDICATION_MODEL,
    ADJUDICATION_THINKING_BUDGET,
)
from app.observability.logger.logger import get_logger


@dataclass
class GraphOutput:
    """output from the fact-checking graph with source data for citation mapping."""
    result: FactCheckResult
    fact_check_results: list[FactCheckApiContext] = field(default_factory=list)
    search_results: dict[str, list[GoogleSearchContext]] = field(default_factory=dict)
    scraped_pages: list[WebScrapeContext] = field(default_factory=list)

logger = get_logger(__name__)


def _build_graph():
    """build the context agent graph with real tool implementations."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from app.agentic_ai.graph import build_graph
    from app.agentic_ai.tools.fact_check_search import FactCheckSearchTool
    from app.agentic_ai.tools.web_search import WebSearchTool
    from app.agentic_ai.tools.page_scraper import PageScraperTool

    model = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, temperature=0)
    adj_model = ChatGoogleGenerativeAI(
        model=ADJUDICATION_MODEL,
        temperature=0,
        thinking_budget=ADJUDICATION_THINKING_BUDGET,
    )
    fact_checker = FactCheckSearchTool()
    web_searcher = WebSearchTool()
    page_scraper = PageScraperTool()

    return build_graph(model, fact_checker, web_searcher, page_scraper, adj_model)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

async def run_fact_check(data_sources: list[DataSource]) -> GraphOutput:
    """run the full agentic fact-checking graph on a list of DataSources.

    args:
        data_sources: one or more DataSource objects to verify.

    returns:
        GraphOutput with FactCheckResult and collected source lists for citation mapping.
    """
    from app.agentic_ai.graph import extract_output

    graph = _build_graph()

    initial_state = {
        "messages": [],
        "data_sources": data_sources,
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": "",
        "adjudication_result": None,
        "retry_count": 0,
        "retry_context": None,
    }

    final_state = await graph.ainvoke(initial_state)
    output = extract_output(final_state)

    # extract source lists from graph state for citation mapping
    fc_results = final_state.get("fact_check_results", [])
    sr_results = final_state.get("search_results", {})
    sp_results = final_state.get("scraped_pages", [])

    if isinstance(output, FactCheckResult):
        return GraphOutput(
            result=output,
            fact_check_results=fc_results,
            search_results=sr_results,
            scraped_pages=sp_results,
        )

    # fallback: no adjudication result (shouldn't happen in production)
    logger.warning("graph returned raw context output instead of FactCheckResult")
    return GraphOutput(
        result=FactCheckResult(results=[], overall_summary="Nenhuma verificação foi produzida."),
        fact_check_results=fc_results,
        search_results=sr_results,
        scraped_pages=sp_results,
    )
