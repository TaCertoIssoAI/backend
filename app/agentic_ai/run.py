"""
public API for running the agentic fact-checking graph.

usage from FastAPI or any async caller:

    from app.agentic_ai.run import run_fact_check
    result = await run_fact_check(data_sources)
"""

from __future__ import annotations

from app.models.commondata import DataSource
from app.models.factchecking import FactCheckResult
from app.agentic_ai.config import (
    DEFAULT_MODEL,
    ADJUDICATION_MODEL,
    ADJUDICATION_THINKING_BUDGET,
)
from app.observability.logger.logger import get_logger

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

async def run_fact_check(data_sources: list[DataSource]) -> FactCheckResult:
    """run the full agentic fact-checking graph on a list of DataSources.

    args:
        data_sources: one or more DataSource objects to verify.

    returns:
        a FactCheckResult with claim verdicts and overall summary.
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

    # extract_output may return ContextNodeOutput as fallback — wrap it
    if isinstance(output, FactCheckResult):
        return output

    # fallback: no adjudication result (shouldn't happen in production)
    logger.warning("graph returned raw context output instead of FactCheckResult")
    return FactCheckResult(results=[], overall_summary="Nenhuma verificação foi produzida.")
