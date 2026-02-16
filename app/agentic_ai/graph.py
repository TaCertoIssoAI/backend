"""
LangGraph state graph definition for the context search loop.

compiles the graph with: context_agent → check_edges → (wait_for_async | adjudication).
after adjudication, prepare_retry decides whether to retry with different queries.
tool calls are handled by the built-in LangGraph ToolNode.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.models.agenticai import (
    ContextNodeOutput,
    FactCheckApiContext,
    GoogleSearchContext,
    ScrapeTarget,
    WebScrapeContext,
)
from app.models.factchecking import FactCheckResult
from app.agentic_ai.state import ContextAgentState, _merge_search_results
from app.agentic_ai.nodes.context_agent import make_context_agent_node
from app.agentic_ai.nodes.adjudication import make_adjudication_node
from app.agentic_ai.nodes.check_edges import check_edges as check_edges_router
from app.agentic_ai.nodes.format_input import format_input_node
from app.agentic_ai.nodes.retry_context_agent import make_retry_context_agent_node
from app.agentic_ai.controlflow.wait_for_async import wait_for_async_node
from app.agentic_ai.controlflow.prepare_retry import (
    prepare_retry_node,
    route_after_prepare_retry,
)
from app.agentic_ai.tools.protocols import (
    FactCheckSearchProtocol,
    WebSearchProtocol,
    PageScraperProtocol,
)

logger = logging.getLogger(__name__)


def _make_tools(
    fact_checker: FactCheckSearchProtocol,
    web_searcher: WebSearchProtocol,
    page_scraper: PageScraperProtocol,
) -> list:
    """create LangChain @tool functions that delegate to protocol implementations."""

    @tool
    async def search_fact_check_api(queries: list[str]) -> str:
        """search fact-checking databases for existing verdicts on claims.
        returns results classified as 'Muito confiável'.
        queries: list of search query strings."""
        results = await fact_checker.search(queries)
        items = [
            {
                "id": r.id,
                "title": r.title,
                "publisher": r.publisher,
                "rating": r.rating,
                "claim_text": r.claim_text,
                "url": r.url,
                "review_date": r.review_date,
            }
            for r in results
        ]
        return json.dumps(
            {
                "results": items,
                "_summary": {"total_results": len(items)},
            },
            ensure_ascii=False,
        )

    @tool
    async def search_web(
        queries: list[str],
        max_results_per_domain: int = 5,
        max_results_general: int = 5,
    ) -> str:
        """search the web across general and domain-specific sources (G1, Estadão, Aos Fatos, Folha).
        queries: list of search query strings.
        max_results_per_domain: max results per domain-specific source per query (default 5).
        max_results_general: max results for the general web search per query (default 5).
        """
        results = await web_searcher.search(queries, max_results_per_domain, max_results_general)
        output = {}
        for domain_key, entries in results.items():
            output[domain_key] = [
                {
                    "id": e.id,
                    "title": e.title,
                    "url": e.url,
                    "snippet": e.snippet,
                    "domain": e.domain,
                }
                for e in entries
            ]
        total = sum(len(entries) for entries in output.values())
        per_domain = {k: len(v) for k, v in output.items() if v}
        output["_summary"] = {"total_results": total, "per_domain": per_domain}
        return json.dumps(output, ensure_ascii=False)

    @tool
    async def scrape_pages(targets: list[dict]) -> str:
        """extract full content from web pages.
        targets: list of objects with 'url' and 'title' fields."""
        parsed_targets = [ScrapeTarget(url=t["url"], title=t["title"]) for t in targets]
        results = await page_scraper.scrape(parsed_targets)
        return json.dumps(
            [
                {
                    "id": r.id,
                    "title": r.title,
                    "url": r.url,
                    "extraction_status": r.extraction_status,
                    "content_preview": r.content[:500] if r.content else "",
                }
                for r in results
            ],
            ensure_ascii=False,
        )

    # attach protocol refs so tool_node_with_state_update can access them
    search_fact_check_api._protocol = fact_checker  # type: ignore[attr-defined]
    search_web._protocol = web_searcher  # type: ignore[attr-defined]
    scrape_pages._protocol = page_scraper  # type: ignore[attr-defined]

    return [search_fact_check_api, search_web, scrape_pages]


def _make_tool_node_with_state_update(
    tools: list,
    fact_checker: FactCheckSearchProtocol,
    web_searcher: WebSearchProtocol,
    page_scraper: PageScraperProtocol,
) -> Any:
    """
    create a node that runs the ToolNode and also updates typed state fields.

    LangGraph's built-in ToolNode only updates the messages list.
    this wrapper additionally parses tool results and appends to the
    typed context lists in state.
    """
    tool_node = ToolNode(tools)

    async def tool_node_with_state_update(state: ContextAgentState) -> dict:
        # run the actual tool node
        result = await tool_node.ainvoke(state)
        messages = result.get("messages", [])

        # parse tool outputs and accumulate into typed state
        new_fact_checks: list[FactCheckApiContext] = []
        new_search_results: dict[str, list[GoogleSearchContext]] = {}
        new_scraped: list[WebScrapeContext] = []

        for msg in messages:
            if not hasattr(msg, "name") or not hasattr(msg, "content"):
                continue

            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except (json.JSONDecodeError, TypeError):
                continue

            if msg.name == "search_fact_check_api":
                items = data.get("results", data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    from app.models.agenticai import SourceReliability as SR

                    for item in items:
                        new_fact_checks.append(
                            FactCheckApiContext(
                                id=item.get("id", str(uuid4())),
                                url=item.get("url", ""),
                                parent_id=None,
                                reliability=SR.MUITO_CONFIAVEL,
                                title=item.get("title", ""),
                                publisher=item.get("publisher", ""),
                                rating=item.get("rating", ""),
                                claim_text=item.get("claim_text", ""),
                                review_date=item.get("review_date"),
                            )
                        )

            elif msg.name == "search_web":
                if isinstance(data, dict):
                    from app.models.agenticai import SourceReliability
                    from app.agentic_ai.config import DOMAIN_SEARCHES

                    for domain_key, entries in data.items():
                        if domain_key == "_summary":
                            continue
                        reliability = DOMAIN_SEARCHES.get(
                            domain_key, {}
                        ).get("reliability", SourceReliability.NEUTRO)
                        if domain_key not in new_search_results:
                            new_search_results[domain_key] = []
                        for item in entries:
                            new_search_results[domain_key].append(
                                GoogleSearchContext(
                                    id=item.get("id", str(uuid4())),
                                    url=item.get("url", ""),
                                    parent_id=None,
                                    reliability=reliability,
                                    title=item.get("title", ""),
                                    snippet=item.get("snippet", ""),
                                    domain=item.get("domain", ""),
                                )
                            )

            elif msg.name == "scrape_pages":
                if isinstance(data, list):
                    from app.models.agenticai import SourceReliability

                    for item in data:
                        new_scraped.append(
                            WebScrapeContext(
                                id=item.get("id", str(uuid4())),
                                url=item.get("url", ""),
                                parent_id=None,
                                reliability=SourceReliability.POUCO_CONFIAVEL,
                                title=item.get("title", ""),
                                content=item.get("content_preview", ""),
                                extraction_status=item.get("extraction_status", ""),
                                extraction_tool="",
                            )
                        )

        # count actual new items (not just truthy dict with empty lists)
        new_search_count = sum(len(v) for v in new_search_results.values())
        existing_fc = len(state.get("fact_check_results", []))
        existing_search = sum(len(v) for v in state.get("search_results", {}).values())
        existing_scraped = len(state.get("scraped_pages", []))

        logger.debug(
            f"tool_node parsed: {len(new_fact_checks)} new fact_check, "
            f"{new_search_count} new search, {len(new_scraped)} new scraped "
            f"(state has {existing_fc} fc, {existing_search} search, {existing_scraped} scraped)"
        )

        update: dict[str, Any] = {"messages": messages}
        if new_fact_checks:
            update["fact_check_results"] = state.get("fact_check_results", []) + new_fact_checks
        if new_search_count:
            update["search_results"] = _merge_search_results(
                state.get("search_results", {}), new_search_results
            )
        if new_scraped:
            update["scraped_pages"] = state.get("scraped_pages", []) + new_scraped

        return update

    return tool_node_with_state_update


def _make_route_after_agent(tools_node_name: str):
    """create a router for an agent node that directs to the correct tools node."""

    def router(state: ContextAgentState) -> str:
        last_msg = state["messages"][-1] if state.get("messages") else None
        has_tool_calls = (
            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        ) if last_msg else False

        if has_tool_calls:
            return tools_node_name

        edge = check_edges_router(state)
        if edge == "wait_for_async":
            return "wait_for_async"
        return "adjudication"

    return router


def build_graph(
    model: Any,
    fact_checker: FactCheckSearchProtocol,
    web_searcher: WebSearchProtocol,
    page_scraper: PageScraperProtocol,
    adjudication_model: Any = None,
):
    """
    build and compile the context search loop graph.

    args:
        model: LLM with .bind_tools() and .ainvoke() support
        fact_checker: fact-check search implementation
        web_searcher: web search implementation
        page_scraper: page scraper implementation
        adjudication_model: optional LLM for adjudication (defaults to model)

    returns:
        compiled LangGraph graph
    """
    tools = _make_tools(fact_checker, web_searcher, page_scraper)
    model_with_tools = model.bind_tools(tools)

    context_agent_node = make_context_agent_node(model_with_tools)
    tool_node = _make_tool_node_with_state_update(
        tools, fact_checker, web_searcher, page_scraper
    )
    adjudication_node = make_adjudication_node(adjudication_model or model)
    retry_context_agent_node = make_retry_context_agent_node(model_with_tools)

    graph = StateGraph(ContextAgentState)

    graph.add_node("format_input", format_input_node)
    graph.add_node("context_agent", context_agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("wait_for_async", wait_for_async_node)
    graph.add_node("adjudication", adjudication_node)
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("retry_context_agent", retry_context_agent_node)
    graph.add_node("retry_tools", tool_node)  # same function, different graph name

    graph.add_edge(START, "format_input")
    graph.add_edge("format_input", "context_agent")

    # after context_agent: route based on tool calls + check_edges
    graph.add_conditional_edges(
        "context_agent",
        _make_route_after_agent("tools"),
        {
            "tools": "tools",
            "wait_for_async": "wait_for_async",
            "adjudication": "adjudication",
        },
    )

    # after tools: always go back to context_agent
    graph.add_edge("tools", "context_agent")

    # after wait_for_async: go back to context_agent
    graph.add_edge("wait_for_async", "context_agent")

    # adjudication -> prepare_retry (normal) or END (on timeout error)
    def _route_after_adjudication(state: ContextAgentState) -> str:
        if state.get("adjudication_error"):
            return END
        return "prepare_retry"

    graph.add_conditional_edges(
        "adjudication",
        _route_after_adjudication,
        {"prepare_retry": "prepare_retry", END: END},
    )
    graph.add_conditional_edges(
        "prepare_retry",
        route_after_prepare_retry,
        {"retry_context_agent": "retry_context_agent", END: END},
    )

    # retry agent loop (mirrors context_agent loop)
    graph.add_conditional_edges(
        "retry_context_agent",
        _make_route_after_agent("retry_tools"),
        {
            "retry_tools": "retry_tools",
            "wait_for_async": "wait_for_async",
            "adjudication": "adjudication",
        },
    )
    graph.add_edge("retry_tools", "retry_context_agent")

    return graph.compile()


def extract_output(state: dict) -> FactCheckResult | ContextNodeOutput:
    """extract the adjudication result or raw context from final graph state."""
    if state.get("adjudication_result"):
        return state["adjudication_result"]
    # fallback to raw context output for backwards compatibility
    return ContextNodeOutput(
        fact_check_results=state.get("fact_check_results", []),
        search_results=state.get("search_results", {}),
        scraped_pages=state.get("scraped_pages", []),
    )
