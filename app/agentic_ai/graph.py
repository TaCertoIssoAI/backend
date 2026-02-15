"""
LangGraph state graph definition for the context search loop.

compiles the graph with: context_agent → check_edges → (wait_for_async | END).
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
from app.agentic_ai.state import ContextAgentState
from app.agentic_ai.nodes.context_agent import make_context_agent_node
from app.agentic_ai.nodes.check_edges import check_edges as check_edges_router
from app.agentic_ai.nodes.wait_for_async import wait_for_async_node
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
        return json.dumps(
            [
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
            ],
            ensure_ascii=False,
        )

    @tool
    async def search_web(
        queries: list[str], max_results_per_search: int = 5
    ) -> str:
        """search the web across general and domain-specific sources (G1, Estadão, Aos Fatos, Folha).
        queries: list of search query strings.
        max_results_per_search: max results per domain per query (default 5)."""
        results = await web_searcher.search(queries, max_results_per_search)
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
                if isinstance(data, list):
                    from app.models.agenticai import SourceReliability as SR

                    for item in data:
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

        update: dict[str, Any] = {"messages": messages}
        if new_fact_checks:
            update["fact_check_results"] = new_fact_checks
        if new_search_results:
            update["search_results"] = new_search_results
        if new_scraped:
            update["scraped_pages"] = new_scraped

        return update

    return tool_node_with_state_update


def build_graph(
    model: Any,
    fact_checker: FactCheckSearchProtocol,
    web_searcher: WebSearchProtocol,
    page_scraper: PageScraperProtocol,
):
    """
    build and compile the context search loop graph.

    args:
        model: LLM with .bind_tools() and .ainvoke() support
        fact_checker: fact-check search implementation
        web_searcher: web search implementation
        page_scraper: page scraper implementation

    returns:
        compiled LangGraph graph
    """
    tools = _make_tools(fact_checker, web_searcher, page_scraper)
    model_with_tools = model.bind_tools(tools)

    context_agent_node = make_context_agent_node(model_with_tools)
    tool_node = _make_tool_node_with_state_update(
        tools, fact_checker, web_searcher, page_scraper
    )

    def _route_after_agent(state: ContextAgentState) -> str:
        """combined router: check for tool calls first, then check_edges logic."""
        last_msg = state["messages"][-1] if state.get("messages") else None
        has_tool_calls = (
            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
        ) if last_msg else False

        if has_tool_calls:
            return "tools"

        # no tool calls — apply check_edges routing
        edge = check_edges_router(state)
        if edge == "wait_for_async":
            return "wait_for_async"
        return "__end__"

    graph = StateGraph(ContextAgentState)

    graph.add_node("context_agent", context_agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("wait_for_async", wait_for_async_node)

    graph.add_edge(START, "context_agent")

    # after context_agent: route based on tool calls + check_edges
    graph.add_conditional_edges(
        "context_agent",
        _route_after_agent,
        {
            "tools": "tools",
            "wait_for_async": "wait_for_async",
            "__end__": END,
        },
    )

    # after tools: always go back to context_agent
    graph.add_edge("tools", "context_agent")

    # after wait_for_async: go back to context_agent
    graph.add_edge("wait_for_async", "context_agent")

    return graph.compile()


def extract_output(state: dict) -> ContextNodeOutput:
    """extract the ContextNodeOutput from final graph state."""
    return ContextNodeOutput(
        fact_check_results=state.get("fact_check_results", []),
        search_results=state.get("search_results", {}),
        scraped_pages=state.get("scraped_pages", []),
    )
