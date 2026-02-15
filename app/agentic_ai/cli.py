"""
interactive CLI for testing the context search agent.

usage: python -m app.agentic_ai.cli
"""

from __future__ import annotations

import asyncio
import sys
import os
from uuid import uuid4

# add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.playground.common import (
    Colors,
    Menu,
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
    print_warning,
    prompt_input,
    prompt_multiline,
    with_spinner,
)

from app.models.agenticai import ContextNodeOutput
from app.agentic_ai.config import MAX_ITERATIONS, DEFAULT_MODEL


def _build_graph():
    """build the context agent graph with real tool implementations."""
    from langchain_openai import ChatOpenAI
    from app.agentic_ai.graph import build_graph
    from app.agentic_ai.tools.fact_check_search import FactCheckSearchTool
    from app.agentic_ai.tools.web_search import WebSearchTool
    from app.agentic_ai.tools.page_scraper import PageScraperTool

    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    fact_checker = FactCheckSearchTool()
    web_searcher = WebSearchTool()
    page_scraper = PageScraperTool()

    return build_graph(model, fact_checker, web_searcher, page_scraper)


async def run_context_agent(text: str) -> ContextNodeOutput:
    """run the full context search loop on a text input."""
    from app.agentic_ai.graph import extract_output
    from langchain_core.messages import HumanMessage

    graph = _build_graph()

    initial_state = {
        "messages": [HumanMessage(content=text)],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": text,
    }

    final_state = await graph.ainvoke(initial_state)
    return extract_output(final_state)


def display_result(output: ContextNodeOutput) -> None:
    """display the context agent output in a formatted way."""

    # fact-check results
    print_section("Fact-Check API Results")
    if output.fact_check_results:
        for i, entry in enumerate(output.fact_check_results, 1):
            print(f"\n  {Colors.BOLD}[{i}]{Colors.END} {entry.title}")
            print(f"      Publisher: {entry.publisher}")
            print(f"      Rating: {Colors.BOLD}{entry.rating}{Colors.END}")
            print(f"      Claim: {entry.claim_text[:100]}")
            print(f"      URL: {Colors.CYAN}{entry.url}{Colors.END}")
            if entry.review_date:
                print(f"      Date: {entry.review_date}")
    else:
        print_warning("  No fact-check results found")

    # web search results
    print_section("Web Search Results")
    for domain_key, entries in output.search_results.items():
        if entries:
            print(f"\n  {Colors.BOLD}{domain_key.upper()}{Colors.END} ({len(entries)} results)")
            for j, entry in enumerate(entries[:3], 1):
                print(f"    [{j}] {entry.title}")
                print(f"        {Colors.CYAN}{entry.url}{Colors.END}")
                if entry.snippet:
                    print(f"        {entry.snippet[:120]}...")

    total_search = sum(len(v) for v in output.search_results.values())
    if total_search == 0:
        print_warning("  No web search results found")

    # scraped pages
    print_section("Scraped Pages")
    if output.scraped_pages:
        for i, entry in enumerate(output.scraped_pages, 1):
            status_color = Colors.GREEN if entry.extraction_status == "success" else Colors.RED
            print(f"\n  [{i}] {entry.title}")
            print(f"      URL: {Colors.CYAN}{entry.url}{Colors.END}")
            print(f"      Status: {status_color}{entry.extraction_status}{Colors.END}")
            if entry.content:
                print(f"      Content preview: {entry.content[:150]}...")
    else:
        print_info("  No pages scraped")

    # summary
    print_section("Summary")
    print_success(f"  Fact-checks: {len(output.fact_check_results)}")
    print_success(f"  Search results: {total_search}")
    print_success(f"  Scraped pages: {len(output.scraped_pages)}")


def handle_text_input() -> None:
    """handle text verification input."""
    text = prompt_multiline("Cole o texto para verificar")
    if not text.strip():
        print_error("Empty input, skipping")
        return

    print_info(f"Verifying: {text[:100]}...")

    def run():
        return asyncio.run(run_context_agent(text))

    result = with_spinner(run, "Pesquisando fontes...")
    display_result(result)


def handle_url_input() -> None:
    """handle URL verification input."""
    url = prompt_input("URL para verificar")
    if not url.strip():
        print_error("Empty URL, skipping")
        return

    print_info(f"Verifying URL: {url}")

    def run():
        return asyncio.run(run_context_agent(f"Verifique o conteúdo desta URL: {url}"))

    result = with_spinner(run, "Pesquisando fontes...")
    display_result(result)


def show_config() -> None:
    """display current configuration."""
    print_section("Configuration")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  GOOGLE_API_KEY: {'set' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    print(f"  GOOGLE_SEARCH_API_KEY: {'set' if os.getenv('GOOGLE_SEARCH_API_KEY') else 'NOT SET'}")
    print(f"  GOOGLE_CSE_CX: {'set' if os.getenv('GOOGLE_CSE_CX') else 'NOT SET'}")
    print(f"  OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"  APIFY_TOKEN: {'set' if os.getenv('APIFY_TOKEN') else 'NOT SET'}")


def main():
    print_header("Context Agent — Interactive CLI")

    menu = Menu("Context Agent")
    menu.add_option("Verificar texto", handle_text_input)
    menu.add_option("Verificar URL", handle_url_input)
    menu.add_separator()
    menu.add_option("Configurações", show_config)
    menu.run()


if __name__ == "__main__":
    main()
