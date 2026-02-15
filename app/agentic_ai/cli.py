"""
interactive CLI for testing the context search agent.

usage: python -m app.agentic_ai.cli
"""

from __future__ import annotations

import asyncio
import sys
import os

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
from app.models.factchecking import FactCheckResult
from app.agentic_ai.config import MAX_ITERATIONS, DEFAULT_MODEL, ADJUDICATION_MODEL


def _build_graph():
    """build the context agent graph with real tool implementations."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from app.agentic_ai.graph import build_graph
    from app.agentic_ai.tools.fact_check_search import FactCheckSearchTool
    from app.agentic_ai.tools.web_search import WebSearchTool
    from app.agentic_ai.tools.page_scraper import PageScraperTool

    model = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, temperature=0)
    adj_model = ChatGoogleGenerativeAI(model=ADJUDICATION_MODEL, temperature=0)
    fact_checker = FactCheckSearchTool()
    web_searcher = WebSearchTool()
    page_scraper = PageScraperTool()

    return build_graph(model, fact_checker, web_searcher, page_scraper, adj_model)


async def run_context_agent(text: str) -> FactCheckResult | ContextNodeOutput:
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
        "adjudication_result": None,
    }

    final_state = await graph.ainvoke(initial_state)
    return extract_output(final_state)


def display_result(output: FactCheckResult | ContextNodeOutput) -> None:
    """display the output in a formatted way."""
    if isinstance(output, FactCheckResult):
        _display_fact_check_result(output)
    else:
        _display_context_output(output)


def _display_fact_check_result(result: FactCheckResult) -> None:
    """display a FactCheckResult with verdicts and summary."""
    print_section("Adjudication Result")

    for ds_result in result.results:
        for i, cv in enumerate(ds_result.claim_verdicts, 1):
            verdict_color = {
                "Verdadeiro": Colors.GREEN,
                "Falso": Colors.RED,
                "Fora de Contexto": Colors.YELLOW,
            }.get(cv.verdict, Colors.BLUE)

            print(f"\n  {Colors.BOLD}[Claim {i}]{Colors.END} {cv.claim_text}")
            print(f"      Verdict: {verdict_color}{Colors.BOLD}{cv.verdict}{Colors.END}")
            print(f"      {cv.justification}")

            if cv.citations_used:
                print(f"      {Colors.CYAN}Citations:{Colors.END}")
                for c in cv.citations_used:
                    print(f"        - {c.title} ({c.publisher}): {c.url}")

    if result.overall_summary:
        print_section("Overall Summary")
        print(f"  {result.overall_summary}")


def _display_context_output(output: ContextNodeOutput) -> None:
    """display the raw context agent output."""
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
    print(f"  Context model: {DEFAULT_MODEL}")
    print(f"  Adjudication model: {ADJUDICATION_MODEL}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  GOOGLE_API_KEY: {'set' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    print(f"  GOOGLE_SEARCH_API_KEY: {'set' if os.getenv('GOOGLE_SEARCH_API_KEY') else 'NOT SET'}")
    print(f"  GOOGLE_CSE_CX: {'set' if os.getenv('GOOGLE_CSE_CX') else 'NOT SET'}")
    print(f"  OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"  APIFY_TOKEN: {'set' if os.getenv('APIFY_TOKEN') else 'NOT SET'}")


# ===== interactive session =====

def _make_empty_state() -> dict:
    """create a fresh empty state for the context agent graph."""
    return {
        "messages": [],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": "",
        "adjudication_result": None,
    }


def _handle_cmd_state(session_state: dict) -> None:
    """show accumulated sources using the context formatter."""
    from app.agentic_ai.prompts.context_formatter import format_context

    formatted = format_context(
        session_state.get("fact_check_results", []),
        session_state.get("search_results", {}),
        session_state.get("scraped_pages", []),
    )
    print_section("Collected Sources")
    print(formatted)

    fc = len(session_state.get("fact_check_results", []))
    sr = sum(len(v) for v in session_state.get("search_results", {}).values())
    sp = len(session_state.get("scraped_pages", []))
    print(f"\n  {Colors.BOLD}Totals:{Colors.END} {fc} fact-checks, {sr} search results, {sp} scraped pages")


def _handle_cmd_prompt(session_state: dict) -> None:
    """rebuild and display the current system prompt from state."""
    from app.agentic_ai.prompts.system_prompt import build_system_prompt

    prompt = build_system_prompt(
        formatted_data_sources=session_state.get("formatted_data_sources", ""),
        iteration_count=session_state.get("iteration_count", 0),
        fact_check_results=session_state.get("fact_check_results", []),
        search_results=session_state.get("search_results", {}),
        scraped_pages=session_state.get("scraped_pages", []),
    )
    print_section("Current System Prompt")
    print(prompt)


def _handle_cmd_messages(session_state: dict) -> None:
    """display message history with role, content preview, and tool calls."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

    messages = session_state.get("messages", [])
    if not messages:
        print_info("No messages in history")
        return

    print_section(f"Message History ({len(messages)} messages)")

    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            role = f"{Colors.GREEN}human{Colors.END}"
        elif isinstance(msg, AIMessage):
            role = f"{Colors.CYAN}ai{Colors.END}"
        elif isinstance(msg, ToolMessage):
            role = f"{Colors.YELLOW}tool({msg.name}){Colors.END}"
        elif isinstance(msg, SystemMessage):
            role = f"{Colors.BLUE}system{Colors.END}"
        else:
            role = type(msg).__name__

        content_preview = ""
        if msg.content:
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_preview = content_str[:120].replace("\n", " ")
            if len(content_str) > 120:
                content_preview += "..."

        print(f"\n  {Colors.BOLD}[{i}]{Colors.END} {role}")
        if content_preview:
            print(f"      {content_preview}")

        # show tool calls on AI messages
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            _print_tool_calls(msg.tool_calls, indent="      ")


def _print_tool_calls(tool_calls: list, indent: str = "  ") -> None:
    """print tool call details from an AI message."""
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("args", {})
        # print each arg on its own line for readability
        if not args:
            print(f"{indent}{Colors.YELLOW}-> {name}(){Colors.END}")
            continue
        print(f"{indent}{Colors.YELLOW}-> {name}({Colors.END}")
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            print(f"{indent}   {Colors.YELLOW}{k}={v_str}{Colors.END}")
        print(f"{indent}{Colors.YELLOW}){Colors.END}")


async def _run_streaming_query(graph, session_state: dict, text: str) -> None:
    """run a query with streaming, printing node events in real-time.

    mutates session_state in place by accumulating updates from the stream.
    each node yields its raw output (before reducer application), so we apply
    the same merge logic that ContextAgentState uses.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from app.agentic_ai.state import _merge_search_results

    # prepare state for this run
    session_state["messages"].append(HumanMessage(content=text))
    session_state["formatted_data_sources"] = text
    session_state["iteration_count"] = 0

    # snapshot the input state so the graph gets a clean copy
    run_input = {
        "messages": list(session_state["messages"]),
        "fact_check_results": list(session_state["fact_check_results"]),
        "search_results": {k: list(v) for k, v in session_state["search_results"].items()},
        "scraped_pages": list(session_state["scraped_pages"]),
        "iteration_count": session_state["iteration_count"],
        "pending_async_count": session_state["pending_async_count"],
        "formatted_data_sources": session_state["formatted_data_sources"],
    }

    iteration = 0

    async for event in graph.astream(run_input, stream_mode="updates"):
        for node_name, update in event.items():
            # accumulate streamed updates into session_state
            if "messages" in update:
                session_state["messages"].extend(update["messages"])
            if "fact_check_results" in update:
                session_state["fact_check_results"].extend(update["fact_check_results"])
            if "search_results" in update:
                session_state["search_results"] = _merge_search_results(
                    session_state["search_results"], update["search_results"]
                )
            if "scraped_pages" in update:
                session_state["scraped_pages"].extend(update["scraped_pages"])
            if "iteration_count" in update:
                session_state["iteration_count"] = update["iteration_count"]
            if "pending_async_count" in update:
                session_state["pending_async_count"] = update["pending_async_count"]
            if "adjudication_result" in update:
                session_state["adjudication_result"] = update["adjudication_result"]

            # display
            if node_name == "context_agent":
                iteration += 1
                print(f"\n{Colors.BOLD}[iteration {iteration}]{Colors.END} {Colors.CYAN}context_agent{Colors.END}")

                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
                        if has_tools:
                            _print_tool_calls(msg.tool_calls)
                            # show a short reasoning preview when there are more tool calls
                            if msg.content:
                                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                                if content_str.strip():
                                    preview = content_str.strip()[:200]
                                    print(f"  {Colors.BLUE}reasoning: {preview}{Colors.END}")
                        else:
                            print(f"  {Colors.GREEN}-> no tool calls, ending{Colors.END}")
                            # print the full agent response
                            content_str = ""
                            if msg.content:
                                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                            if content_str.strip():
                                print(f"\n{Colors.BOLD}Agent response:{Colors.END}")
                                print(content_str.strip())
                            else:
                                # model returned empty content — show a state summary
                                fc = len(session_state.get("fact_check_results", []))
                                sr = sum(len(v) for v in session_state.get("search_results", {}).values())
                                sp = len(session_state.get("scraped_pages", []))
                                print(f"\n  {Colors.CYAN}(agent returned no text — "
                                      f"collected {fc} fact-checks, {sr} search results, "
                                      f"{sp} scraped pages. Use /state to inspect){Colors.END}")

            elif node_name == "tools":
                tool_msgs = update.get("messages", [])
                count = len(tool_msgs)
                print(f"{Colors.YELLOW}[tools]{Colors.END} executed {count} tool call{'s' if count != 1 else ''}")

            elif node_name == "wait_for_async":
                print(f"{Colors.BLUE}[wait_for_async]{Colors.END} waiting for pending operations...")

            elif node_name == "adjudication":
                adj_result = update.get("adjudication_result")
                if adj_result:
                    print(f"\n{Colors.BOLD}[adjudication]{Colors.END}")
                    _display_fact_check_result(adj_result)

    print(f"\n{Colors.GREEN}Done ({iteration} iterations){Colors.END}")


def _handle_cmd_verdict(session_state: dict) -> None:
    """re-display the last adjudication result from state."""
    adj_result = session_state.get("adjudication_result")
    if not adj_result:
        print_info("No adjudication result yet. Run a query first.")
        return
    _display_fact_check_result(adj_result)


def _print_session_help() -> None:
    """display available session commands."""
    print(f"  {Colors.BOLD}/state{Colors.END}     — show collected sources")
    print(f"  {Colors.BOLD}/verdict{Colors.END}   — show last adjudication verdict")
    print(f"  {Colors.BOLD}/prompt{Colors.END}    — show current system prompt")
    print(f"  {Colors.BOLD}/messages{Colors.END}  — show message history")
    print(f"  {Colors.BOLD}/config{Colors.END}    — show configuration")
    print(f"  {Colors.BOLD}/reset{Colors.END}     — clear state, start fresh")
    print(f"  {Colors.BOLD}/help{Colors.END}      — show this help")
    print(f"  {Colors.BOLD}/quit{Colors.END}      — exit session")


async def _interactive_session_async() -> None:
    """async REPL loop for the interactive session."""
    import copy

    print_header("Interactive Session")
    print_info("Type a claim to verify, or use /commands to inspect state")
    _print_session_help()
    print()

    graph = _build_graph()
    session_state = _make_empty_state()

    while True:
        try:
            user_input = input(f"{Colors.BOLD}agent> {Colors.END}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print_info("Exiting session")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print_info("Exiting session")
            break

        # slash commands
        if user_input == "/state":
            _handle_cmd_state(session_state)
            continue
        if user_input == "/verdict":
            _handle_cmd_verdict(session_state)
            continue
        if user_input == "/prompt":
            _handle_cmd_prompt(session_state)
            continue
        if user_input == "/messages":
            _handle_cmd_messages(session_state)
            continue
        if user_input == "/config":
            show_config()
            continue
        if user_input == "/help":
            _print_session_help()
            continue
        if user_input == "/reset":
            session_state = _make_empty_state()
            print_success("State cleared")
            continue
        if user_input.startswith("/"):
            print_warning(f"Unknown command: {user_input}. Type /help for available commands")
            continue

        # run the claim through the graph with streaming
        state_backup = copy.deepcopy(session_state)
        try:
            await _run_streaming_query(graph, session_state, user_input)
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # restore state to pre-query snapshot
            session_state.clear()
            session_state.update(state_backup)
            print_warning("State restored to pre-query snapshot due to error")


def handle_interactive_session() -> None:
    """launch the interactive stateful session."""
    asyncio.run(_interactive_session_async())


def main():
    print_header("Context Agent — Interactive CLI")

    menu = Menu("Context Agent")
    menu.add_option("Verificar texto", handle_text_input)
    menu.add_option("Verificar URL", handle_url_input)
    menu.add_separator()
    menu.add_option("Sessão interativa", handle_interactive_session)
    menu.add_option("Configurações", show_config)
    menu.run()


if __name__ == "__main__":
    main()
