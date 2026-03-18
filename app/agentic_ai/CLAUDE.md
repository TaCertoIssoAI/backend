# Agentic AI — LangGraph Fact-Checking Pipeline

This module implements the core fact-checking engine as a LangGraph state graph. An LLM agent iteratively gathers evidence using tools, then a separate adjudication step produces grounded verdicts.

## Architecture

```
START → format_input → context_agent ⇄ tools (loop, max 5 iterations)
                            ↓
                      adjudication → prepare_retry → END
                                          ↓ (if insufficient sources)
                                   retry_context_agent ⇄ retry_tools → adjudication
```

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | public API — `run_fact_check(data_sources)` builds the graph and returns `GraphOutput` |
| `graph.py` | graph definition — wires nodes, tools, and conditional edges |
| `state.py` | `ContextAgentState` typed state schema extending LangGraph `MessagesState` |
| `config.py` | constants: iteration limits, timeouts, domain config, model names |

## Subfolders

- **nodes/** — graph node functions: `format_input`, `context_agent`, `adjudication`, `retry_context_agent`, `check_edges`
- **tools/** — LangChain `@tool` implementations: `WebSearchTool`, `FactCheckSearchTool`, `PageScraperTool`, plus `protocols.py` for dependency injection
- **prompts/** — system prompt, adjudication prompt, and context formatter that builds the evidence block for the LLM
- **controlflow/** — routing logic: `wait_for_async` (link expansion), `prepare_retry` (retry decision)
- **utils/** — helpers like `link_expander.py` for expanding URLs found in input
- **context/** — context-building utilities used by nodes
- **tests/** — unit tests for nodes, tools, and prompts

## Graph State

`ContextAgentState` holds all data flowing through the graph:
- `messages` — LangChain message history (agent ↔ tools conversation)
- `data_sources` — append-only input `DataSource` list
- `fact_check_results`, `search_results`, `scraped_pages` — accumulated evidence
- `iteration_count` — loop counter (max `MAX_ITERATIONS=5`)
- `retry_count` — retry counter (max `MAX_RETRY_COUNT=1`)
- `adjudication_result` — final `FactCheckResult` set by adjudication node

## Tool Node Pattern

The `ToolNode` wrapper in `graph.py` (`_make_tool_node_with_state_update`) runs LangGraph's built-in `ToolNode` then parses tool outputs into the typed state fields. This is necessary because the default `ToolNode` only updates messages.

## Models and Timeouts

- Context agent: `gemini-2.5-flash-lite` (temperature 0)
- Adjudication: `gemini-2.5-flash-lite` with `thinking_budget=1024`
- Search timeout: 35s per query
- Scrape timeout: 30s per page
- Adjudication timeout: 20s per attempt, up to 2 retries

## How to Add a New Tool

1. Create a class in `tools/` implementing the relevant protocol from `tools/protocols.py`
2. Add a `@tool` wrapper function in `graph.py:_make_tools()`
3. Add state parsing logic in `_make_tool_node_with_state_update()`
4. Update the system prompt in `prompts/system_prompt.py` to describe the new tool

## Running

```python
from app.agentic_ai.run import run_fact_check
result = await run_fact_check(data_sources)  # returns GraphOutput
```
