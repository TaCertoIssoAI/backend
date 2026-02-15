# Agentic AI — Context Search Loop Implementation Plan

## Context

The current fact-checking pipeline uses a fixed, linear flow: extract claims → retrieve evidence → adjudicate. This works but has limitations — it always runs all retrievers regardless of evidence quality, can't loop back when sources are insufficient, and separates claim understanding from search.

This plan implements an **agentic state machine** using LangGraph where a context-gathering agent decides what to search, evaluates whether it has enough sources, and only proceeds to adjudication when confident. The agent holds typed context dataclasses, uses tools for search/scraping, and builds its evidence iteratively through a loop.

**Scope**: Context agent graph only. Adjudication agent is out of scope but we define the output contract for it.

---

## 1. Graph Nodes and Edges

```
        ┌──────────┐
        │  START    │
        └────┬─────┘
             │
             ▼
    ┌────────────────┐
    │  context_agent │◄─────────────────┐
    │  (LLM + tools) │                  │
    └───────┬────────┘                  │
            │                           │
            ▼                           │
    ┌────────────────┐    ┌─────────────┴──────┐
    │  check_edges   │───►│  wait_for_async    │
    │  (router)      │    │  (wait + re-enter) │
    └───────┬────────┘    └────────────────────┘
            │
            ▼
    ┌────────────────┐
    │     END        │
    └────────────────┘
```

### Nodes

| Node | Type | Description |
|------|------|-------------|
| `context_agent` | LangGraph `ToolNode` + LLM | The core agent. Receives system prompt with formatted context, calls tools, accumulates `ContextEntry` objects in state. Runs until it makes 0 tool calls (signals "done"). |
| `check_edges` | Python function (router) | Evaluates 3 conditions after `context_agent` returns with no tool calls. |
| `wait_for_async` | Python function | Blocks until pending async DataSources resolve, appends them to state, resets `no_tool_call` flag, routes back to `context_agent`. |

### Edges

| From | To | Condition |
|------|------|-----------|
| `START` | `context_agent` | Always |
| `context_agent` | `context_agent` | Tool calls pending (LangGraph built-in tool loop) |
| `context_agent` | `check_edges` | No tool calls (agent returned text-only) |
| `check_edges` | `wait_for_async` | `pending_async_count > 0` |
| `check_edges` | `END` | `pending_async_count == 0` OR `iteration_count >= MAX_ITERATIONS` |
| `wait_for_async` | `context_agent` | Always (after async resolves) |

---

## 2. Tools

### 2.1 Tool: `search_fact_check_api`

**Interface:**
```python
def search_fact_check_api(queries: list[str]) -> list[FactCheckApiContext]
```

**Implementation:** Wraps existing `GoogleFactCheckGatherer._parse_response()` logic. For each query, calls the Fact-Check API endpoint, parses results into `FactCheckApiContext` objects with `reliability = MUITO_CONFIAVEL`.

**Reuses:**
- `app/ai/context/factcheckapi/google_factcheck_gatherer.py` — API call logic, rating mapping (`map_english_rating_to_portuguese`)
- Base URL and parameter structure already implemented

**Async:** Uses `httpx.AsyncClient` (already async in existing code). Runs all queries concurrently with `asyncio.gather()`.

### 2.2 Tool: `search_web`

**Interface:**
```python
def search_web(queries: list[str], max_results_per_search: int = 5) -> dict[str, list[GoogleSearchContext]]
```

**Returns:** Dict keyed by domain group:
```python
{
    "geral": [GoogleSearchContext(...)],      # reliability = NEUTRO
    "g1": [GoogleSearchContext(...)],          # reliability = NEUTRO
    "estadao": [GoogleSearchContext(...)],     # reliability = NEUTRO
    "aosfatos": [GoogleSearchContext(...)],    # reliability = MUITO_CONFIAVEL
    "folha": [GoogleSearchContext(...)],       # reliability = NEUTRO
}
```

**Implementation:** For each query, fires 5 parallel searches:
1. General (no domain filter) — uses trusted domains from `app/config/trusted_domains.py` via `_build_search_query_with_domains()` pattern from `WebSearchGatherer`
2. `site_search="g1.globo.com", site_search_filter="i"`
3. `site_search="estadao.com.br", site_search_filter="i"`
4. `site_search="aosfatos.org", site_search_filter="i"`
5. `site_search="folha.uol.com.br", site_search_filter="i"`

Domains are hardcoded inside the tool. `aosfatos` gets `MUITO_CONFIAVEL` (it's a fact-checker), the rest get `NEUTRO`.

**Reuses:**
- `app/ai/context/web/google_search.py` — `google_search()` function (already async, supports `site_search` param)
- `app/ai/context/web/web_search_gatherer.py` — `_build_search_query_with_domains()` pattern for general search
- `app/config/trusted_domains.py` — `get_trusted_domains()` for allowlisted domains
- `scripts/playground/google/google_search_cli.py` — domain filtering pattern reference

**Async:** `google_search()` is already async. Uses `asyncio.gather()` to run all 5 searches × N queries concurrently.

### 2.3 Tool: `scrape_pages`

**Interface:**
```python
class ScrapeTarget(BaseModel):
    url: str
    title: str

def scrape_pages(targets: list[ScrapeTarget]) -> list[WebScrapeContext]
```

**Implementation:** For each target, scrapes the page content and returns a `WebScrapeContext`. Sets `parent_id` from the `GoogleSearchContext.id` that produced the URL (tracked via state).

**Reuses:**
- `app/ai/context/web/apify_utils.py` — `scrapeGenericUrl()` (production-ready, handles platform detection, fallbacks)
- `app/ai/context/web/models.py` — `WebContentResult` for intermediate parsing
- `app/ai/pipeline/link_context_expander.py` — `expand_link_context()` async wrapper pattern
- Domain-specific extractors from `scripts/playground/` (g1, estadao, folha, aosfatos) as reference for field extraction

**Async:** `scrapeGenericUrl()` is already async. Uses `asyncio.gather()` with per-target timeout.

### 2.4 Tool Protocol (for mocking)

All tools are wrapped behind a protocol so they can be swapped in tests:

```python
class FactCheckSearchProtocol(Protocol):
    async def search(self, queries: list[str]) -> list[FactCheckApiContext]: ...

class WebSearchProtocol(Protocol):
    async def search(self, queries: list[str], max_results_per_search: int = 5) -> dict[str, list[GoogleSearchContext]]: ...

class PageScraperProtocol(Protocol):
    async def scrape(self, targets: list[ScrapeTarget]) -> list[WebScrapeContext]: ...
```

---

## 3. Agent State

### 3.1 Graph State Schema

```python
from langgraph.graph import MessagesState

class ContextAgentState(MessagesState):
    # inputs
    data_sources: list[DataSource]

    # accumulated context (append-only)
    fact_check_results: list[FactCheckApiContext]
    search_results: dict[str, list[GoogleSearchContext]]
    scraped_pages: list[WebScrapeContext]

    # control flow
    iteration_count: int           # incremented each time context_agent runs
    pending_async_count: int       # decremented as async DataSources arrive
    new_data_sources: list[DataSource]  # async DataSources that just arrived
```

`MessagesState` gives us the built-in `messages` list for the LLM conversation. The typed context fields sit alongside it.

### 3.2 Context Data Storage

Each tool call appends to the corresponding typed list:
- `search_fact_check_api` → appends to `fact_check_results`
- `search_web` → merges into `search_results` dict (extends each domain key's list)
- `scrape_pages` → appends to `scraped_pages`

Context is **append-only** — the LLM naturally ignores weak entries when reasoning. All entries are preserved for logging/analytics.

### 3.3 Context Dataclasses

**File:** `app/models/agenticai.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class SourceReliability(str, Enum):
    MUITO_CONFIAVEL = "Muito confiável"
    POUCO_CONFIAVEL = "Pouco Confiável"
    NEUTRO = "Neutro"

@dataclass
class BaseContext:
    id: str
    url: str
    parent_id: Optional[str]
    reliability: SourceReliability

@dataclass
class FactCheckApiContext(BaseContext):
    title: str = ""
    publisher: str = ""
    rating: str = ""              # VerdictType value
    rating_comment: Optional[str] = None
    claim_text: str = ""          # the claim as seen by the fact-checker
    review_date: Optional[str] = None

@dataclass
class GoogleSearchContext(BaseContext):
    title: str = ""
    snippet: str = ""
    domain: str = ""
    position: int = 0            # rank in search results

@dataclass
class WebScrapeContext(BaseContext):
    title: str = ""
    content: str = ""
    extraction_status: str = ""  # success, failed, timeout
    extraction_tool: str = ""    # apify, beautifulsoup, trafilatura

class ScrapeTarget(BaseModel):
    url: str
    title: str
```

### 3.4 Output Contract (for future adjudication agent)

```python
@dataclass
class ContextNodeOutput:
    data_sources: list[DataSource]
    fact_check_results: list[FactCheckApiContext]
    search_results: dict[str, list[GoogleSearchContext]]
    scraped_pages: list[WebScrapeContext]
```

Built from graph state after the context agent finishes. The adjudication agent receives only structured dataclasses — no prompt artifacts.

---

## 4. Context Agent System Prompt

### 4.1 Prompt Structure (Portuguese)

The system prompt is assembled dynamically from a template + formatted context sections:

```
<ROLE_AND_TASK>
Você é um agente de pesquisa para verificação de fatos. Sua tarefa é reunir
fontes suficientes para que um agente adjudicador possa emitir um veredito
sobre o conteúdo recebido.

<TOOLS_DESCRIPTION>
Ferramentas disponíveis:
1. search_fact_check_api(queries) — busca em bases de fact-checking (Muito confiável)
2. search_web(queries, max_results_per_search) — busca web geral + domínios específicos (G1, Estadão, Aos Fatos, Folha)
3. scrape_pages(targets) — extrai conteúdo completo de páginas web

<STOP_CRITERIA>
Critérios para considerar fontes SUFICIENTES:
- Para cada afirmação identificada no conteúdo, deve existir ao menos:
  • 1 fonte "Muito confiável" que cubra o tema, OU
  • 2+ fontes "Neutro" que corroborem a mesma informação, sem fontes de
    confiabilidade igual ou maior dizendo o contrário
- Fontes "Pouco confiável" NUNCA são suficientes sozinhas
- Todas as afirmações verificáveis devem ter cobertura

Se esses critérios estão atendidos, NÃO chame mais ferramentas.
Se NÃO estão atendidos, faça mais buscas com queries diferentes ou mais específicas.

<ITERATION_INFO>
Iteração atual: {iteration_count}/{max_iterations}

<USER_CONTENT>
## Conteúdo recebido para verificação
{formatted_data_sources}

<CONTEXT_SECTIONS>
{formatted_context}
```

### 4.2 Context Formatting Function

The `formatted_context` is built by a function that reads the typed lists and assembles sections ordered by reliability:

```
## Fontes — Muito confiável

### Fact-Check API
[1] Publisher: Agência Lupa | Rating: Falso
    URL: https://lupa.uol.com.br/...
    Afirmação verificada: "Vacina X causa infertilidade"
    Data da revisão: 2025-01-10

[2] ...

### Busca Web — Aos Fatos
[3] Title: "Verificamos: vacina X não causa infertilidade"
    URL: https://aosfatos.org/... | Domain: aosfatos.org
    Snippet: "Segundo estudos clínicos..."

## Fontes — Neutro

### Busca Web — Geral
[4] Title: "Estudo sobre segurança de vacinas"
    URL: https://bbc.com/... | Domain: bbc.com
    Snippet: "..."

### Busca Web — G1
[5] ...

### Busca Web — Estadão
[6] ...

### Busca Web — Folha
[7] ...

## Fontes — Pouco confiável

### Conteúdo Extraído de Páginas
[8] Title: "..." | URL: https://...
    Status: success | Ferramenta: beautifulsoup
    Conteúdo (primeiros 500 chars): "..."
```

**Global numbering** across all sections so the agent can reference `[N]` unambiguously. Each entry is formatted by a `format_*` method on its dataclass.

### 4.3 New DataSource Prompt Section

When `wait_for_async` adds new DataSources and re-enters the agent:

```
## ⚠️ NOVO CONTEÚDO RECEBIDO — requer verificação
[N+1] (link_context) URL: https://folha.uol.com.br/...
      "Conteúdo completo extraído da página: ..."

## Contexto já coletado (não buscar novamente)
(existing sections listed normally)
```

This section is only present when `new_data_sources` is non-empty in state. After the agent processes it, `new_data_sources` is cleared.

---

## 5. Conditional Edges

### 5.1 `check_edges` Router Logic

```python
def check_edges(state: ContextAgentState) -> str:
    # safety cap
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"

    # async DataSources still pending
    if state["pending_async_count"] > 0:
        return "wait_for_async"

    # done
    return "end"
```

### 5.2 Async DataSource Handling

The `wait_for_async` node:
1. Blocks on an `asyncio.Event` or polls a shared queue until async DataSources resolve
2. Appends resolved DataSources to `state["data_sources"]` and `state["new_data_sources"]`
3. Decrements `pending_async_count`
4. Increments `iteration_count`
5. Returns to `context_agent`

The `context_agent` node sees the new DataSources highlighted in its prompt (section 4.3) and decides whether to make additional tool calls.

### 5.3 Iteration Counter

- `iteration_count` starts at 0, incremented each time `context_agent` runs
- `MAX_ITERATIONS` defaults to 5 (configurable)
- When hit, the graph proceeds to END regardless of source sufficiency
- The iteration count is shown to the agent in the prompt so it can prioritize

### 5.4 No-Tool-Call Detection

LangGraph's built-in `tools_condition` handles the tool loop. When the agent returns a message without tool calls, it routes to `check_edges`. This is standard LangGraph behavior — no custom logic needed.

---

## 6. Implementation Plan

### 6.1 File Structure

```
app/agentic-ai/
├── __init__.py
├── graph.py                  # LangGraph graph definition (nodes, edges, compilation)
├── state.py                  # ContextAgentState TypedDict
├── nodes/
│   ├── __init__.py
│   ├── context_agent.py      # context_agent node (builds prompt, invokes LLM)
│   ├── check_edges.py        # check_edges router function
│   └── wait_for_async.py     # wait_for_async node
├── tools/
│   ├── __init__.py
│   ├── protocols.py          # Tool protocols (FactCheckSearchProtocol, etc.)
│   ├── fact_check_search.py  # search_fact_check_api implementation
│   ├── web_search.py         # search_web implementation
│   └── page_scraper.py       # scrape_pages implementation
├── prompts/
│   ├── __init__.py
│   ├── system_prompt.py      # system prompt template and builder
│   └── context_formatter.py  # format context entries into prompt sections
├── config.py                 # agentic pipeline config (MAX_ITERATIONS, timeouts, etc.)
└── cli.py                    # CLI for testing the context agent

app/models/agenticai.py       # SourceReliability, BaseContext, FactCheckApiContext,
                              # GoogleSearchContext, WebScrapeContext, ScrapeTarget,
                              # ContextNodeOutput
```

### 6.2 New Files — What Each Does

**`app/models/agenticai.py`** — Data models (section 3.3). Pure dataclasses, no IO.

**`app/agentic-ai/state.py`** — `ContextAgentState(MessagesState)` TypedDict (section 3.1).

**`app/agentic-ai/tools/protocols.py`** — Protocol classes for all 3 tools (section 2.4). Used for dependency injection and testing.

**`app/agentic-ai/tools/fact_check_search.py`** — Implements `FactCheckSearchProtocol`.
- Reuses: `google_factcheck_gatherer.py` API call logic and `map_english_rating_to_portuguese()`
- Creates `FactCheckApiContext` objects instead of `Citation`
- All entries get `reliability = MUITO_CONFIAVEL`

**`app/agentic-ai/tools/web_search.py`** — Implements `WebSearchProtocol`.
- Reuses: `google_search()` from `app/ai/context/web/google_search.py`
- Reuses: `get_trusted_domains()` from `app/config/trusted_domains.py` for general search
- Reuses: Domain filtering pattern from `scripts/playground/google/google_search_cli.py` (the `site_search` + `site_search_filter="i"` pattern)
- Hardcoded domains: `g1.globo.com`, `estadao.com.br`, `aosfatos.org`, `folha.uol.com.br`
- `aosfatos` → `MUITO_CONFIAVEL`, rest → `NEUTRO`

**`app/agentic-ai/tools/page_scraper.py`** — Implements `PageScraperProtocol`.
- Reuses: `scrapeGenericUrl()` from `app/ai/context/web/apify_utils.py`
- Reuses: `WebContentResult.from_dict()` from `app/ai/context/web/models.py`
- Reuses: domain-specific extraction patterns from `scripts/playground/` (g1, estadao, folha, aosfatos explorers)
- Sets `parent_id` linking to the search result that provided the URL

**`app/agentic-ai/prompts/system_prompt.py`** — Template string and `build_system_prompt(state)` function (section 4.1).

**`app/agentic-ai/prompts/context_formatter.py`** — `format_context(state) -> str` function (section 4.2). Reads typed lists, groups by reliability, applies global numbering.

**`app/agentic-ai/nodes/context_agent.py`** — The main agent node.
- Builds system prompt via `build_system_prompt(state)`
- Binds tools to the LLM via `model.bind_tools([...])`
- Increments `iteration_count`
- Returns updated messages + state

**`app/agentic-ai/nodes/check_edges.py`** — Router (section 5.1). Pure function, no IO.

**`app/agentic-ai/nodes/wait_for_async.py`** — Waits for async DataSources (section 5.2). Appends to state, sets `new_data_sources` flag.

**`app/agentic-ai/graph.py`** — Compiles the LangGraph `StateGraph`:
```python
graph = StateGraph(ContextAgentState)
graph.add_node("context_agent", context_agent_node)
graph.add_node("check_edges", check_edges_node)
graph.add_node("wait_for_async", wait_for_async_node)

graph.add_edge(START, "context_agent")
graph.add_conditional_edges("context_agent", tools_condition, {
    "tools": "context_agent",  # built-in tool loop
    "__end__": "check_edges",  # no tool calls
})
graph.add_conditional_edges("check_edges", check_edges_router, {
    "wait_for_async": "wait_for_async",
    "end": END,
})
graph.add_edge("wait_for_async", "context_agent")
```

**`app/agentic-ai/config.py`** — Constants and config:
- `MAX_ITERATIONS = 5`
- `SEARCH_TIMEOUT_PER_QUERY = 15.0`
- `SCRAPE_TIMEOUT_PER_PAGE = 30.0`
- Domain mappings and reliability assignments

**`app/agentic-ai/cli.py`** — Interactive CLI (section 9).

### 6.3 Existing Code Reuse Map

| New Module | Reuses From | What |
|------------|-------------|------|
| `tools/fact_check_search.py` | `app/ai/context/factcheckapi/google_factcheck_gatherer.py` | API call, `_parse_response`, `map_english_rating_to_portuguese` |
| `tools/web_search.py` | `app/ai/context/web/google_search.py` | `google_search()` async function |
| `tools/web_search.py` | `app/config/trusted_domains.py` | `get_trusted_domains()` for general search domain filter |
| `tools/web_search.py` | `app/ai/context/web/web_search_gatherer.py` | `_build_search_query_with_domains()` pattern |
| `tools/page_scraper.py` | `app/ai/context/web/apify_utils.py` | `scrapeGenericUrl()`, platform detection |
| `tools/page_scraper.py` | `app/ai/context/web/models.py` | `WebContentResult` |
| `tools/page_scraper.py` | `app/ai/pipeline/link_context_expander.py` | `extract_links()`, async wrapper pattern |
| `cli.py` | `scripts/playground/common.py` | `Colors`, `print_header`, `print_section`, `with_spinner`, `prompt_input`, `Menu` |
| `models/agenticai.py` | `app/models/factchecking.py` | `VerdictType` for rating field |

---

## 7. Testing

Every module gets unit tests. Test files mirror source structure:

```
app/agentic-ai/tests/
├── __init__.py
├── test_state.py                  # state initialization, append semantics
├── test_graph.py                  # graph compilation, edge routing
├── nodes/
│   ├── test_context_agent.py      # prompt building, tool binding, iteration increment
│   ├── test_check_edges.py        # all 3 routing conditions
│   └── test_wait_for_async.py     # async resolution, state update
├── tools/
│   ├── test_fact_check_search.py  # API parsing, rating mapping, error handling
│   ├── test_web_search.py         # 5-way parallel search, domain filtering, result grouping
│   └── test_page_scraper.py       # scraping, timeout handling, parent_id linking
└── prompts/
    ├── test_system_prompt.py      # prompt assembly, section ordering
    └── test_context_formatter.py  # formatting, global numbering, reliability grouping
```

**Key test scenarios:**
- **Tools**: Mock `httpx` / `scrapeGenericUrl` via protocols. Test happy path, timeouts, empty results, API errors.
- **Router**: Test all 3 edge conditions: `pending_async > 0`, `iteration >= MAX`, normal end.
- **Prompt**: Test that context sections appear in reliability order, global numbering is contiguous, new DataSources get the warning header.
- **Graph integration**: Use mock tool protocols to run the full graph with fake API responses and verify state accumulation.

---

## 8. Code Modelling — Mockable IO

All IO operations are behind protocols (defined in `tools/protocols.py`):

```python
class FactCheckSearchProtocol(Protocol):
    async def search(self, queries: list[str]) -> list[FactCheckApiContext]: ...

class WebSearchProtocol(Protocol):
    async def search(self, queries: list[str], max_results_per_search: int = 5) -> dict[str, list[GoogleSearchContext]]: ...

class PageScraperProtocol(Protocol):
    async def scrape(self, targets: list[ScrapeTarget]) -> list[WebScrapeContext]: ...

class LLMProtocol(Protocol):
    def bind_tools(self, tools: list) -> Any: ...
    async def ainvoke(self, messages: list) -> Any: ...
```

The graph accepts these protocols via dependency injection:

```python
def build_graph(
    llm: LLMProtocol,
    fact_check_searcher: FactCheckSearchProtocol,
    web_searcher: WebSearchProtocol,
    page_scraper: PageScraperProtocol,
) -> CompiledGraph:
```

Tests inject mock implementations that return canned data without hitting any external APIs.

---

## 9. CLI

**File:** `app/agentic-ai/cli.py`

Interactive CLI for testing the context agent from the command line. Reuses CLI utilities from `scripts/playground/common.py`.

**Features:**
- Input text directly or paste a URL
- Watch the agent loop in real-time (show tool calls, context accumulation, iteration count)
- Display final `ContextNodeOutput` formatted with context sections
- Configurable: model, max iterations, timeouts

**Structure:**
```python
# usage: python -m app.agentic-ai.cli

def main():
    print_header("Context Agent — Interactive CLI")

    # menu: enter text, enter URL, configure, quit
    menu = Menu("Context Agent")
    menu.add_option("Verificar texto", handle_text_input)
    menu.add_option("Verificar URL", handle_url_input)
    menu.add_option("Configurações", show_config)
    menu.run()

def handle_text_input():
    text = prompt_multiline("Cole o texto para verificar")
    data_sources = [DataSource(id=uuid4(), source_type="original_text", original_text=text)]
    result = with_spinner(lambda: asyncio.run(run_context_agent(data_sources)), "Pesquisando...")
    display_result(result)

def display_result(output: ContextNodeOutput):
    print_section("Fact-Check API Results")
    for entry in output.fact_check_results:
        # formatted display
    print_section("Web Search Results")
    for domain, entries in output.search_results.items():
        # formatted display per domain
    print_section("Scraped Pages")
    for entry in output.scraped_pages:
        # formatted display
```

---

## Verification

### How to test end-to-end:

1. **Unit tests**: `pytest app/agentic-ai/tests/ -v`
2. **CLI smoke test**: `python -m app.agentic-ai.cli` → paste a claim → verify tools are called, context accumulates, agent stops when satisfied
3. **Integration test with mocks**: Run the compiled graph with mock tool protocols, assert state contains expected context entries after N iterations
4. **Edge case tests**:
   - Claim with 0 fact-check results → agent should fall back to web search + scraping
   - All API timeouts → agent should stop at MAX_ITERATIONS with whatever it has
   - Async DataSource arrives after agent's first pass → verify re-entry with new content highlighted

### Environment requirements:
```bash
export GOOGLE_SEARCH_API_KEY=...
export GOOGLE_CSE_CX=...
export GOOGLE_API_KEY=...
export APIFY_TOKEN=...
export OPENAI_API_KEY=...
pip install langgraph  # if not already installed
```
