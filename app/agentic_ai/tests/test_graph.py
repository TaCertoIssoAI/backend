"""tests for graph compilation and structure."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    ScrapeTarget,
    WebScrapeContext,
    SourceReliability,
    ContextNodeOutput,
)
from app.models.factchecking import (
    FactCheckResult,
    DataSourceResult,
    ClaimVerdict,
    LLMAdjudicationOutput,
    LLMDataSourceResult,
    LLMClaimVerdict,
)
from app.models.commondata import DataSource
from app.agentic_ai.graph import build_graph, extract_output


class MockFactChecker:
    async def search(self, queries):
        return [
            FactCheckApiContext(
                id="fc-1",
                url="https://factcheck.org/test",
                parent_id=None,
                reliability=SourceReliability.MUITO_CONFIAVEL,
                title="Test FC",
                publisher="TestPub",
                rating="Falso",
                claim_text="test claim",
            )
        ]


class MockWebSearcher:
    async def search(self, queries, max_results_per_domain=5, max_results_general=5):
        return {
            "geral": [
                GoogleSearchContext(
                    id="gs-1",
                    url="https://test.com",
                    parent_id=None,
                    reliability=SourceReliability.NEUTRO,
                    title="Test Search",
                    snippet="snippet",
                    domain="test.com",
                )
            ],
            "g1": [],
            "estadao": [],
            "aosfatos": [],
            "folha": [],
        }


class MockScraper:
    async def scrape(self, targets):
        return []


def _make_mock_model():
    """create a mock LLM that returns no tool calls (immediate stop)."""
    from langchain_core.messages import AIMessage

    model = MagicMock()
    # bind_tools returns another mock that has ainvoke
    bound = AsyncMock()
    bound.ainvoke = AsyncMock(
        return_value=AIMessage(content="Sources are sufficient. Done.")
    )
    model.bind_tools = MagicMock(return_value=bound)
    return model


def _make_mock_adjudication_model():
    """create a mock adjudication LLM that returns structured output."""
    model = MagicMock()

    adjudication_output = LLMAdjudicationOutput(
        results=[
            LLMDataSourceResult(
                data_source_id=None,
                claim_verdicts=[
                    LLMClaimVerdict(
                        claim_id=None,
                        claim_text="Test claim",
                        verdict="Falso",
                        justification="Contradicted by sources [1].",
                        citations_used=[],
                    )
                ],
            )
        ],
        overall_summary="Test claim is false.",
    )

    structured = AsyncMock()
    structured.ainvoke = AsyncMock(return_value=adjudication_output)
    model.with_structured_output = MagicMock(return_value=structured)
    return model


def test_graph_compiles():
    model = _make_mock_model()
    adj_model = _make_mock_adjudication_model()
    graph = build_graph(model, MockFactChecker(), MockWebSearcher(), MockScraper(), adj_model)
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_runs_to_completion():
    model = _make_mock_model()
    adj_model = _make_mock_adjudication_model()
    graph = build_graph(model, MockFactChecker(), MockWebSearcher(), MockScraper(), adj_model)

    initial_state = {
        "messages": [],
        "data_sources": [DataSource(id="ds-1", source_type="original_text", original_text="Test claim")],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": "",
        "run_id": "test-run",
        "adjudication_result": None,
        "retry_count": 0,
        "retry_context": None,
        "seen_source_keys": set(),
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["iteration_count"] >= 1
    assert len(final_state["messages"]) >= 2
    # adjudication node should have produced a result
    assert final_state["adjudication_result"] is not None
    assert isinstance(final_state["adjudication_result"], FactCheckResult)


def test_extract_output_with_adjudication():
    """extract_output returns FactCheckResult when adjudication_result is present."""
    adj_result = FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[
                    ClaimVerdict(
                        claim_id="c-1",
                        claim_text="test",
                        verdict="Falso",
                        justification="reason",
                    )
                ],
            )
        ],
        overall_summary="summary",
    )
    state = {
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "adjudication_result": adj_result,
    }
    output = extract_output(state)
    assert isinstance(output, FactCheckResult)
    assert output.overall_summary == "summary"


def test_extract_output_fallback():
    """extract_output returns ContextNodeOutput when no adjudication_result."""
    state = {
        "fact_check_results": [
            FactCheckApiContext(
                id="1",
                url="url",
                parent_id=None,
                reliability=SourceReliability.MUITO_CONFIAVEL,
            )
        ],
        "search_results": {"geral": []},
        "scraped_pages": [],
        "adjudication_result": None,
    }
    output = extract_output(state)
    assert isinstance(output, ContextNodeOutput)
    assert len(output.fact_check_results) == 1
    assert "geral" in output.search_results


def _make_mock_adjudication_model_with_retry():
    """create adjudication model that returns insufficient on first call, Falso on second."""
    model = MagicMock()
    call_count = 0

    insufficient_output = LLMAdjudicationOutput(
        results=[
            LLMDataSourceResult(
                data_source_id=None,
                claim_verdicts=[
                    LLMClaimVerdict(
                        claim_id=None,
                        claim_text="Test claim",
                        verdict="Fontes insuficientes para verificar",
                        justification="No reliable sources found.",
                        citations_used=[],
                    )
                ],
            )
        ],
        overall_summary="Insufficient sources.",
    )

    falso_output = LLMAdjudicationOutput(
        results=[
            LLMDataSourceResult(
                data_source_id=None,
                claim_verdicts=[
                    LLMClaimVerdict(
                        claim_id=None,
                        claim_text="Test claim",
                        verdict="Falso",
                        justification="Contradicted by new sources.",
                        citations_used=[],
                    )
                ],
            )
        ],
        overall_summary="Claim is false after retry.",
    )

    async def _invoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return insufficient_output
        return falso_output

    structured = AsyncMock()
    structured.ainvoke = _invoke
    model.with_structured_output = MagicMock(return_value=structured)
    return model


@pytest.mark.asyncio
async def test_graph_retries_on_all_insufficient():
    """graph retries when first adjudication returns all-insufficient verdicts."""
    from langchain_core.messages import AIMessage

    model = _make_mock_model()
    adj_model = _make_mock_adjudication_model_with_retry()
    graph = build_graph(model, MockFactChecker(), MockWebSearcher(), MockScraper(), adj_model)

    initial_state = {
        "messages": [AIMessage(content="Test")],
        "data_sources": [DataSource(id="ds-1", source_type="original_text", original_text="Test claim")],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": "Test claim text",
        "run_id": "test-run",
        "adjudication_result": None,
        "retry_count": 0,
        "retry_context": None,
        "seen_source_keys": set(),
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["retry_count"] == 1
    assert final_state["adjudication_result"] is not None
    result = final_state["adjudication_result"]
    assert isinstance(result, FactCheckResult)
    # second adjudication should produce "Falso"
    assert result.results[0].claim_verdicts[0].verdict == "Falso"


# ── dedup tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_node_deduplicates_by_url(monkeypatch):
    """tool_node drops results whose (context_type, url) is already in seen_source_keys."""
    from langchain_core.messages import ToolMessage
    from app.agentic_ai.graph import _make_tool_node_with_state_update, _make_tools

    search_json = json.dumps({
        "geral": [{"id": "gs-1", "title": "Dup", "url": "https://dup.com", "snippet": "s", "domain": "dup.com"}]
    })
    tool_msg = ToolMessage(content=search_json, name="search_web", tool_call_id="tc-1")

    mock_tool_node = AsyncMock()
    mock_tool_node.ainvoke = AsyncMock(return_value={"messages": [tool_msg]})
    monkeypatch.setattr("app.agentic_ai.graph.ToolNode", lambda tools: mock_tool_node)

    tools = _make_tools(MockFactChecker(), MockWebSearcher(), MockScraper())
    tool_node_fn = _make_tool_node_with_state_update(
        tools, MockFactChecker(), MockWebSearcher(), MockScraper(),
    )

    # first call: empty seen set → url is kept
    state1 = {
        "messages": [], "fact_check_results": [], "search_results": {},
        "scraped_pages": [], "seen_source_keys": set(),
    }
    result1 = await tool_node_fn(state1)
    assert ("search", "https://dup.com") in result1["seen_source_keys"]
    assert len(result1["search_results"]["geral"]) == 1

    # second call: url already seen → duplicate dropped
    state2 = {
        "messages": [], "fact_check_results": [],
        "search_results": result1["search_results"],
        "scraped_pages": [], "seen_source_keys": result1["seen_source_keys"],
    }
    result2 = await tool_node_fn(state2)
    total = sum(len(v) for v in result2.get("search_results", {}).values())
    # merged result still has exactly the original 1 entry, no growth
    assert total == 1


@pytest.mark.asyncio
async def test_tool_node_allows_same_url_different_context_type(monkeypatch):
    """same URL as fact_check and scraped are both kept because (context_type, url) differs."""
    from langchain_core.messages import ToolMessage
    from app.agentic_ai.graph import _make_tool_node_with_state_update, _make_tools

    shared_url = "https://shared.com/article"
    fc_msg = ToolMessage(
        content=json.dumps([{
            "id": "fc-1", "title": "FC", "url": shared_url,
            "publisher": "P", "rating": "Falso", "claim_text": "c", "review_date": None,
        }]),
        name="search_fact_check_api", tool_call_id="tc-1",
    )
    scrape_msg = ToolMessage(
        content=json.dumps([{
            "id": "sc-1", "title": "SC", "url": shared_url,
            "content_preview": "text", "extraction_status": "success",
        }]),
        name="scrape_pages", tool_call_id="tc-2",
    )

    call_count = 0

    async def _mock_ainvoke(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"messages": [fc_msg]}
        return {"messages": [scrape_msg]}

    mock_tool_node = MagicMock()
    mock_tool_node.ainvoke = _mock_ainvoke
    monkeypatch.setattr("app.agentic_ai.graph.ToolNode", lambda tools: mock_tool_node)

    tools = _make_tools(MockFactChecker(), MockWebSearcher(), MockScraper())
    tool_node_fn = _make_tool_node_with_state_update(
        tools, MockFactChecker(), MockWebSearcher(), MockScraper(),
    )

    # first call: fact_check result
    state1 = {
        "messages": [], "fact_check_results": [], "search_results": {},
        "scraped_pages": [], "seen_source_keys": set(),
    }
    result1 = await tool_node_fn(state1)
    assert ("fact_check", shared_url) in result1["seen_source_keys"]
    assert len(result1.get("fact_check_results", [])) == 1

    # second call: scraped result with same URL, different context type → kept
    state2 = {
        "messages": [], "fact_check_results": result1["fact_check_results"],
        "search_results": {}, "scraped_pages": [],
        "seen_source_keys": result1["seen_source_keys"],
    }
    result2 = await tool_node_fn(state2)
    assert ("scraped", shared_url) in result2["seen_source_keys"]
    assert len(result2.get("scraped_pages", [])) == 1
