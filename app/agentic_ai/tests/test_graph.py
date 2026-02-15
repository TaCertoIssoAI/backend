"""tests for graph compilation and structure."""

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
    async def search(self, queries, max_results_per_search=5):
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
    from langchain_core.messages import HumanMessage

    model = _make_mock_model()
    adj_model = _make_mock_adjudication_model()
    graph = build_graph(model, MockFactChecker(), MockWebSearcher(), MockScraper(), adj_model)

    initial_state = {
        "messages": [HumanMessage(content="Test claim")],
        "data_sources": [DataSource(id="ds-1", source_type="original_text", original_text="Test claim")],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": "",
        "run_id": "test-run",
        "adjudication_result": None,
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
