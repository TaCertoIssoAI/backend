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
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["retry_count"] == 1
    assert final_state["adjudication_result"] is not None
    result = final_state["adjudication_result"]
    assert isinstance(result, FactCheckResult)
    # second adjudication should produce "Falso"
    assert result.results[0].claim_verdicts[0].verdict == "Falso"
