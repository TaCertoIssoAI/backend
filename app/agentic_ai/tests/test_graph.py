"""tests for graph compilation and structure."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    ScrapeTarget,
    WebScrapeContext,
    SourceReliability,
)
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


def test_graph_compiles():
    model = _make_mock_model()
    graph = build_graph(model, MockFactChecker(), MockWebSearcher(), MockScraper())
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_runs_to_completion():
    from langchain_core.messages import HumanMessage

    model = _make_mock_model()
    graph = build_graph(model, MockFactChecker(), MockWebSearcher(), MockScraper())

    initial_state = {
        "messages": [HumanMessage(content="Test claim")],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "iteration_count": 0,
        "pending_async_count": 0,
        "formatted_data_sources": "Test claim",
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["iteration_count"] >= 1
    assert len(final_state["messages"]) >= 2


def test_extract_output():
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
    }
    output = extract_output(state)
    assert len(output.fact_check_results) == 1
    assert "geral" in output.search_results
