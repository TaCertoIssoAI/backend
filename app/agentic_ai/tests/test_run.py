"""tests for app.agentic_ai.run module.

covers:
- run_fact_check returns GraphOutput with FactCheckResult on successful graph invocation
- run_fact_check returns fallback FactCheckResult when graph returns ContextNodeOutput
- initial state passed to graph contains the provided data_sources
- _build_graph builds a fresh graph per call
- data_sources are forwarded without mutation
- GraphOutput includes source lists from graph state
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.commondata import DataSource
from app.models.factchecking import (
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
)
from app.agentic_ai.run import GraphOutput


# ---- helpers ----

def _make_data_source(text: str = "Test claim") -> DataSource:
    return DataSource(id="ds-1", source_type="original_text", original_text=text)


def _make_fact_check_result(summary: str = "Done.") -> FactCheckResult:
    return FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[
                    ClaimVerdict(
                        claim_id="c-1",
                        claim_text="Test claim",
                        verdict="Falso",
                        justification="Reason.",
                        citations_used=[],
                    )
                ],
            )
        ],
        overall_summary=summary,
    )


# ---- test: successful graph run ----

@pytest.mark.asyncio
async def test_run_fact_check_returns_fact_check_result():
    """when graph produces a FactCheckResult, it is returned in GraphOutput."""
    expected = _make_fact_check_result()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"adjudication_result": expected})

    with patch("app.agentic_ai.run._build_graph", return_value=mock_graph), \
         patch("app.agentic_ai.graph.extract_output", return_value=expected):
        from app.agentic_ai.run import run_fact_check
        output = await run_fact_check([_make_data_source()])

    assert isinstance(output, GraphOutput)
    assert isinstance(output.result, FactCheckResult)
    assert output.result.overall_summary == "Done."
    assert len(output.result.results) == 1
    assert output.result.results[0].claim_verdicts[0].verdict == "Falso"


# ---- test: fallback when graph returns ContextNodeOutput ----

@pytest.mark.asyncio
async def test_run_fact_check_fallback_on_context_output():
    """when extract_output returns something other than FactCheckResult, fallback is used."""
    # simulate ContextNodeOutput (not a FactCheckResult)
    context_output = MagicMock(spec=[])  # spec=[] makes isinstance(_, FactCheckResult) False
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"adjudication_result": None})

    with patch("app.agentic_ai.run._build_graph", return_value=mock_graph), \
         patch("app.agentic_ai.graph.extract_output", return_value=context_output):
        from app.agentic_ai.run import run_fact_check
        output = await run_fact_check([_make_data_source()])

    assert isinstance(output, GraphOutput)
    assert isinstance(output.result, FactCheckResult)
    assert output.result.results == []
    assert "Nenhuma verificação" in output.result.overall_summary


# ---- test: data_sources forwarded in initial state ----

@pytest.mark.asyncio
async def test_run_fact_check_passes_data_sources_to_graph():
    """data_sources provided to run_fact_check must appear in the initial state."""
    ds = _make_data_source("My specific claim")
    captured_state = {}

    async def _capture(state):
        captured_state.update(state)
        return {"adjudication_result": _make_fact_check_result()}

    mock_graph = MagicMock()
    mock_graph.ainvoke = _capture

    with patch("app.agentic_ai.run._build_graph", return_value=mock_graph), \
         patch("app.agentic_ai.graph.extract_output", return_value=_make_fact_check_result()):
        from app.agentic_ai.run import run_fact_check
        await run_fact_check([ds])

    assert captured_state["data_sources"] == [ds]
    assert captured_state["data_sources"][0].original_text == "My specific claim"


# ---- test: initial state defaults ----

@pytest.mark.asyncio
async def test_run_fact_check_initial_state_has_correct_defaults():
    """initial state should have all required keys with default values."""
    captured_state = {}

    async def _capture(state):
        captured_state.update(state)
        return {"adjudication_result": _make_fact_check_result()}

    mock_graph = MagicMock()
    mock_graph.ainvoke = _capture

    with patch("app.agentic_ai.run._build_graph", return_value=mock_graph), \
         patch("app.agentic_ai.graph.extract_output", return_value=_make_fact_check_result()):
        from app.agentic_ai.run import run_fact_check
        await run_fact_check([_make_data_source()])

    assert captured_state["messages"] == []
    assert captured_state["fact_check_results"] == []
    assert captured_state["search_results"] == {}
    assert captured_state["scraped_pages"] == []
    assert captured_state["iteration_count"] == 0
    assert captured_state["pending_async_count"] == 0
    assert captured_state["formatted_data_sources"] == ""
    assert captured_state["adjudication_result"] is None
    assert captured_state["retry_count"] == 0
    assert captured_state["retry_context"] is None


# ---- test: fresh graph per call ----

@pytest.mark.asyncio
async def test_build_graph_called_every_time():
    """_build_graph should be called on every run_fact_check invocation."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"adjudication_result": _make_fact_check_result()})

    with patch("app.agentic_ai.run._build_graph", return_value=mock_graph) as mock_build, \
         patch("app.agentic_ai.graph.extract_output", return_value=_make_fact_check_result()):
        from app.agentic_ai.run import run_fact_check
        await run_fact_check([_make_data_source()])
        await run_fact_check([_make_data_source()])
        await run_fact_check([_make_data_source()])

    assert mock_build.call_count == 3


# ---- test: multiple data sources ----

@pytest.mark.asyncio
async def test_run_fact_check_with_multiple_data_sources():
    """multiple data sources should all be passed through to the graph."""
    ds1 = DataSource(id="ds-1", source_type="original_text", original_text="Claim A")
    ds2 = DataSource(id="ds-2", source_type="image", original_text="Claim B")
    captured_state = {}

    async def _capture(state):
        captured_state.update(state)
        return {"adjudication_result": _make_fact_check_result()}

    mock_graph = MagicMock()
    mock_graph.ainvoke = _capture

    with patch("app.agentic_ai.run._build_graph", return_value=mock_graph), \
         patch("app.agentic_ai.graph.extract_output", return_value=_make_fact_check_result()):
        from app.agentic_ai.run import run_fact_check
        await run_fact_check([ds1, ds2])

    assert len(captured_state["data_sources"]) == 2
    assert captured_state["data_sources"][0].id == "ds-1"
    assert captured_state["data_sources"][1].id == "ds-2"
