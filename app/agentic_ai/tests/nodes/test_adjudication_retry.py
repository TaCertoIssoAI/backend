"""tests for the adjudication node timeout retry policy.

covers:
- happy path (no timeout)
- single timeout then success
- two timeouts then success
- all attempts time out → error propagation
- same input on every retry attempt
- timeout boundary (just under the limit)
- non-timeout errors (None result) still use existing fallback
- graph integration with timeout → END routing
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agentic_ai.nodes.adjudication import (
    make_adjudication_node,
    _make_timeout_error_result,
)
from app.models.factchecking import (
    DataSourceResult,
    FactCheckResult,
    LLMAdjudicationOutput,
    LLMClaimVerdict,
    LLMDataSourceResult,
)


# ---- helpers ----

def _make_llm_output(verdict: str = "Falso", summary: str = "Summary.") -> LLMAdjudicationOutput:
    """build a simple LLMAdjudicationOutput for testing."""
    return LLMAdjudicationOutput(
        results=[
            LLMDataSourceResult(
                data_source_id=None,
                claim_verdicts=[
                    LLMClaimVerdict(
                        claim_id=None,
                        claim_text="Test claim",
                        verdict=verdict,
                        justification="Reason [1].",
                        citations_used=[],
                    )
                ],
            )
        ],
        overall_summary=summary,
    )


def _make_state(**overrides) -> dict:
    """build a minimal ContextAgentState dict."""
    state = {
        "messages": [],
        "formatted_data_sources": "original text",
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
    }
    state.update(overrides)
    return state


def _make_mock_model(ainvoke_side_effect=None, ainvoke_return=None):
    """create a mock model whose .with_structured_output().ainvoke() is controllable."""
    model = MagicMock()
    structured = MagicMock()
    if ainvoke_side_effect is not None:
        structured.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    else:
        structured.ainvoke = AsyncMock(return_value=ainvoke_return)
    model.with_structured_output = MagicMock(return_value=structured)
    return model, structured


# ---- test: _make_timeout_error_result ----

def test_timeout_error_result_has_empty_verdicts():
    result = _make_timeout_error_result(3, 20.0)
    assert isinstance(result, FactCheckResult)
    assert len(result.results) == 1
    assert result.results[0].claim_verdicts == []


def test_timeout_error_result_summary_contains_info():
    result = _make_timeout_error_result(3, 20.0)
    assert "20.0" in result.overall_summary
    assert "3" in result.overall_summary


# ---- test: happy path — no timeout ----

@pytest.mark.asyncio
async def test_no_timeout_returns_normal_result():
    """LLM responds immediately, no retries needed."""
    llm_output = _make_llm_output()
    model, _ = _make_mock_model(ainvoke_return=llm_output)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 20.0), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert "adjudication_result" in result
    assert isinstance(result["adjudication_result"], FactCheckResult)
    assert result["adjudication_result"].results[0].claim_verdicts[0].verdict == "Falso"
    assert "adjudication_error" not in result


@pytest.mark.asyncio
async def test_no_timeout_calls_ainvoke_once():
    """on success, ainvoke is called exactly once."""
    llm_output = _make_llm_output()
    model, structured = _make_mock_model(ainvoke_return=llm_output)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 20.0), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        await node(_make_state())

    assert structured.ainvoke.call_count == 1


# ---- test: single timeout then success ----

@pytest.mark.asyncio
async def test_one_timeout_then_success():
    """first call times out, second succeeds."""
    llm_output = _make_llm_output()

    async def _slow_then_fast(messages):
        if _slow_then_fast.call_count == 0:
            _slow_then_fast.call_count += 1
            await asyncio.sleep(100)  # will be cancelled by timeout
        _slow_then_fast.call_count += 1
        return llm_output

    _slow_then_fast.call_count = 0
    model, _ = _make_mock_model(ainvoke_side_effect=_slow_then_fast)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert isinstance(result["adjudication_result"], FactCheckResult)
    assert result["adjudication_result"].results[0].claim_verdicts[0].verdict == "Falso"
    assert "adjudication_error" not in result


# ---- test: two timeouts then success ----

@pytest.mark.asyncio
async def test_two_timeouts_then_success():
    """first two calls time out, third succeeds."""
    llm_output = _make_llm_output(verdict="Verdadeiro")

    async def _two_slow(messages):
        if _two_slow.call_count < 2:
            _two_slow.call_count += 1
            await asyncio.sleep(100)
        _two_slow.call_count += 1
        return llm_output

    _two_slow.call_count = 0
    model, _ = _make_mock_model(ainvoke_side_effect=_two_slow)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert isinstance(result["adjudication_result"], FactCheckResult)
    assert result["adjudication_result"].results[0].claim_verdicts[0].verdict == "Verdadeiro"
    assert "adjudication_error" not in result


# ---- test: all attempts time out → error ----

@pytest.mark.asyncio
async def test_all_attempts_timeout_returns_error():
    """all 3 attempts time out → adjudication_error is set."""
    async def _always_slow(messages):
        await asyncio.sleep(100)

    model, _ = _make_mock_model(ainvoke_side_effect=_always_slow)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert "adjudication_error" in result
    assert "timed out" in result["adjudication_error"]
    assert "3 attempt(s)" in result["adjudication_error"]
    assert isinstance(result["adjudication_result"], FactCheckResult)
    assert result["adjudication_result"].results[0].claim_verdicts == []


@pytest.mark.asyncio
async def test_all_attempts_timeout_with_one_retry():
    """with max_retries=1, only 2 attempts total."""
    async def _always_slow(messages):
        await asyncio.sleep(100)

    model, _ = _make_mock_model(ainvoke_side_effect=_always_slow)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 1):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert "adjudication_error" in result
    assert "2 attempt(s)" in result["adjudication_error"]


# ---- test: same input on every retry ----

@pytest.mark.asyncio
async def test_same_messages_on_every_retry():
    """messages passed to ainvoke must be identical across all retry attempts."""
    captured_messages = []

    async def _capture_and_timeout(messages):
        captured_messages.append(messages)
        await asyncio.sleep(100)

    model, _ = _make_mock_model(ainvoke_side_effect=_capture_and_timeout)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        await node(_make_state())

    assert len(captured_messages) == 3
    # all three calls received the exact same messages list object
    assert captured_messages[0] is captured_messages[1]
    assert captured_messages[1] is captured_messages[2]


@pytest.mark.asyncio
async def test_same_messages_content_on_retry():
    """verify the actual content of messages is the same on retry."""
    captured_messages = []
    llm_output = _make_llm_output()

    async def _timeout_then_ok(messages):
        captured_messages.append(messages)
        if len(captured_messages) == 1:
            await asyncio.sleep(100)
        return llm_output

    model, _ = _make_mock_model(ainvoke_side_effect=_timeout_then_ok)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        await node(_make_state())

    assert len(captured_messages) == 2
    first_contents = [m.content for m in captured_messages[0]]
    second_contents = [m.content for m in captured_messages[1]]
    assert first_contents == second_contents


# ---- test: timeout boundary ----

@pytest.mark.asyncio
async def test_completes_just_under_timeout():
    """call that finishes just under the limit should succeed (no retry)."""
    llm_output = _make_llm_output()

    async def _just_under(messages):
        await asyncio.sleep(0.05)  # well under the 0.2s timeout
        return llm_output

    model, structured = _make_mock_model(ainvoke_side_effect=_just_under)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.2), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert isinstance(result["adjudication_result"], FactCheckResult)
    assert "adjudication_error" not in result
    assert structured.ainvoke.call_count == 1


# ---- test: non-timeout errors (None result) use existing fallback ----

@pytest.mark.asyncio
async def test_none_result_uses_existing_fallback():
    """LLM returning None (schema parse failure) should NOT trigger timeout retries."""
    model, _ = _make_mock_model(ainvoke_return=None)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 20.0), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 2):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert "adjudication_error" not in result
    assert isinstance(result["adjudication_result"], FactCheckResult)
    assert "resposta estruturada" in result["adjudication_result"].overall_summary


# ---- test: zero retries config ----

@pytest.mark.asyncio
async def test_zero_retries_times_out_immediately():
    """with max_retries=0, only one attempt is made."""
    async def _always_slow(messages):
        await asyncio.sleep(100)

    model, structured = _make_mock_model(ainvoke_side_effect=_always_slow)

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 0):
        node = make_adjudication_node(model)
        result = await node(_make_state())

    assert "adjudication_error" in result
    assert "1 attempt(s)" in result["adjudication_error"]
    assert structured.ainvoke.call_count == 1


# ---- test: graph integration — error routes to END ----

@pytest.mark.asyncio
async def test_graph_routes_to_end_on_adjudication_error():
    """when adjudication times out, graph should route to END, not prepare_retry."""
    from langchain_core.messages import AIMessage
    from app.agentic_ai.graph import build_graph
    from app.models.commondata import DataSource
    from app.models.agenticai import (
        FactCheckApiContext,
        GoogleSearchContext,
        SourceReliability,
    )

    # mock context agent model (returns no tool calls → goes to adjudication)
    agent_model = MagicMock()
    bound = AsyncMock()
    bound.ainvoke = AsyncMock(
        return_value=AIMessage(content="Sources are sufficient. Done.")
    )
    agent_model.bind_tools = MagicMock(return_value=bound)

    # mock adjudication model that always times out
    adj_model = MagicMock()
    structured = MagicMock()

    async def _always_slow(messages):
        await asyncio.sleep(100)

    structured.ainvoke = _always_slow
    adj_model.with_structured_output = MagicMock(return_value=structured)

    # mock tool protocol implementations
    class MockFC:
        async def search(self, queries):
            return []

    class MockWS:
        async def search(self, queries, max_results_specific_search=5, max_results_general=5):
            return {"geral": [], "especifico": []}

    class MockScraper:
        async def scrape(self, targets):
            return []

    with patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_TIMEOUT", 0.1), \
         patch("app.agentic_ai.nodes.adjudication.ADJUDICATION_MAX_RETRIES", 1):
        graph = build_graph(agent_model, MockFC(), MockWS(), MockScraper(), adj_model)

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
            "adjudication_error": None,
        }

        final_state = await graph.ainvoke(initial_state)

    assert final_state.get("adjudication_error") is not None
    assert "timed out" in final_state["adjudication_error"]
    assert isinstance(final_state["adjudication_result"], FactCheckResult)
    assert final_state["adjudication_result"].results[0].claim_verdicts == []
