"""tests for prepare_retry node and helpers."""

import pytest
from unittest.mock import MagicMock

from langgraph.graph import END

from app.agentic_ai.config import MAX_RETRY_COUNT
from app.agentic_ai.controlflow.prepare_retry import (
    _all_verdicts_insufficient,
    _extract_used_queries,
    _build_retry_context,
    prepare_retry_node,
    route_after_prepare_retry,
)
from app.models.factchecking import (
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
    VerdictTypeEnum,
)


def _make_claim_verdict(verdict: str, claim_text: str = "test claim", justification: str = "reason") -> ClaimVerdict:
    return ClaimVerdict(
        claim_id="c-1",
        claim_text=claim_text,
        verdict=verdict,
        justification=justification,
    )


def _make_fact_check_result(verdicts: list[str], summary: str = "summary") -> FactCheckResult:
    return FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[_make_claim_verdict(v) for v in verdicts],
            )
        ],
        overall_summary=summary,
    )


# --- _all_verdicts_insufficient tests ---

def test_all_insufficient_returns_true():
    result = _make_fact_check_result([
        "Fontes insuficientes para verificar",
        "Fontes insuficientes para verificar",
    ])
    assert _all_verdicts_insufficient(result) is True


def test_mixed_verdicts_returns_false():
    result = _make_fact_check_result([
        "Fontes insuficientes para verificar",
        "Falso",
    ])
    assert _all_verdicts_insufficient(result) is False


def test_all_false_returns_false():
    result = _make_fact_check_result(["Falso", "Falso"])
    assert _all_verdicts_insufficient(result) is False


def test_none_result_returns_false():
    assert _all_verdicts_insufficient(None) is False


def test_empty_results_returns_false():
    result = FactCheckResult(results=[], overall_summary=None)
    assert _all_verdicts_insufficient(result) is False


def test_no_claim_verdicts_returns_false():
    result = FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[],
            )
        ],
        overall_summary=None,
    )
    assert _all_verdicts_insufficient(result) is False


def test_single_insufficient_verdict_returns_true():
    """single insufficient verdict returns true."""
    result = _make_fact_check_result(["Fontes insuficientes para verificar"])
    assert _all_verdicts_insufficient(result) is True


# --- _extract_used_queries tests ---

def _make_ai_msg_with_tool_calls(tool_calls: list) -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.id = "msg-1"
    return msg


def test_extract_queries_from_search_web():
    msg = _make_ai_msg_with_tool_calls([
        {"name": "search_web", "args": {"queries": ["query1", "query2"]}},
    ])
    queries = _extract_used_queries([msg])
    assert queries == ["query1", "query2"]


def test_extract_queries_from_fact_check_api():
    msg = _make_ai_msg_with_tool_calls([
        {"name": "search_fact_check_api", "args": {"queries": ["fc query"]}},
    ])
    queries = _extract_used_queries([msg])
    assert queries == ["fc query"]


def test_extract_queries_deduplicates():
    msg1 = _make_ai_msg_with_tool_calls([
        {"name": "search_web", "args": {"queries": ["Query1"]}},
    ])
    msg2 = _make_ai_msg_with_tool_calls([
        {"name": "search_web", "args": {"queries": ["query1"]}},
    ])
    queries = _extract_used_queries([msg1, msg2])
    assert len(queries) == 1
    assert queries[0] == "Query1"


def test_extract_queries_skips_scrape():
    msg = _make_ai_msg_with_tool_calls([
        {"name": "scrape_pages", "args": {"targets": []}},
    ])
    queries = _extract_used_queries([msg])
    assert queries == []


def test_extract_queries_from_empty_messages():
    queries = _extract_used_queries([])
    assert queries == []


# --- _build_retry_context tests ---

def test_build_retry_context_includes_justifications():
    result = _make_fact_check_result(
        ["Fontes insuficientes para verificar"],
        summary="Overall summary",
    )
    result.results[0].claim_verdicts[0].claim_text = "Earth is flat"
    result.results[0].claim_verdicts[0].justification = "No sources found"

    context = _build_retry_context(result, [])
    assert "Earth is flat" in context
    assert "No sources found" in context


def test_build_retry_context_includes_queries():
    result = _make_fact_check_result(["Fontes insuficientes para verificar"])
    context = _build_retry_context(result, ["old query 1", "old query 2"])
    assert "NAO repita" in context
    assert "old query 1" in context
    assert "old query 2" in context


def test_build_retry_context_includes_summary():
    result = _make_fact_check_result(
        ["Fontes insuficientes para verificar"],
        summary="Not enough evidence found",
    )
    context = _build_retry_context(result, [])
    assert "Not enough evidence found" in context


# --- prepare_retry_node tests ---

def _make_state(
    verdict_strings: list[str] | None = None,
    retry_count: int = 0,
    messages: list | None = None,
) -> dict:
    adj_result = None
    if verdict_strings is not None:
        adj_result = _make_fact_check_result(verdict_strings)

    return {
        "adjudication_result": adj_result,
        "retry_count": retry_count,
        "messages": messages or [],
        "iteration_count": 3,
    }


@pytest.mark.asyncio
async def test_prepare_retry_triggers_on_all_insufficient():
    state = _make_state(["Fontes insuficientes para verificar"])
    result = await prepare_retry_node(state)

    assert result["retry_count"] == 1
    assert result["iteration_count"] == 0
    assert result["adjudication_result"] is None
    assert result["retry_context"] is not None
    assert len(result["retry_context"]) > 0


@pytest.mark.asyncio
async def test_prepare_retry_does_not_trigger_on_mixed():
    state = _make_state(["Fontes insuficientes para verificar", "Falso"])
    result = await prepare_retry_node(state)
    assert result == {}


@pytest.mark.asyncio
async def test_prepare_retry_respects_max_retry_count():
    state = _make_state(
        ["Fontes insuficientes para verificar"],
        retry_count=MAX_RETRY_COUNT,
    )
    result = await prepare_retry_node(state)
    assert result == {}


# --- route_after_prepare_retry tests ---

def test_route_to_retry_agent_on_retry():
    state = {"adjudication_result": None, "retry_count": 1}
    assert route_after_prepare_retry(state) == "retry_context_agent"


def test_route_to_end_when_no_retry():
    state = {
        "adjudication_result": _make_fact_check_result(["Falso"]),
        "retry_count": 0,
    }
    assert route_after_prepare_retry(state) == END


def test_route_to_end_at_max_retries():
    state = {
        "adjudication_result": _make_fact_check_result(
            ["Fontes insuficientes para verificar"]
        ),
        "retry_count": MAX_RETRY_COUNT,
    }
    assert route_after_prepare_retry(state) == END
