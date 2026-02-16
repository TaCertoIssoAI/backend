"""tests for prepare_retry node and helpers."""

import pytest
from unittest.mock import MagicMock

from langgraph.graph import END

from app.agentic_ai.config import MAX_RETRY_COUNT
from app.agentic_ai.controlflow.prepare_retry import (
    _all_verdicts_insufficient,
    _extract_used_queries,
    _build_retry_context,
    _get_cited_numbers,
    _filter_to_cited_sources,
    prepare_retry_node,
    route_after_prepare_retry,
)
from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
    SourceReliability,
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
    fact_check_results: list | None = None,
    search_results: dict | None = None,
    scraped_pages: list | None = None,
) -> dict:
    adj_result = None
    if verdict_strings is not None:
        adj_result = _make_fact_check_result(verdict_strings)

    return {
        "adjudication_result": adj_result,
        "retry_count": retry_count,
        "messages": messages or [],
        "iteration_count": 3,
        "fact_check_results": fact_check_results or [],
        "search_results": search_results if search_results is not None else {},
        "scraped_pages": scraped_pages or [],
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


# --- source model helpers ---

def _make_fc(id="fc-1"):
    return FactCheckApiContext(
        id=id,
        url=f"https://factcheck.org/{id}",
        parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        title=f"FC {id}",
        publisher="Lupa",
        rating="Falso",
        claim_text="claim text",
    )


def _make_gs(id="gs-1", domain_key="geral", domain="bbc.com"):
    muito = {"aosfatos", "g1", "estadao", "folha"}
    reliability = SourceReliability.MUITO_CONFIAVEL if domain_key in muito else SourceReliability.NEUTRO
    return GoogleSearchContext(
        id=id,
        url=f"https://{domain}/{id}",
        parent_id=None,
        reliability=reliability,
        title=f"Search {id}",
        snippet="snippet",
        domain=domain,
    )


def _make_sc(id="sc-1"):
    return WebScrapeContext(
        id=id,
        url=f"https://example.com/{id}",
        parent_id=None,
        reliability=SourceReliability.POUCO_CONFIAVEL,
        title=f"Scraped {id}",
        content="page content",
        extraction_status="success",
        extraction_tool="beautifulsoup",
    )


# --- _get_cited_numbers tests ---

def test_get_cited_numbers_from_justification():
    fc1 = _make_fc("fc-1")
    fc2 = _make_fc("fc-2")
    gs = _make_gs("ge-1", "geral", "bbc.com")

    # sources: [1] fc1, [2] fc2, [3] geral
    adj = _make_fact_check_result(
        ["Fontes insuficientes para verificar"],
        summary="summary",
    )
    adj.results[0].claim_verdicts[0].justification = "Based on [1] and [3]."
    cited = _get_cited_numbers([fc1, fc2], {"geral": [gs]}, [], adj)
    assert cited == {1, 3}


def test_get_cited_numbers_from_summary():
    fc = _make_fc("fc-1")
    gs1 = _make_gs("ge-1", "geral", "bbc.com")
    gs2 = _make_gs("ge-2", "geral", "cnn.com")
    sc1 = _make_sc("sc-1")
    sc2 = _make_sc("sc-2")

    # sources: [1] fc, [2] geral, [3] geral, [4] scraped, [5] scraped
    adj = _make_fact_check_result(
        ["Fontes insuficientes para verificar"],
        summary="See [2][5] for details.",
    )
    adj.results[0].claim_verdicts[0].justification = "no refs here"
    cited = _get_cited_numbers([fc], {"geral": [gs1, gs2]}, [sc1, sc2], adj)
    assert cited == {2, 5}


def test_get_cited_numbers_union_across_claims():
    fc = _make_fc("fc-1")
    aos = _make_gs("af-1", "aosfatos", "aosfatos.org")
    gs1 = _make_gs("ge-1", "geral", "bbc.com")
    gs2 = _make_gs("ge-2", "geral", "cnn.com")

    # sources: [1] fc, [2] aosfatos, [3] geral, [4] geral
    adj = FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[
                    ClaimVerdict(
                        claim_id="c-1", claim_text="a",
                        verdict="Fontes insuficientes para verificar",
                        justification="Source [1].",
                    ),
                    ClaimVerdict(
                        claim_id="c-2", claim_text="b",
                        verdict="Fontes insuficientes para verificar",
                        justification="Source [4].",
                    ),
                ],
            )
        ],
        overall_summary="Overall [2].",
    )
    cited = _get_cited_numbers([fc], {"aosfatos": [aos], "geral": [gs1, gs2]}, [], adj)
    assert cited == {1, 2, 4}


def test_get_cited_numbers_empty():
    fc = _make_fc("fc-1")
    adj = _make_fact_check_result(
        ["Fontes insuficientes para verificar"],
        summary="no refs",
    )
    adj.results[0].claim_verdicts[0].justification = "nothing"
    cited = _get_cited_numbers([fc], {}, [], adj)
    assert cited == set()


# --- _filter_to_cited_sources tests ---

def test_filter_to_cited_sources_keeps_only_cited():
    # sources: [1] fc, [2] aosfatos, [3] geral, [4] geral, [5] scraped
    fc = _make_fc("fc-1")
    aos = _make_gs("af-1", "aosfatos", "aosfatos.org")
    g1 = _make_gs("ge-1", "geral", "bbc.com")
    g2 = _make_gs("ge-2", "geral", "cnn.com")
    sc = _make_sc("sc-1")

    r_fc, r_search, r_scraped = _filter_to_cited_sources(
        [fc], {"aosfatos": [aos], "geral": [g1, g2]}, [sc], {1, 3},
    )
    assert len(r_fc) == 1
    assert r_fc[0].id == "fc-1"
    assert r_search["aosfatos"] == []
    assert len(r_search["geral"]) == 1
    assert r_search["geral"][0].id == "ge-1"
    assert r_scraped == []


def test_filter_to_cited_sources_empty_when_none_cited():
    fc = _make_fc("fc-1")
    gs = _make_gs("ge-1", "geral", "bbc.com")
    sc = _make_sc("sc-1")

    r_fc, r_search, r_scraped = _filter_to_cited_sources(
        [fc], {"geral": [gs]}, [sc], set(),
    )
    assert r_fc == []
    assert r_search["geral"] == []
    assert r_scraped == []


def test_filter_to_cited_sources_keeps_all_when_all_cited():
    fc = _make_fc("fc-1")
    gs = _make_gs("ge-1", "geral", "bbc.com")
    sc = _make_sc("sc-1")

    # [1] fc, [2] geral, [3] scraped
    r_fc, r_search, r_scraped = _filter_to_cited_sources(
        [fc], {"geral": [gs]}, [sc], {1, 2, 3},
    )
    assert len(r_fc) == 1
    assert len(r_search["geral"]) == 1
    assert len(r_scraped) == 1


def test_filter_to_cited_sources_numbering_matches_format_context():
    """after filtering, format_context on filtered output produces contiguous numbering."""
    from app.agentic_ai.prompts.context_formatter import format_context
    import re

    fc = _make_fc("fc-1")
    aos = _make_gs("af-1", "aosfatos", "aosfatos.org")
    gs = _make_gs("ge-1", "geral", "bbc.com")
    sc = _make_sc("sc-1")

    # sources: [1] fc, [2] aosfatos, [3] geral, [4] scraped — cite [1] and [3]
    r_fc, r_search, r_scraped = _filter_to_cited_sources(
        [fc], {"aosfatos": [aos], "geral": [gs]}, [sc], {1, 3},
    )

    formatted = format_context(r_fc, r_search, r_scraped)
    numbers = [int(m) for m in re.findall(r"\[(\d+)\]", formatted)]
    assert numbers == [1, 2]


# --- prepare_retry_node returns source overwrite keys ---

@pytest.mark.asyncio
async def test_prepare_retry_overwrites_sources():
    fc = _make_fc("fc-1")
    gs = _make_gs("ge-1", "geral", "bbc.com")
    sc = _make_sc("sc-1")

    state = _make_state(
        verdict_strings=["Fontes insuficientes para verificar"],
        fact_check_results=[fc],
        search_results={"geral": [gs]},
        scraped_pages=[sc],
    )
    # patch justification to cite [1] only (the fact check)
    state["adjudication_result"].results[0].claim_verdicts[0].justification = "See [1]."

    result = await prepare_retry_node(state)

    assert "fact_check_results" in result
    assert "search_results" in result
    assert "scraped_pages" in result
    # only [1] (fact check) retained
    assert len(result["fact_check_results"]) == 1
    assert result["search_results"]["geral"] == []
    assert result["scraped_pages"] == []


@pytest.mark.asyncio
async def test_prepare_retry_rebuilds_seen_source_keys():
    """seen_source_keys is rebuilt from cited sources only, allowing uncited URLs to be re-found."""
    fc = _make_fc("fc-1")
    gs = _make_gs("ge-1", "geral", "bbc.com")
    sc = _make_sc("sc-1")

    state = _make_state(
        verdict_strings=["Fontes insuficientes para verificar"],
        fact_check_results=[fc],
        search_results={"geral": [gs]},
        scraped_pages=[sc],
    )
    # pre-populate seen keys as if all three were discovered
    state["seen_source_keys"] = {
        ("fact_check", fc.url),
        ("search", gs.url),
        ("scraped", sc.url),
    }
    # cite only [1] (fact check)
    state["adjudication_result"].results[0].claim_verdicts[0].justification = "See [1]."

    result = await prepare_retry_node(state)

    # only the cited source's key survives
    assert result["seen_source_keys"] == {("fact_check", fc.url)}
    # uncited URLs are no longer blocked — retry agent can re-find them
    assert ("search", gs.url) not in result["seen_source_keys"]
    assert ("scraped", sc.url) not in result["seen_source_keys"]
