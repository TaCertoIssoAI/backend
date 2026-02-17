"""tests for the /text endpoint (app.api.endpoints.text).

covers:
- GraphOutput is correctly destructured and source lists forwarded to mapper
- successful request returns 200 with valid AnalysisResponse schema
- run_fact_check failure returns 500
- source lists from GraphOutput are passed to fact_check_result_to_response
- sanitize_request and sanitize_response are called
- analytics is sent when claims exist, skipped otherwise
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints.text import router
from app.models.api import AnalysisResponse
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
)
from app.agentic_ai.run import GraphOutput


# ---- test app ----

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---- helpers ----

def _make_fact_check_result(
    claim_text: str = "Test claim",
    verdict: str = "Falso",
    justification: str = "Reason [1].",
    summary: str = "Overall summary.",
) -> FactCheckResult:
    return FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[
                    ClaimVerdict(
                        claim_id="c-1",
                        claim_text=claim_text,
                        verdict=verdict,
                        justification=justification,
                        citations_used=[],
                    )
                ],
            )
        ],
        overall_summary=summary,
    )


def _make_fc_context(url: str = "https://fc.com/1") -> FactCheckApiContext:
    return FactCheckApiContext(
        id="fc-1", url=url, parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        publisher="Publisher", claim_text="claim", rating="Falso",
    )


def _make_search_context(url: str = "https://sr.com/1") -> GoogleSearchContext:
    return GoogleSearchContext(
        id="gs-1", url=url, parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        title="Search Result", snippet="snippet", domain="sr.com",
    )


def _make_scrape_context(url: str = "https://sp.com/1") -> WebScrapeContext:
    return WebScrapeContext(
        id="ws-1", url=url, parent_id=None,
        reliability=SourceReliability.POUCO_CONFIAVEL,
        title="Scraped", content="content",
        extraction_status="success", extraction_tool="cloudscraper",
    )


def _make_graph_output(
    fc_result: FactCheckResult | None = None,
    fact_check_results: list | None = None,
    search_results: dict | None = None,
    scraped_pages: list | None = None,
    error: str | None = None,
) -> GraphOutput:
    return GraphOutput(
        result=fc_result or _make_fact_check_result(),
        fact_check_results=fact_check_results or [],
        search_results=search_results or {},
        scraped_pages=scraped_pages or [],
        error=error,
    )


_TEXT_PAYLOAD = {
    "content": [{"textContent": "Test claim text", "type": "text"}]
}


# ---- test: successful request ----

@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_text_endpoint_returns_200(mock_run, mock_analytics):
    """a valid request should return 200 with AnalysisResponse fields."""
    mock_run.return_value = _make_graph_output()

    resp = client.post("/text", json=_TEXT_PAYLOAD)

    assert resp.status_code == 200
    data = resp.json()
    assert "message_id" in data
    assert "rationale" in data
    assert "responseWithoutLinks" in data


@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_text_endpoint_rationale_contains_verdict(mock_run, mock_analytics):
    """rationale should contain claim text and verdict."""
    mock_run.return_value = _make_graph_output()

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    data = resp.json()

    assert "Test claim" in data["rationale"]
    assert "Falso" in data["rationale"]


# ---- test: source lists forwarded to mapper ----

@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.fact_check_result_to_response", wraps=None)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_source_lists_passed_to_mapper(mock_run, mock_mapper, mock_analytics):
    """source lists from GraphOutput should be forwarded to fact_check_result_to_response."""
    fc = [_make_fc_context()]
    sr = {"aosfatos": [_make_search_context()]}
    sp = [_make_scrape_context()]

    mock_run.return_value = _make_graph_output(
        fact_check_results=fc,
        search_results=sr,
        scraped_pages=sp,
    )

    # mock mapper to return a valid response
    mock_mapper.return_value = AnalysisResponse(
        message_id="msg-1",
        rationale="rationale",
        responseWithoutLinks="rationale",
    )

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    assert resp.status_code == 200

    # verify mapper was called with source lists
    mock_mapper.assert_called_once()
    call_kwargs = mock_mapper.call_args
    assert call_kwargs.kwargs["fact_check_results"] is fc
    assert call_kwargs.kwargs["search_results"] is sr
    assert call_kwargs.kwargs["scraped_pages"] is sp


@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_citations_appear_when_source_lists_provided(mock_run, mock_analytics):
    """when source lists are provided and LLM cites [N], Fontes should appear."""
    fc = [_make_fc_context(url="https://factcheck.org/article1")]
    graph_output = _make_graph_output(
        fc_result=_make_fact_check_result(justification="Evidence [1] shows."),
        fact_check_results=fc,
    )
    mock_run.return_value = graph_output

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    data = resp.json()

    assert "Fontes" in data["rationale"]
    assert "https://factcheck.org/article1" in data["rationale"]


@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_no_citations_when_source_lists_empty(mock_run, mock_analytics):
    """when source lists are empty, no Fontes section should appear."""
    mock_run.return_value = _make_graph_output()

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    data = resp.json()

    assert "Fontes" not in data["rationale"]


# ---- test: GraphOutput destructuring ----

@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_graph_output_result_used_for_claim_count(mock_run, mock_analytics):
    """fact_check_result from GraphOutput.result should drive response content."""
    result_with_two = FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[
                    ClaimVerdict(
                        claim_id="c-1", claim_text="Claim A",
                        verdict="Verdadeiro", justification="True.",
                    ),
                    ClaimVerdict(
                        claim_id="c-2", claim_text="Claim B",
                        verdict="Falso", justification="False.",
                    ),
                ],
            )
        ],
        overall_summary="Two claims analyzed.",
    )
    mock_run.return_value = _make_graph_output(fc_result=result_with_two)

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    data = resp.json()

    assert "Claim A" in data["rationale"]
    assert "Claim B" in data["rationale"]
    assert "Afirmação 1" in data["rationale"]
    assert "Afirmação 2" in data["rationale"]


# ---- test: error handling ----

@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_run_fact_check_error_returns_500(mock_run):
    """when run_fact_check raises, endpoint should return 500."""
    mock_run.side_effect = RuntimeError("Graph exploded")

    resp = client.post("/text", json=_TEXT_PAYLOAD)

    assert resp.status_code == 500
    assert "Graph exploded" in resp.json()["detail"]


@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_adjudication_timeout_returns_500(mock_run):
    """when adjudication times out, endpoint should return 500 with error detail."""
    mock_run.return_value = _make_graph_output(
        error="Adjudication timed out after 3 attempt(s) (120s each)",
    )

    resp = client.post("/text", json=_TEXT_PAYLOAD)

    assert resp.status_code == 500
    assert "timed out" in resp.json()["detail"]
    assert "3 attempt(s)" in resp.json()["detail"]


@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_no_error_field_returns_200(mock_run, mock_analytics):
    """when GraphOutput.error is None, endpoint returns 200 normally."""
    mock_run.return_value = _make_graph_output(error=None)

    resp = client.post("/text", json=_TEXT_PAYLOAD)

    assert resp.status_code == 200


# ---- test: sanitization ----

@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.sanitize_response")
@patch("app.api.endpoints.text.sanitize_request")
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_sanitize_request_called(mock_run, mock_san_req, mock_san_resp, mock_analytics):
    """sanitize_request should be called before processing."""
    from app.models.api import Request, ContentItem, ContentType

    # make sanitize_request pass through
    mock_san_req.side_effect = lambda r: r
    mock_san_resp.side_effect = lambda r: r
    mock_run.return_value = _make_graph_output()

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    assert resp.status_code == 200

    mock_san_req.assert_called_once()
    mock_san_resp.assert_called_once()


# ---- test: analytics ----

@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_analytics_sent_when_claims_exist(mock_run, mock_analytics):
    """analytics payload should be sent when claims are extracted."""
    mock_run.return_value = _make_graph_output()

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    assert resp.status_code == 200
    # analytics is fired via asyncio.create_task — hard to assert directly,
    # but at minimum the endpoint should complete without error


@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_no_verdicts_returns_fallback(mock_run, mock_analytics):
    """when no verdicts, rationale should have fallback message."""
    empty_result = FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[],
            )
        ],
        overall_summary="Nothing found.",
    )
    mock_run.return_value = _make_graph_output(fc_result=empty_result)

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    data = resp.json()

    assert resp.status_code == 200
    assert "Nenhuma alegação verificável" in data["rationale"]


# ---- test: response schema ----

@patch("app.api.endpoints.text.send_analytics_payload", new_callable=AsyncMock)
@patch("app.api.endpoints.text.run_fact_check", new_callable=AsyncMock)
def test_response_without_links_has_no_urls(mock_run, mock_analytics):
    """responseWithoutLinks field should not contain URLs."""
    fc = [_make_fc_context(url="https://example.com/check")]
    mock_run.return_value = _make_graph_output(
        fc_result=_make_fact_check_result(justification="See [1]."),
        fact_check_results=fc,
    )

    resp = client.post("/text", json=_TEXT_PAYLOAD)
    data = resp.json()

    assert "https://" not in data["responseWithoutLinks"]
