"""tests for the mapper's fact_check_result_to_response function.

covers:
- single verdict with citations_used → rationale includes citations
- multiple verdicts with deduplicated citations
- empty citations_used → no sources section
- no verdicts → fallback message
- overall_summary inclusion
- verdict count summary generation
- responseWithoutLinks strips URLs
"""

from __future__ import annotations

import pytest
from app.api.mapper.mapper import (
    fact_check_result_to_response,
    request_to_data_sources,
    _get_verdict_summary,
)
from app.models.api import Request, ContentItem, ContentType, AnalysisResponse
from app.models.factchecking import (
    Citation,
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
)


# ---- helpers ----

def _make_citation(
    url: str = "https://example.com/article",
    title: str = "Example Article",
    publisher: str = "Example Publisher",
    citation_text: str = "Relevant quote.",
    date: str | None = "2024-01-01",
) -> Citation:
    return Citation(
        url=url,
        title=title,
        publisher=publisher,
        citation_text=citation_text,
        date=date,
    )


def _make_verdict(
    claim_text: str = "Test claim",
    verdict: str = "Falso",
    justification: str = "Reason [1].",
    citations: list[Citation] | None = None,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id="claim-1",
        claim_text=claim_text,
        verdict=verdict,
        justification=justification,
        citations_used=citations or [],
    )


def _make_fact_check_result(
    verdicts: list[ClaimVerdict] | None = None,
    summary: str | None = "Overall summary.",
) -> FactCheckResult:
    return FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=verdicts or [],
            )
        ],
        overall_summary=summary,
    )


# ---- test: basic response structure ----

def test_single_verdict_produces_valid_response():
    """a single verdict should produce a valid AnalysisResponse."""
    verdict = _make_verdict(citations=[_make_citation()])
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-1", result)

    assert isinstance(resp, AnalysisResponse)
    assert resp.message_id == "msg-1"
    assert "Afirmação 1" in resp.rationale
    assert "Test claim" in resp.rationale
    assert "Veredito:" in resp.rationale
    assert "Falso" in resp.rationale


def test_no_verdicts_returns_fallback_message():
    """no verdicts → 'no verifiable claims' fallback text."""
    result = _make_fact_check_result(verdicts=[], summary="Some summary")
    resp = fact_check_result_to_response("msg-2", result)

    assert "Nenhuma alegação verificável" in resp.rationale
    assert "Some summary" in resp.rationale


def test_overall_summary_included_in_rationale():
    """overall_summary appears in the rationale when present."""
    verdict = _make_verdict()
    result = _make_fact_check_result(verdicts=[verdict], summary="Important summary here")
    resp = fact_check_result_to_response("msg-3", result)

    assert "Resumo Geral" in resp.rationale
    assert "Important summary here" in resp.rationale


# ---- test: citations from citations_used ----

def test_citation_from_citations_used_appears_in_sources():
    """citations from verdict.citations_used should appear in the Fontes section."""
    citation = _make_citation(
        url="https://factcheck.org/claim1",
        title="Fact Check Article",
        publisher="FactCheck.org",
    )
    verdict = _make_verdict(
        justification="This is false [1].",
        citations=[citation],
    )
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-4", result)

    assert "Fontes" in resp.rationale
    assert "Fact Check Article" in resp.rationale
    assert "FactCheck.org" in resp.rationale


def test_citations_deduplicated_by_url():
    """same URL appearing in multiple verdicts should only appear once in sources."""
    shared_citation = _make_citation(url="https://example.com/shared")
    verdict1 = _make_verdict(
        claim_text="Claim A",
        justification="Reason A [1].",
        citations=[shared_citation],
    )
    verdict2 = _make_verdict(
        claim_text="Claim B",
        justification="Reason B [1].",
        citations=[shared_citation],
    )
    result = _make_fact_check_result(verdicts=[verdict1, verdict2])
    resp = fact_check_result_to_response("msg-5", result)

    # URL should only appear once in sources
    sources_section = resp.rationale.split("*Fontes*:")[-1] if "*Fontes*:" in resp.rationale else ""
    assert sources_section.count("https://example.com/shared") == 1


def test_empty_citations_used_no_sources_section():
    """when no verdict has citations, no Fontes section should appear."""
    verdict = _make_verdict(justification="No sources cited.", citations=[])
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-6", result)

    assert "Fontes" not in resp.rationale


def test_unreferenced_citations_excluded():
    """citations not referenced by [N] in text should be excluded from sources."""
    citation = _make_citation(url="https://example.com/not-referenced")
    # justification has no [N] references
    verdict = _make_verdict(
        justification="This is false based on evidence.",
        citations=[citation],
    )
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-7", result)

    # citation exists in all_citations but is NOT in the final output
    # because _add_citations_to_final_msg filters by [N] pattern
    assert "not-referenced" not in resp.rationale


def test_citation_date_included_when_present():
    """citation date should appear in sources section when set."""
    citation = _make_citation(
        url="https://example.com/dated",
        title="Dated Article",
        publisher="Publisher",
        date="2024-06-15",
    )
    verdict = _make_verdict(justification="See [1].", citations=[citation])
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-8", result)

    assert "2024-06-15" in resp.rationale


# ---- test: multiple verdicts ----

def test_multiple_verdicts_numbered():
    """multiple verdicts should be numbered Afirmação 1, Afirmação 2, etc."""
    v1 = _make_verdict(claim_text="Claim one", verdict="Verdadeiro")
    v2 = _make_verdict(claim_text="Claim two", verdict="Falso")
    result = _make_fact_check_result(verdicts=[v1, v2])
    resp = fact_check_result_to_response("msg-9", result)

    assert "Afirmação 1" in resp.rationale
    assert "Afirmação 2" in resp.rationale
    assert "Claim one" in resp.rationale
    assert "Claim two" in resp.rationale


# ---- test: responseWithoutLinks ----

def test_response_without_links_strips_urls():
    """responseWithoutLinks should not contain any URLs."""
    citation = _make_citation(url="https://example.com/strip-me")
    verdict = _make_verdict(justification="See [1].", citations=[citation])
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-10", result)

    assert "https://" not in resp.responseWithoutLinks


# ---- test: verdict summary helper ----

def test_verdict_summary_all_true():
    verdicts = [_make_verdict(verdict="Verdadeiro"), _make_verdict(verdict="Verdadeiro")]
    summary = _get_verdict_summary(verdicts)
    assert "verdadeiras" in summary.lower()


def test_verdict_summary_all_false():
    verdicts = [_make_verdict(verdict="Falso"), _make_verdict(verdict="Falso")]
    summary = _get_verdict_summary(verdicts)
    assert "Falsas" in summary


def test_verdict_summary_mixed():
    verdicts = [_make_verdict(verdict="Verdadeiro"), _make_verdict(verdict="Falso")]
    summary = _get_verdict_summary(verdicts)
    assert "1 de 2" in summary


def test_verdict_summary_empty():
    assert _get_verdict_summary([]) == ""


# ---- test: request_to_data_sources ----

def test_request_to_data_sources_creates_correct_count():
    """one content item → one DataSource."""
    req = Request(content=[ContentItem(textContent="Hello world", type=ContentType.TEXT)])
    sources = request_to_data_sources(req)
    assert len(sources) == 1
    assert sources[0].original_text == "Hello world"
    assert sources[0].source_type == "original_text"


def test_request_to_data_sources_maps_types():
    """different content types map to correct source types."""
    req = Request(content=[
        ContentItem(textContent="text", type=ContentType.TEXT),
        ContentItem(textContent="image", type=ContentType.IMAGE),
        ContentItem(textContent="audio", type=ContentType.AUDIO),
    ])
    sources = request_to_data_sources(req)
    assert sources[0].source_type == "original_text"
    assert sources[1].source_type == "image"
    assert sources[2].source_type == "audio_transcript"
