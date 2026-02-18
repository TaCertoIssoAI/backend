"""tests for AnalyticsCollector.populate_from_graph_output."""

from __future__ import annotations

import pytest

from app.observability.analytics.collector import AnalyticsCollector
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


# ---- helpers ----

def _make_fact_check_result(
    claim_text: str = "Test claim",
    verdict: str = "Falso",
    justification: str = "Reason [1].",
    summary: str = "Overall summary.",
    claim_id: str = "c-1",
    data_source_id: str = "ds-1",
    source_type: str = "original_text",
) -> FactCheckResult:
    return FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id=data_source_id,
                source_type=source_type,
                claim_verdicts=[
                    ClaimVerdict(
                        claim_id=claim_id,
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


def _make_fc_context(url: str = "https://fc.com/1", publisher: str = "Publisher", claim_text: str = "claim") -> FactCheckApiContext:
    return FactCheckApiContext(
        id="fc-1", url=url, parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        publisher=publisher, claim_text=claim_text, rating="Falso",
    )


def _make_search_context(
    url: str = "https://sr.com/1",
    title: str = "Search Result",
    snippet: str = "snippet text",
    domain: str = "sr.com",
) -> GoogleSearchContext:
    return GoogleSearchContext(
        id="gs-1", url=url, parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        title=title, snippet=snippet, domain=domain,
    )


def _make_scrape_context(
    url: str = "https://sp.com/1",
    extraction_status: str = "success",
    content: str = "scraped content",
    title: str = "Scraped Page",
) -> WebScrapeContext:
    return WebScrapeContext(
        id="ws-1", url=url, parent_id=None,
        reliability=SourceReliability.POUCO_CONFIAVEL,
        title=title, content=content,
        extraction_status=extraction_status, extraction_tool="cloudscraper",
    )


def _collector() -> AnalyticsCollector:
    return AnalyticsCollector(msg_id="msg-test")


# ---- tests ----

def test_populate_fills_response_by_data_source():
    col = _collector()
    fc_result = _make_fact_check_result()

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )

    assert len(col.analytics.ResponseByDataSource) == 1
    ds = col.analytics.ResponseByDataSource[0]
    assert ds.data_source_id == "ds-1"
    assert len(ds.claim_verdicts) == 1


def test_populate_fills_response_by_claim():
    col = _collector()
    fc_result = _make_fact_check_result(claim_text="Earth is flat", verdict="Falso", justification="No refs.")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )

    assert "1" in col.analytics.ResponseByClaim
    claim_resp = col.analytics.ResponseByClaim["1"]
    assert claim_resp.claim_text == "Earth is flat"
    assert claim_resp.Result == "Falso"
    assert claim_resp.reasoningText == "No refs."


def test_populate_fills_overall_summary():
    col = _collector()
    fc_result = _make_fact_check_result(summary="This is the summary.")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )

    assert col.analytics.CommentAboutCompleteContext == "This is the summary."


def test_scraped_links_populated_from_successful_scrapes():
    col = _collector()
    fc_result = _make_fact_check_result()
    success_page = _make_scrape_context(url="https://good.com/page", extraction_status="success", content="content")
    failed_page = _make_scrape_context(url="https://bad.com/page", extraction_status="failed")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[success_page, failed_page],
    )

    urls = [link.url for link in col.analytics.ScrapedLinks]
    assert "https://good.com/page" in urls
    assert "https://bad.com/page" not in urls


def test_reasoning_sources_patched_via_filter():
    col = _collector()
    fc_entry = _make_fc_context(url="https://fc.com/1", publisher="FactCheck", claim_text="some claim")
    fc_result = _make_fact_check_result(justification="Evidence from [1] is clear.")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[fc_entry],
        search_results={},
        scraped_pages=[],
    )

    sources = col.analytics.ResponseByClaim["1"].reasoningSources
    assert len(sources) == 1
    assert sources[0].url == "https://fc.com/1"
    assert sources[0].publisher == "FactCheck"


def test_reasoning_sources_on_response_by_data_source():
    col = _collector()
    fc_entry = _make_fc_context(url="https://fc.com/1", publisher="Checker")
    fc_result = _make_fact_check_result(justification="See [1].")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[fc_entry],
        search_results={},
        scraped_pages=[],
    )

    ds = col.analytics.ResponseByDataSource[0]
    cv_sources = ds.claim_verdicts[0].reasoningSources
    assert len(cv_sources) == 1
    assert cv_sources[0].url == "https://fc.com/1"


def test_reasoning_sources_publisher_from_named_domain():
    col = _collector()
    g1_entry = _make_search_context(url="https://g1.globo.com/article", snippet="g1 snippet")
    # g1 is ref [1] in the source list (no fact_check_results, no named domains before g1)
    fc_result = _make_fact_check_result(justification="Reported by [1].")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={"g1": [g1_entry]},
        scraped_pages=[],
    )

    sources = col.analytics.ResponseByClaim["1"].reasoningSources
    assert len(sources) == 1
    assert sources[0].url == "https://g1.globo.com/article"
    assert sources[0].publisher == "g1"
    assert sources[0].citation_text == "g1 snippet"


def test_reasoning_sources_publisher_from_fact_check():
    col = _collector()
    fc_entry = _make_fc_context(
        url="https://aosfatos.org/1",
        publisher="Aos Fatos",
        claim_text="The claim checked here",
    )
    fc_result = _make_fact_check_result(justification="According to [1].")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[fc_entry],
        search_results={},
        scraped_pages=[],
    )

    sources = col.analytics.ResponseByClaim["1"].reasoningSources
    assert len(sources) == 1
    assert sources[0].publisher == "Aos Fatos"
    assert sources[0].citation_text == "The claim checked here"


def test_reasoning_sources_empty_when_no_refs_in_justification():
    col = _collector()
    fc_entry = _make_fc_context(url="https://fc.com/1")
    fc_result = _make_fact_check_result(justification="No references mentioned here at all.")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[fc_entry],
        search_results={},
        scraped_pages=[],
    )

    sources = col.analytics.ResponseByClaim["1"].reasoningSources
    assert sources == []


def test_has_extracted_claims_true_after_populate():
    col = _collector()
    fc_result = _make_fact_check_result()

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )

    assert col.has_extracted_claims() is True


def test_empty_source_lists():
    col = _collector()
    fc_result = _make_fact_check_result(justification="No sources.")

    col.populate_from_graph_output(
        fact_check_result=fc_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )

    assert col.analytics.ScrapedLinks == []
    sources = col.analytics.ResponseByClaim["1"].reasoningSources
    assert sources == []


# ---- has_extracted_claims edge cases ----

def test_has_extracted_claims_false_when_no_claims():
    """fresh collector with no data should return False."""
    col = _collector()
    assert col.has_extracted_claims() is False


def test_has_extracted_claims_false_with_empty_claim_verdicts():
    """ResponseByDataSource entries with zero claim_verdicts should not count."""
    col = _collector()
    empty_result = FactCheckResult(
        results=[
            DataSourceResult(
                data_source_id="ds-1",
                source_type="original_text",
                claim_verdicts=[],
            )
        ],
        overall_summary="Nothing.",
    )

    col.populate_from_graph_output(
        fact_check_result=empty_result,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )

    # ResponseByDataSource has an entry, but with zero claim_verdicts
    assert len(col.analytics.ResponseByDataSource) == 1
    assert col.has_extracted_claims() is False


def test_has_extracted_claims_true_with_response_by_claim():
    """ResponseByClaim with entries should be enough."""
    col = _collector()
    from app.models.analytics import ClaimResponseAnalytics
    col.analytics.ResponseByClaim["1"] = ClaimResponseAnalytics(
        claim_id="c-1", claim_text="claim", Result="Falso", reasoningText="reason"
    )
    assert col.has_extracted_claims() is True
