"""tests for the mapper's fact_check_result_to_response function.

covers:
- basic response structure and fields
- citation numbering matches context_formatter ordering exactly
- source type ordering: fact_check → aosfatos → g1 → estadao → folha → geral → scraped
- filter_cited_references integration (only cited [N] appear)
- no source lists → no citations section
- no verdicts → fallback message
- overall_summary inclusion
- verdict count summary generation
- responseWithoutLinks strips URLs
"""

from __future__ import annotations

import pytest
from app.api.mapper.mapper import (
    fact_check_result_to_response,
    _build_citations_from_source_refs,
    request_to_data_sources,
    _get_verdict_summary,
)
from app.models.api import Request, ContentItem, ContentType, AnalysisResponse
from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
    SourceReliability,
)
from app.models.factchecking import (
    Citation,
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
)
from app.agentic_ai.prompts.context_formatter import (
    build_source_reference_list,
    filter_cited_references,
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


def _make_fc_context(
    publisher: str = "Aos Fatos",
    claim_text: str = "test claim",
    url: str = "https://aosfatos.org/check1",
    rating: str = "Falso",
) -> FactCheckApiContext:
    return FactCheckApiContext(
        id="fc-1",
        url=url,
        parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        title=f"Check: {claim_text}",
        publisher=publisher,
        rating=rating,
        claim_text=claim_text,
    )


def _make_search_context(
    title: str = "Search Result",
    url: str = "https://example.com/result",
    domain: str = "example.com",
    snippet: str = "Some snippet",
) -> GoogleSearchContext:
    return GoogleSearchContext(
        id="gs-1",
        url=url,
        parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        title=title,
        snippet=snippet,
        domain=domain,
    )


def _make_scrape_context(
    title: str = "Scraped Page",
    url: str = "https://example.com/page",
    content: str = "Page content here",
) -> WebScrapeContext:
    return WebScrapeContext(
        id="ws-1",
        url=url,
        parent_id=None,
        reliability=SourceReliability.POUCO_CONFIAVEL,
        title=title,
        content=content,
        extraction_status="success",
        extraction_tool="cloudscraper",
    )


# ---- test: basic response structure ----

def test_single_verdict_produces_valid_response():
    """a single verdict should produce a valid AnalysisResponse."""
    verdict = _make_verdict()
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


# ---- test: no source lists → no citations ----

def test_no_source_lists_no_fontes_section():
    """when no source lists are provided, no Fontes section should appear."""
    verdict = _make_verdict(justification="Reason [1].")
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-no-src", result)

    assert "Fontes" not in resp.rationale


def test_empty_source_lists_no_fontes_section():
    """when source lists are empty, no Fontes section should appear."""
    verdict = _make_verdict(justification="Reason [1].")
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response(
        "msg-empty", result,
        fact_check_results=[], search_results={}, scraped_pages=[],
    )

    assert "Fontes" not in resp.rationale


# ---- test: citations from source refs ----

def test_fact_check_source_appears_as_citation_1():
    """a single fact_check_results entry should appear as [1] in citations."""
    fc = _make_fc_context(
        publisher="Estadão",
        claim_text="some claim",
        url="https://estadao.com.br/check",
    )
    verdict = _make_verdict(justification="According to evidence [1].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-fc", result,
        fact_check_results=[fc], search_results={}, scraped_pages=[],
    )

    assert "*Fontes*:" in resp.rationale
    assert "[1]" in resp.rationale.split("*Fontes*:")[-1]
    assert "https://estadao.com.br/check" in resp.rationale


def test_search_result_in_aosfatos_domain():
    """a search result in 'aosfatos' domain key should get correct numbering."""
    sr = _make_search_context(
        title="Aos Fatos Article",
        url="https://aosfatos.org/article",
        domain="aosfatos.org",
    )
    verdict = _make_verdict(justification="Evidence [1] confirms.")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-aof", result,
        fact_check_results=[],
        search_results={"aosfatos": [sr]},
        scraped_pages=[],
    )

    assert "*Fontes*:" in resp.rationale
    assert "Aos Fatos Article" in resp.rationale


def test_scraped_page_appears_in_citations():
    """a scraped page should appear in citations when referenced."""
    sp = _make_scrape_context(
        title="Scraped Blog Post",
        url="https://blog.example.com/post",
    )
    verdict = _make_verdict(justification="Blog evidence [1].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-sp", result,
        fact_check_results=[], search_results={}, scraped_pages=[sp],
    )

    assert "*Fontes*:" in resp.rationale
    assert "Scraped Blog Post" in resp.rationale
    assert "https://blog.example.com/post" in resp.rationale


def test_unreferenced_sources_excluded():
    """sources not referenced by [N] in text should be excluded from Fontes."""
    fc = _make_fc_context(url="https://factcheck.com/1")
    sr = _make_search_context(title="Unreferenced", url="https://unreferenced.com")

    # justification only references [1] (fact_check), not [2] (search)
    verdict = _make_verdict(justification="Only this source [1].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-unref", result,
        fact_check_results=[fc],
        search_results={"aosfatos": [sr]},
        scraped_pages=[],
    )

    assert "https://factcheck.com/1" in resp.rationale
    assert "https://unreferenced.com" not in resp.rationale


# ---- test: citation numbering matches context_formatter ordering ----

def test_numbering_matches_format_context_order():
    """citation numbers in mapper must match build_source_reference_list ordering.

    ordering: fact_check → aosfatos → g1 → estadao → folha → geral → scraped
    """
    fc1 = _make_fc_context(publisher="P1", claim_text="fc claim", url="https://fc.com/1")
    aof1 = _make_search_context(title="AosFatos 1", url="https://aosfatos.org/1", domain="aosfatos.org")
    g1_1 = _make_search_context(title="G1 Article", url="https://g1.com/1", domain="g1.com")
    est1 = _make_search_context(title="Estadao Article", url="https://estadao.com/1", domain="estadao.com")
    fol1 = _make_search_context(title="Folha Article", url="https://folha.com/1", domain="folha.com")
    gen1 = _make_search_context(title="General Result", url="https://general.com/1", domain="general.com")
    sp1 = _make_scrape_context(title="Scraped Page", url="https://scraped.com/1")

    search_results = {
        "aosfatos": [aof1],
        "g1": [g1_1],
        "estadao": [est1],
        "folha": [fol1],
        "geral": [gen1],
    }

    # build expected refs using the same function the mapper uses
    expected_refs = build_source_reference_list([fc1], search_results, [sp1])

    # assert expected ordering
    assert expected_refs[0] == (1, "P1: fc claim", "https://fc.com/1")
    assert expected_refs[1] == (2, "AosFatos 1", "https://aosfatos.org/1")
    assert expected_refs[2] == (3, "G1 Article", "https://g1.com/1")
    assert expected_refs[3] == (4, "Estadao Article", "https://estadao.com/1")
    assert expected_refs[4] == (5, "Folha Article", "https://folha.com/1")
    assert expected_refs[5] == (6, "General Result", "https://general.com/1")
    assert expected_refs[6] == (7, "Scraped Page", "https://scraped.com/1")

    # now verify the mapper produces citations with the same numbering
    verdict = _make_verdict(justification="See [1] and [3] and [7].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-order", result,
        fact_check_results=[fc1],
        search_results=search_results,
        scraped_pages=[sp1],
    )

    fontes_section = resp.rationale.split("*Fontes*:")[-1]

    # [1] → fc.com, [3] → g1.com, [7] → scraped.com
    assert "[1]" in fontes_section
    assert "https://fc.com/1" in fontes_section
    assert "[3]" in fontes_section
    assert "https://g1.com/1" in fontes_section
    assert "[7]" in fontes_section
    assert "https://scraped.com/1" in fontes_section

    # unreferenced ones should NOT appear
    assert "https://aosfatos.org/1" not in fontes_section
    assert "https://estadao.com/1" not in fontes_section
    assert "https://folha.com/1" not in fontes_section
    assert "https://general.com/1" not in fontes_section


def test_numbering_fact_check_comes_first():
    """fact_check_results should always be numbered before search results."""
    fc = _make_fc_context(publisher="Pub", claim_text="claim", url="https://fc.com/1")
    sr = _make_search_context(title="Search", url="https://search.com/1")

    verdict = _make_verdict(justification="[1] and [2]")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-fc-first", result,
        fact_check_results=[fc],
        search_results={"aosfatos": [sr]},
        scraped_pages=[],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]
    fc_pos = fontes.index("https://fc.com/1")
    sr_pos = fontes.index("https://search.com/1")
    assert fc_pos < sr_pos, "fact_check source should appear before search result"


def test_numbering_domain_order_within_search():
    """search results should follow domain order: aosfatos → g1 → estadao → folha → geral."""
    aof = _make_search_context(title="AoF", url="https://aof.com/1")
    g1 = _make_search_context(title="G1", url="https://g1.com/1")
    est = _make_search_context(title="Est", url="https://est.com/1")
    fol = _make_search_context(title="Fol", url="https://fol.com/1")
    gen = _make_search_context(title="Gen", url="https://gen.com/1")

    search_results = {
        "aosfatos": [aof],
        "g1": [g1],
        "estadao": [est],
        "folha": [fol],
        "geral": [gen],
    }

    # reference all 5
    verdict = _make_verdict(justification="[1][2][3][4][5]")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-domain-order", result,
        fact_check_results=[],
        search_results=search_results,
        scraped_pages=[],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]
    positions = {
        "aof": fontes.index("https://aof.com/1"),
        "g1": fontes.index("https://g1.com/1"),
        "est": fontes.index("https://est.com/1"),
        "fol": fontes.index("https://fol.com/1"),
        "gen": fontes.index("https://gen.com/1"),
    }

    assert positions["aof"] < positions["g1"]
    assert positions["g1"] < positions["est"]
    assert positions["est"] < positions["fol"]
    assert positions["fol"] < positions["gen"]


def test_numbering_scraped_pages_come_last():
    """scraped pages should always be numbered after all search results."""
    gen = _make_search_context(title="General", url="https://general.com/1")
    sp = _make_scrape_context(title="Scraped", url="https://scraped.com/1")

    verdict = _make_verdict(justification="[1] and [2]")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-scraped-last", result,
        fact_check_results=[],
        search_results={"geral": [gen]},
        scraped_pages=[sp],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]
    gen_pos = fontes.index("https://general.com/1")
    sp_pos = fontes.index("https://scraped.com/1")
    assert gen_pos < sp_pos, "scraped page should appear after general search result"


def test_multiple_entries_per_domain():
    """multiple entries within a domain should be numbered consecutively."""
    sr1 = _make_search_context(title="AoF Article 1", url="https://aof.com/1")
    sr2 = _make_search_context(title="AoF Article 2", url="https://aof.com/2")

    refs = build_source_reference_list([], {"aosfatos": [sr1, sr2]}, [])
    assert refs[0] == (1, "AoF Article 1", "https://aof.com/1")
    assert refs[1] == (2, "AoF Article 2", "https://aof.com/2")


def test_mapper_and_context_formatter_produce_same_refs():
    """the mapper's citation output must match build_source_reference_list exactly.

    this is the core consistency test — if the mapper used a different ordering,
    the [N] numbers in LLM justifications would point to wrong sources.
    """
    fc1 = _make_fc_context(publisher="CheckOrg", claim_text="vax claim", url="https://check.org/1")
    fc2 = _make_fc_context(publisher="Lupa", claim_text="economy claim", url="https://lupa.com/1")
    aof = _make_search_context(title="AosFatos Result", url="https://aosfatos.org/r1")
    g1 = _make_search_context(title="G1 Result", url="https://g1.com/r1")
    gen1 = _make_search_context(title="General 1", url="https://gen.com/1")
    gen2 = _make_search_context(title="General 2", url="https://gen.com/2")
    sp = _make_scrape_context(title="Blog", url="https://blog.com/1")

    search_results = {
        "aosfatos": [aof],
        "g1": [g1],
        "geral": [gen1, gen2],
    }

    # build the canonical reference list
    canonical_refs = build_source_reference_list([fc1, fc2], search_results, [sp])

    # [1]=fc1, [2]=fc2, [3]=aof, [4]=g1, [5]=gen1, [6]=gen2, [7]=sp
    assert len(canonical_refs) == 7
    assert canonical_refs[0][2] == "https://check.org/1"   # [1]
    assert canonical_refs[1][2] == "https://lupa.com/1"     # [2]
    assert canonical_refs[2][2] == "https://aosfatos.org/r1"  # [3]
    assert canonical_refs[3][2] == "https://g1.com/r1"     # [4]
    assert canonical_refs[4][2] == "https://gen.com/1"      # [5]
    assert canonical_refs[5][2] == "https://gen.com/2"      # [6]
    assert canonical_refs[6][2] == "https://blog.com/1"     # [7]

    # now call the mapper with justifications citing [2], [4], [7]
    verdict = _make_verdict(justification="See [2] and also [4]. Blog says [7].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-consistency", result,
        fact_check_results=[fc1, fc2],
        search_results=search_results,
        scraped_pages=[sp],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]

    # cited ones should appear with correct URLs
    assert "[2]" in fontes and "https://lupa.com/1" in fontes
    assert "[4]" in fontes and "https://g1.com/r1" in fontes
    assert "[7]" in fontes and "https://blog.com/1" in fontes

    # non-cited ones should NOT appear
    assert "https://check.org/1" not in fontes
    assert "https://aosfatos.org/r1" not in fontes
    assert "https://gen.com/1" not in fontes
    assert "https://gen.com/2" not in fontes


def test_citations_from_overall_summary_also_included():
    """[N] references in overall_summary (not just justifications) should be picked up."""
    fc = _make_fc_context(publisher="P", claim_text="c", url="https://fc.com/1")
    sr = _make_search_context(title="SR", url="https://sr.com/1")

    # [1] only in summary, [2] only in justification
    verdict = _make_verdict(justification="Evidence from [2].")
    result = _make_fact_check_result(
        verdicts=[verdict],
        summary="Overall: see [1] for context.",
    )

    resp = fact_check_result_to_response(
        "msg-summary-cite", result,
        fact_check_results=[fc],
        search_results={"aosfatos": [sr]},
        scraped_pages=[],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]
    assert "[1]" in fontes and "https://fc.com/1" in fontes
    assert "[2]" in fontes and "https://sr.com/1" in fontes


def test_citations_from_multiple_verdicts():
    """citations referenced across multiple verdicts should all appear."""
    fc = _make_fc_context(publisher="P", claim_text="c", url="https://fc.com/1")
    sr = _make_search_context(title="SR", url="https://sr.com/1")
    sp = _make_scrape_context(title="SP", url="https://sp.com/1")

    v1 = _make_verdict(claim_text="Claim A", justification="Ref [1].")
    v2 = _make_verdict(claim_text="Claim B", justification="Ref [3].")
    result = _make_fact_check_result(verdicts=[v1, v2])

    resp = fact_check_result_to_response(
        "msg-multi-v", result,
        fact_check_results=[fc],
        search_results={"aosfatos": [sr]},
        scraped_pages=[sp],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]
    assert "[1]" in fontes and "https://fc.com/1" in fontes
    assert "[3]" in fontes and "https://sp.com/1" in fontes
    # [2] not cited, should not appear
    assert "https://sr.com/1" not in fontes


def test_no_citations_referenced_no_fontes_section():
    """when justification has no [N] patterns, Fontes section should not appear."""
    fc = _make_fc_context(url="https://fc.com/1")
    verdict = _make_verdict(justification="No numbered citations here.")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-no-cite", result,
        fact_check_results=[fc], search_results={}, scraped_pages=[],
    )

    assert "Fontes" not in resp.rationale


def test_high_citation_numbers():
    """mapper should handle [N] with large numbers (e.g. 15+ sources)."""
    # create 20 general search results with unique URLs
    search_results = {"geral": [
        _make_search_context(title=f"Result {i}", url=f"https://gen.com/article-{i}")
        for i in range(1, 21)
    ]}

    # reference [5], [10], [20]
    verdict = _make_verdict(justification="See [5], [10], and [20].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-high-nums", result,
        fact_check_results=[],
        search_results=search_results,
        scraped_pages=[],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]
    assert "https://gen.com/article-5" in fontes
    assert "https://gen.com/article-10" in fontes
    assert "https://gen.com/article-20" in fontes
    # non-cited should not appear
    assert "https://gen.com/article-1\n" not in fontes
    assert "https://gen.com/article-3\n" not in fontes


# ---- test: _build_citations_from_source_refs directly ----

def test_build_citations_returns_empty_for_no_sources():
    """_build_citations_from_source_refs returns empty when no source lists."""
    result = _make_fact_check_result(
        verdicts=[_make_verdict(justification="[1]")],
    )
    text = _build_citations_from_source_refs(
        "rationale [1]", result, [], {}, [],
    )
    assert text == ""


def test_build_citations_returns_empty_for_no_references():
    """_build_citations_from_source_refs returns empty when text has no [N]."""
    fc = _make_fc_context(url="https://fc.com/1")
    result = _make_fact_check_result(
        verdicts=[_make_verdict(justification="no refs here")],
    )
    text = _build_citations_from_source_refs(
        "rationale with no refs", result, [fc], {}, [],
    )
    assert text == ""


# ---- test: responseWithoutLinks ----

def test_response_without_links_strips_urls():
    """responseWithoutLinks should not contain any URLs."""
    fc = _make_fc_context(url="https://example.com/strip-me")
    verdict = _make_verdict(justification="See [1].")
    result = _make_fact_check_result(verdicts=[verdict])

    resp = fact_check_result_to_response(
        "msg-10", result,
        fact_check_results=[fc], search_results={}, scraped_pages=[],
    )

    assert "https://" not in resp.responseWithoutLinks


# ---- test: backward compatibility ----

def test_backward_compat_no_source_args():
    """calling without source list args should still work (no citations)."""
    verdict = _make_verdict(justification="See [1].")
    result = _make_fact_check_result(verdicts=[verdict])
    resp = fact_check_result_to_response("msg-compat", result)

    assert isinstance(resp, AnalysisResponse)
    assert "Fontes" not in resp.rationale


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


# ---- test: citation ordering stability (integration) ----

def test_citation_number_stability_across_calls():
    """the same source configuration should always produce the same numbering."""
    fc = _make_fc_context(publisher="P", claim_text="c", url="https://fc.com/1")
    sr = _make_search_context(title="SR", url="https://sr.com/1")

    search_results = {"aosfatos": [sr]}

    refs_a = build_source_reference_list([fc], search_results, [])
    refs_b = build_source_reference_list([fc], search_results, [])

    assert refs_a == refs_b


def test_full_pipeline_citation_order_matches_cli():
    """simulate the full pipeline and verify mapper output matches CLI logic.

    this replicates the exact approach used in cli.py:
    1. build_source_reference_list from state
    2. filter_cited_references with justification texts
    3. display the cited refs with [N] numbering
    """
    fc = _make_fc_context(publisher="Lupa", claim_text="economia", url="https://lupa.com/eco")
    aof = _make_search_context(title="AosFatos: economia", url="https://aosfatos.org/eco")
    g1 = _make_search_context(title="G1 Report", url="https://g1.com/report")
    gen = _make_search_context(title="Wikipedia", url="https://pt.wikipedia.org/eco")
    sp = _make_scrape_context(title="Blog Analysis", url="https://blog.com/analysis")

    search_results = {
        "aosfatos": [aof],
        "g1": [g1],
        "geral": [gen],
    }

    # --- CLI approach ---
    cli_refs = build_source_reference_list([fc], search_results, [sp])
    # [1]=lupa, [2]=aosfatos, [3]=g1, [4]=wikipedia, [5]=blog

    justification_text = "According to [1], the economy [3]. Blog analysis [5] also confirms."
    summary_text = "See also [2]."

    cli_cited = filter_cited_references(
        cli_refs, justification_text, summary_text,
    )

    # should have [1], [2], [3], [5]
    cli_cited_nums = [ref[0] for ref in cli_cited]
    assert cli_cited_nums == [1, 2, 3, 5]
    assert cli_cited[0][2] == "https://lupa.com/eco"
    assert cli_cited[1][2] == "https://aosfatos.org/eco"
    assert cli_cited[2][2] == "https://g1.com/report"
    assert cli_cited[3][2] == "https://blog.com/analysis"

    # --- Mapper approach ---
    verdict = _make_verdict(justification=justification_text)
    result = _make_fact_check_result(verdicts=[verdict], summary=summary_text)

    resp = fact_check_result_to_response(
        "msg-cli-match", result,
        fact_check_results=[fc],
        search_results=search_results,
        scraped_pages=[sp],
    )

    fontes = resp.rationale.split("*Fontes*:")[-1]

    # verify same URLs appear in the same order
    for num, _title, url in cli_cited:
        assert f"[{num}]" in fontes
        assert url in fontes

    # [4] (wikipedia) should NOT appear — not cited
    assert "https://pt.wikipedia.org/eco" not in fontes

    # verify ordering: [1] before [2] before [3] before [5]
    pos_1 = fontes.index("[1]")
    pos_2 = fontes.index("[2]")
    pos_3 = fontes.index("[3]")
    pos_5 = fontes.index("[5]")
    assert pos_1 < pos_2 < pos_3 < pos_5
