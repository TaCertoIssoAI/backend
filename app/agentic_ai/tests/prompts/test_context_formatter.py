"""tests for context_formatter."""

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
    SourceReliability,
)
from app.agentic_ai.prompts.context_formatter import format_context


def _make_fact_check(id="fc-1", title="FC Title", publisher="Lupa", rating="Falso"):
    return FactCheckApiContext(
        id=id,
        url="https://lupa.uol.com.br/test",
        parent_id=None,
        reliability=SourceReliability.MUITO_CONFIAVEL,
        title=title,
        publisher=publisher,
        rating=rating,
        claim_text="test claim",
        review_date="2025-01-10",
    )


def _make_search(id="gs-1", domain_key="geral", domain="bbc.com"):
    muito_confiavel_domains = {"aosfatos", "g1", "estadao", "folha"}
    reliability = (
        SourceReliability.MUITO_CONFIAVEL
        if domain_key in muito_confiavel_domains
        else SourceReliability.NEUTRO
    )
    return GoogleSearchContext(
        id=id,
        url=f"https://{domain}/test",
        parent_id=None,
        reliability=reliability,
        title="Search Title",
        snippet="search snippet",
        domain=domain,
        position=1,
    )


def _make_scrape(id="sc-1"):
    return WebScrapeContext(
        id=id,
        url="https://example.com/page",
        parent_id=None,
        reliability=SourceReliability.POUCO_CONFIAVEL,
        title="Scraped Page",
        content="page content here",
        extraction_status="success",
        extraction_tool="beautifulsoup",
    )


def test_empty_context_returns_placeholder():
    result = format_context([], {}, [])
    assert "nenhuma fonte coletada" in result


def test_fact_check_results_appear_in_muito_confiavel():
    fc = _make_fact_check()
    result = format_context([fc], {}, [])
    assert "Muito confiável" in result
    assert "Fact-Check API" in result
    assert "Lupa" in result
    assert "[1]" in result


def test_aosfatos_appears_in_muito_confiavel():
    entry = _make_search(domain_key="aosfatos", domain="aosfatos.org")
    result = format_context([], {"aosfatos": [entry]}, [])
    assert "Muito confiável" in result
    assert "Aos Fatos" in result


def test_general_search_appears_in_neutro():
    entry = _make_search(domain_key="geral", domain="bbc.com")
    result = format_context([], {"geral": [entry]}, [])
    assert "Neutro" in result
    assert "Geral" in result


def test_scraped_pages_appear_in_pouco_confiavel():
    scrape = _make_scrape()
    result = format_context([], {}, [scrape])
    assert "Pouco confiável" in result
    assert "Conteúdo Extraído" in result


def test_global_numbering_is_contiguous():
    fc = _make_fact_check(id="fc-1")
    search = _make_search(id="gs-1", domain_key="geral")
    scrape = _make_scrape(id="sc-1")
    result = format_context([fc], {"geral": [search]}, [scrape])
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


def test_multiple_domains_ordered_correctly():
    g1 = _make_search(id="g1-1", domain_key="g1", domain="g1.globo.com")
    estadao = _make_search(id="es-1", domain_key="estadao", domain="estadao.com.br")
    result = format_context(
        [], {"g1": [g1], "estadao": [estadao]}, []
    )
    g1_pos = result.index("G1")
    estadao_pos = result.index("Estadão")
    assert g1_pos < estadao_pos


def test_g1_appears_in_muito_confiavel():
    entry = _make_search(domain_key="g1", domain="g1.globo.com")
    result = format_context([], {"g1": [entry]}, [])
    assert "Muito confiável" in result
    assert "G1" in result


def test_estadao_appears_in_muito_confiavel():
    entry = _make_search(domain_key="estadao", domain="estadao.com.br")
    result = format_context([], {"estadao": [entry]}, [])
    assert "Muito confiável" in result
    assert "Estadão" in result


def test_folha_appears_in_muito_confiavel():
    entry = _make_search(domain_key="folha", domain="folha.uol.com.br")
    result = format_context([], {"folha": [entry]}, [])
    assert "Muito confiável" in result
    assert "Folha" in result
