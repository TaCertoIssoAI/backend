"""tests for context_formatter."""

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
    SourceReliability,
)
from app.agentic_ai.prompts.context_formatter import (
    format_context,
    build_source_reference_list,
    filter_cited_references,
)


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
    muito_confiavel_domains = {"especifico"}
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


def test_especifico_appears_in_muito_confiavel():
    entry = _make_search(domain_key="especifico", domain="aosfatos.org")
    result = format_context([], {"especifico": [entry]}, [])
    assert "Muito confiável" in result
    assert "Sites específicos" in result


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


def test_especifico_section_appears_before_neutro():
    especifico = _make_search(id="es-1", domain_key="especifico", domain="g1.globo.com")
    geral = _make_search(id="ge-1", domain_key="geral", domain="bbc.com")
    result = format_context([], {"especifico": [especifico], "geral": [geral]}, [])
    especifico_pos = result.index("Sites específicos")
    geral_pos = result.index("Geral")
    assert especifico_pos < geral_pos


def test_especifico_appears_in_muito_confiavel_label():
    entry = _make_search(domain_key="especifico", domain="g1.globo.com")
    result = format_context([], {"especifico": [entry]}, [])
    assert "Muito confiável" in result
    assert "Sites específicos" in result


# ===== build_source_reference_list tests =====


def test_ref_list_empty():
    refs = build_source_reference_list([], {}, [])
    assert refs == []


def test_ref_list_fact_check_uses_publisher_and_claim():
    fc = _make_fact_check(publisher="Lupa", id="fc-1")
    refs = build_source_reference_list([fc], {}, [])
    assert len(refs) == 1
    num, title, url = refs[0]
    assert num == 1
    assert "Lupa" in title
    assert "test claim" in title
    assert url == "https://lupa.uol.com.br/test"


def test_ref_list_fact_check_no_claim_text():
    fc = _make_fact_check(publisher="AosFatos")
    fc.claim_text = ""
    refs = build_source_reference_list([fc], {}, [])
    _, title, _ = refs[0]
    assert title == "AosFatos"


def test_ref_list_ordering_matches_format_context():
    """numbering must follow the same order as format_context: fact-checks, especifico, geral, scraped."""
    fc = _make_fact_check(id="fc-1")
    especifico = _make_search(id="es-1", domain_key="especifico", domain="g1.globo.com")
    geral = _make_search(id="ge-1", domain_key="geral", domain="bbc.com")
    scrape = _make_scrape(id="sc-1")

    refs = build_source_reference_list(
        [fc],
        {"especifico": [especifico], "geral": [geral]},
        [scrape],
    )

    numbers = [r[0] for r in refs]
    assert numbers == [1, 2, 3, 4]


def test_ref_list_numbering_contiguous_with_multiple_per_domain():
    es_a = _make_search(id="es-1", domain_key="especifico", domain="g1.globo.com")
    es_b = _make_search(id="es-2", domain_key="especifico", domain="g1.globo.com")
    geral = _make_search(id="ge-1", domain_key="geral", domain="bbc.com")

    refs = build_source_reference_list(
        [], {"especifico": [es_a, es_b], "geral": [geral]}, []
    )
    numbers = [r[0] for r in refs]
    assert numbers == [1, 2, 3]


def test_ref_list_urls_preserved():
    scrape = _make_scrape(id="sc-1")
    refs = build_source_reference_list([], {}, [scrape])
    assert refs[0][2] == "https://example.com/page"


# ===== filter_cited_references tests =====


def _sample_refs() -> list[tuple[int, str, str]]:
    return [
        (1, "Source A", "https://a.com"),
        (2, "Source B", "https://b.com"),
        (3, "Source C", "https://c.com"),
        (4, "Source D", "https://d.com"),
        (5, "Source E", "https://e.com"),
    ]


def test_filter_single_citation():
    refs = _sample_refs()
    result = filter_cited_references(refs, "According to [3], this is true.")
    assert len(result) == 1
    assert result[0][0] == 3


def test_filter_multiple_citations():
    refs = _sample_refs()
    result = filter_cited_references(refs, "Sources [1][3][5] agree.")
    assert [r[0] for r in result] == [1, 3, 5]


def test_filter_preserves_original_order():
    refs = _sample_refs()
    result = filter_cited_references(refs, "Confirmed by [5][1][3].")
    assert [r[0] for r in result] == [1, 3, 5]


def test_filter_across_multiple_texts():
    refs = _sample_refs()
    result = filter_cited_references(
        refs,
        "Claim A says [1].",
        "Claim B says [4].",
        "Summary mentions [2].",
    )
    assert [r[0] for r in result] == [1, 2, 4]


def test_filter_deduplicates_repeated_citations():
    refs = _sample_refs()
    result = filter_cited_references(refs, "See [2][2][2] for details.")
    assert len(result) == 1
    assert result[0][0] == 2


def test_filter_no_citations_returns_empty():
    refs = _sample_refs()
    result = filter_cited_references(refs, "No citations here.")
    assert result == []


def test_filter_empty_text_returns_empty():
    refs = _sample_refs()
    result = filter_cited_references(refs, "")
    assert result == []


def test_filter_none_text_skipped():
    refs = _sample_refs()
    result = filter_cited_references(refs, None, "[1] is valid")
    assert [r[0] for r in result] == [1]


def test_filter_citation_not_in_refs_ignored():
    refs = _sample_refs()
    result = filter_cited_references(refs, "See [99] for more.")
    assert result == []


def test_filter_adjacent_citations():
    refs = _sample_refs()
    result = filter_cited_references(refs, "Confirmed [1][2][3].")
    assert [r[0] for r in result] == [1, 2, 3]


def test_filter_mixed_text_and_citations():
    refs = _sample_refs()
    text = "According to the study [2], the data [4] shows evidence. See also [1]."
    result = filter_cited_references(refs, text)
    assert [r[0] for r in result] == [1, 2, 4]


# ===== numbering consistency: format_context vs build_source_reference_list =====

import re


def test_format_context_and_ref_list_numbering_match():
    """[N] numbers from format_context match those from build_source_reference_list."""
    fc1 = _make_fact_check(id="fc-1")
    fc2 = _make_fact_check(id="fc-2", publisher="AosFatos")
    es = _make_search(id="es-1", domain_key="especifico", domain="g1.globo.com")
    ge1 = _make_search(id="ge-1", domain_key="geral", domain="bbc.com")
    ge2 = _make_search(id="ge-2", domain_key="geral", domain="cnn.com")
    sc = _make_scrape(id="sc-1")

    search = {"especifico": [es], "geral": [ge1, ge2]}

    formatted = format_context([fc1, fc2], search, [sc])
    formatted_numbers = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", formatted)))

    refs = build_source_reference_list([fc1, fc2], search, [sc])
    ref_numbers = sorted(r[0] for r in refs)

    assert formatted_numbers == ref_numbers


def test_format_context_and_ref_list_urls_match():
    """URL at each [N] position matches between format_context and build_source_reference_list."""
    fc = _make_fact_check(id="fc-1")
    es = _make_search(id="es-1", domain_key="especifico", domain="g1.globo.com")
    ge = _make_search(id="ge-1", domain_key="geral", domain="bbc.com")
    sc = _make_scrape(id="sc-1")

    search = {"especifico": [es], "geral": [ge]}
    refs = build_source_reference_list([fc], search, [sc])
    formatted = format_context([fc], search, [sc])

    # extract (number, url) pairs from formatted text
    # pattern: [N] ... URL: <url>
    lines = formatted.split("\n")
    formatted_urls: dict[int, str] = {}
    current_num = None
    for line in lines:
        num_match = re.match(r"\[(\d+)\]", line.strip())
        if num_match:
            current_num = int(num_match.group(1))
        url_match = re.search(r"URL:\s*(https?://\S+)", line)
        if url_match and current_num is not None:
            formatted_urls[current_num] = url_match.group(1)

    for num, title, url in refs:
        assert num in formatted_urls, f"[{num}] not found in formatted output"
        assert formatted_urls[num] == url, (
            f"URL mismatch at [{num}]: formatted={formatted_urls[num]}, ref_list={url}"
        )


# ===== end-to-end: adjudication prompt numbering matches build_source_reference_list =====


def test_adjudication_prompt_numbering_matches_ref_list():
    """[N] in the adjudication user prompt (what the LLM sees) must match build_source_reference_list."""
    from app.agentic_ai.prompts.adjudication_prompt import build_adjudication_prompt

    fc = _make_fact_check(id="fc-1")
    es = _make_search(id="es-1", domain_key="especifico", domain="g1.globo.com")
    ge = _make_search(id="ge-1", domain_key="geral", domain="bbc.com")
    sc = _make_scrape(id="sc-1")

    search = {"especifico": [es], "geral": [ge]}

    _, user_prompt = build_adjudication_prompt(
        formatted_data_sources="Test claim text",
        fact_check_results=[fc],
        search_results=search,
        scraped_pages=[sc],
    )
    refs = build_source_reference_list([fc], search, [sc])

    # extract [N] numbers from the prompt
    prompt_numbers = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", user_prompt)))
    ref_numbers = sorted(r[0] for r in refs)
    assert prompt_numbers == ref_numbers

    # extract URLs at each [N] from the prompt and verify they match ref_list
    lines = user_prompt.split("\n")
    prompt_urls: dict[int, str] = {}
    current_num = None
    for line in lines:
        num_match = re.match(r"\[(\d+)\]", line.strip())
        if num_match:
            current_num = int(num_match.group(1))
        url_match = re.search(r"URL:\s*(https?://\S+)", line)
        if url_match and current_num is not None:
            prompt_urls[current_num] = url_match.group(1)

    for num, title, url in refs:
        assert num in prompt_urls, f"[{num}] not in adjudication prompt"
        assert prompt_urls[num] == url, (
            f"URL mismatch at [{num}]: prompt={prompt_urls[num]}, ref_list={url}"
        )
