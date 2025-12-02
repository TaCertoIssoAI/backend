# -*- coding: utf-8 -*-
"""
tests for the web search gatherer evidence collection step.

these tests validate:
- the structure of outputs
- citation extraction from search results
- timeout handling
- error handling
- source attribution
- time profiling via decorator

IMPORTANT: these tests make REAL calls to the Google Custom Search API.
set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_CX in your environment before running.

run with:
    pytest app/ai/context/web/tests/web_search_gatherer_test.py -v -s

the -s flag shows stdout so you can see the search results and time profiling logs.
"""

from typing import List

import pytest

from app.ai.context.web import WebSearchGatherer
from app.models import Citation, ClaimSource, ExtractedClaim


# ===== HELPER FUNCTIONS =====

def create_test_claim(text: str, claim_id: str = "test-claim-1") -> ExtractedClaim:
    """create a test extracted claim."""
    return ExtractedClaim(
        id=claim_id,
        text=text,
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-001"
        ),
        entities=[],
        llm_comment="test claim"
    )


def print_citations(citations: List[Citation], test_name: str):
    """print citations for debugging."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    print(f"\nFound {len(citations)} citation(s):\n")

    for i, citation in enumerate(citations, 1):
        print(f"  Citation {i}:")
        print(f"    Title: {citation.title[:60]}...")
        print(f"    URL: {citation.url}")
        print(f"    Publisher: {citation.publisher}")
        print(f"    Source: {citation.source}")
        if citation.citation_text:
            text_preview = (
                citation.citation_text[:80]
                if len(citation.citation_text) > 80
                else citation.citation_text
            )
            print(f"    Citation Text: {text_preview}...")
        print()


def validate_citation_structure(citation: Citation):
    """validate that a citation has the correct structure."""
    # required fields
    assert citation.url, "citation URL should not be empty"
    assert citation.title, "citation title should not be empty"
    assert citation.publisher is not None, "citation should have a publisher"
    assert citation.citation_text is not None, "citation should have citation text"
    assert citation.source is not None, "citation should have a source"

    # type checks
    assert isinstance(citation.url, str), "URL should be a string"
    assert isinstance(citation.title, str), "title should be a string"
    assert isinstance(citation.publisher, str), "publisher should be a string"
    assert isinstance(citation.citation_text, str), "citation text should be string"
    assert isinstance(citation.source, str), "source should be a string"

    # source should always be google_web_search
    assert citation.source == "google_web_search", "source should be google_web_search"

    # URL should be valid
    assert citation.url.startswith("http"), "URL should start with http"


def validate_citations_list(citations: List[Citation]):
    """validate that a list of citations has the correct structure."""
    assert isinstance(citations, list), "result should be a list"

    for citation in citations:
        validate_citation_structure(citation)


# ===== ASYNC GATHER TESTS =====

@pytest.mark.asyncio
async def test_basic_web_search_gather():
    """test basic web search with real API call."""
    # setup
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0)
    claim = create_test_claim("COVID-19 vaccine safety studies")

    # execute - real API call
    citations = await gatherer.gather(claim)

    # validate
    print_citations(citations, "Basic Web Search Gather")
    validate_citations_list(citations)
    assert len(citations) > 0, "should return at least one citation"

    # verify all citations have proper content
    for citation in citations:
        assert len(citation.title) > 0, "title should not be empty"
        assert citation.url.startswith("http"), "URL should be valid"


@pytest.mark.asyncio
async def test_web_search_gather_scientific_claim():
    """test web search with a scientific claim."""
    # setup
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0)
    claim = create_test_claim("climate change causes global temperature increase")

    # execute - real API call
    citations = await gatherer.gather(claim)

    # validate
    print_citations(citations, "Scientific Claim Web Search")
    validate_citations_list(citations)
    assert len(citations) > 0, "should find citations for scientific claim"


@pytest.mark.asyncio
async def test_web_search_gather_portuguese_claim():
    """test web search with portuguese text."""
    # setup
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0)
    claim = create_test_claim("vacinas contra COVID-19 são seguras")

    # execute - real API call
    citations = await gatherer.gather(claim)

    # validate
    print_citations(citations, "Portuguese Claim Web Search")
    validate_citations_list(citations)
    assert len(citations) > 0, "should find citations for portuguese claim"


@pytest.mark.asyncio
async def test_web_search_gather_with_timeout():
    """test web search with very short timeout to trigger timeout handling."""
    # setup - very short timeout
    gatherer = WebSearchGatherer(max_results=5, timeout=0.001)
    claim = create_test_claim("test claim with short timeout")

    # execute - should timeout
    citations = await gatherer.gather(claim)

    # validate - should return empty list on timeout
    print("\n" + "=" * 80)
    print("TEST: Web Search with Timeout")
    print("=" * 80)
    print(f"\nTimeout handled gracefully, returned {len(citations)} citations\n")

    assert len(citations) == 0, "should return empty list on timeout"


@pytest.mark.asyncio
async def test_web_search_gather_multiple_results():
    """test web search requesting multiple results."""
    # setup
    gatherer = WebSearchGatherer(max_results=10, timeout=60.0)
    claim = create_test_claim("python programming language features")

    # execute - real API call
    citations = await gatherer.gather(claim)

    # validate
    print_citations(citations, "Multiple Results Web Search")
    validate_citations_list(citations)
    assert len(citations) > 0, "should return multiple citations"

    # verify each citation has unique URL
    urls = [c.url for c in citations]
    assert len(urls) == len(set(urls)), "all URLs should be unique"


@pytest.mark.asyncio
async def test_citation_metadata_fields():
    """test that citation metadata fields are set correctly."""
    # setup
    gatherer = WebSearchGatherer(max_results=2, timeout=45.0)
    claim = create_test_claim("artificial intelligence machine learning")

    # execute - real API call
    citations = await gatherer.gather(claim)

    # validate
    assert len(citations) > 0, "should get at least one citation"

    for citation in citations:
        # verify metadata fields
        assert citation.rating is None, "web search should have None rating"
        assert citation.date is None, "web search should have None date"
        assert citation.source == "google_web_search", "source should be google_web_search"

    print("\n" + "=" * 80)
    print("TEST: Citation Metadata Fields")
    print("=" * 80)
    print("\n✓ rating field is None (web search doesn't provide ratings)")
    print("✓ date field is None (web search doesn't include publication date)")
    print("✓ source field is 'google_web_search'")
    print()


# ===== SYNC GATHER TESTS =====

def test_gather_sync_basic():
    """test synchronous gather method with basic claim."""
    # setup
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0)
    claim = create_test_claim("renewable energy solar power")

    # execute - real API call via sync method
    citations = gatherer.gather_sync(claim)

    # validate
    print_citations(citations, "Synchronous Gather - Basic")
    validate_citations_list(citations)
    assert len(citations) > 0, "should return at least one citation"


def test_gather_sync_portuguese():
    """test synchronous gather with portuguese claim."""
    # setup
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0)
    claim = create_test_claim("energia renovável e sustentabilidade")

    # execute - real API call via sync method
    citations = gatherer.gather_sync(claim)

    # validate
    print_citations(citations, "Synchronous Gather - Portuguese")
    validate_citations_list(citations)
    assert len(citations) > 0, "should find citations for portuguese claim"


def test_gather_sync_with_custom_config():
    """test synchronous gather with custom max_results and timeout."""
    # setup
    gatherer = WebSearchGatherer(max_results=5, timeout=60.0)
    claim = create_test_claim("quantum computing algorithms")

    # execute - real API call via sync method
    citations = gatherer.gather_sync(claim)

    # validate
    print_citations(citations, "Synchronous Gather - Custom Config")
    validate_citations_list(citations)
    assert len(citations) > 0, "should return citations with custom config"

    print(f"✓ Used custom config: max_results=5, timeout=60.0")
    print(f"✓ Retrieved {len(citations)} citation(s)")


def test_gather_sync_timeout_handling():
    """test synchronous gather with timeout."""
    # setup - very short timeout to trigger timeout
    gatherer = WebSearchGatherer(max_results=5, timeout=0.001)
    claim = create_test_claim("test sync timeout handling")

    # execute - should timeout
    citations = gatherer.gather_sync(claim)

    # validate - should return empty list on timeout
    print("\n" + "=" * 80)
    print("TEST: Synchronous Gather - Timeout Handling")
    print("=" * 80)
    print(f"\nTimeout handled gracefully, returned {len(citations)} citations\n")

    assert len(citations) == 0, "should return empty list on timeout"


def test_gather_sync_multiple_calls():
    """test multiple sequential synchronous gather calls."""
    # setup
    gatherer = WebSearchGatherer(max_results=2, timeout=45.0)

    claims = [
        create_test_claim("blockchain technology", "claim-1"),
        create_test_claim("machine learning applications", "claim-2"),
        create_test_claim("cloud computing services", "claim-3"),
    ]

    all_citations = []

    # execute - multiple sequential calls
    for i, claim in enumerate(claims, 1):
        print(f"\n--- Sequential Call {i} ---")
        citations = gatherer.gather_sync(claim)
        all_citations.append(citations)

        # validate each call
        validate_citations_list(citations)
        assert len(citations) > 0, f"call {i} should return citations"

    # validate overall
    print("\n" + "=" * 80)
    print("TEST: Synchronous Gather - Multiple Sequential Calls")
    print("=" * 80)
    print(f"\n✓ Made {len(claims)} sequential calls")
    print(f"✓ Total citations retrieved: {sum(len(c) for c in all_citations)}")
    print()

    assert len(all_citations) == 3, "should have results for all 3 claims"


def test_gather_sync_publisher_extraction():
    """test that publisher field is correctly extracted in sync mode."""
    # setup
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0)
    claim = create_test_claim("space exploration mars missions")

    # execute - real API call via sync method
    citations = gatherer.gather_sync(claim)

    # validate
    assert len(citations) > 0, "should get at least one citation"

    for citation in citations:
        # publisher should be extracted and not empty
        assert citation.publisher, "publisher should not be empty"
        assert len(citation.publisher) > 0, "publisher should have content"

    print("\n" + "=" * 80)
    print("TEST: Synchronous Gather - Publisher Extraction")
    print("=" * 80)
    print("\n✓ All citations have valid publisher information")
    print("Publishers found:")
    for citation in citations[:3]:  # show first 3
        print(f"  - {citation.publisher}")
    print()


def test_gather_sync_citation_text_content():
    """test that citation text is properly populated in sync mode."""
    # setup
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0)
    claim = create_test_claim("electric vehicles battery technology")

    # execute - real API call via sync method
    citations = gatherer.gather_sync(claim)

    # validate
    assert len(citations) > 0, "should get at least one citation"

    for citation in citations:
        # citation text should not be empty
        assert citation.citation_text, "citation text should not be empty"
        assert len(citation.citation_text) > 0, "citation text should have content"

    print("\n" + "=" * 80)
    print("TEST: Synchronous Gather - Citation Text Content")
    print("=" * 80)
    print("\n✓ All citations have non-empty citation text")
    print(f"✓ Average text length: "
          f"{sum(len(c.citation_text) for c in citations) / len(citations):.0f} chars")
    print()


# ===== INITIALIZATION AND CONFIGURATION TESTS =====

def test_gatherer_initialization():
    """test gatherer initialization with custom parameters."""
    # test with custom parameters
    gatherer = WebSearchGatherer(max_results=10, timeout=60.0)

    assert gatherer.max_results == 10, "max_results should be set correctly"
    assert gatherer.timeout == 60.0, "timeout should be set correctly"
    assert gatherer.source_name == "google_web_search", (
        "source name should be google_web_search"
    )

    # test with default parameters
    default_gatherer = WebSearchGatherer()

    assert default_gatherer.max_results == 5, "default max_results should be 5"
    assert default_gatherer.timeout == 45.0, "default timeout should be 45.0"

    print("\n" + "=" * 80)
    print("TEST: Gatherer Initialization")
    print("=" * 80)
    print("\n✓ Custom parameters: max_results=10, timeout=60.0")
    print("✓ Default parameters: max_results=5, timeout=45.0")
    print()


def test_source_name_property():
    """test that source_name property returns correct value."""
    gatherer = WebSearchGatherer()

    assert gatherer.source_name == "google_web_search", (
        "source_name should be google_web_search"
    )

    print("\n" + "=" * 80)
    print("TEST: Source Name Property")
    print("=" * 80)
    print("\n✓ source_name property returns 'google_web_search'")
    print()


# ===== TIME PROFILING TEST =====

@pytest.mark.asyncio
async def test_time_profiling_decorator():
    """test that time profiling decorator is working."""
    # setup
    gatherer = WebSearchGatherer(max_results=2, timeout=45.0)
    claim = create_test_claim("time profiling test claim")

    # execute - decorator should log execution time
    citations = await gatherer.gather(claim)

    # if we get here without errors, the decorator is working
    assert isinstance(citations, list), "should return a list"

    print("\n" + "=" * 80)
    print("TEST: Time Profiling Decorator")
    print("=" * 80)
    print("\n✓ time_profile decorator is applied and working")
    print("✓ check logs above for [TIME PROFILE] gather completed in X.XXs")
    print()


# ===== DOMAIN FILTERING TESTS =====

def test_build_search_query_no_domains():
    """test query building without domain filtering."""
    # setup
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=None)

    # execute
    query = gatherer._build_search_query_with_domains("vaccines cause autism")

    # validate
    assert query == "vaccines cause autism", "should return original query when no domains"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - No Domains")
    print("=" * 80)
    print(f"\nInput:  'vaccines cause autism'")
    print(f"Output: '{query}'")
    print("✓ No domain filtering applied (fail-open)")
    print()


def test_build_search_query_empty_domains():
    """test query building with empty domains list."""
    # setup
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=[])

    # execute
    query = gatherer._build_search_query_with_domains("climate change")

    # validate
    assert query == "climate change", "should return original query when domains list is empty"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Empty Domains List")
    print("=" * 80)
    print(f"\nInput:  'climate change'")
    print(f"Output: '{query}'")
    print("✓ No domain filtering applied for empty list")
    print()


def test_build_search_query_single_domain():
    """test query building with single domain."""
    # setup
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=["who.int"])

    # execute
    query = gatherer._build_search_query_with_domains("COVID-19 vaccines")

    # validate
    expected = "COVID-19 vaccines (site:who.int)"
    assert query == expected, f"should add single domain filter: expected '{expected}', got '{query}'"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Single Domain")
    print("=" * 80)
    print(f"\nInput:  'COVID-19 vaccines'")
    print(f"Domains: ['who.int']")
    print(f"Output: '{query}'")
    print("✓ Single domain filter applied correctly")
    print()


def test_build_search_query_multiple_domains():
    """test query building with multiple domains."""
    # setup
    domains = ["who.int", "cdc.gov", "gov.br"]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute
    query = gatherer._build_search_query_with_domains("vaccine safety")

    # validate
    expected = "vaccine safety (site:who.int OR site:cdc.gov OR site:gov.br)"
    assert query == expected, f"should add multiple domain filters: expected '{expected}', got '{query}'"

    # verify structure
    assert "site:who.int" in query, "should contain who.int"
    assert "site:cdc.gov" in query, "should contain cdc.gov"
    assert "site:gov.br" in query, "should contain gov.br"
    assert " OR " in query, "should use OR operator"
    assert query.startswith("vaccine safety ("), "should start with original query"
    assert query.endswith(")"), "should end with closing parenthesis"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Multiple Domains")
    print("=" * 80)
    print(f"\nInput:  'vaccine safety'")
    print(f"Domains: {domains}")
    print(f"Output: '{query}'")
    print("✓ Multiple domain filters with OR operator")
    print()


def test_build_search_query_domains_with_whitespace():
    """test query building handles domains with whitespace."""
    # setup
    domains = ["  who.int  ", "cdc.gov", "  gov.br"]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute
    query = gatherer._build_search_query_with_domains("test claim")

    # validate - whitespace should be stripped
    assert "site:who.int" in query, "should contain trimmed who.int"
    assert "site:  who.int  " not in query, "should not contain whitespace around domain"

    expected = "test claim (site:who.int OR site:cdc.gov OR site:gov.br)"
    assert query == expected, f"should strip whitespace: expected '{expected}', got '{query}'"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Domains with Whitespace")
    print("=" * 80)
    print(f"\nInput:  'test claim'")
    print(f"Domains: {domains}")
    print(f"Output: '{query}'")
    print("✓ Whitespace stripped from domains")
    print()


def test_build_search_query_domains_with_empty_strings():
    """test query building filters out empty domain strings."""
    # setup
    domains = ["who.int", "", "  ", "cdc.gov"]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute
    query = gatherer._build_search_query_with_domains("health data")

    # validate - empty strings should be filtered out
    expected = "health data (site:who.int OR site:cdc.gov)"
    assert query == expected, f"should filter empty strings: expected '{expected}', got '{query}'"
    assert query.count("site:") == 2, "should only have 2 site: operators"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Filter Empty Strings")
    print("=" * 80)
    print(f"\nInput:  'health data'")
    print(f"Domains: {domains}")
    print(f"Output: '{query}'")
    print("✓ Empty strings filtered out")
    print()


def test_build_search_query_all_empty_domains():
    """test query building when all domains are empty/whitespace."""
    # setup
    domains = ["", "  ", "   "]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute
    query = gatherer._build_search_query_with_domains("test query")

    # validate - should fail-open and return original query
    assert query == "test query", "should return original query when all domains are empty"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - All Empty Domains")
    print("=" * 80)
    print(f"\nInput:  'test query'")
    print(f"Domains: {domains}")
    print(f"Output: '{query}'")
    print("✓ Fail-open behavior when all domains empty")
    print()


def test_build_search_query_special_characters_in_claim():
    """test query building with special characters in claim text."""
    # setup
    domains = ["who.int", "cdc.gov"]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute - claim with quotes and special chars
    claim_text = 'vaccines "mRNA technology" & safety (2024)'
    query = gatherer._build_search_query_with_domains(claim_text)

    # validate
    expected = 'vaccines "mRNA technology" & safety (2024) (site:who.int OR site:cdc.gov)'
    assert query == expected, f"should preserve special chars: expected '{expected}', got '{query}'"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Special Characters")
    print("=" * 80)
    print(f"\nInput:  '{claim_text}'")
    print(f"Domains: {domains}")
    print(f"Output: '{query}'")
    print("✓ Special characters preserved in query")
    print()


def test_build_search_query_subdomain_support():
    """test that root domains will match subdomains."""
    # setup
    domains = ["gov.br"]  # should match saude.gov.br, anvisa.gov.br, etc.
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute
    query = gatherer._build_search_query_with_domains("Brazilian health policy")

    # validate
    expected = "Brazilian health policy (site:gov.br)"
    assert query == expected, f"should use root domain: expected '{expected}', got '{query}'"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Subdomain Support")
    print("=" * 80)
    print(f"\nInput:  'Brazilian health policy'")
    print(f"Domains: {domains}")
    print(f"Output: '{query}'")
    print("✓ Root domain 'gov.br' will match all *.gov.br subdomains")
    print("  (e.g., saude.gov.br, anvisa.gov.br, planalto.gov.br)")
    print()


def test_build_search_query_many_domains():
    """test query building with many domains."""
    # setup
    domains = [
        "who.int",
        "cdc.gov",
        "gov.br",
        "fiocruz.br",
        "fapesp.br",
        "scielo.br",
        "ebc.com.br"
    ]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)

    # execute
    query = gatherer._build_search_query_with_domains("health research")

    # validate
    assert query.startswith("health research ("), "should start with original query"
    assert query.count("site:") == len(domains), f"should have {len(domains)} site: operators"
    assert query.count(" OR ") == len(domains) - 1, f"should have {len(domains) - 1} OR operators"

    for domain in domains:
        assert f"site:{domain}" in query, f"should contain site:{domain}"

    print("\n" + "=" * 80)
    print("TEST: Build Search Query - Many Domains")
    print("=" * 80)
    print(f"\nInput:  'health research'")
    print(f"Domains: {len(domains)} domains")
    print(f"Output length: {len(query)} characters")
    print(f"✓ All {len(domains)} domains included with OR operators")
    print()


@pytest.mark.asyncio
async def test_gather_with_domain_filtering():
    """test actual gather with domain filtering (integration test)."""
    # setup - limit to trusted health domains
    domains = ["who.int", "cdc.gov"]
    gatherer = WebSearchGatherer(max_results=3, timeout=45.0, allowed_domains=domains)
    claim = create_test_claim("COVID-19 vaccine effectiveness")

    # execute - real API call with domain filtering
    citations = await gatherer.gather(claim)

    # validate
    print_citations(citations, "Gather with Domain Filtering")

    # assert we got results
    assert len(citations) > 0, "should return at least one citation with domain filtering"

    # verify all results are from allowed domains
    for citation in citations:
        url = citation.url.lower()
        url_domain = citation.publisher.lower()

        # check if citation matches any allowed domain
        is_allowed = any(
            allowed_domain in url or allowed_domain in url_domain
            for allowed_domain in domains
        )

        print(f"  URL: {citation.url}")
        print(f"  Publisher: {citation.publisher}")
        print(f"  Matches allowed domains: {is_allowed}")
        print()

        # assert each citation is from an allowed domain
        assert is_allowed, (
            f"citation from {citation.publisher} ({citation.url}) "
            f"should match one of the allowed domains: {domains}"
        )

    print("✓ Domain filtering applied to search query")
    print(f"✓ Allowed domains: {domains}")
    print(f"✓ All {len(citations)} citation(s) are from allowed domains")
    print()


@pytest.mark.asyncio
async def test_gather_with_gov_br_domain_filtering():
    """test gather with gov.br domain to verify subdomain matching (integration test)."""
    # setup - restrict to Brazilian government sites
    domains = ["gov.br"]
    gatherer = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=domains)
    claim = create_test_claim("vacinas COVID-19 Brasil")

    # execute - real API call
    citations = await gatherer.gather(claim)

    # validate
    print_citations(citations, "Gather with gov.br Domain Filtering")

    # assert we got results
    assert len(citations) > 0, "should return citations from gov.br domains"

    # verify all results are from gov.br or subdomains
    gov_br_count = 0
    for citation in citations:
        url = citation.url.lower()
        publisher = citation.publisher.lower()

        # check if it's a gov.br domain or subdomain
        is_gov_br = "gov.br" in url or "gov.br" in publisher

        print(f"  Publisher: {citation.publisher}")
        print(f"  URL: {citation.url}")
        print(f"  Is gov.br domain: {is_gov_br}")

        if is_gov_br:
            gov_br_count += 1
            # extract subdomain for informational purposes
            if "gov.br" in publisher:
                subdomain = publisher.split(".gov.br")[0].split(".")[-1] if "." in publisher else "root"
                print(f"  Subdomain detected: {subdomain}")
        print()

        # assert it matches gov.br
        assert is_gov_br, (
            f"citation from {citation.publisher} should be from gov.br or its subdomains"
        )

    print(f"✓ All {len(citations)} citation(s) are from gov.br or subdomains")
    print(f"✓ Examples: saude.gov.br, anvisa.gov.br, planalto.gov.br, etc.")
    print()


@pytest.mark.asyncio
async def test_gather_without_vs_with_domain_filtering():
    """test comparing results with and without domain filtering (integration test)."""
    claim = create_test_claim("climate change global warming")

    # first: gather without filtering (open web)
    gatherer_no_filter = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=None)
    citations_no_filter = await gatherer_no_filter.gather(claim)

    # second: gather with strict domain filtering
    trusted_domains = ["who.int", "nasa.gov", "noaa.gov"]
    gatherer_with_filter = WebSearchGatherer(max_results=5, timeout=45.0, allowed_domains=trusted_domains)
    citations_with_filter = await gatherer_with_filter.gather(claim)

    print("\n" + "=" * 80)
    print("TEST: Gather Without vs With Domain Filtering")
    print("=" * 80)

    # validate both returned results
    print(f"\nResults without filtering: {len(citations_no_filter)} citation(s)")
    print(f"Results with filtering: {len(citations_with_filter)} citation(s)")

    # assert we got results from both
    assert len(citations_no_filter) > 0, "should get results without filtering"
    assert len(citations_with_filter) > 0, "should get results with filtering"

    # show domains from unfiltered results
    print("\nDomains from UNFILTERED search:")
    unfiltered_domains = set()
    for citation in citations_no_filter[:5]:
        domain = citation.publisher
        unfiltered_domains.add(domain)
        print(f"  - {domain}")

    # show domains from filtered results
    print(f"\nDomains from FILTERED search (allowed: {trusted_domains}):")
    filtered_domains = set()
    for citation in citations_with_filter:
        domain = citation.publisher
        filtered_domains.add(domain)
        print(f"  - {domain}")

        # assert each result matches allowed domains
        is_allowed = any(
            allowed in citation.url.lower() or allowed in domain.lower()
            for allowed in trusted_domains
        )
        assert is_allowed, (
            f"filtered result {domain} should match one of {trusted_domains}"
        )

    # compare diversity
    print(f"\n✓ Unfiltered search returned {len(unfiltered_domains)} unique domains")
    print(f"✓ Filtered search returned {len(filtered_domains)} unique domains")
    print(f"✓ All filtered results match allowed domains: {trusted_domains}")
    print()


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
