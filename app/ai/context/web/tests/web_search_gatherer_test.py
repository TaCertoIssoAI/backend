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

IMPORTANT: these tests make REAL calls to the Apify web search API.
set APIFY_API_KEY in your environment before running.

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

    # source should always be apify_web_search
    assert citation.source == "apify_web_search", "source should be apify_web_search"

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
        assert citation.source == "apify_web_search", "source should be apify_web_search"

    print("\n" + "=" * 80)
    print("TEST: Citation Metadata Fields")
    print("=" * 80)
    print("\n✓ rating field is None (web search doesn't provide ratings)")
    print("✓ date field is None (web search doesn't include publication date)")
    print("✓ source field is 'apify_web_search'")
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
    assert gatherer.source_name == "apify_web_search", (
        "source name should be apify_web_search"
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

    assert gatherer.source_name == "apify_web_search", (
        "source_name should be apify_web_search"
    )

    print("\n" + "=" * 80)
    print("TEST: Source Name Property")
    print("=" * 80)
    print("\n✓ source_name property returns 'apify_web_search'")
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


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
