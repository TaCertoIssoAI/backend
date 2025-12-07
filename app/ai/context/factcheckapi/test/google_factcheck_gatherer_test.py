"""
Tests for Google Fact-Check API Evidence Gatherer

This test suite covers:
- Unit tests for response parsing
- Integration tests with real Google Fact-Check API calls
- Error handling and edge cases
- Composition with other gatherers
"""

import pytest
import os

pytest_plugins = ('pytest_asyncio',)

from app.ai.context.factcheckapi import GoogleFactCheckGatherer
from app.models import ExtractedClaim, ClaimSource


# ===== UNIT TESTS FOR PARSING =====

def test_parse_response_with_valid_data():
    """should parse valid google api response into citations"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")

    mock_response = {
        "claims": [
            {
                "text": "Vaccines cause autism",
                "claimant": "Unknown",
                "claimReview": [
                    {
                        "publisher": {"name": "FactCheck.org"},
                        "url": "https://factcheck.org/article1",
                        "title": "Vaccines Do Not Cause Autism",
                        "textualRating": "False",
                        "reviewDate": "2024-01-15"
                    }
                ]
            }
        ]
    }

    citations = gatherer._parse_response(mock_response)

    assert len(citations) == 1
    assert citations[0].url == "https://factcheck.org/article1"
    assert citations[0].title == "Vaccines Do Not Cause Autism"
    assert citations[0].publisher == "FactCheck.org"
    assert citations[0].rating == "Falso"  # "False" mapped to "Falso"
    assert citations[0].rating_comment is None  # no comment in this rating
    assert citations[0].date == "2024-01-15"
    assert citations[0].source == "google_fact_checking_api"


def test_parse_response_with_multiple_reviews():
    """should handle multiple fact-check reviews for same claim"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")

    mock_response = {
        "claims": [
            {
                "text": "Climate change is a hoax",
                "claimReview": [
                    {
                        "publisher": {"name": "FactCheck.org"},
                        "url": "https://factcheck.org/climate1",
                        "title": "Climate Change Is Real",
                        "textualRating": "False"
                    },
                    {
                        "publisher": {"name": "Snopes"},
                        "url": "https://snopes.com/climate1",
                        "title": "Climate Change Science Is Valid",
                        "textualRating": "False"
                    }
                ]
            }
        ]
    }

    citations = gatherer._parse_response(mock_response)

    assert len(citations) == 2
    assert citations[0].publisher == "FactCheck.org"
    assert citations[1].publisher == "Snopes"


def test_parse_response_empty():
    """should handle empty response gracefully"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")

    empty_response = {}
    citations = gatherer._parse_response(empty_response)
    assert citations == []

    no_claims_response = {"claims": []}
    citations = gatherer._parse_response(no_claims_response)
    assert citations == []


def test_parse_rating_with_comment():
    """should split textualRating into rating and rating_comment"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")

    mock_response = {
        "claims": [
            {
                "text": "5G causes cancer",
                "claimReview": [
                    {
                        "publisher": {"name": "Snopes"},
                        "url": "https://snopes.com/5g-cancer",
                        "title": "Does 5G Cause Cancer?",
                        "textualRating": "False. Multiple scientific studies have found no evidence linking 5G to cancer"
                    }
                ]
            }
        ]
    }

    citations = gatherer._parse_response(mock_response)

    assert len(citations) == 1
    citation = citations[0]

    # check that rating was split correctly and mapped to Portuguese
    assert citation.rating == "Falso"  # "False" mapped to "Falso"
    assert citation.rating_comment == "Multiple scientific studies have found no evidence linking 5G to cancer"
    assert citation.source == "google_fact_checking_api"


def test_parse_rating_variations():
    """should handle different rating formats correctly"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")

    # test case 1: rating without comment
    claim_data = {"text": "Test claim"}
    review_simple = {
        "url": "https://example.com/1",
        "title": "Test",
        "publisher": {"name": "Test Publisher"},
        "textualRating": "True"
    }
    citation = gatherer._parse_claim_review(claim_data, review_simple)
    assert citation.rating == "Verdadeiro"  # "True" mapped to "Verdadeiro"
    assert citation.rating_comment is None

    # test case 2: rating with comment
    review_with_comment = {
        "url": "https://example.com/2",
        "title": "Test",
        "publisher": {"name": "Test Publisher"},
        "textualRating": "Misleading. The claim lacks important context"
    }
    citation = gatherer._parse_claim_review(claim_data, review_with_comment)
    assert citation.rating == "Fora de Contexto"  # "Misleading" mapped to "Fora de Contexto"
    assert citation.rating_comment == "The claim lacks important context"

    # test case 3: rating with multiple periods in comment
    review_multiple_periods = {
        "url": "https://example.com/3",
        "title": "Test",
        "publisher": {"name": "Test Publisher"},
        "textualRating": "False. This is incorrect. Multiple sources confirm this."
    }
    citation = gatherer._parse_claim_review(claim_data, review_multiple_periods)
    assert citation.rating == "Falso"  # "False" mapped to "Falso"
    # should keep everything after first period
    assert citation.rating_comment == "This is incorrect. Multiple sources confirm this."

    # test case 4: empty rating
    review_empty = {
        "url": "https://example.com/4",
        "title": "Test",
        "publisher": {"name": "Test Publisher"},
        "textualRating": ""
    }
    citation = gatherer._parse_claim_review(claim_data, review_empty)
    assert citation.rating is None
    assert citation.rating_comment is None


def test_parse_claim_review_missing_required_fields():
    """should skip reviews with missing url or title"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")

    claim_data = {"text": "Some claim"}

    # missing URL
    review_no_url = {
        "title": "Fact-check title",
        "publisher": {"name": "Publisher"}
    }
    citation = gatherer._parse_claim_review(claim_data, review_no_url)
    assert citation is None

    # missing title
    review_no_title = {
        "url": "https://example.com",
        "publisher": {"name": "Publisher"}
    }
    citation = gatherer._parse_claim_review(claim_data, review_no_title)
    assert citation is None


def test_source_name():
    """should return correct source identifier"""
    gatherer = GoogleFactCheckGatherer(api_key="test_key")
    assert gatherer.source_name == "google_fact_checking_api"


# ===== INTEGRATION TESTS WITH REAL API =====
# these tests make REAL network calls to Google Fact-Check API

@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_gather_real_claim_vaccines():
    """should search google fact-check api for vaccine claim"""
    gatherer = GoogleFactCheckGatherer(max_results=5)

    claim = ExtractedClaim(
        id="claim-test-001",
        text="Vaccines cause autism",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-001"
        ),
        entities=["vaccines", "autism"]
    )

    citations = await gatherer.gather(claim)

    print(f"\n{'=' * 80}")
    print(f"TEST: Google Fact-Check API - Vaccine Claim")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")
    print(f"Citations found: {len(citations)}")

    # validate results
    assert isinstance(citations, list)
    # note: google api might return 0 results for some queries
    # so we just check structure, not count

    for i, citation in enumerate(citations[:3], 1):
        print(f"\n--- Citation {i} ---")
        print(f"Title: {citation.title}")
        print(f"Publisher: {citation.publisher}")
        print(f"URL: {citation.url}")
        print(f"Rating: {citation.rating}")
        print(f"Rating Comment: {citation.rating_comment}")
        print(f"Date: {citation.date}")
        print(f"Source: {citation.source}")

        # validate structure
        assert citation.url != ""
        assert citation.title != ""
        assert citation.publisher != ""
        assert citation.source == "google_fact_checking_api"
        # rating, rating_comment, and date are optional

        # if rating exists, verify it was mapped to portuguese
        if citation.rating:
            print(f"✓ Rating mapped to Portuguese: {citation.rating}")
            assert isinstance(citation.rating, str)
            # rating must be one of the portuguese verdict types
            assert citation.rating in ["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]
            # rating_comment should be None or a non-empty string
            if citation.rating_comment:
                assert isinstance(citation.rating_comment, str)
                assert len(citation.rating_comment) > 0
        else:
            print("⚠ No rating available for this citation")

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_gather_real_claim_climate():
    """should search google fact-check api for climate claim"""
    gatherer = GoogleFactCheckGatherer(max_results=3)

    claim = ExtractedClaim(
        id="claim-test-002",
        text="Climate change is a hoax",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-002"
        ),
        entities=["climate change"]
    )

    citations = await gatherer.gather(claim)

    print(f"\n{'=' * 80}")
    print(f"TEST: Google Fact-Check API - Climate Claim")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")
    print(f"Citations found: {len(citations)}")

    # validate citations list
    assert isinstance(citations, list)
    assert len(citations) <= 3  # should respect max_results

    # validate rating mapping for each citation
    for i, citation in enumerate(citations, 1):
        print(f"  Citation {i}: {citation.title[:60]}...")
        print(f"    Publisher: {citation.publisher}")
        print(f"    Rating: {citation.rating}")
        assert citation.source == "google_fact_checking_api"
        if citation.rating:
            print(f"    ✓ Rating mapped to Portuguese: {citation.rating}")
            assert citation.rating in ["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]
        else:
            print(f"    ⚠ No rating available")

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_gather_portuguese_claim():
    """should handle portuguese language claims"""
    gatherer = GoogleFactCheckGatherer(max_results=5)

    claim = ExtractedClaim(
        id="claim-test-003",
        text="A Terra é plana",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-003"
        ),
        entities=["Terra plana"]
    )

    citations = await gatherer.gather(claim)

    print(f"\n{'=' * 80}")
    print(f"TEST: Google Fact-Check API - Portuguese Claim")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")
    print(f"Citations found: {len(citations)}")

    assert isinstance(citations, list)

    # validate rating mapping for each citation
    for i, citation in enumerate(citations, 1):
        print(f"  Citation {i}: {citation.title[:60]}...")
        print(f"    Publisher: {citation.publisher}")
        print(f"    Rating: {citation.rating}")
        assert citation.source == "google_fact_checking_api"
        if citation.rating:
            print(f"    ✓ Rating mapped to Portuguese: {citation.rating}")
            assert citation.rating in ["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]
        else:
            print(f"    ⚠ No rating available")

    print(f"{'=' * 80}\n")


# ===== INTEGRATION WITH EVIDENCE RETRIEVAL PIPELINE =====

@pytest.mark.asyncio
async def test_compose_with_other_gatherers():
    """should work alongside other evidence gatherers"""
    from app.ai.pipeline.evidence_retrieval import gather_evidence_async
    from app.models import EvidenceRetrievalInput

    claim = ExtractedClaim(
        id="claim-compose-001",
        text="The moon landing was faked",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-compose-001"
        )
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    # use google fact-check gatherer
    google_gatherer = GoogleFactCheckGatherer(max_results=3)

    result = await gather_evidence_async(
        retrieval_input,
        gatherers=[google_gatherer]
    )

    # validate result
    assert claim.id in result.claim_evidence_map
    enriched = result.claim_evidence_map[claim.id]

    # all citations should be from google with proper rating mapping
    print(f"\n{'=' * 80}")
    print(f"TEST: Compose Google Gatherer with Pipeline")
    print(f"{'=' * 80}")
    print(f"Claim: {enriched.text}")
    print(f"Citations from Google: {len(enriched.citations)}")

    for i, citation in enumerate(enriched.citations, 1):
        print(f"  Citation {i}: {citation.title[:60]}...")
        print(f"    Rating: {citation.rating}")
        assert citation.source == "google_fact_checking_api"
        if citation.rating:
            print(f"    ✓ Rating mapped to Portuguese: {citation.rating}")
            assert citation.rating in ["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]
        else:
            print(f"    ⚠ No rating available")

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_combine_google_and_web_search():
    """should combine google fact-check with web search results"""
    from app.ai.pipeline.evidence_retrieval import (
        gather_evidence_async,
        WebSearchGatherer
    )
    from app.models import EvidenceRetrievalInput

    claim = ExtractedClaim(
        id="claim-multi-001",
        text="Drinking lemon water helps weight loss",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-multi-001"
        )
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    # use both gatherers
    google_gatherer = GoogleFactCheckGatherer(max_results=3)
    web_gatherer = WebSearchGatherer(max_results=3)

    result = await gather_evidence_async(
        retrieval_input,
        gatherers=[google_gatherer, web_gatherer]
    )

    enriched = result.claim_evidence_map[claim.id]

    # should have citations from both sources
    sources = {cit.source for cit in enriched.citations}

    print(f"\n{'=' * 80}")
    print(f"TEST: Combine Google + Web Search")
    print(f"{'=' * 80}")
    print(f"Claim: {enriched.text}")
    print(f"Total citations: {len(enriched.citations)}")
    print(f"Sources used: {sources}")

    # count citations by source and validate google ratings
    google_count = 0
    web_count = 0

    print(f"\nCitation details:")
    for i, cit in enumerate(enriched.citations, 1):
        if cit.source == "google_fact_checking_api":
            google_count += 1
            print(f"  {i}. [Google] {cit.title[:50]}...")
            print(f"     Rating: {cit.rating}")
            # validate rating mapping for google citations
            if cit.rating:
                print(f"     ✓ Rating mapped to Portuguese: {cit.rating}")
                assert cit.rating in ["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]
            else:
                print(f"     ⚠ No rating available")
        elif cit.source == "apify_web_search":
            web_count += 1
            print(f"  {i}. [Web Search] {cit.title[:50]}...")

    print(f"\nSummary:")
    print(f"  Google Fact-Check: {google_count}")
    print(f"  Web Search: {web_count}")
    print(f"{'=' * 80}\n")
