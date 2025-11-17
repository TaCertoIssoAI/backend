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
    assert citations[0].rating == "False"
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
        print(f"Date: {citation.date}")
        print(f"Source: {citation.source}")

        # validate structure
        assert citation.url != ""
        assert citation.title != ""
        assert citation.publisher != ""
        assert citation.source == "google_fact_checking_api"
        # rating and date are optional

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

    # all citations should be from google
    for citation in enriched.citations:
        assert citation.source == "google_fact_checking_api"

    print(f"\n{'=' * 80}")
    print(f"TEST: Compose Google Gatherer with Pipeline")
    print(f"{'=' * 80}")
    print(f"Claim: {enriched.text}")
    print(f"Citations from Google: {len(enriched.citations)}")
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

    # count citations by source
    google_count = sum(
        1 for cit in enriched.citations
        if cit.source == "google_fact_checking_api"
    )
    web_count = sum(
        1 for cit in enriched.citations
        if cit.source == "apify_web_search"
    )

    print(f"Google Fact-Check: {google_count}")
    print(f"Web Search: {web_count}")
    print(f"{'=' * 80}\n")
