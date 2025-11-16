import pytest

# configure pytest to automatically handle async tests
pytest_plugins = ('pytest_asyncio',)

from app.ai.pipeline.evidence_retrieval import (
    WebSearchGatherer,
    gather_evidence_async,
    gather_and_filter_evidence,
    deduplicate_citations,
    filter_low_quality_citations,
)
from app.models import (
    EvidenceRetrievalInput,
    ExtractedClaim,
    ClaimSource,
    Citation,
)


# ===== UNIT TESTS FOR HELPER FUNCTIONS =====

def test_deduplicate_citations_removes_duplicates():
    """should remove citations with duplicate URLs"""
    citations = [
        Citation(
            url="https://example.com/article1",
            title="Article 1",
            publisher="Example",
            citation_text="Some text",
            source="apify_web_search"
        ),
        Citation(
            url="https://example.com/article2",
            title="Article 2",
            publisher="Example",
            citation_text="Other text",
            source="apify_web_search"
        ),
        Citation(
            url="https://example.com/article1",  # duplicate
            title="Article 1 Again",
            publisher="Example",
            citation_text="Different text",
            source="apify_web_search"
        ),
    ]

    result = deduplicate_citations(citations)

    assert len(result) == 2
    assert result[0].url == "https://example.com/article1"
    assert result[1].url == "https://example.com/article2"


def test_deduplicate_citations_case_insensitive():
    """should treat URLs as case-insensitive when deduplicating"""
    citations = [
        Citation(
            url="https://Example.com/Article",
            title="Article 1",
            publisher="Example",
            citation_text="Text",
            source="apify_web_search"
        ),
        Citation(
            url="https://example.com/article",  # same URL, different case
            title="Article 2",
            publisher="Example",
            citation_text="Text",
            source="apify_web_search"
        ),
    ]

    result = deduplicate_citations(citations)

    assert len(result) == 1


def test_deduplicate_citations_empty_list():
    """should handle empty list"""
    result = deduplicate_citations([])
    assert result == []


def test_filter_low_quality_citations_removes_short_text():
    """should remove citations with very short citation text"""
    citations = [
        Citation(
            url="https://example.com/1",
            title="Good Article",
            publisher="Example",
            citation_text="This is a good citation with enough text content",
            source="apify_web_search"
        ),
        Citation(
            url="https://example.com/2",
            title="Bad Article",
            publisher="Example",
            citation_text="short",  # too short
            source="apify_web_search"
        ),
    ]

    result = filter_low_quality_citations(citations, min_text_length=10)

    assert len(result) == 1
    assert result[0].url == "https://example.com/1"


def test_filter_low_quality_citations_removes_missing_fields():
    """should remove citations with missing critical fields"""
    citations = [
        Citation(
            url="https://example.com/1",
            title="Good Article",
            publisher="Example",
            citation_text="Good content here",
            source="apify_web_search"
        ),
        Citation(
            url="",  # missing URL
            title="Bad Article",
            publisher="Example",
            citation_text="Some content",
            source="apify_web_search"
        ),
        Citation(
            url="https://example.com/3",
            title="",  # missing title
            publisher="Example",
            citation_text="Some content",
            source="apify_web_search"
        ),
    ]

    result = filter_low_quality_citations(citations)

    assert len(result) == 1
    assert result[0].url == "https://example.com/1"


def test_filter_low_quality_citations_empty_list():
    """should handle empty list"""
    result = filter_low_quality_citations([])
    assert result == []


# ===== INTEGRATION TESTS FOR WEB SEARCH GATHERER =====
# these tests make REAL network calls to the Apify web search API

@pytest.mark.asyncio
async def test_web_search_gatherer_real_claim():
    """should search the web for a real claim and return citations"""
    gatherer = WebSearchGatherer(max_results=3)

    claim = ExtractedClaim(
        id="claim-test-001",
        text="A vacina contra COVID-19 é segura para mulheres grávidas",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-001"
        ),
        entities=["COVID-19", "vacina", "mulheres grávidas"]
    )

    citations = await gatherer.gather(claim)

    print(f"\n{'=' * 80}")
    print(f"TEST: Web Search for COVID-19 Vaccine Safety Claim")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")
    print(f"Citations found: {len(citations)}")

    # validate structure
    assert isinstance(citations, list), "Should return a list"
    assert len(citations) > 0, "Should find at least one citation"
    assert len(citations) <= 3, "Should respect max_results limit"

    # validate each citation
    for i, citation in enumerate(citations, 1):
        print(f"\n--- Citation {i} ---")
        print(f"Title: {citation.title}")
        print(f"URL: {citation.url}")
        print(f"Publisher: {citation.publisher}")
        print(f"Source: {citation.source}")
        print(f"Text preview: {citation.citation_text[:150]}...")

        assert citation.url != "", "URL should not be empty"
        assert citation.title != "", "Title should not be empty"
        assert citation.source == "apify_web_search", "Source should be apify_web_search"
        assert citation.rating is None, "Web search shouldn't provide ratings"

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_web_search_gatherer_english_claim():
    """should handle English language claims"""
    gatherer = WebSearchGatherer(max_results=3)

    claim = ExtractedClaim(
        id="claim-test-002",
        text="Climate change is caused by human activities",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-002"
        ),
        entities=["climate change", "human activities"]
    )

    citations = await gatherer.gather(claim)

    print(f"\n{'=' * 80}")
    print(f"TEST: Web Search for Climate Change Claim (English)")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")
    print(f"Citations found: {len(citations)}")

    assert len(citations) > 0, "Should find citations for English claims"

    for i, citation in enumerate(citations, 1):
        print(f"\n--- Citation {i} ---")
        print(f"Title: {citation.title}")
        print(f"URL: {citation.url}")
        print(f"Publisher: {citation.publisher}")

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_web_search_gatherer_source_name():
    """should return correct source name"""
    gatherer = WebSearchGatherer(max_results=5)
    assert gatherer.source_name == "apify_web_search"


# ===== INTEGRATION TESTS FOR MAIN EVIDENCE RETRIEVAL =====

@pytest.mark.asyncio
async def test_gather_evidence_async_single_claim():
    """should gather evidence for a single claim"""
    claim = ExtractedClaim(
        id="claim-single-001",
        text="A vitamina D ajuda a prevenir gripes e resfriados",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-single-001"
        ),
        entities=["vitamina D", "gripe", "resfriado"]
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    result = await gather_evidence_async(retrieval_input)

    print(f"\n{'=' * 80}")
    print(f"TEST: Evidence Gathering for Single Claim")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")

    # validate result structure
    assert claim.id in result.claim_evidence_map
    enriched_claim = result.claim_evidence_map[claim.id]

    print(f"Citations gathered: {len(enriched_claim.citations)}")

    # enriched claim should preserve original fields
    assert enriched_claim.id == claim.id
    assert enriched_claim.text == claim.text
    assert enriched_claim.source == claim.source
    assert enriched_claim.entities == claim.entities

    # should have citations
    assert len(enriched_claim.citations) > 0

    for i, citation in enumerate(enriched_claim.citations[:3], 1):
        print(f"\n--- Citation {i} ---")
        print(f"Title: {citation.title}")
        print(f"URL: {citation.url}")

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_gather_evidence_async_multiple_claims():
    """should gather evidence for multiple claims"""
    claims = [
        ExtractedClaim(
            id="claim-multi-001",
            text="Beber água com limão em jejum emagrece",
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-multi-001"
            ),
            entities=["água com limão", "jejum", "emagrecer"]
        ),
        ExtractedClaim(
            id="claim-multi-002",
            text="O 5G causa câncer",
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-multi-001"
            ),
            entities=["5G", "câncer"]
        ),
    ]

    retrieval_input = EvidenceRetrievalInput(claims=claims)

    result = await gather_evidence_async(retrieval_input)

    print(f"\n{'=' * 80}")
    print(f"TEST: Evidence Gathering for Multiple Claims")
    print(f"{'=' * 80}")

    # should have evidence for both claims
    assert len(result.claim_evidence_map) == 2
    assert "claim-multi-001" in result.claim_evidence_map
    assert "claim-multi-002" in result.claim_evidence_map

    for claim in claims:
        enriched = result.claim_evidence_map[claim.id]
        print(f"\nClaim: {enriched.text}")
        print(f"Citations: {len(enriched.citations)}")

        # both claims should have citations
        assert len(enriched.citations) > 0

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_gather_evidence_async_empty_claims():
    """should handle empty claims list"""
    retrieval_input = EvidenceRetrievalInput(claims=[])

    result = await gather_evidence_async(retrieval_input)

    assert result.claim_evidence_map == {}


# ===== INTEGRATION TESTS FOR CONVENIENCE FUNCTION =====

@pytest.mark.asyncio
async def test_gather_and_filter_evidence_deduplicates():
    """should deduplicate and filter citations"""
    claim = ExtractedClaim(
        id="claim-filter-001",
        text="Tomar café diariamente faz bem para a saúde",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-filter-001"
        ),
        entities=["café", "saúde"]
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    # gather with filtering enabled
    result = await gather_and_filter_evidence(
        retrieval_input,
        deduplicate=True,
        filter_quality=True
    )

    print(f"\n{'=' * 80}")
    print(f"TEST: Gather and Filter Evidence")
    print(f"{'=' * 80}")
    print(f"Claim: {claim.text}")

    enriched = result.claim_evidence_map[claim.id]
    print(f"Filtered citations: {len(enriched.citations)}")

    # all citations should be high quality (no empty URLs or titles)
    for citation in enriched.citations:
        assert citation.url != ""
        assert citation.title != ""
        assert len(citation.citation_text) >= 10

    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_gather_and_filter_evidence_no_filters():
    """should work without filters"""
    claim = ExtractedClaim(
        id="claim-no-filter-001",
        text="Exercícios físicos melhoram a saúde mental",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-no-filter-001"
        ),
        entities=["exercícios físicos", "saúde mental"]
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    # gather without filters
    result = await gather_and_filter_evidence(
        retrieval_input,
        deduplicate=False,
        filter_quality=False
    )

    enriched = result.claim_evidence_map[claim.id]

    # should still have citations
    assert len(enriched.citations) > 0


# ===== TESTS FOR CUSTOM GATHERERS =====

@pytest.mark.asyncio
async def test_custom_gatherer_composition():
    """should support custom evidence gatherers"""

    # create a mock gatherer for testing
    class MockGatherer:
        @property
        def source_name(self) -> str:
            return "mock_source"

        async def gather(self, claim: ExtractedClaim):
            # return a fixed citation for testing
            return [
                Citation(
                    url="https://mock.com/article",
                    title="Mock Article",
                    publisher="Mock Publisher",
                    citation_text="This is a mock citation for testing purposes",
                    source="mock_source"
                )
            ]

    claim = ExtractedClaim(
        id="claim-custom-001",
        text="Test claim for custom gatherer",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-custom-001"
        )
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    # use custom gatherer
    result = await gather_evidence_async(
        retrieval_input,
        gatherers=[MockGatherer()]
    )

    enriched = result.claim_evidence_map[claim.id]

    # should have exactly one citation from mock gatherer
    assert len(enriched.citations) == 1
    assert enriched.citations[0].source == "mock_source"
    assert enriched.citations[0].url == "https://mock.com/article"


@pytest.mark.asyncio
async def test_multiple_gatherers_composition():
    """should combine citations from multiple gatherers"""

    class MockGatherer1:
        @property
        def source_name(self) -> str:
            return "mock_source_1"

        async def gather(self, claim: ExtractedClaim):
            return [
                Citation(
                    url="https://mock1.com/article",
                    title="Mock Article 1",
                    publisher="Mock Publisher 1",
                    citation_text="Citation from source 1",
                    source="mock_source_1"
                )
            ]

    class MockGatherer2:
        @property
        def source_name(self) -> str:
            return "mock_source_2"

        async def gather(self, claim: ExtractedClaim):
            return [
                Citation(
                    url="https://mock2.com/article",
                    title="Mock Article 2",
                    publisher="Mock Publisher 2",
                    citation_text="Citation from source 2",
                    source="mock_source_2"
                )
            ]

    claim = ExtractedClaim(
        id="claim-multi-gatherer-001",
        text="Test claim for multiple gatherers",
        source=ClaimSource(
            source_type="original_text",
            source_id="msg-multi-gatherer-001"
        )
    )

    retrieval_input = EvidenceRetrievalInput(claims=[claim])

    # use both gatherers
    result = await gather_evidence_async(
        retrieval_input,
        gatherers=[MockGatherer1(), MockGatherer2()]
    )

    enriched = result.claim_evidence_map[claim.id]

    # should have citations from both gatherers
    assert len(enriched.citations) == 2
    sources = {cit.source for cit in enriched.citations}
    assert "mock_source_1" in sources
    assert "mock_source_2" in sources
