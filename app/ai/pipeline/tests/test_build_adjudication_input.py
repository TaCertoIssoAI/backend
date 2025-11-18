# -*- coding: utf-8 -*-
"""
tests for build_adjudication_input function.

these tests validate that:
- claim IDs are correctly preserved throughout the pipeline
- data source tracking works properly via ClaimSource.source_id
- enriched claims are correctly grouped by their original data sources
- the function handles edge cases (empty claims, missing evidence, etc.)

run with:
    pytest app/ai/pipeline/tests/test_build_adjudication_input.py -v -s

the -s flag shows stdout so you can see detailed output for debugging.
"""

import pytest
from typing import List, Dict

from app.models import (
    DataSource,
    ClaimExtractionOutput,
    ExtractedClaim,
    ClaimSource,
    EnrichedClaim,
    Citation,
    EvidenceRetrievalResult,
    AdjudicationInput,
    DataSourceWithClaims,
)
from app.ai.main_pipeline import build_adjudication_input


# ===== HELPER FUNCTIONS =====

def create_test_data_source(ds_id: str, source_type: str, text: str) -> DataSource:
    """create a test data source"""
    return DataSource(
        id=ds_id,
        source_type=source_type,
        original_text=text,
        metadata={},
        locale="pt-BR",
        timestamp="2024-11-18T00:00:00Z"
    )


def create_test_extracted_claim(
    claim_id: str,
    claim_text: str,
    source_id: str,
    source_type: str
) -> ExtractedClaim:
    """create a test extracted claim"""
    return ExtractedClaim(
        id=claim_id,
        text=claim_text,
        source=ClaimSource(
            source_type=source_type,
            source_id=source_id
        ),
        entities=["test", "entity"],
        llm_comment="test comment"
    )


def create_test_enriched_claim(
    claim_id: str,
    claim_text: str,
    source_id: str,
    source_type: str,
    citations: List[Citation]
) -> EnrichedClaim:
    """create a test enriched claim with citations"""
    return EnrichedClaim(
        id=claim_id,
        text=claim_text,
        source=ClaimSource(
            source_type=source_type,
            source_id=source_id
        ),
        entities=["test", "entity"],
        llm_comment="test comment",
        citations=citations
    )


def create_test_citation(url: str, title: str) -> Citation:
    """create a test citation"""
    return Citation(
        url=url,
        title=title,
        publisher="Test Publisher",
        citation_text="Test citation text",
        source="google_fact_checking_api",
        rating="Falso",
        date="2024-11-18"
    )


def print_adjudication_input(adj_input: AdjudicationInput, test_name: str):
    """print adjudication input for debugging"""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    print(f"\n📦 ADJUDICATION INPUT:")
    print(f"  Total data sources: {len(adj_input.sources_with_claims)}")

    for i, ds_with_claims in enumerate(adj_input.sources_with_claims, 1):
        ds = ds_with_claims.data_source
        claims = ds_with_claims.enriched_claims
        print(f"\n  {i}. DataSource: {ds.id} ({ds.source_type})")
        print(f"     Text: {ds.original_text[:60]}...")
        print(f"     Enriched claims: {len(claims)}")

        for j, claim in enumerate(claims, 1):
            print(f"       {j}) Claim ID: {claim.id}")
            print(f"          Text: {claim.text[:60]}...")
            print(f"          Source: {claim.source.source_type} ({claim.source.source_id})")
            print(f"          Citations: {len(claim.citations)}")


# ===== TESTS =====

def test_claim_ids_are_preserved():
    """
    test that claim IDs are correctly preserved from extraction through enrichment.

    validates:
    - extracted claim ID matches enriched claim ID
    - claim ID is used as key in evidence map
    - same claim ID appears in final adjudication input
    """
    # setup: create data source
    ds = create_test_data_source("ds-001", "original_text", "Test message")

    # setup: create extracted claims
    claim_1 = create_test_extracted_claim(
        "claim-123",
        "Test claim 1",
        "ds-001",
        "original_text"
    )
    claim_2 = create_test_extracted_claim(
        "claim-456",
        "Test claim 2",
        "ds-001",
        "original_text"
    )

    claim_output = ClaimExtractionOutput(
        data_source=ds,
        claims=[claim_1, claim_2]
    )

    # setup: create enriched claims with same IDs
    enriched_1 = create_test_enriched_claim(
        "claim-123",
        "Test claim 1",
        "ds-001",
        "original_text",
        [create_test_citation("https://example.com/1", "Test 1")]
    )
    enriched_2 = create_test_enriched_claim(
        "claim-456",
        "Test claim 2",
        "ds-001",
        "original_text",
        [create_test_citation("https://example.com/2", "Test 2")]
    )

    evidence_result = EvidenceRetrievalResult(
        claim_evidence_map={
            "claim-123": enriched_1,
            "claim-456": enriched_2
        }
    )

    # execute: build adjudication input
    adj_input = build_adjudication_input([claim_output], evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Claim IDs Preserved")

    # assert: claim IDs are preserved
    assert len(adj_input.sources_with_claims) == 1

    ds_with_claims = adj_input.sources_with_claims[0]
    assert len(ds_with_claims.enriched_claims) == 2

    claim_ids = {claim.id for claim in ds_with_claims.enriched_claims}
    assert "claim-123" in claim_ids
    assert "claim-456" in claim_ids

    print("\n✅ PASSED: All claim IDs preserved correctly")


def test_source_tracking_preserved():
    """
    test that source tracking (source_id and source_type) is preserved.

    validates:
    - ClaimSource.source_id matches DataSource.id
    - ClaimSource.source_type matches DataSource.source_type
    - enriched claims can be traced back to their original data source
    """
    # setup: create multiple data sources
    ds1 = create_test_data_source("msg-001", "original_text", "Original message")
    ds2 = create_test_data_source("link-002", "link_context", "Link content")

    # setup: create claims from different sources
    claim_from_msg = create_test_extracted_claim(
        "claim-msg-1",
        "Claim from original message",
        "msg-001",
        "original_text"
    )
    claim_from_link = create_test_extracted_claim(
        "claim-link-1",
        "Claim from link",
        "link-002",
        "link_context"
    )

    claim_outputs = [
        ClaimExtractionOutput(data_source=ds1, claims=[claim_from_msg]),
        ClaimExtractionOutput(data_source=ds2, claims=[claim_from_link])
    ]

    # setup: create enriched claims
    enriched_msg = create_test_enriched_claim(
        "claim-msg-1",
        "Claim from original message",
        "msg-001",
        "original_text",
        []
    )
    enriched_link = create_test_enriched_claim(
        "claim-link-1",
        "Claim from link",
        "link-002",
        "link_context",
        []
    )

    evidence_result = EvidenceRetrievalResult(
        claim_evidence_map={
            "claim-msg-1": enriched_msg,
            "claim-link-1": enriched_link
        }
    )

    # execute
    adj_input = build_adjudication_input(claim_outputs, evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Source Tracking Preserved")

    # assert: source tracking is correct
    assert len(adj_input.sources_with_claims) == 2

    # find DataSourceWithClaims for msg-001
    msg_sources = [
        ds for ds in adj_input.sources_with_claims
        if ds.data_source.id == "msg-001"
    ]
    assert len(msg_sources) == 1
    assert len(msg_sources[0].enriched_claims) == 1

    msg_claim = msg_sources[0].enriched_claims[0]
    assert msg_claim.id == "claim-msg-1"
    assert msg_claim.source.source_id == "msg-001"
    assert msg_claim.source.source_type == "original_text"

    # find DataSourceWithClaims for link-002
    link_sources = [
        ds for ds in adj_input.sources_with_claims
        if ds.data_source.id == "link-002"
    ]
    assert len(link_sources) == 1
    assert len(link_sources[0].enriched_claims) == 1

    link_claim = link_sources[0].enriched_claims[0]
    assert link_claim.id == "claim-link-1"
    assert link_claim.source.source_id == "link-002"
    assert link_claim.source.source_type == "link_context"

    print("\n✅ PASSED: Source tracking preserved correctly")


def test_grouping_by_data_source():
    """
    test that claims are correctly grouped by their original data source.

    validates:
    - multiple claims from same source are grouped together
    - claims from different sources are kept separate
    - all claims for a source are included in the group
    """
    # setup: one data source with multiple claims
    ds = create_test_data_source("ds-multi", "original_text", "Message with multiple claims")

    claim_1 = create_test_extracted_claim("claim-1", "First claim", "ds-multi", "original_text")
    claim_2 = create_test_extracted_claim("claim-2", "Second claim", "ds-multi", "original_text")
    claim_3 = create_test_extracted_claim("claim-3", "Third claim", "ds-multi", "original_text")

    claim_output = ClaimExtractionOutput(
        data_source=ds,
        claims=[claim_1, claim_2, claim_3]
    )

    # setup: enriched claims
    enriched_1 = create_test_enriched_claim("claim-1", "First claim", "ds-multi", "original_text", [])
    enriched_2 = create_test_enriched_claim("claim-2", "Second claim", "ds-multi", "original_text", [])
    enriched_3 = create_test_enriched_claim("claim-3", "Third claim", "ds-multi", "original_text", [])

    evidence_result = EvidenceRetrievalResult(
        claim_evidence_map={
            "claim-1": enriched_1,
            "claim-2": enriched_2,
            "claim-3": enriched_3
        }
    )

    # execute
    adj_input = build_adjudication_input([claim_output], evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Grouping By Data Source")

    # assert: all claims grouped under same data source
    assert len(adj_input.sources_with_claims) == 1

    ds_with_claims = adj_input.sources_with_claims[0]
    assert ds_with_claims.data_source.id == "ds-multi"
    assert len(ds_with_claims.enriched_claims) == 3

    claim_ids = {claim.id for claim in ds_with_claims.enriched_claims}
    assert claim_ids == {"claim-1", "claim-2", "claim-3"}

    print("\n✅ PASSED: Claims correctly grouped by data source")


def test_empty_claims_handled():
    """
    test edge case: data source with no claims extracted.

    validates:
    - data sources with zero claims are included in output
    - empty enriched_claims list is created
    - no errors occur
    """
    # setup: data source with no claims
    ds = create_test_data_source("ds-empty", "original_text", "Message with no claims")

    claim_output = ClaimExtractionOutput(
        data_source=ds,
        claims=[]  # no claims extracted
    )

    evidence_result = EvidenceRetrievalResult(claim_evidence_map={})

    # execute
    adj_input = build_adjudication_input([claim_output], evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Empty Claims Handled")

    # assert: data source included with empty claims
    assert len(adj_input.sources_with_claims) == 1

    ds_with_claims = adj_input.sources_with_claims[0]
    assert ds_with_claims.data_source.id == "ds-empty"
    assert len(ds_with_claims.enriched_claims) == 0

    print("\n✅ PASSED: Empty claims handled correctly")


def test_missing_evidence_for_claim():
    """
    test edge case: claim exists but no evidence was found.

    validates:
    - claims without evidence in the map are skipped
    - only claims with evidence appear in adjudication input
    - no errors occur for missing claims
    """
    # setup: data source with claims
    ds = create_test_data_source("ds-001", "original_text", "Test message")

    claim_1 = create_test_extracted_claim("claim-has-evidence", "Claim with evidence", "ds-001", "original_text")
    claim_2 = create_test_extracted_claim("claim-no-evidence", "Claim without evidence", "ds-001", "original_text")

    claim_output = ClaimExtractionOutput(
        data_source=ds,
        claims=[claim_1, claim_2]
    )

    # setup: only one claim has evidence
    enriched_1 = create_test_enriched_claim(
        "claim-has-evidence",
        "Claim with evidence",
        "ds-001",
        "original_text",
        [create_test_citation("https://example.com", "Test")]
    )
    # claim-no-evidence is NOT in the evidence map

    evidence_result = EvidenceRetrievalResult(
        claim_evidence_map={
            "claim-has-evidence": enriched_1
            # "claim-no-evidence" is missing
        }
    )

    # execute
    adj_input = build_adjudication_input([claim_output], evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Missing Evidence For Claim")

    # assert: only claim with evidence is included
    assert len(adj_input.sources_with_claims) == 1

    ds_with_claims = adj_input.sources_with_claims[0]
    assert len(ds_with_claims.enriched_claims) == 1
    assert ds_with_claims.enriched_claims[0].id == "claim-has-evidence"

    print("\n✅ PASSED: Missing evidence handled correctly (claim skipped)")


def test_citations_preserved():
    """
    test that citations are preserved in enriched claims.

    validates:
    - citations from evidence gathering are present in adjudication input
    - citation count matches
    - citation details are intact
    """
    # setup
    ds = create_test_data_source("ds-001", "original_text", "Test message")

    claim = create_test_extracted_claim("claim-1", "Test claim", "ds-001", "original_text")
    claim_output = ClaimExtractionOutput(data_source=ds, claims=[claim])

    # setup: enriched claim with multiple citations
    citations = [
        create_test_citation("https://example.com/1", "Source 1"),
        create_test_citation("https://example.com/2", "Source 2"),
        create_test_citation("https://example.com/3", "Source 3"),
    ]

    enriched = create_test_enriched_claim(
        "claim-1",
        "Test claim",
        "ds-001",
        "original_text",
        citations
    )

    evidence_result = EvidenceRetrievalResult(claim_evidence_map={"claim-1": enriched})

    # execute
    adj_input = build_adjudication_input([claim_output], evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Citations Preserved")

    # assert: citations preserved
    ds_with_claims = adj_input.sources_with_claims[0]
    claim_with_citations = ds_with_claims.enriched_claims[0]

    assert len(claim_with_citations.citations) == 3
    assert claim_with_citations.citations[0].url == "https://example.com/1"
    assert claim_with_citations.citations[1].title == "Source 2"
    assert claim_with_citations.citations[2].publisher == "Test Publisher"

    print("\n✅ PASSED: Citations preserved correctly")


def test_complex_pipeline_flow():
    """
    integration test simulating full pipeline flow with multiple sources and claims.

    validates:
    - original_text source with 2 claims
    - link_context source with 1 claim
    - all IDs, sources, and citations preserved correctly
    """
    # setup: multiple data sources
    ds_original = create_test_data_source(
        "msg-001",
        "original_text",
        "Original message mentioning vaccine and climate change"
    )
    ds_link = create_test_data_source(
        "link-001",
        "link_context",
        "Link content about vaccine safety"
    )

    # setup: claims from different sources
    claim_vaccine_msg = create_test_extracted_claim(
        "claim-vac-msg",
        "Vaccine X causes infertility",
        "msg-001",
        "original_text"
    )
    claim_climate_msg = create_test_extracted_claim(
        "claim-climate-msg",
        "Global warming is accelerating",
        "msg-001",
        "original_text"
    )
    claim_vaccine_link = create_test_extracted_claim(
        "claim-vac-link",
        "Vaccine X was tested on 50000 participants",
        "link-001",
        "link_context"
    )

    claim_outputs = [
        ClaimExtractionOutput(data_source=ds_original, claims=[claim_vaccine_msg, claim_climate_msg]),
        ClaimExtractionOutput(data_source=ds_link, claims=[claim_vaccine_link])
    ]

    # setup: enriched claims with citations
    enriched_vac_msg = create_test_enriched_claim(
        "claim-vac-msg",
        "Vaccine X causes infertility",
        "msg-001",
        "original_text",
        [
            create_test_citation("https://health.gov/vaccine-study", "Vaccine Study"),
            create_test_citation("https://who.int/vaccines", "WHO Report")
        ]
    )
    enriched_climate = create_test_enriched_claim(
        "claim-climate-msg",
        "Global warming is accelerating",
        "msg-001",
        "original_text",
        [create_test_citation("https://ipcc.ch/report", "IPCC Report")]
    )
    enriched_vac_link = create_test_enriched_claim(
        "claim-vac-link",
        "Vaccine X was tested on 50000 participants",
        "link-001",
        "link_context",
        [create_test_citation("https://clinicaltrials.gov/study", "Clinical Trial")]
    )

    evidence_result = EvidenceRetrievalResult(
        claim_evidence_map={
            "claim-vac-msg": enriched_vac_msg,
            "claim-climate-msg": enriched_climate,
            "claim-vac-link": enriched_vac_link
        }
    )

    # execute
    adj_input = build_adjudication_input(claim_outputs, evidence_result)

    # print for debugging
    print_adjudication_input(adj_input, "Complex Pipeline Flow")

    # assert: correct structure
    assert len(adj_input.sources_with_claims) == 2

    # find original_text source
    original_sources = [
        ds for ds in adj_input.sources_with_claims
        if ds.data_source.source_type == "original_text"
    ]
    assert len(original_sources) == 1
    assert len(original_sources[0].enriched_claims) == 2

    # find link_context source
    link_sources = [
        ds for ds in adj_input.sources_with_claims
        if ds.data_source.source_type == "link_context"
    ]
    assert len(link_sources) == 1
    assert len(link_sources[0].enriched_claims) == 1

    # verify specific claims
    original_claim_ids = {claim.id for claim in original_sources[0].enriched_claims}
    assert "claim-vac-msg" in original_claim_ids
    assert "claim-climate-msg" in original_claim_ids

    link_claim = link_sources[0].enriched_claims[0]
    assert link_claim.id == "claim-vac-link"
    assert len(link_claim.citations) == 1

    print("\n✅ PASSED: Complex pipeline flow handled correctly")


# ===== RUN ALL TESTS =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
