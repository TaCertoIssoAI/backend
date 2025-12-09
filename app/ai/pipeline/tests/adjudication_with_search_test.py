# -*- coding: utf-8 -*-
"""
Tests for the adjudication with Google Search pipeline step.

These tests make REAL calls to the Google GenAI API with Search grounding.

IMPORTANT: Set GOOGLE_API_KEY in your environment before running.

Run with:
    pytest app/ai/pipeline/tests/adjudication_with_search_test.py -v -s

The -s flag shows stdout so you can see the LLM responses for debugging.
"""

import pytest
from app.models import (
    ExtractedClaim,
    ClaimSource,
    FactCheckResult,
    DataSource,
    DataSourceWithExtractedClaims,
)
from app.ai.pipeline.adjudication_with_search import (
    adjudicate_claims_with_search,
    adjudicate_claims_with_search_async,
)


# ===== HELPER FUNCTIONS =====

def create_source_with_claims(claims: list[ExtractedClaim], source_id: str = "test-source") -> DataSourceWithExtractedClaims:
    """Helper to create DataSourceWithExtractedClaims for testing."""
    data_source = DataSource(
        id=source_id,
        source_type="original_text",
        original_text="Test message for fact-checking",
        metadata={},
    )

    return DataSourceWithExtractedClaims(
        data_source=data_source,
        extracted_claims=claims
    )


def print_result(result: FactCheckResult, test_name: str):
    """Print fact-check result for debugging."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)

    for ds_result in result.results:
        print(f"\nData Source: {ds_result.data_source_id}")
        for verdict in ds_result.claim_verdicts:
            print(f"\n  Claim: {verdict.claim_text}")
            print(f"  Verdict: {verdict.verdict}")
            print(f"  Justification: {verdict.justification[:200]}...")

    if result.overall_summary:
        print(f"\nOverall Summary: {result.overall_summary}")

    print("\n" + "=" * 80)


def validate_result(result: FactCheckResult):
    """Validate that result has correct structure."""
    print("\n[TEST DEBUG] Validating result structure...")

    assert isinstance(result, FactCheckResult), f"Expected FactCheckResult, got {type(result)}"
    assert isinstance(result.results, list), f"Expected list for results, got {type(result.results)}"
    assert len(result.results) > 0, f"Expected at least one result, got {len(result.results)}"

    print(f"[TEST DEBUG] Result has {len(result.results)} data source result(s)")

    for idx, ds_result in enumerate(result.results, 1):
        print(f"[TEST DEBUG] Validating data source result {idx}...")
        print(f"  - data_source_id: {ds_result.data_source_id}")
        print(f"  - source_type: {ds_result.source_type}")
        print(f"  - Number of verdicts: {len(ds_result.claim_verdicts)}")

        assert ds_result.data_source_id, f"Result {idx} missing data_source_id"
        assert ds_result.source_type, f"Result {idx} missing source_type"
        assert isinstance(ds_result.claim_verdicts, list), f"Result {idx} verdicts should be list, got {type(ds_result.claim_verdicts)}"

        for v_idx, verdict in enumerate(ds_result.claim_verdicts, 1):
            print(f"  - Verdict {v_idx}:")
            print(f"      claim_id: {verdict.claim_id}")
            print(f"      verdict: {verdict.verdict}")
            print(f"      claim_text: {verdict.claim_text[:60]}...")

            assert verdict.claim_id, f"Result {idx}, Verdict {v_idx} missing claim_id"
            assert verdict.claim_text, f"Result {idx}, Verdict {v_idx} missing claim_text"

            valid_verdicts = [
                "Verdadeiro",
                "Falso",
                "Fora de Contexto",
                "Fontes insuficientes para verificar"
            ]
            assert verdict.verdict in valid_verdicts, \
                f"Result {idx}, Verdict {v_idx}: Invalid verdict '{verdict.verdict}'. Must be one of: {valid_verdicts}"

            assert verdict.justification, f"Result {idx}, Verdict {v_idx} missing justification"

    print("[TEST DEBUG] Validation passed!")


# ===== BASIC TESTS =====

def test_single_claim_true():
    """Test adjudication with search for a single true claim."""
    claim = ExtractedClaim(
        id="claim-test-1",
        text="A Terra orbita ao redor do Sol",
        source=ClaimSource(source_type="original_text", source_id="msg-test-1"),
        entities=["Terra", "Sol"]
    )

    source_with_claims = create_source_with_claims([claim], source_id="msg-test-1")
    result = adjudicate_claims_with_search([source_with_claims])

    print_result(result, "Single Claim - True")
    validate_result(result)

    # Should have one verdict
    print(f"\n[TEST DEBUG] Checking verdict count...")
    print(f"  Expected: 1 verdict")
    print(f"  Got: {len(result.results[0].claim_verdicts)} verdicts")
    assert len(result.results[0].claim_verdicts) == 1, \
        f"Expected 1 verdict, got {len(result.results[0].claim_verdicts)}"

    verdict = result.results[0].claim_verdicts[0]

    # Should be classified as Verdadeiro (this is a well-known scientific fact)
    print(f"\n[TEST DEBUG] Checking verdict value...")
    print(f"  Claim: {verdict.claim_text}")
    print(f"  Expected: Verdadeiro")
    print(f"  Got: {verdict.verdict}")
    print(f"  Justification: {verdict.justification[:150]}...")

    assert verdict.verdict == "Verdadeiro", \
        f"Expected verdict 'Verdadeiro' for well-known fact, but got '{verdict.verdict}'. " \
        f"Justification: {verdict.justification[:200]}"

    assert len(verdict.justification) > 50, \
        f"Justification should be detailed (>50 chars), got {len(verdict.justification)} chars"


def test_single_claim_false():
    """Test adjudication with search for a single false claim."""
    claim = ExtractedClaim(
        id="claim-test-2",
        text="A Terra é plana",
        source=ClaimSource(source_type="original_text", source_id="msg-test-2"),
        entities=["Terra"]
    )

    source_with_claims = create_source_with_claims([claim], source_id="msg-test-2")
    result = adjudicate_claims_with_search([source_with_claims])

    print_result(result, "Single Claim - False")
    validate_result(result)

    # Should have one verdict
    assert len(result.results[0].claim_verdicts) == 1
    verdict = result.results[0].claim_verdicts[0]

    # Should be classified as Falso
    assert verdict.verdict == "Falso", f"Expected Falso, got {verdict.verdict}"
    assert len(verdict.justification) > 50, "Justification should be detailed"


def test_multiple_claims():
    """Test adjudication with search for multiple claims."""
    claims = [
        ExtractedClaim(
            id="claim-test-3a",
            text="A água ferve a 100°C ao nível do mar",
            source=ClaimSource(source_type="original_text", source_id="msg-test-3"),
            entities=["água", "100°C"]
        ),
        ExtractedClaim(
            id="claim-test-3b",
            text="A velocidade da luz é aproximadamente 300.000 km/s",
            source=ClaimSource(source_type="original_text", source_id="msg-test-3"),
            entities=["luz", "300.000 km/s"]
        )
    ]

    source_with_claims = create_source_with_claims(claims, source_id="msg-test-3")
    result = adjudicate_claims_with_search([source_with_claims])

    print_result(result, "Multiple Claims - Scientific Facts")
    validate_result(result)

    # Should have verdicts for both claims
    assert len(result.results[0].claim_verdicts) == 2

    # Both should be Verdadeiro
    for verdict in result.results[0].claim_verdicts:
        assert verdict.verdict == "Verdadeiro", f"Scientific facts should be Verdadeiro, got {verdict.verdict}"


def test_recent_event():
    """Test adjudication with search for a recent event (uses Google Search)."""
    claim = ExtractedClaim(
        id="claim-test-4",
        text="Portugal venceu a Eurocopa de 2024",
        source=ClaimSource(source_type="original_text", source_id="msg-test-4"),
        entities=["Portugal", "Eurocopa", "2024"]
    )

    source_with_claims = create_source_with_claims([claim], source_id="msg-test-4")
    result = adjudicate_claims_with_search([source_with_claims])

    print_result(result, "Recent Event - Euro 2024")
    validate_result(result)

    # Should have one verdict
    verdict = result.results[0].claim_verdicts[0]

    # This should be Falso (Spain won Euro 2024)
    # But we accept any verdict as long as it's justified
    assert verdict.verdict in ["Verdadeiro", "Falso", "Fora de Contexto"], "Should have a clear verdict"
    assert len(verdict.justification) > 50, "Should have detailed justification with search results"


def test_unverifiable_claim():
    """Test adjudication with search for an unverifiable claim."""
    claim = ExtractedClaim(
        id="claim-test-5",
        text="Existe um mineral secreto chamado Vibranium na Antártida",
        source=ClaimSource(source_type="original_text", source_id="msg-test-5"),
        entities=["Vibranium", "Antártida"]
    )

    source_with_claims = create_source_with_claims([claim], source_id="msg-test-5")
    result = adjudicate_claims_with_search([source_with_claims])

    print_result(result, "Unverifiable Claim - Fictional Element")
    validate_result(result)

    verdict = result.results[0].claim_verdicts[0]

    # Should likely be Falso or Fontes insuficientes
    assert verdict.verdict in ["Falso", "Fontes insuficientes para verificar"], \
        f"Fictional claim should be Falso or unverifiable, got {verdict.verdict}"


# ===== ASYNC TESTS =====

@pytest.mark.asyncio
async def test_async_adjudication():
    """Test async version of adjudication with search."""
    claim = ExtractedClaim(
        id="claim-test-6",
        text="O Brasil ganhou 5 Copas do Mundo de futebol",
        source=ClaimSource(source_type="original_text", source_id="msg-test-6"),
        entities=["Brasil", "Copa do Mundo", "5"]
    )

    source_with_claims = create_source_with_claims([claim], source_id="msg-test-6")
    result = await adjudicate_claims_with_search_async([source_with_claims])

    print_result(result, "Async - Brazil World Cups")
    validate_result(result)

    verdict = result.results[0].claim_verdicts[0]

    # Should be Verdadeiro (Brazil won in 1958, 1962, 1970, 1994, 2002)
    assert verdict.verdict == "Verdadeiro", f"Expected Verdadeiro, got {verdict.verdict}"


# ===== INTEGRATION TESTS =====

def test_full_pipeline_mixed_verdicts():
    """Test full pipeline with claims that should get different verdicts."""
    claims = [
        ExtractedClaim(
            id="claim-mixed-1",
            text="A Lua é feita de queijo",
            source=ClaimSource(source_type="original_text", source_id="msg-mixed"),
            entities=["Lua", "queijo"]
        ),
        ExtractedClaim(
            id="claim-mixed-2",
            text="A capital do Brasil é Brasília",
            source=ClaimSource(source_type="original_text", source_id="msg-mixed"),
            entities=["Brasil", "Brasília"]
        ),
        ExtractedClaim(
            id="claim-mixed-3",
            text="Vacinas contêm microchips de rastreamento",
            source=ClaimSource(source_type="original_text", source_id="msg-mixed"),
            entities=["vacinas", "microchips"]
        )
    ]

    source_with_claims = create_source_with_claims(claims, source_id="msg-mixed")
    result = adjudicate_claims_with_search([source_with_claims])

    print_result(result, "Mixed Verdicts - True, False, Conspiracy")
    validate_result(result)

    verdicts = result.results[0].claim_verdicts
    assert len(verdicts) == 3, "Should have verdicts for all 3 claims"

    # Check that we have at least some variety in verdicts
    verdict_types = [v.verdict for v in verdicts]
    assert "Falso" in verdict_types, "Should have at least one Falso verdict"
    assert "Verdadeiro" in verdict_types, "Should have at least one Verdadeiro verdict"

    # Overall summary should be present
    assert result.overall_summary, "Should have overall summary"
    assert len(result.overall_summary) > 20, "Summary should be substantive"


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    """Run tests manually with: python -m app.ai.pipeline.tests.adjudication_with_search_test"""
    pytest.main([__file__, "-v", "-s"])
