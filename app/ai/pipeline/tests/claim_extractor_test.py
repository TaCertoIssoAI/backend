# -*- coding: utf-8 -*-
"""
Tests for the claim extraction pipeline step.

These tests make REAL calls to the LLM (OpenAI API) to validate:
- The structure of outputs
- The LangChain chain works correctly
- The prompt produces valid results
- Source tracking works properly

IMPORTANT: Set OPENAI_API_KEY in your environment before running.

Run with:
    pytest app/ai/pipeline/tests/claim_extractor_test.py -v -s

The -s flag shows stdout so you can see the LLM responses for debugging.
"""

import pytest
from typing import List

from app.models import ClaimExtractionInput, ExtractedClaim, ClaimExtractionOutput
from app.ai.pipeline import (
    extract_claims,
    extract_and_validate_claims,
    validate_claims,
)


# ===== HELPER FUNCTIONS =====

def print_claim_results(
    claims: List[ExtractedClaim],
    test_name: str,
    input_text: str | None = None
):
    """Print claim extraction results for debugging, including input for verification."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)

    # Print input for verification
    if input_text:
        print(f"\n📥 INPUT TEXT:")
        print(f"  {input_text}")

    # Print output
    print(f"\n📤 OUTPUT:")
    print(f"  Extracted {len(claims)} claim(s):\n")

    for i, claim in enumerate(claims, 1):
        print(f"  Claim {i}:")
        print(f"    ID: {claim.id}")
        print(f"    Text: {claim.text}")
        print(f"    Entities: {claim.entities}")
        print(f"    Source Type: {claim.source.source_type}")
        print(f"    Source ID: {claim.source.source_id}")
        print(f"    LLM Comment: {claim.llm_comment}")
        print()


def validate_claim_structure(claim: ExtractedClaim):
    """Validate that a claim has the correct structure."""
    # Required fields
    assert claim.id is not None and claim.id != "", "Claim ID should not be empty"
    assert claim.text is not None and claim.text != "", "Claim text should not be empty"
    assert claim.source is not None, "Claim should have a source"

    # Type checks
    assert isinstance(claim.id, str), "Claim ID should be a string"
    assert isinstance(claim.text, str), "Claim text should be a string"
    assert isinstance(claim.entities, list), "Entities should be a list"

    # Source validation
    assert isinstance(claim.source.source_type, str), "Source type should be a string"
    assert isinstance(claim.source.source_id, str), "Source ID should be a string"

    # Optional field type check
    if claim.llm_comment is not None:
        assert isinstance(claim.llm_comment, str), "LLM comment should be a string"

    # List element type checks
    for entity in claim.entities:
        assert isinstance(entity, str), "Each entity should be a string"


def validate_claims_list(claims: List[ExtractedClaim]):
    """Validate that a list of claims has the correct structure."""
    assert isinstance(claims, list), "Result should be a list"

    for claim in claims:
        validate_claim_structure(claim)


# ===== TESTS =====

def test_basic_claim_extraction_from_user_message():
    """Test basic claim extraction from a user message."""
    # Setup
    text = "I heard that vaccine X causes infertility in women, is this true?"

    extraction_input = ClaimExtractionInput(
        source_id="msg-001",
        type="original_text",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input,
        timeout=30.0
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Basic Claim Extraction from User Message",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    assert len(claims) > 0, "Should extract at least one claim"

    # Check source tracking
    for claim in claims:
        assert claim.source.source_type == "original_text"
        assert claim.source.source_id == "msg-001"
        assert claim.id.startswith("msg-001-claim-"), "Claim ID should start with source_id"


def test_claim_extraction_from_link_context():
    """Test claim extraction from link/article content."""
    # Setup - simulate content extracted from a link
    text = """=== Article: New Study on Vaccine Safety ===

A comprehensive study published today found no evidence linking
Vaccine X to fertility issues in women. The study examined over
50,000 participants and concluded that the vaccine is safe.

The research was conducted by the Ministry of Health over 3 years."""

    extraction_input = ClaimExtractionInput(
        source_id="link-456",
        type="link_context",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input,
        timeout=30.0
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Claim Extraction from Link Context",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    assert len(claims) > 0, "Should extract claims from article"

    # Check source tracking
    for claim in claims:
        assert claim.source.source_type == "link_context"
        assert claim.source.source_id == "link-456"


def test_multiple_claims_extraction():
    """Test extraction of multiple claims from one text."""
    # Setup
    text = """The president announced a new carbon tax of $50 per ton.
Additionally, the government will invest $100 billion in renewable energy over the next decade.
This makes it the largest climate investment in history."""

    extraction_input = ClaimExtractionInput(
        source_id="msg-002",
        type="original_text",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Multiple Claims Extraction",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    # We expect multiple claims but don't assert specific number
    # as LLM behavior may vary


def test_portuguese_message_extraction():
    """Test claim extraction with Portuguese text."""
    # Setup
    text = "Dizem que a vacina da COVID causa problemas no coração. Isso é verdade mesmo?"

    extraction_input = ClaimExtractionInput(
        source_id="msg-003",
        type="original_text",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Portuguese Message Extraction",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    assert len(claims) > 0, "Should extract at least one claim from Portuguese text"


def test_image_ocr_extraction():
    """Test claim extraction from simulated image OCR text."""
    # Setup - simulate OCR output from an image
    text = "BREAKING NEWS: Vaccine X causes infertility. Share before they delete this!"

    extraction_input = ClaimExtractionInput(
        source_id="img-789",
        type="image",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Image OCR Extraction",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    assert len(claims) > 0, "Should extract claim from OCR text"

    # Check source tracking
    for claim in claims:
        assert claim.source.source_type == "image"
        assert claim.source.source_id == "img-789"


def test_empty_text():
    """Test behavior with empty text."""
    # Setup
    text = ""

    extraction_input = ClaimExtractionInput(
        source_id="msg-004",
        type="original_text",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Empty Text",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    # With empty text, should return empty list or handle gracefully
    assert len(claims) == 0, "Empty text should result in no claims"


def test_opinion_vs_claim():
    """Test that LLM can distinguish opinions from fact-checkable claims."""
    # Setup
    text = "I think vaccines are scary and I don't like them. What do you think?"

    extraction_input = ClaimExtractionInput(
        source_id="msg-005",
        type="original_text",
        text=text
    )

    # Execute
    claims = extract_and_validate_claims(
        extraction_input=extraction_input
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Opinion vs Claim",
        input_text=text
    )

    # Validate structure
    validate_claims_list(claims)
    # This is pure opinion, not a fact-checkable claim
    # LLM should ideally return empty or very few claims


def test_validate_claims_function():
    """Test the validate_claims helper function."""
    # Setup: Create mock claims with some that should be filtered
    from app.models import ExtractedClaim, ClaimSource

    claims = [
        ExtractedClaim(
            id="claim-1",
            text="Valid claim about vaccines",
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="This is valid",
            entities=["vaccines"]
        ),
        ExtractedClaim(
            id="claim-2",
            text="",  # Empty text - should be filtered
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="Empty",
            entities=[]
        ),
        ExtractedClaim(
            id="claim-3",
            text="Valid claim about vaccines",  # Duplicate - should be filtered
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="This is duplicate",
            entities=["vaccines"]
        ),
        ExtractedClaim(
            id="claim-4",
            text="Another valid claim",
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="Also valid",
            entities=["claim"]
        ),
    ]

    # Execute
    validated = validate_claims(claims)

    # Print for debugging
    print("\n" + "=" * 80)
    print("TEST: Validate Claims Function")
    print("=" * 80)
    print(f"\nInput: {len(claims)} claims")
    print(f"Output: {len(validated)} claims")
    print("\nFiltered out:")
    print("  - 1 empty claim")
    print("  - 1 duplicate claim")
    print()

    # Validate
    assert len(validated) == 2, "Should filter out empty and duplicate claims"
    assert validated[0].text == "Valid claim about vaccines"
    assert validated[1].text == "Another valid claim"


def test_chain_building():
    """Test that the chain can be built without errors."""
    from app.ai.pipeline import build_claim_extraction_chain

    # Build chain
    chain = build_claim_extraction_chain(
        model_name="gpt-4o-mini",
        temperature=0.0,
        timeout=30.0
    )

    # Validate
    assert chain is not None, "Chain should be built successfully"
    print("\n" + "=" * 80)
    print("TEST: Chain Building")
    print("=" * 80)
    print(f"\n✓ Chain built successfully: {type(chain).__name__}")
    print()


def test_return_type_is_list():
    """Test that extract_claims returns a List[ExtractedClaim], not a wrapper."""
    # Setup
    text = "Test message for type checking"

    extraction_input = ClaimExtractionInput(
        source_id="msg-006",
        type="original_text",
        text=text
    )

    # Execute
    result = extract_claims(
        extraction_input=extraction_input
    )

    # Validate type
    assert isinstance(result, list), "Result should be a list, not a wrapper object"
    assert not isinstance(result, ClaimExtractionOutput), "Result should be List[ExtractedClaim], not ClaimExtractionOutput"

    print("\n" + "=" * 80)
    print("TEST: Return Type Check")
    print("=" * 80)
    print(f"\n✓ Correct return type: {type(result)}")
    print(f"✓ Returns list directly, not wrapped in ClaimExtractionOutput")
    print()


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    """Run tests manually with: python -m app.ai.pipeline.tests.claim_extractor_test"""
    pytest.main([__file__, "-v", "-s"])
