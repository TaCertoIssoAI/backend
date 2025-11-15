# -*- coding: utf-8 -*-
"""
Tests for the claim extraction pipeline step.

These tests make REAL calls to the LLM (OpenAI API) to validate:
- The structure of outputs
- The LangChain chain works correctly
- The prompt produces valid results

IMPORTANT: Set OPENAI_API_KEY in your environment before running.

Run with:
    pytest app/ai/pipeline/tests/claim_extractor_test.py -v -s

The -s flag shows stdout so you can see the LLM responses for debugging.
"""

import pytest
from typing import List

from app.models import ExpandedUserInput, ExtractedClaim, ClaimExtractionOutput
from app.models.commondata import CommonPipelineData
from app.ai.pipeline import (
    extract_claims,
    extract_and_validate_claims,
    validate_claims,
)


# ===== HELPER FUNCTIONS =====

def print_claim_results(
    claims: List[ExtractedClaim],
    test_name: str,
    user_message: str| None = None,
    expanded_context: str | None = None
):
    """Print claim extraction results for debugging, including input for verification."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    # Print input for verification
    if user_message:
        print(f"\n📥 INPUT:")
        print(f"  User Message: {user_message}")

    if expanded_context:
        print(f"\n  Expanded Context:")
        # Print first 200 chars to avoid clutter
        print(f"  {expanded_context}")

    # Print output
    print(f"\n📤 OUTPUT:")
    print(f"  Extracted {len(claims)} claim(s):\n")

    for i, claim in enumerate(claims, 1):
        print(f"  Claim {i}:")
        print(f"    ID: {claim.id}")
        print(f"    Text: {claim.text}")
        print(f"    Entities: {claim.entities}")
        print(f"    Links: {claim.links}")
        print(f"    LLM Comment: {claim.llm_comment}")
        print()


def validate_claim_structure(claim: ExtractedClaim):
    """Validate that a claim has the correct structure."""
    # Required fields
    assert claim.id is not None and claim.id != "", "Claim ID should not be empty"
    assert claim.text is not None and claim.text != "", "Claim text should not be empty"

    # Type checks
    assert isinstance(claim.id, str), "Claim ID should be a string"
    assert isinstance(claim.text, str), "Claim text should be a string"
    assert isinstance(claim.links, list), "Links should be a list"
    assert isinstance(claim.entities, list), "Entities should be a list"

    # Optional field type check
    if claim.llm_comment is not None:
        assert isinstance(claim.llm_comment, str), "LLM comment should be a string"

    # List element type checks
    for link in claim.links:
        assert isinstance(link, str), "Each link should be a string"

    for entity in claim.entities:
        assert isinstance(entity, str), "Each entity should be a string"


def validate_claims_list(claims: List[ExtractedClaim]):
    """Validate that a list of claims has the correct structure."""
    assert isinstance(claims, list), "Result should be a list"

    for claim in claims:
        validate_claim_structure(claim)


# ===== TESTS =====

def test_basic_claim_extraction():
    """Test basic claim extraction with one claim and context."""
    # Setup
    user_message = "I heard that vaccine X causes infertility in women, is this true?"

    expanded_context = """=== Context from https://example.com/vaccine-article ===
Title: New Study on Vaccine Safety

A comprehensive study published today found no evidence linking
Vaccine X to fertility issues in women. The study examined over
50,000 participants and concluded that the vaccine is safe.
"""

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context=expanded_context,
        expanded_context_by_source={
            "https://example.com/vaccine-article": expanded_context
        }
    )

    common_data = CommonPipelineData(
        message_id="test-msg-001",
        message_text=user_message,
        locale="en-US"
    )

    # Execute
    claims = extract_and_validate_claims(
        expanded_input=expanded_input,
        common_data=common_data,
        timeout=30.0
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Basic Claim Extraction",
        user_message=user_message,
        expanded_context=expanded_context
    )

    # Validate structure
    validate_claims_list(claims)
    assert len(claims) > 0, "Should extract at least one claim"

    # Check ID format (should include message_id)
    for claim in claims:
        assert claim.id.startswith("test-msg-001"), "Claim ID should start with message_id"


def test_multiple_claims_extraction():
    """Test extraction of multiple claims from one message."""
    # Setup
    user_message = """Did you see this article? The president is imposing a huge carbon tax of $50 per ton,
and apparently they're spending $100 billion on renewable energy. Is this really happening?"""

    expanded_context = """=== Context from https://news.example.com/politics ===
Breaking News: President Announces Climate Policy

The president announced a new carbon tax of $50 per ton.
Additionally, the government will invest $100 billion in renewable energy over the next decade.
"""

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context=expanded_context,
        expanded_context_by_source={
            "https://news.example.com/politics": expanded_context
        }
    )

    common_data = CommonPipelineData(
        message_id="test-msg-002",
        message_text=user_message,
        locale="en-US"
    )

    # Execute
    claims = extract_and_validate_claims(
        expanded_input=expanded_input,
        common_data=common_data
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Multiple Claims Extraction",
        user_message=user_message,
        expanded_context=expanded_context_text
    )

    # Validate structure
    validate_claims_list(claims)
    # Note: We don't assert a specific number because LLM might interpret differently
    # But we do validate the structure


def test_portuguese_message_extraction():
    """Test claim extraction with Portuguese message."""
    # Setup
    user_message = "Vi esse artigo aqui. Dizem que a vacina da COVID causa problemas no cora��o. Isso � verdade mesmo?"

    expanded_context = """=== Context from https://g1.globo.com/exemplo ===
T�tulo: Estudo sobre Vacinas no Brasil

Pesquisa realizada pela Fiocruz demonstra que a vacina contra COVID-19
� segura e eficaz. N�o h� evid�ncias de efeitos colaterais graves no cora��o.
"""

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context=expanded_context,
        expanded_context_by_source={
            "https://g1.globo.com/exemplo": expanded_context
        }
    )

    common_data = CommonPipelineData(
        message_id="test-msg-003",
        message_text=user_message,
        locale="pt-BR"
    )

    # Execute
    claims = extract_and_validate_claims(
        expanded_input=expanded_input,
        common_data=common_data
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Portuguese Message Extraction",
        user_message=user_message,
        expanded_context=expanded_context
    )

    # Validate structure
    validate_claims_list(claims)
    assert len(claims) > 0, "Should extract at least one claim from Portuguese message"


def test_no_context_extraction():
    """Test claim extraction with no expanded context (only user message)."""
    # Setup
    user_message = "The Earth is flat and NASA is hiding the truth from us."

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context="",
        expanded_context_by_source={}
    )

    common_data = CommonPipelineData(
        message_id="test-msg-004",
        message_text=user_message,
        locale="en-US"
    )

    # Execute
    claims = extract_and_validate_claims(
        expanded_input=expanded_input,
        common_data=common_data
    )

    # Print for debugging
    print_claim_results(
        claims,
        "No Context Extraction",
        user_message=user_message,
        expanded_context=""
    )

    # Validate structure
    validate_claims_list(claims)
    # LLM should still be able to extract claims even without context


def test_empty_message():
    """Test behavior with empty user message."""
    # Setup
    user_message = ""

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context="",
        expanded_context_by_source={}
    )

    common_data = CommonPipelineData(
        message_id="test-msg-005",
        message_text=user_message,
        locale="en-US"
    )

    # Execute
    claims = extract_and_validate_claims(
        expanded_input=expanded_input,
        common_data=common_data
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Empty Message",
        user_message=user_message,
        expanded_context=""
    )

    # Validate structure
    validate_claims_list(claims)
    # With empty message, should return empty list or handle gracefully
    assert len(claims) == 0, "Empty message should result in no claims"


def test_opinion_vs_claim():
    """Test that LLM can distinguish opinions from fact-checkable claims."""
    # Setup
    user_message = "I think vaccines are scary and I don't like them. What do you think?"

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context="",
        expanded_context_by_source={}
    )

    common_data = CommonPipelineData(
        message_id="test-msg-006",
        message_text=user_message,
        locale="en-US"
    )

    # Execute
    claims = extract_and_validate_claims(
        expanded_input=expanded_input,
        common_data=common_data
    )

    # Print for debugging
    print_claim_results(
        claims,
        "Opinion vs Claim",
        user_message=user_message,
        expanded_context=""
    )

    # Validate structure
    validate_claims_list(claims)
    # This is an opinion, not a fact-checkable claim
    # LLM should ideally return empty or very few claims


def test_validate_claims_function():
    """Test the validate_claims helper function."""
    # Setup: Create mock claims with some that should be filtered
    from app.models import ExtractedClaim

    claims = [
        ExtractedClaim(
            id="claim-1",
            text="Valid claim about vaccines",
            links=["https://example.com"],
            llm_comment="This is valid",
            entities=["vaccines"]
        ),
        ExtractedClaim(
            id="claim-2",
            text="",  # Empty text - should be filtered
            links=[],
            llm_comment="Empty",
            entities=[]
        ),
        ExtractedClaim(
            id="claim-3",
            text="Valid claim about vaccines",  # Duplicate - should be filtered
            links=["https://example.com"],
            llm_comment="This is duplicate",
            entities=["vaccines"]
        ),
        ExtractedClaim(
            id="claim-4",
            text="Another valid claim",
            links=[],
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
    print(f"\n Chain built successfully: {type(chain).__name__}")
    print()


def test_return_type_is_list():
    """Test that extract_claims returns a List[ExtractedClaim], not a wrapper."""
    # Setup
    user_message = "Test message for type checking"

    expanded_input = ExpandedUserInput(
        user_text=user_message,
        expanded_context="",
        expanded_context_by_source={}
    )

    common_data = CommonPipelineData(
        message_id="test-msg-007",
        message_text=user_message,
        locale="en-US"
    )

    # Execute
    result = extract_claims(
        expanded_input=expanded_input,
        common_data=common_data
    )

    # Validate type
    assert isinstance(result, list), "Result should be a list, not a wrapper object"
    assert not isinstance(result, ClaimExtractionOutput), "Result should be List[ExtractedClaim], not ClaimExtractionOutput"

    print("\n" + "=" * 80)
    print("TEST: Return Type Check")
    print("=" * 80)
    print(f"\n Correct return type: {type(result)}")
    print(f" Returns list directly, not wrapped in ClaimExtractionOutput")
    print()


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    """Run tests manually with: python -m app.ai.pipeline.tests.claim_extractor_test"""
    pytest.main([__file__, "-v", "-s"])
