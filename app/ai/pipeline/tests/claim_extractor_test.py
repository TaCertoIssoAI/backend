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
from langchain_openai import ChatOpenAI

from app.models import ClaimExtractionInput, ExtractedClaim, ClaimExtractionOutput, LLMConfig, DataSource
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
    text = "Ouvi dizer que a vacina X causa infertilidade em mulheres, isso é verdade?"

    data_source = DataSource(
        id="msg-001",
        source_type="original_text",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            timeout=30.0
        )
    )

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Basic Claim Extraction from User Message",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    assert len(result.claims) > 0, "Should extract at least one claim"

    # Check source tracking
    for claim in result.claims:
        assert claim.source.source_type == "original_text"
        assert claim.source.source_id == "msg-001"


def test_claim_extraction_from_link_context():
    """Test claim extraction from link/article content."""
    # Setup - simulate content extracted from a link
    text = """=== Artigo: Novo Estudo sobre Segurança de Vacinas ===

Um estudo abrangente publicado hoje não encontrou evidências ligando
a Vacina X a problemas de fertilidade em mulheres. O estudo examinou mais de
50.000 participantes e concluiu que a vacina é segura.

A pesquisa foi conduzida pelo Ministério da Saúde ao longo de 3 anos."""

    data_source = DataSource(
        id="link-456",
        source_type="link_context",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            timeout=30.0
        )
    )

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Claim Extraction from Link Context",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    assert len(result.claims) > 0, "Should extract claims from article"

    # Check source tracking
    for claim in result.claims:
        assert claim.source.source_type == "link_context"
        assert claim.source.source_id == "link-456"


def test_multiple_claims_extraction():
    """Test extraction of multiple claims from one text."""
    # Setup
    text = """O presidente anunciou um novo imposto sobre carbono de R$250 por tonelada.
Além disso, o governo vai investir R$500 bilhões em energia renovável na próxima década.
Isso torna o maior investimento climático da história."""

    data_source = DataSource(
        id="msg-002",
        source_type="original_text",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0))

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Multiple Claims Extraction",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    # We expect multiple claims but don't assert specific number
    # as LLM behavior may vary


def test_portuguese_message_extraction():
    """Test claim extraction with Portuguese text."""
    # Setup
    text = "Dizem que a vacina da COVID causa problemas no coração. Isso é verdade mesmo?"

    data_source = DataSource(
        id="msg-003",
        source_type="original_text",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0))

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Portuguese Message Extraction",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    assert len(result.claims) > 0, "Should extract at least one claim from Portuguese text"


def test_image_ocr_extraction():
    """Test claim extraction from simulated image OCR text."""
    # Setup - simulate OCR output from an image
    text = "URGENTE: Vacina X causa infertilidade. Compartilhe antes que apaguem isso!"

    data_source = DataSource(
        id="img-789",
        source_type="image",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0))

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Image OCR Extraction",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    assert len(result.claims) > 0, "Should extract claim from OCR text"

    # Check source tracking
    for claim in result.claims:
        assert claim.source.source_type == "image"
        assert claim.source.source_id == "img-789"


def test_empty_text():
    """Test behavior with empty text."""
    # Setup
    text = ""

    data_source = DataSource(
        id="msg-004",
        source_type="original_text",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0))

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Empty Text",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    # With empty text, should return empty list or handle gracefully
    assert len(result.claims) == 0, "Empty text should result in no claims"


def test_opinion_vs_claim():
    """Test that LLM can distinguish opinions from fact-checkable claims."""
    # Setup
    text = "Acho que vacinas são assustadoras e não gosto delas. O que você acha?"

    data_source = DataSource(
        id="msg-005",
        source_type="original_text",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0))

    # Execute
    result = extract_and_validate_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate wrapper type
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput"

    # Print for debugging
    print_claim_results(
        result.claims,
        "Opinion vs Claim",
        input_text=text
    )

    # Validate structure
    validate_claims_list(result.claims)
    # This is pure opinion, not a fact-checkable claim
    # LLM should ideally return empty or very few claims


def test_validate_claims_function():
    """Test the validate_claims helper function."""
    # Setup: Create mock claims with some that should be filtered
    from app.models import ExtractedClaim, ClaimSource

    claims = [
        ExtractedClaim(
            id="claim-1",
            text="Afirmação válida sobre vacinas",
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="Esta é válida",
            entities=["vacinas"]
        ),
        ExtractedClaim(
            id="claim-2",
            text="",  # Empty text - should be filtered
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="Vazia",
            entities=[]
        ),
        ExtractedClaim(
            id="claim-3",
            text="Afirmação válida sobre vacinas",  # Duplicate - should be filtered
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="Esta é duplicada",
            entities=["vacinas"]
        ),
        ExtractedClaim(
            id="claim-4",
            text="Outra afirmação válida",
            source=ClaimSource(
                source_type="original_text",
                source_id="msg-1"
            ),
            llm_comment="Também válida",
            entities=["afirmação"]
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
    assert validated[0].text == "Afirmação válida sobre vacinas"
    assert validated[1].text == "Outra afirmação válida"


def test_chain_building():
    """Test that the chain can be built without errors."""
    from app.ai.pipeline import build_claim_extraction_chain

    llm_config = LLMConfig(
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            timeout=30.0
        )
    )

    # Build chain with default source type
    chain = build_claim_extraction_chain(
        llm_config=llm_config,
        source_type="original_text"
    )

    # Validate
    assert chain is not None, "Chain should be built successfully"
    print("\n" + "=" * 80)
    print("TEST: Chain Building")
    print("=" * 80)
    print(f"\n✓ Chain built successfully: {type(chain).__name__}")
    print()


def test_return_type_is_wrapper():
    """Test that extract_claims returns ClaimExtractionOutput wrapper for type safety."""
    # Setup
    text = "Mensagem de teste para verificação de tipo"

    data_source = DataSource(
        id="msg-006",
        source_type="original_text",
        original_text=text
    )

    extraction_input = ClaimExtractionInput(data_source=data_source)

    llm_config = LLMConfig(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0))

    # Execute
    result = extract_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    # Validate type - should be wrapper, not raw list
    assert isinstance(result, ClaimExtractionOutput), "Result should be ClaimExtractionOutput wrapper"
    assert hasattr(result, 'claims'), "Wrapper should have 'claims' attribute"
    assert isinstance(result.claims, list), "The 'claims' attribute should be a list"

    print("\n" + "=" * 80)
    print("TEST: Return Type Check")
    print("=" * 80)
    print(f"\n✓ Correct return type: {type(result).__name__}")
    print(f"✓ Returns ClaimExtractionOutput wrapper for type safety")
    print(f"✓ Wrapper contains {len(result.claims)} claim(s)")
    print()


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    """Run tests manually with: python -m app.ai.pipeline.tests.claim_extractor_test"""
    pytest.main([__file__, "-v", "-s"])
