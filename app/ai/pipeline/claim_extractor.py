"""
Claim Extraction Step for the Fact-Checking Pipeline.

Follows LangChain best practices:
- LCEL composition for declarative chains
- Structured outputs with Pydantic
- Stateless design with explicit state passing
- Type annotations throughout
- Support for both sync and async operations

Architecture:
- Receives a ClaimExtractionInput wrapping a DataSource
- Extracts all fact-checkable claims from the text
- Returns claims with proper source tracking
- Source-agnostic: works for any text (user message, link, image OCR, etc.)
"""

from typing import List, Optional
import uuid

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.models import (
    ClaimExtractionInput,
    ExtractedClaim,
    ClaimExtractionOutput,
    ClaimSource,
    LLMConfig,
)
from .prompts import get_claim_extraction_prompt


# ===== INTERNAL LLM SCHEMAS =====
# These are what the LLM returns - simple claim data without ID or source

class _LLMExtractedClaim(BaseModel):
    """Internal schema for what the LLM returns - just the claim content."""
    text: str = Field(..., description="The normalized claim text")
    entities: List[str] = Field(default_factory=list, description="Named entities in the claim")
    llm_comment: Optional[str] = Field(None, description="LLM's analysis of why this is fact-checkable")


class _LLMClaimOutput(BaseModel):
    """Internal schema for LLM output - wrapper containing list of claims."""
    claims: List[_LLMExtractedClaim] = Field(
        default_factory=list,
        description="List of extracted claims"
    )


# ===== CHAIN CONSTRUCTION =====

def build_claim_extraction_chain(
    llm_config: LLMConfig
) -> Runnable:
    """
    Builds the LCEL chain for claim extraction.

    The chain follows this structure:
        prompt | model.with_structured_output() -> ClaimExtractionOutput

    Args:
        llm_config: LLM configuration (model name, temperature, timeout).

    Returns:
        A Runnable chain that takes dict input and returns ClaimExtractionOutput

    Best practices applied:
    - Structured output binding for type safety
    - Low temperature for consistent extractions
    - Stateless design - no global state
    """
    # Get the prompt template
    prompt = get_claim_extraction_prompt()

    # Initialize the model with structured output
    model = ChatOpenAI(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        timeout=llm_config.timeout,
    )

    # Bind the structured output schema to enforce JSON format
    # Use internal schema - LLM only returns claim content, not ID or source
    structured_model = model.with_structured_output(
        _LLMClaimOutput,
        method="json_mode"  # Use JSON mode for reliable parsing
    )

    # Compose the chain using LCEL
    chain = prompt | structured_model

    return chain


# ===== MAIN EXTRACTION FUNCTIONS =====

def extract_claims(
    extraction_input: ClaimExtractionInput,
    llm_config: LLMConfig
) -> ClaimExtractionOutput:
    """
    Extracts fact-checkable claims from a text chunk.

    This is the main synchronous entry point for claim extraction.
    Source-agnostic: works for user messages, link content, OCR text, transcripts, etc.

    Args:
        extraction_input: Input wrapping a DataSource with id, source_type, original_text, and metadata
        llm_config: LLM configuration (model name, temperature, timeout).

    Returns:
        ClaimExtractionOutput containing list of ExtractedClaim objects with unique IDs and source tracking

    Example:
        >>> from app.models import ClaimExtractionInput, DataSource, LLMConfig
        >>> data_source = DataSource(
        ...     id="msg-123",
        ...     source_type="original_text",
        ...     original_text="I heard vaccine X causes infertility in women."
        ... )
        >>> input_data = ClaimExtractionInput(data_source=data_source)
        >>> config = LLMConfig(model_name="gpt-4o-mini", temperature=0.0)
        >>> result = extract_claims(input_data, llm_config=config)
        >>> print(len(result.claims))
        1
        >>> print(result.claims[0].text)
        "Vaccine X causes infertility in women"
        >>> print(result.claims[0].source.source_type)
        "original_text"
    """
    # Build the chain
    chain = build_claim_extraction_chain(llm_config=llm_config)

    # Prepare input for the prompt template
    chain_input = {
        "text": extraction_input.data_source.original_text
    }

    # Invoke the chain - gets LLM output (just claim content)
    result: _LLMClaimOutput = chain.invoke(chain_input)

    # Convert LLM output to full ExtractedClaim objects with ID and source
    claims: List[ExtractedClaim] = []
    for llm_claim in result.claims:
        # Generate unique ID with source prefix
        claim_id = f"{uuid.uuid4()}"

        # Build the ClaimSource object
        source = ClaimSource(
            source_type=extraction_input.data_source.source_type,
            source_id=extraction_input.data_source.id
        )

        # Create the full ExtractedClaim with all fields
        claim = ExtractedClaim(
            id=claim_id,
            text=llm_claim.text,
            source=source,
            llm_comment=llm_claim.llm_comment,
            entities=llm_claim.entities
        )
        claims.append(claim)

    return ClaimExtractionOutput(claims=claims)


async def extract_claims_async(
    extraction_input: ClaimExtractionInput,
    llm_config: LLMConfig
) -> ClaimExtractionOutput:
    """
    Async version of extract_claims.

    Follows LangChain best practice: provide async methods for IO-bound operations.

    Args:
        extraction_input: Input wrapping a DataSource with id, source_type, original_text, and metadata
        llm_config: LLM configuration (model name, temperature, timeout).

    Returns:
        ClaimExtractionOutput containing list of ExtractedClaim objects with unique IDs and source tracking
    """
    # Build the chain
    chain = build_claim_extraction_chain(llm_config=llm_config)

    # Prepare input for the prompt template
    chain_input = {
        "text": extraction_input.data_source.original_text
    }

    # Invoke the chain asynchronously - gets LLM output (just claim content)
    result: _LLMClaimOutput = await chain.ainvoke(chain_input)

    # Convert LLM output to full ExtractedClaim objects with ID and source
    claims: List[ExtractedClaim] = []
    for llm_claim in result.claims:
        # Generate unique ID
        claim_id = f"{uuid.uuid4()}"

        # Build the ClaimSource object
        source = ClaimSource(
            source_type=extraction_input.data_source.source_type,
            source_id=extraction_input.data_source.id
        )

        # Create the full ExtractedClaim with all fields
        claim = ExtractedClaim(
            id=claim_id,
            text=llm_claim.text,
            source=source,
            llm_comment=llm_claim.llm_comment,
            entities=llm_claim.entities
        )
        claims.append(claim)

    return ClaimExtractionOutput(claims=claims)


# ===== HELPER FUNCTIONS =====

def validate_claims(claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
    """
    Validates and filters extracted claims.

    Filters out:
    - Claims with empty text
    - Duplicate claims (same text)

    Args:
        claims: List of extracted claims

    Returns:
        Filtered and validated list of claims
    """
    if not claims:
        return []

    validated = []
    seen_texts: set[str] = set()

    for claim in claims:
        # Skip empty or very short claims
        if not claim.text or len(claim.text.strip()) < 3:
            continue

        # Skip duplicates
        normalized_text = claim.text.strip().lower()
        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)
        validated.append(claim)

    return validated


# ===== CONVENIENCE FUNCTION =====

def extract_and_validate_claims(
    extraction_input: ClaimExtractionInput,
    llm_config: LLMConfig
) -> ClaimExtractionOutput:
    """
    Extracts claims and validates them in one call.

    This is the recommended entry point for most use cases.

    Args:
        extraction_input: Input wrapping a DataSource with id, source_type, original_text, and metadata
        llm_config: LLM configuration (model name, temperature, timeout).

    Returns:
        ClaimExtractionOutput containing validated list of ExtractedClaim objects with source tracking
    """
    result = extract_claims(
        extraction_input=extraction_input,
        llm_config=llm_config
    )

    validated_claims = validate_claims(result.claims)
    return ClaimExtractionOutput(claims=validated_claims)
