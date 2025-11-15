"""
Claim Extraction Step for the Fact-Checking Pipeline.

Follows LangChain best practices:
- LCEL composition for declarative chains
- Structured outputs with Pydantic
- Stateless design with explicit state passing
- Type annotations throughout
- Support for both sync and async operations
"""

from typing import Any, List, Optional
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.models import ExtractedClaim, ExpandedUserInput, ClaimExtractionOutput
from app.models.commondata import CommonPipelineData
from .prompts import get_claim_extraction_prompt


# ===== CHAIN CONSTRUCTION =====

def build_claim_extraction_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: Optional[float] = None
) -> Runnable:
    """
    Builds the LCEL chain for claim extraction.

    The chain follows this structure:
        prompt | model.with_structured_output() -> List[ExtractedClaim]

    Args:
        model_name: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        temperature: Model temperature (default: 0.0 for deterministic extraction)
        timeout: Optional timeout in seconds for the model call

    Returns:
        A Runnable chain that takes dict input and returns List[ExtractedClaim]

    Best practices applied:
    - Structured output binding for type safety
    - Low temperature for consistent extractions
    - Stateless design - no global state
    """
    # Get the prompt template
    prompt = get_claim_extraction_prompt()

    # Initialize the model with structured output
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
    )

    # Bind the structured output schema to enforce JSON format
    structured_model = model.with_structured_output(
        ClaimExtractionOutput,
        method="json_mode"  # Use JSON mode for reliable parsing
    )

    # Compose the chain using LCEL
    chain = prompt | structured_model

    return chain


# ===== MAIN EXTRACTION FUNCTIONS =====

def extract_claims(
    expanded_input: ExpandedUserInput,
    common_data: CommonPipelineData,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: Optional[float] = 30.0
) -> List[ExtractedClaim]:
    """
    Extracts fact-checkable claims from a user message with expanded context.

    This is the main synchronous entry point for claim extraction.

    Args:
        expanded_input: User input with expanded context from links
        common_data: Common pipeline data (for locale, message_id, original message)
        model_name: OpenAI model to use
        temperature: Model temperature
        timeout: Timeout in seconds for the model call

    Returns:
        List of ExtractedClaim objects with unique IDs

    Example:
        >>> from app.models import ExpandedUserInput
        >>> expanded = ExpandedUserInput(
        ...     user_text="I heard vaccine X causes infertility",
        ...     expanded_context="=== Context from https://example.com ===\\nArticle...",
        ...     expanded_context_by_source={"https://example.com": "Article..."}
        ... )
        >>> common = CommonPipelineData(
        ...     message_id="msg-1",
        ...     message_text="I heard vaccine X causes infertility"
        ... )
        >>> claims = extract_claims(expanded, common)
        >>> print(len(claims))
        1
        >>> print(claims[0].text)
        "Vaccine X causes infertility in women"
    """
    # Build the chain
    chain = build_claim_extraction_chain(
        model_name=model_name,
        temperature=temperature,
        timeout=timeout
    )

    # Prepare input for the prompt template
    chain_input = {
        "expanded_context": expanded_input.expanded_context or "(No additional context from links)",
        "user_message": common_data.message_text
    }

    # Invoke the chain
    result: ClaimExtractionOutput = chain.invoke(chain_input)

    # Post-process: ensure unique IDs if not provided by the model
    claims = result.claims
    for i, claim in enumerate(claims):
        if not claim.id or claim.id == f"claim-{i+1}":
            # Generate proper UUID for production use, prefixed with message_id for traceability
            claim.id = f"{common_data.message_id}-claim-{uuid.uuid4()}"

    return claims


async def extract_claims_async(
    expanded_input: ExpandedUserInput,
    common_data: CommonPipelineData,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: Optional[float] = 30.0
) -> List[ExtractedClaim]:
    """
    Async version of extract_claims.

    Follows LangChain best practice: provide async methods for IO-bound operations.

    Args:
        expanded_input: User input with expanded context from links
        common_data: Common pipeline data (for locale, message_id, original message)
        model_name: OpenAI model to use
        temperature: Model temperature
        timeout: Timeout in seconds

    Returns:
        List of ExtractedClaim objects with unique IDs
    """
    # Build the chain
    chain = build_claim_extraction_chain(
        model_name=model_name,
        temperature=temperature,
        timeout=timeout
    )

    # Prepare input for the prompt template
    chain_input = {
        "expanded_context": expanded_input.expanded_context or "(No additional context from links)",
        "user_message": common_data.message_text
    }

    # Invoke the chain asynchronously
    result: ClaimExtractionOutput = await chain.ainvoke(chain_input)

    # Post-process: ensure unique IDs
    claims = result.claims
    for i, claim in enumerate(claims):
        if not claim.id or claim.id == f"claim-{i+1}":
            # Generate proper UUID for production use, prefixed with message_id for traceability
            claim.id = f"{common_data.message_id}-claim-{uuid.uuid4()}"

    return claims


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
    seen_texts = set[Any]()

    for claim in claims:
        # Skip empty or very short claims
        if not claim.text:
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
    expanded_input: ExpandedUserInput,
    common_data: CommonPipelineData,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: Optional[float] = 30.0
) -> List[ExtractedClaim]:
    """
    Extracts claims and validates them in one call.

    This is the recommended entry point for most use cases.

    Args:
        expanded_input: User input with expanded context from links
        common_data: Common pipeline data (for locale, message_id, original message)
        model_name: OpenAI model to use
        temperature: Model temperature
        timeout: Timeout in seconds

    Returns:
        Validated list of ExtractedClaim objects
    """
    claims = extract_claims(
        expanded_input=expanded_input,
        common_data=common_data,
        model_name=model_name,
        temperature=temperature,
        timeout=timeout
    )

    return validate_claims(claims)
