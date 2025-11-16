"""
Fact-Checking Pipeline Module

This module contains the individual steps of the fact-checking pipeline:
- Claim Extraction: Extract verifiable claims from user messages
- Evidence Gathering: Search for supporting/refuting evidence
- Final Adjudication: Produce verdict based on claims and evidence

Each step follows LangChain best practices with LCEL chains, structured outputs,
and stateless design.
"""

# Note: Main pipeline orchestration functions are in app.ai.pipeline module (file)
# not in this directory. This directory contains step-by-step implementations.

from .claim_extractor import (
    extract_claims,
    extract_claims_async,
    extract_and_validate_claims,
    validate_claims,
    build_claim_extraction_chain,
)

from .link_context_expander import (
    expand_link_contexts,
    expand_link_context,
    extract_links,
)

from .judgement import (
    adjudicate_claims,
    adjudicate_claims_async,
    build_adjudication_chain,
    format_adjudication_input,
)

from .prompts import (
    get_claim_extraction_prompt,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_USER_PROMPT,
    get_adjudication_prompt,
    ADJUDICATION_SYSTEM_PROMPT,
    ADJUDICATION_USER_PROMPT,
)

# Re-export data models from app.models for convenience
from app.models import ClaimExtractionOutput, FactCheckResult

__all__ = [
    # Claim extraction functions
    "extract_claims",
    "extract_claims_async",
    "extract_and_validate_claims",
    "validate_claims",
    "build_claim_extraction_chain",

    # Link context expansion functions
    "expand_link_contexts",
    "expand_link_context",
    "extract_links",

    # Adjudication functions
    "adjudicate_claims",
    "adjudicate_claims_async",
    "build_adjudication_chain",
    "format_adjudication_input",

    # Data models
    "ClaimExtractionOutput",
    "FactCheckResult",

    # Claim extraction prompts
    "get_claim_extraction_prompt",
    "CLAIM_EXTRACTION_SYSTEM_PROMPT",
    "CLAIM_EXTRACTION_USER_PROMPT",

    # Adjudication prompts
    "get_adjudication_prompt",
    "ADJUDICATION_SYSTEM_PROMPT",
    "ADJUDICATION_USER_PROMPT",
]
