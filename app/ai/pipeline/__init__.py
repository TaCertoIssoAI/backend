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

from .prompts import (
    get_claim_extraction_prompt,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    CLAIM_EXTRACTION_USER_PROMPT,
)

# Re-export the data model from app.models for convenience
from app.models import ClaimExtractionOutput

__all__ = [
    # Main extraction functions
    "extract_claims",
    "extract_claims_async",
    "extract_and_validate_claims",

    # Helper functions
    "validate_claims",
    "build_claim_extraction_chain",

    # Data models
    "ClaimExtractionOutput",

    # Prompts
    "get_claim_extraction_prompt",
    "CLAIM_EXTRACTION_SYSTEM_PROMPT",
    "CLAIM_EXTRACTION_USER_PROMPT",
]
