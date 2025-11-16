"""
Adjudication/Judgment Step for the Fact-Checking Pipeline.

Follows LangChain best practices:
- LCEL composition for declarative chains
- Structured outputs with Pydantic
- Stateless design with explicit state passing
- Type annotations throughout
- Support for both sync and async operations

Architecture:
- Receives AdjudicationInput with DataSources paired with their EnrichedClaims
- Analyzes evidence and generates verdicts for each claim
- Returns structured FactCheckResult with verdicts grouped by data source
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.models import (
    AdjudicationInput,
    FactCheckResult,
    DataSourceResult,
    ClaimVerdict,
    VerdictType,
    LLMConfig,
    DataSourceWithClaims,
    EnrichedClaim,
)
from .prompts import get_adjudication_prompt


# ===== INTERNAL LLM SCHEMAS =====
# These are what the LLM returns - only verdicts and justifications
# IDs and source types are populated programmatically from input based on order

class _LLMClaimVerdict(BaseModel):
    """Internal schema for what the LLM returns for a single claim verdict."""
    claim_id: Optional[str] = Field(
        None,
        description="ID of the claim (optional - if missing, matched by claim_text)"
    )
    claim_text: str = Field(..., description="The claim text (for fallback matching)")
    verdict: VerdictType = Field(..., description="The verdict for this claim")
    justification: str = Field(..., description="Detailed justification citing evidence sources")


class _LLMDataSourceResult(BaseModel):
    """
    Internal schema for LLM output for a single data source.
    data_source_id is optional - if missing, we'll match by order as fallback.
    source_type is populated programmatically to avoid validation errors.
    """
    data_source_id: Optional[str] = Field(
        None,
        description="ID of the data source (from the formatted input) - if missing, matched by order"
    )
    claim_verdicts: List[_LLMClaimVerdict] = Field(
        default_factory=list,
        description="Verdicts for all claims from this data source"
    )


class _LLMAdjudicationOutput(BaseModel):
    """Internal schema for complete LLM adjudication output."""
    results: List[_LLMDataSourceResult] = Field(
        default_factory=list,
        description="Results grouped by data source (in same order as input)"
    )
    overall_summary: str = Field(
        default="",
        description="High-level summary of all fact-check results"
    )


# ===== HELPER FUNCTIONS FOR INPUT FORMATTING =====

def format_enriched_claim(claim: EnrichedClaim) -> str:
    """
    Formats an EnrichedClaim into a string representation for the LLM.
    
    Args:
        claim: EnrichedClaim with citations and evidence
        
    Returns:
        Formatted string with claim text, citations, and search queries
    """
    lines = []
    lines.append(f"  Afirmação ID: {claim.id}")
    lines.append(f"  Texto: {claim.claim_text}")
    
    if claim.citations:
        lines.append(f"\n  Citações e Evidências ({len(claim.citations)} fonte(s)):")
        for i, citation in enumerate(claim.citations, 1):
            lines.append(f"\n    [{i}] {citation.title}")
            lines.append(f"        Fonte: {citation.publisher}")
            lines.append(f"        URL: {citation.url}")
            lines.append(f"        Trecho: \"{citation.quoted}\"")
            if citation.rating:
                lines.append(f"        Avaliação prévia: {citation.rating}")
            if citation.review_date:
                lines.append(f"        Data da revisão: {citation.review_date}")
    else:
        lines.append("\n  Citações e Evidências: Nenhuma fonte encontrada")
    
    if claim.search_queries:
        lines.append(f"\n  Consultas de busca utilizadas: {', '.join(claim.search_queries)}")
    
    if claim.retrieval_notes:
        lines.append(f"  Notas de busca: {claim.retrieval_notes}")
    
    return "\n".join(lines)


def format_data_source_with_claims(source_with_claims: DataSourceWithClaims) -> str:
    """
    Formats a DataSource with its EnrichedClaims for LLM input.
    
    Args:
        source_with_claims: DataSourceWithClaims object
        
    Returns:
        Formatted string combining DataSource metadata and all claims with evidence
    """
    lines = []
    
    # Format the data source using its to_llm_string method
    lines.append(source_with_claims.data_source.to_llm_string())
    lines.append("\nAfirmações extraídas da fonte e as evidências de cada uma\n")
    
    if source_with_claims.enriched_claims:
        for i, claim in enumerate(source_with_claims.enriched_claims, 1):
            lines.append(f"Afirmação {i}:")
            lines.append(format_enriched_claim(claim))
            lines.append("\n" + "-" * 80 + "\n")
    else:
        lines.append("Nenhuma alegação extraída desta fonte.\n")
    
    return "\n".join(lines)


def format_adjudication_input(adjudication_input: AdjudicationInput) -> str:
    """
    Formats the complete AdjudicationInput for LLM consumption.
    
    Args:
        adjudication_input: AdjudicationInput with all sources and claims
        
    Returns:
        Complete formatted string for the LLM prompt
    """
    lines = []
    
    for source_with_claims in adjudication_input.sources_with_claims:
        lines.append(f"\n{'=' * 80}")
        lines.append("NOVA FONTE DE DADOS \n")
        lines.append(format_data_source_with_claims(source_with_claims))
    
    return "\n".join(lines)


# ===== Helper functions for LLM output parsing ===== 

def get_data_source_with_claims(
    llm_source_result: _LLMDataSourceResult,
    adjudication_input: AdjudicationInput,
    result_index: int
) -> Optional[DataSourceWithClaims]:
    """
    Matches an LLM data source result back to the original input.
    
    Uses hybrid matching strategy:
    1. Try to match by data_source_id (if provided by LLM)
    2. Fall back to matching by position/order
    
    Args:
        llm_source_result: LLM output for one data source
        adjudication_input: Original input with all sources
        result_index: Position of this result in the LLM output list
        
    Returns:
        Matched DataSourceWithClaims or None if no match found
    """
    # Create mapping of data_source_id to original source_with_claims
    source_map = {
        source_with_claims.data_source.id: source_with_claims
        for source_with_claims in adjudication_input.sources_with_claims
    }

    # TODO change the prints to logs
    
    # Try to match by data_source_id first
    if llm_source_result.data_source_id:
        source_with_claims = source_map.get(llm_source_result.data_source_id)
        if source_with_claims:
            return source_with_claims
        
        print(f"[WARNING] LLM returned unknown data_source_id: {llm_source_result.data_source_id}")

    
    # Fallback: match by order (position in list)
    print(f"[INFO] data_source_id missing for result {result_index}, matching by order")
    if result_index < len(adjudication_input.sources_with_claims):
        return adjudication_input.sources_with_claims[result_index]
    
    
    print(f"[WARNING] No source at index {result_index}")
    return None


def get_claim_verdicts(
    llm_source_result: _LLMDataSourceResult,
    source_with_claims: DataSourceWithClaims
) -> List[ClaimVerdict]:
    """
    Converts LLM claim verdicts to ClaimVerdict objects with proper IDs.
    
    Uses hybrid matching strategy for claim IDs:
    1. Try to use claim_id from LLM output (if provided and valid)
    2. Fall back to matching by claim_text
    
    Args:
        llm_source_result: LLM output for one data source
        source_with_claims: Original input for this data source
        
    Returns:
        List of ClaimVerdict objects with proper claim_id populated
    """
    # Create mappings for claim matching
    claim_id_by_id = {claim.id: claim for claim in source_with_claims.enriched_claims}
    claim_id_by_text = {claim.claim_text: claim.id for claim in source_with_claims.enriched_claims}
    
    # Convert LLM verdicts to ClaimVerdict objects
    claim_verdicts: List[ClaimVerdict] = []
    for llm_verdict in llm_source_result.claim_verdicts:
        # Try to get claim_id: first from LLM output, then from claim_text matching
        if llm_verdict.claim_id and llm_verdict.claim_id in claim_id_by_id:
            # Use claim_id from LLM (most reliable)
            claim_id = llm_verdict.claim_id
        else:
            # Fallback: match by claim_text
            claim_id = claim_id_by_text.get(llm_verdict.claim_text, "unknown")
            if llm_verdict.claim_id:
                print(f"[WARNING] LLM returned unknown claim_id: {llm_verdict.claim_id}, matched by text instead")
        
        verdict = ClaimVerdict(
            claim_id=claim_id,
            claim_text=llm_verdict.claim_text,
            verdict=llm_verdict.verdict,
            justification=llm_verdict.justification
        )
        claim_verdicts.append(verdict)
    
    return claim_verdicts


# ===== CHAIN CONSTRUCTION =====

def build_adjudication_chain(llm_config: LLMConfig) -> Runnable:
    """
    Builds the LCEL chain for fact-check adjudication.
    
    The chain follows this structure:
        prompt | model.with_structured_output() -> _LLMAdjudicationOutput
    
    Args:
        llm_config: LLM configuration (model name, temperature, timeout).
    
    Returns:
        A Runnable chain that takes dict input and returns _LLMAdjudicationOutput
    
    Best practices applied:
    - Structured output binding for type safety
    - Uses OpenAI o3 model for advanced reasoning
    - Stateless design - no global state
    """
    # Get the prompt template
    prompt = get_adjudication_prompt()
    
    # Initialize the model with structured output
    # Using o3 model as specified
    model = ChatOpenAI(
        model=llm_config.model_name or "o3-mini",  # Default to o3-mini if not specified
        temperature=llm_config.temperature,
        timeout=llm_config.timeout,
    )
    
    # Bind the structured output schema
    # Note: Using default method instead of json_mode for better reliability
    structured_model = model.with_structured_output(
        _LLMAdjudicationOutput
    )
    
    # Compose the chain using LCEL
    chain = prompt | structured_model
    
    return chain


# ===== MAIN ADJUDICATION FUNCTIONS =====

def adjudicate_claims(
    adjudication_input: AdjudicationInput,
    llm_config: LLMConfig
) -> FactCheckResult:
    """
    Adjudicates fact-checkable claims with evidence-based verdicts.
    
    This is the main synchronous entry point for claim adjudication.
    Analyzes each claim with its evidence and generates structured verdicts.
    
    Args:
        adjudication_input: AdjudicationInput with data sources and enriched claims
        llm_config: LLM configuration (model name, temperature, timeout).
    
    Returns:
        FactCheckResult with structured verdicts grouped by data source
    
    Example:
        >>> from app.models import AdjudicationInput, DataSourceWithClaims, LLMConfig
        >>> # ... create adjudication_input ...
        >>> config = LLMConfig(model_name="o3-mini", temperature=0.0)
        >>> result = adjudicate_claims(adjudication_input, llm_config=config)
        >>> print(len(result.results))
        2
        >>> print(result.results[0].claim_verdicts[0].verdict)
        "Falso"
    """
    # Build the chain
    chain = build_adjudication_chain(llm_config=llm_config)
    
    # Format the input for the LLM
    formatted_sources = format_adjudication_input(adjudication_input)
    
    # Prepare additional context
    additional_context_str = ""
    if adjudication_input.additional_context:
        additional_context_str = f"\n**Contexto Adicional**: {adjudication_input.additional_context}\n"
    
    # Prepare input for the prompt template
    chain_input = {
        "formatted_sources_and_claims": formatted_sources,
        "additional_context": additional_context_str
    }
    
    # Invoke the chain - gets LLM output
    result: _LLMAdjudicationOutput = chain.invoke(chain_input)
    
    # Debug: Print what LLM returned
    print("\n[DEBUG] LLM returned:")
    print(f"  - Number of data source results: {len(result.results)}")
    print(f"  - Overall summary present: {bool(result.overall_summary)}")
    if result.results:
        print(f"  - First result has {len(result.results[0].claim_verdicts)} verdict(s)")
    else:
        print("  - WARNING: results list is empty!")
        print(f"  - Raw result object: {result}")
    
    # Convert LLM output to FactCheckResult using helper functions
    data_source_results: List[DataSourceResult] = []
    
    # Process each LLM result
    for idx, llm_source_result in enumerate(result.results):
        # Match LLM result to original input data source
        source_with_claims = get_data_source_with_claims(
            llm_source_result=llm_source_result,
            adjudication_input=adjudication_input,
            result_index=idx
        )
        if not source_with_claims:
            continue  # Skip if no match found
        
        # Convert LLM verdicts to ClaimVerdict objects with proper IDs
        claim_verdicts = get_claim_verdicts(
            llm_source_result=llm_source_result,
            source_with_claims=source_with_claims
        )
        
        # Create DataSourceResult with info from original input
        source_result = DataSourceResult(
            data_source_id=source_with_claims.data_source.id,
            source_type=source_with_claims.data_source.source_type,
            claim_verdicts=claim_verdicts
        )
        data_source_results.append(source_result)
    
    return FactCheckResult(
        results=data_source_results,
        overall_summary=result.overall_summary if result.overall_summary else None
    )


async def adjudicate_claims_async(
    adjudication_input: AdjudicationInput,
    llm_config: LLMConfig
) -> FactCheckResult:
    """
    Async version of adjudicate_claims.
    
    Follows LangChain best practice: provide async methods for IO-bound operations.
    
    Args:
        adjudication_input: AdjudicationInput with data sources and enriched claims
        llm_config: LLM configuration (model name, temperature, timeout).
    
    Returns:
        FactCheckResult with structured verdicts grouped by data source
    """
    # Build the chain
    chain = build_adjudication_chain(llm_config=llm_config)
    
    # Format the input for the LLM
    formatted_sources = format_adjudication_input(adjudication_input)
    
    # Prepare additional context
    additional_context_str = ""
    if adjudication_input.additional_context:
        additional_context_str = f"\n**Contexto Adicional**: {adjudication_input.additional_context}\n"
    
    # Prepare input for the prompt template
    chain_input = {
        "formatted_sources_and_claims": formatted_sources,
        "additional_context": additional_context_str
    }
    
    # Invoke the chain asynchronously - gets LLM output
    result: _LLMAdjudicationOutput = await chain.ainvoke(chain_input)
    
    # Debug: Print what LLM returned
    print("\n[DEBUG] LLM returned (async):")
    print(f"  - Number of data source results: {len(result.results)}")
    print(f"  - Overall summary present: {bool(result.overall_summary)}")
    if result.results:
        print(f"  - First result has {len(result.results[0].claim_verdicts)} verdict(s)")
    else:
        print("  - WARNING: results list is empty!")
        print(f"  - Raw result object: {result}")
    
    # Convert LLM output to FactCheckResult using helper functions
    data_source_results: List[DataSourceResult] = []
    
    # Process each LLM result
    for idx, llm_source_result in enumerate(result.results):
        # Match LLM result to original input data source
        source_with_claims = get_data_source_with_claims(
            llm_source_result=llm_source_result,
            adjudication_input=adjudication_input,
            result_index=idx
        )
        if not source_with_claims:
            continue  # Skip if no match found
        
        # Convert LLM verdicts to ClaimVerdict objects with proper IDs
        claim_verdicts = get_claim_verdicts(
            llm_source_result=llm_source_result,
            source_with_claims=source_with_claims
        )
        
        # Create DataSourceResult with info from original input
        source_result = DataSourceResult(
            data_source_id=source_with_claims.data_source.id,
            source_type=source_with_claims.data_source.source_type,
            claim_verdicts=claim_verdicts
        )
        data_source_results.append(source_result)
    
    return FactCheckResult(
        results=data_source_results,
        overall_summary=result.overall_summary if result.overall_summary else None
    )

