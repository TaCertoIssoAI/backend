"""
No Claims Fallback Step for the Fact-Checking Pipeline.

This module provides a fallback mechanism when the claim extractor cannot find
any verifiable claims in the user's input. It uses an LLM to generate a friendly
explanation for the user about why no claims could be extracted.

Follows LangChain best practices:
- LCEL composition for declarative chains
- Structured outputs with Pydantic
- Stateless design with explicit state passing
- Type annotations throughout
- Support for both sync and async operations

Architecture:
- Receives text that had no claims extracted
- Uses LLM to generate user-friendly explanation
- Returns structured output with explanation text
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.models import PipelineConfig
from .prompts import get_no_claims_fallback_prompt


# ===== STRUCTURED OUTPUT SCHEMA =====

class NoClaimsFallbackOutput(BaseModel):
    """
    output schema for no claims fallback.

    contains the explanation text that will be shown to the user.
    """
    explanation: str = Field(
        ...,
        description="Friendly explanation for why no claims were found"
    )
    original_text: str = Field(
        ...,
        description="The original text that had no claims"
    )


# ===== CHAIN CONSTRUCTION =====

def build_no_claims_fallback_chain(
    config: PipelineConfig
) -> Runnable:
    """
    builds the LCEL chain for no claims fallback.

    the chain follows this structure:
        prompt | model | output_parser -> str

    args:
        config: Pipeline configuration with fallback LLM config.

    returns:
        a Runnable chain that takes dict input and returns string explanation

    best practices applied:
    - simple string output for user-facing messages
    - moderate temperature for natural, friendly responses
    - stateless design - no global state
    """
    # get the prompt template
    prompt = get_no_claims_fallback_prompt()

    # use the fallback llm from config
    model = config.fallback_llm_config.llm

    # use string output parser for simple text response
    output_parser = StrOutputParser()

    # compose the chain using LCEL
    chain = prompt | model | output_parser

    return chain


# ===== MAIN FALLBACK FUNCTIONS =====

def generate_no_claims_explanation(
    text: str,
    config: PipelineConfig
) -> NoClaimsFallbackOutput:
    """
    generates explanation for why no claims were found in text.

    this is the main synchronous entry point for no claims fallback.

    args:
        text: the original text that had no verifiable claims
        config: Pipeline configuration with fallback LLM config.

    returns:
        NoClaimsFallbackOutput containing explanation and original text

    example:
        >>> from app.config.default import get_default_pipeline_config
        >>> config = get_default_pipeline_config()
        >>> result = generate_no_claims_explanation("Olá, bom dia!", config)
        >>> print(result.explanation)
        "Olá! Não identifiquei nenhuma alegação verificável..."
    """
    # build the chain
    chain = build_no_claims_fallback_chain(config)

    # prepare input for the prompt template
    chain_input = {
        "text": text
    }

    # invoke the chain - gets explanation string
    try:
        explanation: str = chain.invoke(chain_input)
    except Exception as e:
        # if LLM call fails (API overload, timeout, etc.), use default message
        from app.observability.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"no-claims fallback LLM call failed: {type(e).__name__}: {e}")
        logger.info("using default no-claims message")

        # return friendly default message
        explanation = (
            "Não consegui identificar alegações verificáveis em sua mensagem. "
            "Para verificar informações, é útil incluir detalhes concretos como nomes de pessoas, "
            "lugares, datas, números ou eventos específicos. "
            "Posso ajudar com algo assim?"
        )

    # return structured output
    return NoClaimsFallbackOutput(
        explanation=explanation,
        original_text=text
    )


async def generate_no_claims_explanation_async(
    text: str,
    config: PipelineConfig
) -> NoClaimsFallbackOutput:
    """
    async version of generate_no_claims_explanation.

    follows LangChain best practice: provide async methods for IO-bound operations.

    args:
        text: the original text that had no verifiable claims
        config: Pipeline configuration with fallback LLM config.

    returns:
        NoClaimsFallbackOutput containing explanation and original text
    """
    # build the chain
    chain = build_no_claims_fallback_chain(config)

    # prepare input for the prompt template
    chain_input = {
        "text": text
    }

    # invoke the chain asynchronously - gets explanation string
    try:
        explanation: str = await chain.ainvoke(chain_input)
    except Exception as e:
        # if LLM call fails (API overload, timeout, etc.), use default message
        from app.observability.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"no-claims fallback LLM call failed: {type(e).__name__}: {e}")
        logger.info("using default no-claims message")

        # return friendly default message
        explanation = (
            "Não consegui identificar alegações verificáveis em sua mensagem. "
            "Para verificar informações, é útil incluir detalhes concretos como nomes de pessoas, "
            "lugares, datas, números ou eventos específicos. "
            "Posso ajudar com algo assim?"
        )

    # return structured output
    return NoClaimsFallbackOutput(
        explanation=explanation,
        original_text=text
    )


# ===== HELPER FUNCTIONS =====

def should_use_fallback(total_claims_count: int) -> bool:
    """
    determines if fallback should be used based on claims count.

    args:
        total_claims_count: total number of claims extracted from all sources

    returns:
        True if fallback should be used (no claims found), False otherwise
    """
    return total_claims_count == 0


def get_combined_text_from_sources(sources: list) -> str:
    """
    combines text from multiple data sources for fallback.

    args:
        sources: list of DataSource objects

    returns:
        combined text from all sources, joined with newlines
    """
    texts = []
    for source in sources:
        if hasattr(source, 'original_text') and source.original_text:
            texts.append(source.original_text)

    return "\n\n".join(texts) if texts else ""
