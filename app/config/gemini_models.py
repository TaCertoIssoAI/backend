"""
google gemini configuration factory for the fact-checking pipeline.

provides pipeline configurations using ChatGoogleGenerativeAI (configured for
Vertex AI) for adjudication while keeping the same structure as the default config.
"""

from langchain_openai import ChatOpenAI

from app.models import PipelineConfig, LLMConfig, TimeoutConfig
from app.llms.vertex import make_vertex_chat


def get_gemini_fallback_llm_config() -> LLMConfig:
    """
    create and return an LLMConfig using Gemini for no-claims fallback.

    uses gemini-2.5-flash model optimized for generating friendly explanations
    when no verifiable claims are found in user input.

    returns:
        LLMConfig with gemini model configured for fallback responses

    example:
        >>> from app.config.gemini_models import get_gemini_fallback_llm_config
        >>> config = get_gemini_fallback_llm_config()
        >>> config.llm.model
        'gemini-2.5-flash'
    """
    return LLMConfig(
        llm=make_vertex_chat(
            model="gemini-2.5-flash",
            temperature=0.3,  # slightly higher for more natural, friendly responses
        )
    )


def get_gemini_default_pipeline_config() -> PipelineConfig:
    """
    create and return a PipelineConfig using Gemini for adjudication.

    uses the same configuration as get_default_pipeline_config() but replaces
    the adjudication LLM with Google's Gemini model.

    returns:
        PipelineConfig with gemini adjudication model
    """
    return PipelineConfig(
        # claim extraction uses fast, cheap OpenAI model (same as default)
        claim_extraction_llm_config=LLMConfig(
            llm=ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                timeout=30.0
            )
        ),
        # adjudication uses gemini
        adjudication_llm_config=LLMConfig(
            llm=make_vertex_chat(
                model="gemini-2.5-flash",
                temperature=0.0,
            )
        ),
        # fallback uses gemini for friendly explanations
        fallback_llm_config=LLMConfig(
            llm=make_vertex_chat(
                model="gemini-2.5-flash",
                temperature=0.0,
            )
        ),
        # timeout configuration (same as default)
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=45.0,
            link_content_expander_timeout_total=120.0,
            claim_extractor_timeout_per_source=10.0,
            claim_extractor_timeout_total=20.0,
            evidence_retrieval_timeout_per_claim=15.0,
            evidence_retrieval_timeout_total=40.0,
            adjudication_timeout=20.0
        ),
        # pipeline limits (same as default)
        max_links_to_expand=5,
        max_claims_to_extract=10,
        max_evidence_sources_per_claim=5
    )


def get_gemini_fast_pipeline_config() -> PipelineConfig:
    """
    create a PipelineConfig using Gemini optimized for speed.

    useful for development, testing, or when quick responses are more important
    than thoroughness.

    returns:
        PipelineConfig with fast/minimal settings using gemini
    """
    return PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            llm=ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                timeout=15.0
            )
        ),
        adjudication_llm_config=LLMConfig(
            llm=make_vertex_chat(
                model="gemini-3-pro-preview",
                temperature=0.0,
                thinking_budget=1024,  # low thinking budget for fast responses
            )
        ),
        fallback_llm_config=LLMConfig(
            llm=make_vertex_chat(
                model="gemini-2.5-flash",
                temperature=0.0,
            )
        ),
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=10.0,
            link_content_expander_timeout_total=30.0,
            claim_extractor_timeout_per_source=15.0,
            claim_extractor_timeout_total=45.0,
            evidence_retrieval_timeout_per_claim=20.0,
            evidence_retrieval_timeout_total=60.0,
            adjudication_timeout=30.0
        ),
        max_links_to_expand=3,
        max_claims_to_extract=5,
        max_evidence_sources_per_claim=3
    )


def get_gemini_thorough_pipeline_config() -> PipelineConfig:
    """
    create a PipelineConfig using Gemini optimized for thoroughness.

    useful for high-stakes fact-checking where accuracy and coverage are critical.
    uses higher thinking budget for more comprehensive reasoning.

    returns:
        PipelineConfig with thorough/comprehensive settings using gemini
    """
    return PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            llm=ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,
                timeout=60.0
            )
        ),
        adjudication_llm_config=LLMConfig(
            llm=make_vertex_chat(
                model="gemini-3-pro-preview",
                temperature=0.0,
                thinking_budget=8192,  # high thinking budget for thorough analysis
            )
        ),
        fallback_llm_config=LLMConfig(
            llm=make_vertex_chat(
                model="gemini-2.5-flash",
                temperature=0.0,
            )
        ),
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=60.0,
            link_content_expander_timeout_total=300.0,
            claim_extractor_timeout_per_source=60.0,
            claim_extractor_timeout_total=180.0,
            evidence_retrieval_timeout_per_claim=90.0,
            evidence_retrieval_timeout_total=360.0,
            adjudication_timeout=120.0
        ),
        max_links_to_expand=10,
        max_claims_to_extract=20,
        max_evidence_sources_per_claim=10
    )
