"""
google gemini configuration factory for the fact-checking pipeline.

provides pipeline configurations using Gemini (via LangChain or custom wrapper)
while keeping the same structure as the default config.
"""

import os
from langchain_openai import ChatOpenAI

from app.models import PipelineConfig, LLMConfig, TimeoutConfig
from app.llms.gemini_langchain import create_gemini_model


def get_gemini_default_pipeline_config() -> PipelineConfig:
    """
    create and return a PipelineConfig using Gemini for adjudication.

    uses gemini for adjudication with google search grounding.
    uses openai for claim extraction (faster and cheaper).

    returns:
        PipelineConfig with gemini adjudication model

    example:
        >>> from app.config.gemini_models import get_gemini_default_pipeline_config
        >>> config = get_gemini_default_pipeline_config()
        >>> config.adjudication_llm_config.llm.model
        'gemini-2.5-flash'
    """
    # try to use openai for claim extraction, fallback to gemini if no api key
    try:
        claim_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            timeout=30.0
        )
    except Exception:
        # no openai key - use gemini for claim extraction too
        claim_llm = create_gemini_model(
            model="gemini-2.5-flash",
            enable_search=False,
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    return PipelineConfig(
        # claim extraction uses fast model
        claim_extraction_llm_config=LLMConfig(llm=claim_llm),
        # adjudication uses gemini with google search grounding (langchain native)
        adjudication_llm_config=LLMConfig(
            llm=create_gemini_model(
                model="gemini-2.5-flash",
                enable_search=True,
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
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

    example:
        >>> from app.config.gemini_models import get_gemini_fast_pipeline_config
        >>> config = get_gemini_fast_pipeline_config()
        >>> config.timeout_config.link_content_expander_timeout_per_link
        10.0
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
            llm=create_gemini_model(
                model="gemini-2.5-flash",
                enable_search=True,
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
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
    uses higher thinking level for more comprehensive reasoning.

    returns:
        PipelineConfig with thorough/comprehensive settings using gemini

    example:
        >>> from app.config.gemini_models import get_gemini_thorough_pipeline_config
        >>> config = get_gemini_thorough_pipeline_config()
        >>> config.adjudication_llm_config.llm.thinking_level
        'high'
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
            llm=create_gemini_model(
                model="gemini-2.5-flash",
                enable_search=True,
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
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
