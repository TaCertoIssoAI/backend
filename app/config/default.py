"""
default configuration factory for the fact-checking pipeline.

provides a centralized location for creating default PipelineConfig instances
with sensible production defaults.
"""

from langchain_openai import ChatOpenAI

from app.models import PipelineConfig, LLMConfig, TimeoutConfig


def get_default_pipeline_config() -> PipelineConfig:
    """
    create and return a PipelineConfig with default values.

    this is the recommended way to get a default configuration for the pipeline.
    all defaults are production-ready and can be customized as needed.

    returns:
        PipelineConfig with all default values set

    example:
        >>> from app.config.default import get_default_pipeline_config
        >>> config = get_default_pipeline_config()
        >>> config.claim_extraction_llm_config.llm.model_name
        'gpt-4o-mini'
        >>> config.timeout_config.adjudication_timeout
        60.0
    """
    return PipelineConfig(
        # claim extraction uses fast, cheap model
        claim_extraction_llm_config=LLMConfig(
            llm=ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                timeout=30.0
            )
        ),
        # adjudication uses more powerful model (o3-mini doesn't support temperature)
        adjudication_llm_config=LLMConfig(
            llm=ChatOpenAI(
                model="gpt-5-nano",
                timeout=60.0
            )
        ),
        # timeout configuration
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=45.0,  # increased from 20.0 for Facebook scraping
            link_content_expander_timeout_total=120.0,    # increased from 30.0 for multiple links
            claim_extractor_timeout_per_source=10.0,
            claim_extractor_timeout_total=20.0,
            evidence_retrieval_timeout_per_claim=15.0,
            evidence_retrieval_timeout_total=40.0,
            adjudication_timeout=20.0
        ),
        # pipeline limits
        max_links_to_expand=5,
        max_claims_to_extract=10,
        max_evidence_sources_per_claim=5
    )


def get_fast_pipeline_config() -> PipelineConfig:
    """
    create a PipelineConfig optimized for speed (shorter timeouts, fewer sources).

    useful for development, testing, or when quick responses are more important
    than thoroughness.

    returns:
        PipelineConfig with fast/minimal settings

    example:
        >>> from app.config.default import get_fast_pipeline_config
        >>> config = get_fast_pipeline_config()
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
            llm=ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                timeout=30.0
            )
        ),
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=10.0,   # much faster
            link_content_expander_timeout_total=30.0,
            claim_extractor_timeout_per_source=15.0,
            claim_extractor_timeout_total=45.0,
            evidence_retrieval_timeout_per_claim=20.0,
            evidence_retrieval_timeout_total=60.0,
            adjudication_timeout=30.0
        ),
        max_links_to_expand=3,                # fewer links
        max_claims_to_extract=5,              # fewer claims
        max_evidence_sources_per_claim=3      # fewer sources
    )


def get_thorough_pipeline_config() -> PipelineConfig:
    """
    create a PipelineConfig optimized for thoroughness (longer timeouts, more sources).

    useful for high-stakes fact-checking where accuracy and coverage are critical.

    returns:
        PipelineConfig with thorough/comprehensive settings

    example:
        >>> from app.config.default import get_thorough_pipeline_config
        >>> config = get_thorough_pipeline_config()
        >>> config.max_evidence_sources_per_claim
        10
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
            llm=ChatOpenAI(
                model="gpt-4o",
                temperature=0.2,
                timeout=120.0
            )
        ),
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=60.0,    # more time per link
            link_content_expander_timeout_total=300.0,      # 5 minutes total
            claim_extractor_timeout_per_source=60.0,
            claim_extractor_timeout_total=180.0,            # 3 minutes total
            evidence_retrieval_timeout_per_claim=90.0,
            evidence_retrieval_timeout_total=360.0,         # 6 minutes total
            adjudication_timeout=120.0                      # 2 minutes
        ),
        max_links_to_expand=10,               # more links
        max_claims_to_extract=20,             # more claims
        max_evidence_sources_per_claim=10     # more sources
    )
