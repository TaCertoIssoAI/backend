"""
Default configuration factory for the fact-checking pipeline.

Provides a centralized location for creating default PipelineConfig instances
with sensible production defaults.
"""

from app.models import PipelineConfig, LLMConfig, TimeoutConfig


def get_default_pipeline_config() -> PipelineConfig:
    """
    Create and return a PipelineConfig with default values.

    This is the recommended way to get a default configuration for the pipeline.
    All defaults are production-ready and can be customized as needed.

    Returns:
        PipelineConfig with all default values set

    Example:
        >>> from app.config.default import get_default_pipeline_config
        >>> config = get_default_pipeline_config()
        >>> config.claim_extraction_llm_config.model_name
        'gpt-4o-mini'
        >>> config.timeout_config.adjudication_timeout
        60.0
    """
    return PipelineConfig(
        # claim extraction uses fast, cheap model
        claim_extraction_llm_config=LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0.0,
            timeout=30.0
        ),
        # adjudication uses more powerful model
        adjudication_llm_config=LLMConfig(
            model_name="o3-mini",
            temperature=0.2,
            timeout=60.0
        ),
        # timeout configuration
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=20.0,
            link_content_expander_timeout_total=30.0,
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
    Create a PipelineConfig optimized for speed (shorter timeouts, fewer sources).

    Useful for development, testing, or when quick responses are more important
    than thoroughness.

    Returns:
        PipelineConfig with fast/minimal settings

    Example:
        >>> from app.config.default import get_fast_pipeline_config
        >>> config = get_fast_pipeline_config()
        >>> config.timeout_config.link_content_expander_timeout_per_link
        10.0
    """
    return PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0.0,
            timeout=15.0  # faster
        ),
        adjudication_llm_config=LLMConfig(
            model_name="gpt-4o-mini",  # use faster model
            temperature=0.1,
            timeout=30.0  # faster
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
    Create a PipelineConfig optimized for thoroughness (longer timeouts, more sources).

    Useful for high-stakes fact-checking where accuracy and coverage are critical.

    Returns:
        PipelineConfig with thorough/comprehensive settings

    Example:
        >>> from app.config.default import get_thorough_pipeline_config
        >>> config = get_thorough_pipeline_config()
        >>> config.max_evidence_sources_per_claim
        10
    """
    return PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            model_name="gpt-4o",  # use better model
            temperature=0.0,
            timeout=60.0  # more time
        ),
        adjudication_llm_config=LLMConfig(
            model_name="gpt-4o",
            temperature=0.2,
            timeout=120.0  # much more time
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
