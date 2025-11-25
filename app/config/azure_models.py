"""
azure openai configuration factory for the fact-checking pipeline.

provides pipeline configurations using AzureChatOpenAI instead of standard OpenAI.
requires azure-specific environment variables to be set.

required environment variables:
- AZURE_OPENAI_API_KEY: your azure openai api key
- AZURE_OPENAI_ENDPOINT: your azure openai endpoint URL
- AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI: deployment name for gpt-4o-mini
- AZURE_OPENAI_DEPLOYMENT_O3_MINI: deployment name for o3-mini
- AZURE_OPENAI_DEPLOYMENT_GPT4O: deployment name for gpt-4o

note: API version used: 2025-01-01-preview (works for all models)
      defined in AZURE_API_VERSION constant
"""

import os
from langchain_openai import AzureChatOpenAI

from app.models import PipelineConfig, LLMConfig, TimeoutConfig


# azure openai api version - used across all deployments
AZURE_API_VERSION = "2025-01-01-preview"


def _get_azure_chat_openai(
    deployment_name: str,
    temperature: float | None = None,
    timeout: float = 30.0,
    api_version: str | None = None
) -> AzureChatOpenAI:
    """
    create an AzureChatOpenAI instance with standard configuration.

    args:
        deployment_name: azure deployment name for the model
        temperature: model temperature (optional, not supported by o3 models)
        timeout: timeout in seconds
        api_version: api version to use (optional, uses default if not provided)

    returns:
        configured AzureChatOpenAI instance
    """
    kwargs = {
        "azure_deployment": deployment_name,
        "api_version": api_version or os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        "timeout": timeout,
    }

    # only add temperature if provided (o3 models don't support it)
    if temperature is not None:
        kwargs["temperature"] = temperature

    return AzureChatOpenAI(**kwargs)


def get_azure_default_pipeline_config() -> PipelineConfig:
    """
    create and return a PipelineConfig using Azure OpenAI with default values.

    this mirrors get_default_pipeline_config() but uses AzureChatOpenAI.
    all defaults are production-ready and can be customized as needed.

    returns:
        PipelineConfig with azure openai models

    example:
        >>> from app.config.azure_models import get_azure_default_pipeline_config
        >>> config = get_azure_default_pipeline_config()
        >>> config.claim_extraction_llm_config.llm.azure_deployment
        'gpt-4o-mini'
        >>> config.timeout_config.adjudication_timeout
        60.0
    """
    # get deployment names from environment
    gpt4o_mini_deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI",
        "gpt-4o-mini"
    )
    o3_mini_deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_O3_MINI",
        "o3-mini"
    )

    return PipelineConfig(
        # claim extraction uses fast, cheap model
        claim_extraction_llm_config=LLMConfig(
            llm=_get_azure_chat_openai(
                deployment_name=gpt4o_mini_deployment,
                temperature=0.0,
                timeout=30.0,
                api_version=AZURE_API_VERSION
            )
        ),
        # adjudication uses o3-mini
        adjudication_llm_config=LLMConfig(
            llm=_get_azure_chat_openai(
                deployment_name=o3_mini_deployment,
                timeout=60.0,
                api_version=AZURE_API_VERSION
            )
        ),
        # timeout configuration
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=45.0,
            link_content_expander_timeout_total=120.0,
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


def get_azure_fast_pipeline_config() -> PipelineConfig:
    """
    create a PipelineConfig using Azure OpenAI optimized for speed.

    useful for development, testing, or when quick responses are more important
    than thoroughness.

    returns:
        PipelineConfig with fast/minimal settings using azure openai

    example:
        >>> from app.config.azure_models import get_azure_fast_pipeline_config
        >>> config = get_azure_fast_pipeline_config()
        >>> config.timeout_config.link_content_expander_timeout_per_link
        10.0
    """
    gpt4o_mini_deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI",
        "gpt-4o-mini"
    )

    return PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            llm=_get_azure_chat_openai(
                deployment_name=gpt4o_mini_deployment,
                temperature=0.0,
                timeout=15.0,
                api_version=AZURE_API_VERSION
            )
        ),
        adjudication_llm_config=LLMConfig(
            llm=_get_azure_chat_openai(
                deployment_name=gpt4o_mini_deployment,
                temperature=0.1,
                timeout=30.0,
                api_version=AZURE_API_VERSION
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


def get_azure_thorough_pipeline_config() -> PipelineConfig:
    """
    create a PipelineConfig using Azure OpenAI optimized for thoroughness.

    useful for high-stakes fact-checking where accuracy and coverage are critical.

    returns:
        PipelineConfig with thorough/comprehensive settings using azure openai

    example:
        >>> from app.config.azure_models import get_azure_thorough_pipeline_config
        >>> config = get_azure_thorough_pipeline_config()
        >>> config.max_evidence_sources_per_claim
        10
    """
    gpt4o_deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_GPT4O",
        "gpt-4o"
    )

    return PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            llm=_get_azure_chat_openai(
                deployment_name=gpt4o_deployment,
                temperature=0.0,
                timeout=60.0,
                api_version=AZURE_API_VERSION
            )
        ),
        adjudication_llm_config=LLMConfig(
            llm=_get_azure_chat_openai(
                deployment_name=gpt4o_deployment,
                temperature=0.2,
                timeout=120.0,
                api_version=AZURE_API_VERSION
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
