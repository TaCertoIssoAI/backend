from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict


class LLMConfig(BaseModel):
    """Configuration for LLM model calls"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.0,
            "timeout": 30.0
        }
    })

    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name to use"
    )
    temperature: float = Field(
        default=0.0,
        description="Model temperature (0.0 for deterministic, higher for more creative)",
        ge=0.0,
        le=2.0
    )
    timeout: Optional[float] = Field(
        default=30.0,
        description="Timeout in seconds for the model call",
        gt=0
    )


class TimeoutConfig(BaseModel):
    """Configuration for pipeline step timeouts (all in seconds)"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "link_content_expander_timeout_per_link": 30.0,
            "link_content_expander_timeout_total": 120.0,
            "claim_extractor_timeout_per_source": 30.0,
            "claim_extractor_timeout_total": 90.0,
            "evidence_retrieval_timeout_per_claim": 45.0,
            "evidence_retrieval_timeout_total": 180.0,
            "adjudication_timeout": 60.0
        }
    })

    # link content expander timeouts
    link_content_expander_timeout_per_link: float = Field(
        default=30.0,
        description="Timeout in seconds for scraping a single link",
        gt=0
    )
    link_content_expander_timeout_total: float = Field(
        default=120.0,
        description="Total timeout in seconds for all link expansions",
        gt=0
    )

    # claim extractor timeouts
    claim_extractor_timeout_per_source: float = Field(
        default=30.0,
        description="Timeout in seconds for extracting claims from a single data source",
        gt=0
    )
    claim_extractor_timeout_total: float = Field(
        default=90.0,
        description="Total timeout in seconds for extracting claims from all sources",
        gt=0
    )

    # evidence retrieval timeouts
    evidence_retrieval_timeout_per_claim: float = Field(
        default=45.0,
        description="Timeout in seconds for gathering evidence for a single claim",
        gt=0
    )
    evidence_retrieval_timeout_total: float = Field(
        default=180.0,
        description="Total timeout in seconds for gathering evidence for all claims",
        gt=0
    )

    # adjudication timeout
    adjudication_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the final adjudication LLM call",
        gt=0
    )


class PipelineConfig(BaseModel):
    """Complete configuration for the fact-checking pipeline"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "claim_extraction_llm_config": {
                "model_name": "gpt-4o-mini",
                "temperature": 0.0,
                "timeout": 30.0
            },
            "adjudication_llm_config": {
                "model_name": "gpt-4o",
                "temperature": 0.2,
                "timeout": 60.0
            },
            "timeout_config": {
                "link_content_expander_timeout_per_link": 30,
                "link_content_expander_timeout_total": 120,
                "claim_extractor_timeout_per_source": 30,
                "claim_extractor_timeout_total": 90,
                "evidence_retrieval_timeout_per_claim": 45,
                "evidence_retrieval_timeout_total": 180,
                "adjudication_timeout": 60
            },
            "max_links_to_expand": 5,
            "max_claims_to_extract": 10,
            "max_evidence_sources_per_claim": 5
        }
    })

    # LLM configurations
    claim_extraction_llm_config: LLMConfig = Field(
        default_factory=lambda: LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0.0,
            timeout=30.0
        ),
        description="LLM configuration for claim extraction step"
    )

    adjudication_llm_config: LLMConfig = Field(
        default_factory=lambda: LLMConfig(
            model_name="gpt-4o",
            temperature=0.2,
            timeout=60.0
        ),
        description="LLM configuration for final adjudication step"
    )

    # timeout configuration
    timeout_config: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configurations for pipeline steps"
    )

    # pipeline limits
    max_links_to_expand: int = Field(
        default=5,
        description="Maximum number of links to expand from original text",
        gt=0
    )
    max_claims_to_extract: int = Field(
        default=10,
        description="Maximum number of claims to extract per data source",
        gt=0
    )
    max_evidence_sources_per_claim: int = Field(
        default=5,
        description="Maximum number of evidence sources to retrieve per claim",
        gt=0
    )

    # evidence gatherers configuration
    evidence_gatherers: List[Any] = Field(
        default_factory=list,
        description="List of evidence gatherer instances to use for retrieving citations"
    )