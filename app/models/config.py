from pydantic import BaseModel, Field, ConfigDict
from langchain_core.language_models.chat_models import BaseChatModel


class LLMConfig(BaseModel):
    """configuration for LLM model calls using langchain chat models"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "llm": "ChatOpenAI(model='gpt-4o-mini', temperature=0.0)"
            }
        }
    )

    llm: BaseChatModel = Field(
        ...,
        description="langchain BaseChatModel instance (supports ChatOpenAI, AzureChatOpenAI, ChatGoogleGenerativeAI, custom models, etc.)"
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
    """complete configuration for the fact-checking pipeline"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "claim_extraction_llm_config": {
                    "llm": "ChatOpenAI(model='gpt-4o-mini', temperature=0.0)"
                },
                "adjudication_llm_config": {
                    "llm": "ChatOpenAI(model='gpt-4o', temperature=0.2)"
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
        ...,
        description="LLM configuration for claim extraction step"
    )

    adjudication_llm_config: LLMConfig = Field(
        ...,
        description="LLM configuration for final adjudication step"
    )

    fallback_llm_config: LLMConfig = Field(
        ...,
        description="LLM configuration for no-claims fallback explanation"
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