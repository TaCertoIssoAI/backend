from typing import Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from . import (
     UserInput,
     ExpandedUserInput,
     EnrichedLink,
     ExtractedClaim,
     EnrichedClaim,
     EvidenceRetrievalResult,
     AdjudicationInput,
     FactCheckResult,
     ClaimSourceType,
 )

#Common Data defines models that live through the entirety of the pipeline, used both for core fact checking logic and analytics
class DataSource(BaseModel):
    """
    Data source represents a data source from the pipeline, wrapping different modalities of data such as user text, expanded context links (through web search)
    transcribed audio, images or video. This model provides an unique ID, source type, raw text and metadata about the data source.

    Furthermore this class provides a function that concatenates the metadata in string form with the raw text for easier LLM inputs
    """
    
    id: str = Field(..., description="UUID for the data source")
    source_type: ClaimSourceType = Field(..., description="Type of data source (text, image, audio, video, link)")
    original_text: str = Field(..., description="Raw text content from this source")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the source (e.g., title, author, date, url)"
    )
    
    locale: str = Field(default="pt-BR", description="Language locale")
    timestamp: Optional[str] = Field(None, description="When the message was sent")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "source-uuid-123",
                "source_type": "link_context",
                "original_text": "This article discusses the safety and efficacy of vaccines...",
                "metadata": {
                    "title": "Vaccine Safety Study",
                    "url": "https://example.com/vaccine-study",
                    "published_date": "2024-11-05",
                    "author": "Ministry of Health"
                },
                "locale": "pt-BR",
                "timestamp": "2024-11-05T14:30:00Z"
            }
        }
    )
    
    def to_llm_string(self) -> str:
        """
        Concatenate metadata and original text into a formatted string for LLM consumption.
        
        Returns:
            Formatted string with metadata headers followed by the original text.
        """
        parts: List[str] = []
        
        # Always add source header with type and ID
        parts.append(f"Tipo da fonte: {self.source_type}")

        # prominent URL for link_context sources
        if self.source_type == "link_context" and self.metadata.get("url"):
            parts.append(f"URL da fonte: {self.metadata['url']}")

        # Add additional metadata if present
        if self.metadata:
            parts.append("=== Metadados Adicionais ===")
            # Type narrowing for linter
            meta: dict[str, str] = dict[str, str](self.metadata)
            for key in meta:
                value = meta[key]
                # skip url for link_context — already shown above
                if self.source_type == "link_context" and key == "url":
                    continue
                # Capitalize the key and format nicely
                formatted_key = key.replace("_", " ").title()
                parts.append(f"{formatted_key}: {value}")
            parts.append("")  # Empty line separator
        
        # Add separator before original text
        parts.append("\nTexto Original da fonte: ")
        
        # Add the original text
        parts.append('"' + self.original_text + '"')
        
        return "\n".join(parts)

    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataSource):
            return False
        return self.id == other.id

class StepTiming(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "step_name": "claim_extraction",
            "duration_ms": 180,
            "model_name": "gpt-4o-mini",
            "prompt_tokens": 320,
            "completion_tokens": 95,
        }
    })

    step_name: str = Field(..., description="Logical name of the pipeline step")
    duration_ms: int = Field(..., description="Time spent in this step in milliseconds")
    model_name: Optional[str] = Field(
        None,
        description="Name of the model used in this step, if applicable"
    )
    prompt_tokens: Optional[int] = Field(
        None,
        description="Number of prompt tokens consumed in this step, if applicable"
    )
    completion_tokens: Optional[int] = Field(
        None,
        description="Number of completion tokens produced in this step, if applicable"
    )

class EngineeringAnalytics(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_latency_ms": 950,
            "step_timings": [
                {
                    "step_name": "input_expansion",
                    "duration_ms": 120,
                    "model_name": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                },
                {
                    "step_name": "claim_extraction",
                    "duration_ms": 180,
                    "model_name": "gpt-4o-mini",
                    "prompt_tokens": 320,
                    "completion_tokens": 95,
                },
                {
                    "step_name": "evidence_retrieval",
                    "duration_ms": 400,
                    "model_name": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                },
                {
                    "step_name": "adjudication",
                    "duration_ms": 250,
                    "model_name": "gpt-4o",
                    "prompt_tokens": 520,
                    "completion_tokens": 210,
                },
            ],
        }
    })

    total_latency_ms: Optional[int] = Field(
        None,
        description="Total end to end latency for the pipeline in milliseconds"
    )
    step_timings: List[StepTiming] = Field(
        default_factory=list,
        description="Per step timing and token usage information"
    )

class PublicAnalytics(BaseModel):
    """
    Full snapshot of a single pipeline run, used for:
    - AI explainability
    - anonymized analytics datasets
    - dashboards for researchers, journalists, and policy makers
    """
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "user_input": {
                    "text": "I heard that vaccine X causes infertility in women, is this true?",
                    "locale": "pt-BR",
                    "timestamp": "2024-09-20T15:30:00Z",
                },
                "expanded_input": {
                    "original_message": "Check this out: https://example.com/article - I heard that vaccine X causes infertility in women, is this true?",
                    "user_text": "I heard that vaccine X causes infertility in women, is this true?",
                    "expanded_context": "Article Title: Vaccine Study\nContent: This article discusses vaccine safety...\n\n",
                    "expanded_context_by_source": {
                        "https://example.com/article": "Article Title: Vaccine Study\nContent: This article discusses vaccine safety..."
                    },
                    "locale": "pt-BR",
                    "timestamp": "2024-09-20T15:30:00Z",
                },
                "enriched_links": [
                    {
                        "link_id": "link-uuid-1",
                        "url": "https://example.com/vaccine-study",
                        "title": "Study on Vaccine Safety",
                        "content": "This comprehensive study examined the safety profile of vaccines...",
                        "extraction_status": "success",
                        "extraction_tool": "selenium",
                    }
                ],
                "extracted_claims": [
                    {
                        "id": "claim-uuid-1",
                        "text": "Vaccine X causes infertility in women",
                        "links": ["https://example.com/article"],
                        "llm_comment": "Medical claim about vaccine safety that requires scientific evidence",
                        "entities": ["vaccine X", "infertility", "women"],
                    }
                ],
                "enriched_claims": [
                    {
                        "claim_id": "claim-uuid-1",
                        "claim_text": "Vaccine X causes infertility in women",
                        "citations": [
                            {
                                "url": "https://health.gov/vaccine-safety",
                                "title": "Vaccine Safety Study",
                                "publisher": "Ministry of Health",
                                "quoted": "No associations with infertility were observed",
                                "rating": "False",
                                "review_date": "2024-11-05",
                            }
                        ],
                        "search_queries": [
                            "vaccine X infertility",
                            "vaccine safety women fertility",
                        ],
                        "retrieval_notes": "Found 5 sources, selected top 3 most relevant",
                    }
                ],
                "evidence_result": {
                    "claim_evidence_map": {
                        "claim-uuid-1": {
                            "claim_id": "claim-uuid-1",
                            "claim_text": "Vaccine X causes infertility in women",
                            "citations": [],
                            "search_queries": ["vaccine X infertility"],
                            "retrieval_notes": "Found multiple contradicting sources",
                        }
                    },
                    "total_sources_found": 8,
                    "retrieval_time_ms": 2500,
                },
                "adjudication_input": {
                    "original_user_text": "I heard that vaccine X causes infertility in women, is this true?",
                    "evidence_map": {
                        "claim-uuid-1": {
                            "claim_id": "claim-uuid-1",
                            "claim_text": "Vaccine X causes infertility in women",
                            "citations": [],
                            "search_queries": ["vaccine X infertility"],
                            "retrieval_notes": "Found multiple contradicting sources",
                        }
                    },
                    "additional_context": "User seems concerned about vaccine safety",
                    "timestamp": "2024-09-20T15:30:00Z",
                },
                "fact_check_result": {
                    "original_query": "I heard that vaccine X causes infertility in women, is this true?",
                    "analysis_text": "The user asked about a possible link between vaccine X and female infertility. Available evidence does not support this claim. The vaccine has not been associated with infertility in clinical or observational studies.",
                },
            }
        })

    # Original input (after multimodal preprocessing)
    user_input: Optional["UserInput"] = Field(
        None,
        description="Original user input object used by the pipeline",
    )

    # Text plus expanded context from links
    expanded_input: Optional["ExpandedUserInput"] = Field(
        None,
        description="User input enriched with context extracted from links",
    )

    # Enriched links (from input and possibly from other stages)
    enriched_links: List["EnrichedLink"] = Field(
        default_factory=list,
        description="All links that were enriched with extracted content",
    )

    # Claims extracted from the user text
    extracted_claims: List["ExtractedClaim"] = Field(
        default_factory=list,
        description="List of claims extracted from the original text",
    )

    # Claims enriched with external evidence (fact checking, search, etc.)
    enriched_claims: List["EnrichedClaim"] = Field(
        default_factory=list,
        description="Claims with attached evidence and search metadata",
    )

    # Raw result of the evidence retrieval step
    evidence_result: Optional["EvidenceRetrievalResult"] = Field(
        None,
        description="Full object returned by the evidence retrieval step",
    )

    # Structured input that was sent to the final adjudication model
    adjudication_input: Optional["AdjudicationInput"] = Field(
        None,
        description="Structured input passed to the final adjudication step",
    )

    # Final result shown to the user
    fact_check_result: Optional["FactCheckResult"] = Field(
        None,
        description="Final fact checking result that was delivered to the user",
    )