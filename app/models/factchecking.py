from __future__ import annotations

from typing import List, Optional, Dict, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from .commondata import DataSource

# This file defines models from each step of the fact-checking pipeline, with a focus on NEW data from one step to another. Common and repeated data from other steps are saved in
# the DataSource.py file classes, whereas this file focuses on the new aggregated data between each step


# Common data models for several pipeline steps

ClaimSourceType = Literal[
    "original_text",
    "link_context",
    "image",
    "audio_transcript",
    "video_transcript",
    "other",
]

CitationSource = Literal[
    "google_fact_checking_api",
    "google_web_search",
]


VerdictType = Literal["Verdadeiro", "Falso", "Fora de Contexto", "Não foi possível verificar"]




# ===== STEP 1: USER INPUT =====
class UserInput(BaseModel):
    """Raw, unstructured input from the user"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "text": "I heard that vaccine X causes infertility in women, is this true?",
            "locale": "pt-BR",
            "timestamp": "2024-09-20T15:30:00Z",
        }
    })

    text: str = Field(..., description="Raw unstructured text from user")
    locale: str = Field(default="pt-BR", description="Language locale")
    timestamp: Optional[str] = Field(None, description="When the message was sent")

# ===== STEP 2: ORIGINAL CONTEXT ENRICHMENT =====
class ExpandedUserInput(BaseModel):
    """User input augmented with sources included in the message, like external web links"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_text": "I heard that vaccine X causes infertility in women, is this true?",
            "expanded_context": "Article Title: Vaccine Study\nContent: This article discusses vaccine safety...\n\n",
            "expanded_context_by_source": {
                "https://example.com/article": "Article Title: Vaccine Study\nContent: This article discusses vaccine safety..."
            }
        }
    })

    expanded_context: str = Field(...,description="String with all the expanded context that should be pre-pended to the original text")
    expanded_context_by_source: dict[str,str] = Field(default_factory=dict,description=" dict that maps each message source (links) to the extracted context from it")

# ===== STEP 2.1: LINK ENRICHMENT =====
class EnrichedLink(BaseModel):
    """A single enriched link with extracted content"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "link-uuid-123",
            "url": "https://example.com/vaccine-study",
            "title": "Study on Vaccine Safety",
            "content": "This comprehensive study examined the safety profile of vaccines...",
            "extraction_status": "success",
            "extraction_tool": "selenium"
        }
    })

    id: str = Field(..., description="UUID for the link")
    url: str = Field(..., description="The original URL")
    title: str = Field(default="", description="Title extracted from the webpage")
    content: str = Field(default="", description="Main text content extracted from the webpage")
    extraction_status: str = Field(default="pending", description="Status: 'success', 'failed', 'timeout', 'blocked'")
    extraction_tool: str = Field(default="", description="Status: 'selenium', 'cloudscraper'")

# ===== STEP 3: CLAIM EXTRACTION =====

class ClaimExtractionInput(BaseModel):
    """Input for claim extraction step - wraps a DataSource."""

    data_source: "DataSource" = Field(
        ...,
        description="The data source from which claims will be extracted",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data_source": {
                    "id": "img-uuid-123",
                    "source_type": "image",
                    "original_text": "The image caption says that vaccine X causes infertility in women.",
                    "metadata": {},
                    "locale": "pt-BR",
                    "timestamp": "2024-11-05T14:30:00Z"
                }
            }
        }
    )

class ClaimSource(BaseModel):
    """Where this claim was extracted from and which links contributed to it."""

    source_type: ClaimSourceType = Field(
        ...,
        description="Origin modality or channel where the claim was extracted from",
    )

    source_id: str = Field(
        ...,
        description="Id of the media or segment (message, image, audio, etc) that produced the claim",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_type": "link_context",
                "source_id": "uuid-123",
            }
        }
    )

class ExtractedClaim(BaseModel):
    """A single claim extracted from user input or media"""

    id: str = Field(..., description="UUID for the claim")
    text: str = Field(..., description="The normalized claim text")

    source: ClaimSource = Field(
        ...,
        description="Provenance information about where this claim was extracted from",
    )

    llm_comment: Optional[str] = Field( #unsure about this field
        None,
        description="Optional LLM analysis/comment about this claim (for debugging or triage)",
    )

    entities: List[str] = Field(
        default_factory=list,
        description="Named entities in the claim",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "claim-uuid-456",
                "text": "Vaccine X causes infertility in women",
                "source": {
                    "source_type": "link_context",
                    "source_id": "link-uuid-123",
                    "link_ids": ["link-uuid-123"],
                    "snippet": "The article claims that vaccine X causes infertility in women.",
                },
                "llm_comment": "This is a specific medical claim that can be fact checked against scientific literature",
                "entities": ["vaccine X", "infertility", "women"],
            }
        }
    )

class ClaimExtractionOutput(BaseModel):
    """Output of the claim extraction step - a list of extracted claims"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data_source": {
                "id": "msg-001",
                "source_type": "original_text",
                "original_text": "Sample text",
                "metadata": {}
            },
            "claims": [
                {
                    "id": "claim-uuid-456",
                    "text": "Vaccine X causes infertility in women",
                    "links": ["https://example.com/article"],
                    "llm_comment": "This is a specific medical claim that can be fact-checked against scientific literature",
                    "entities": ["vaccine X", "infertility", "women"]
                }
            ]
        }
    })

    data_source: "DataSource" = Field(
        ...,
        description="The data source from which these claims were extracted"
    )

    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="List of fact-checkable claims extracted from the user message"
    )

# ===== STEP 4: EVIDENCE RETRIEVAL =====

class EvidenceRetrievalInput(BaseModel):
    """Input for evidence retrieval step - contains claims to be fact-checked"""
    
    claims: List[ExtractedClaim] = Field(
        ...,
        description="List of extracted claims to retrieve evidence for"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "claims": [
                    {
                        "id": "claim-uuid-456",
                        "text": "Vaccine X causes infertility in women",
                        "source": {
                            "source_type": "original_text",
                            "source_id": "msg-uuid-123"
                        },
                        "llm_comment": "This is a specific medical claim that can be fact-checked",
                        "entities": ["vaccine X", "infertility", "women"]
                    }
                ]
            }
        }
    )

class Citation(BaseModel):
    """A single source citation extracted from external sources (ex: google fact checking API or web search)"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "url": "https://health.gov/vaccine-safety",
            "title": "Vaccine Safety Study Results",
            "publisher": "Ministry of Health",
            "citation_text": "No associations with infertility were observed in clinical studies",
            "source": "google_fact_checking_api",
            "rating": "Falso",
            "date": "2024-11-05"
        }
    })

    url: str
    title: str
    publisher: str
    citation_text: str
    source: Optional[CitationSource] = None  # source API: google_fact_checking_api or google_web_search
    rating: Optional[VerdictType] = None  # Google fact-check rating: "Falso", "Enganoso", "Verdadeiro", etc.
    rating_comment: Optional[str] = None #optional comment about the rating
    date: Optional[str] = None  # When the fact-check was published

class EnrichedClaim(ExtractedClaim):
    """Claim enriched with evidence from external fact checking - extends ExtractedClaim with citations"""
    
    citations: List[Citation] = Field(
        default_factory=list, 
        description="Sources supporting or refuting the claim"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "claim-uuid-789",
                "text": "Vaccine X causes infertility in women",
                "source": {
                    "source_type": "original_text",
                    "source_id": "msg-uuid-123"
                },
                "llm_comment": "This is a specific medical claim that can be fact-checked",
                "entities": ["vaccine X", "infertility", "women"],
                "citations": [
                    {
                        "url": "https://health.gov/vaccine-safety",
                        "title": "Vaccine Safety Study",
                        "publisher": "Ministry of Health",
                        "citation_text": "No associations with infertility were observed",
                        "source": "google_fact_checking_api",
                        "rating": "Falso",
                        "date": "2024-11-05"
                    }
                ]
            }
        }
    )

class EvidenceRetrievalResult(BaseModel):
    """Output of the evidence retrieval step"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "claim_evidence_map": {
                "claim-uuid-1": {
                    "id": "claim-uuid-1",
                    "text": "Vaccine X causes infertility in women",
                    "source": {
                        "source_type": "original_text",
                        "source_id": "msg-uuid-123"
                    },
                    "llm_comment": "Medical claim that can be fact-checked",
                    "entities": ["vaccine X", "infertility", "women"],
                    "citations": []
                }
            }
        }
    })

    claim_evidence_map: Dict[str, EnrichedClaim] = Field(
        ...,
        description="Maps each claim id to its evidence"
    )

# ===== STEP 5: ADJUDICATION =====

class DataSourceWithClaims(BaseModel):
    """A data source paired with its enriched claims for adjudication"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data_source": {
                "id": "msg-001",
                "source_type": "original_text",
                "original_text": "Ouvi dizer que a vacina X causa infertilidade em mulheres",
                "metadata": {},
                "locale": "pt-BR",
                "timestamp": "2024-11-16T10:00:00Z"
            },
            "enriched_claims": [
                {
                    "id": "claim-uuid-1",
                    "text": "Vacina X causa infertilidade em mulheres",
                    "source": {
                        "source_type": "original_text",
                        "source_id": "msg-001"
                    },
                    "llm_comment": "Alegação médica específica que pode ser verificada",
                    "entities": ["vacina X", "infertilidade", "mulheres"],
                    "citations": [
                        {
                            "url": "https://saude.gov.br/vacinas",
                            "title": "Estudo de Segurança de Vacinas",
                            "publisher": "Ministério da Saúde",
                            "citation_text": "Não foram observadas associações com infertilidade",
                            "source": "google_fact_checking_api",
                            "rating": "Falso",
                            "date": "2024-11-05"
                        }
                    ]
                }
            ]
        }
    })

    data_source: "DataSource" = Field(..., description="The data source from which claims were extracted")
    enriched_claims: List[EnrichedClaim] = Field(default_factory=list, description="Claims from this source with their evidence")


class AdjudicationInput(BaseModel):
    """Input to the adjudication step - groups enriched claims by their data source"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "sources_with_claims": [
                {
                    "data_source": {
                        "id": "msg-001",
                        "source_type": "original_text",
                        "original_text": "Ouvi dizer que a vacina X causa infertilidade",
                        "metadata": {},
                        "locale": "pt-BR"
                    },
                    "enriched_claims": [
                        {
                            "id": "claim-uuid-1",
                            "text": "Vacina X causa infertilidade em mulheres",
                            "source": {
                                "source_type": "original_text",
                                "source_id": "msg-001"
                            },
                            "llm_comment": "Alegação médica que pode ser verificada",
                            "entities": ["vacina X", "infertilidade", "mulheres"],
                            "citations": []
                        }
                    ]
                }
            ],
            "additional_context": "Usuário demonstra preocupação com segurança de vacinas"
        }
    })

    sources_with_claims: List[DataSourceWithClaims] = Field(
        ...,
        description="List of data sources paired with their enriched claims"
    )
    additional_context: Optional[str] = Field(None, description="Any additional context for the adjudication")


class ClaimVerdict(BaseModel):
    """Verdict for a single claim with justification"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "claim_id": "claim-uuid-1",
            "claim_text": "Vacina X causa infertilidade em mulheres",
            "verdict": "Falso",
            "justification": "Estudos científicos conduzidos pelo Ministério da Saúde com mais de 50.000 participantes não encontraram evidências ligando a vacina X a problemas de fertilidade. Fonte: https://saude.gov.br/vacinas"
        }
    })

    claim_id: str = Field(..., description="ID of the claim being judged")
    claim_text: str = Field(..., description="The claim text")
    verdict: VerdictType = Field(..., description="The verdict for this claim")
    justification: str = Field(..., description="Detailed justification citing evidence sources")
    citations_used: List[Citation] = Field(
        default_factory=list,
        description="List of citations that were actually used to make this verdict decision"
    )


class DataSourceResult(BaseModel):
    """Fact-check results for all claims from a single data source"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data_source_id": "msg-001",
            "source_type": "original_text",
            "claim_verdicts": [
                {
                    "claim_id": "claim-uuid-1",
                    "claim_text": "Vacina X causa infertilidade em mulheres",
                    "verdict": "Falso",
                    "justification": "Estudos científicos não encontraram evidências..."
                }
            ]
        }
    })

    data_source_id: str = Field(..., description="ID of the data source")
    source_type: ClaimSourceType = Field(..., description="Type of the data source")
    claim_verdicts: List[ClaimVerdict] = Field(
        default_factory=list,
        description="Verdicts for all claims extracted from this source"
    )


class FactCheckResult(BaseModel):
    """Final result of the fact-checking pipeline with structured verdicts per data source"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": [
                {
                    "data_source_id": "msg-001",
                    "source_type": "original_text",
                    "claim_verdicts": [
                        {
                            "claim_id": "claim-uuid-1",
                            "claim_text": "Vacina X causa infertilidade em mulheres",
                            "verdict": "Falso",
                            "justification": "Estudos científicos conduzidos pelo Ministério da Saúde com mais de 50.000 participantes não encontraram evidências..."
                        }
                    ]
                }
            ],
            "overall_summary": "A mensagem contém uma alegação falsa sobre vacinas. Não há evidências científicas que sustentem a afirmação de que a vacina X causa infertilidade."
        }
    })

    results: List[DataSourceResult] = Field(
        default_factory=list,
        description="Structured fact-check results grouped by data source"
    )
    overall_summary: Optional[str] = Field(
        None,
        description="Optional high-level summary of all fact-check results"
    )
    sources_with_claims: List[DataSourceWithClaims] = Field(
        default_factory=list,
        description="Original data sources with their enriched claims (includes all citations for lookup)"
    )

