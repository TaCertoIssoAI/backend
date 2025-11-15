from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

# This file defines models from each step of the fact-checking pipeline, with a focus on NEW data from one step to another. Common and repeated data from other steps are saved in
# the commondata.py file classes, whereas this file focuses on the new aggregated data between each step

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
class ExtractedClaim(BaseModel):
    """A single claim extracted from user input"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "claim-uuid-456",
            "text": "Vaccine X causes infertility in women",
            "links": ["https://example.com/article"],
            "llm_comment": "This is a specific medical claim that can be fact-checked against scientific literature",
            "entities": ["vaccine X", "infertility", "women"]
        }
    })

    id: str = Field(..., description="UUID for the claim")
    text: str = Field(..., description="The normalized claim text")
    links: List[str] = Field(default_factory=list, description="Any URLs found in the original text relating to this claim")
    llm_comment: Optional[str] = Field(..., description="LLM's analysis/comment about this claim") #unsure about this field
    entities: List[str] = Field(default_factory=list, description="Named entities in the claim")


class ClaimExtractionOutput(BaseModel):
    """Output of the claim extraction step - a list of extracted claims"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
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

    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="List of fact-checkable claims extracted from the user message"
    )


# ===== STEP 2.5: LINK ENRICHMENT OUTPUT =====
class LinkEnrichmentOutput(BaseModel):
    """Output of the link enrichment step"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "enriched_claims": [],
            "total_links_processed": 5,
            "successful_extractions": 4,
            "processing_time_ms": 3000,
            "processing_notes": "Successfully enriched 4 out of 5 links"
        }
    })

    enriched_claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="Claims with enriched link content"
    )
    total_links_processed: int = Field(default=0, description="Total number of links processed")
    successful_extractions: int = Field(default=0, description="Number of successful extractions")
    processing_time_ms: int = Field(default=0, description="Processing time in milliseconds")
    processing_notes: str = Field(default="", description="Notes about the enrichment process")


# ===== STEP 4: EVIDENCE RETRIEVAL =====
class Citation(BaseModel):
    """A single source citation extracted from external sources (ex: google fact checking API or web search)"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "url": "https://health.gov/vaccine-safety",
            "title": "Vaccine Safety Study Results",
            "publisher": "Ministry of Health",
            "quoted": "No associations with infertility were observed in clinical studies",
            "rating": "Falso",
            "review_date": "2024-11-05"
        }
    })

    url: str
    title: str
    publisher: str
    quoted: str
    rating: Optional[str] = None  # Google fact-check rating: "Falso", "Enganoso", "Verdadeiro", etc.
    review_date: Optional[str] = None  # When the fact-check was published


class EnrichedClaim(BaseModel):
    """Claim enriched with evidence from external fact checking"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "claim-uuid-789",
            "claim_text": "Vaccine X causes infertility in women",
            "citations": [
                {
                    "url": "https://health.gov/vaccine-safety",
                    "title": "Vaccine Safety Study",
                    "publisher": "Ministry of Health",
                    "quoted": "No associations with infertility were observed",
                    "rating": "Falso",
                    "review_date": "2024-11-05"
                }
            ],
            "search_queries": ["vaccine X infertility", "vaccine safety women fertility"],
            "retrieval_notes": "Found 5 sources, selected top 3 most relevant"
        }
    })

    id:str = Field(..., description="UUID for the claim")
    claim_text: str = Field(..., description="The claim this evidence relates to")
    citations: List[Citation] = Field(default_factory=list, description="Sources supporting or refuting the claim")
    search_queries: List[str] = Field(default_factory=list, description="Queries used to find evidence")
    retrieval_notes: Optional[str] = Field(None, description="Notes about the evidence retrieval process")


class EvidenceRetrievalResult(BaseModel):
    """Output of the evidence retrieval step"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "claim_evidence_map": {
                "claim-uuid-1": {
                    "id": "claim-uuid-1",
                    "claim_text": "Vaccine X causes infertility in women",
                    "citations": [],
                    "search_queries": ["vaccine X infertility"],
                    "retrieval_notes": "Found multiple contradicting sources"
                }
            }
        }
    })

    claim_evidence_map: Dict[str, EnrichedClaim] = Field(
        ...,
        description="Maps each claim id to its evidence"
    )


# ===== STEP 5: ADJUDICATION =====
class AdjudicationInput(BaseModel):
    """Input to the adjudication step"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "evidence_map": {
                "claim-uuid-1": {
                    "id": "claim-uuid-1",
                    "claim_text": "Vaccine X causes infertility in women",
                    "citations": [],
                    "search_queries": ["vaccine X infertility"],
                    "retrieval_notes": "Found multiple contradicting sources"
                }
            },
            "additional_context": "User seems concerned about vaccine safety"
        }
    })

    evidence_map: Dict[str, EnrichedClaim] = Field(..., description="Evidence for each claim id")
    additional_context: Optional[str] = Field(None, description="Any additional context")


class FactCheckResult(BaseModel):
    """Final result of the fact-checking pipeline"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "analysis_text": """O usuário questionou sobre a relação entre vacina X e infertilidade feminina. Esta é uma preocupação comum que circula em redes sociais mas não tem base científica sólida.

                Análise por alegação:
                • Vacina X causa infertilidade em mulheres: FALSE

                Fontes de apoio:
                - Ministerio da Saúde: nenhuma ligação entre vacinas e autismo foi encontrada """
        }
    })

    analysis_text: str = Field(..., description="Complete analysis as formatted text") #this should probably be structured somehow

# ===== PIPELINE FLOW SUMMARY =====
