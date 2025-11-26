from typing import List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# ===== ENUMS FOR TYPE SAFETY =====
class VerdictLabel(str, Enum):
    TRUE = "verdadeiro"
    FALSE = "falso"
    MISLEADING = "enganoso"
    UNVERIFIABLE = "não verificável"


class ContentType(str, Enum):
    """Type of content being analyzed"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"



class ProcessingStage(str, Enum):
    NORMALIZATION = "normalization"
    PRIOR_CHECK = "prior_check"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    PASSAGE_SELECTION = "passage_selection"
    LLM_JUDGMENT = "llm_judgment"
    QUALITY_GATES = "quality_gates"


class ErrorType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    EXTERNAL_API_ERROR = "external_api_error"
    TIMEOUT_ERROR = "timeout_error"
    QUALITY_GATE_FAILURE = "quality_gate_failure"


# ===== API REQUEST/RESPONSE MODELS =====
class ContentItem(BaseModel):
    """Individual content item with text and type"""
    textContent: str = Field(..., description="Text content or description")
    type: ContentType = Field(..., description="Type of content")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "textContent": "Image shows a news headline about government policies",
                "type": "image"
            }
        }
    )

class Request(BaseModel):
    """Unified request model containing array of content items"""
    content: List[ContentItem] = Field(..., description="Array of content items to be fact-checked")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "content": [
                        {
                            "textContent": "The government announced new policies yesterday",
                            "type": "image"
                        }
                    ]
                },
                {
                    "content": [
                        {
                            "textContent": "According to the speaker, inflation reached 10% last month",
                            "type": "audio"
                        }
                    ]
                },
                {
                    "content": [
                        {
                            "textContent": "Check this important announcement",
                            "type": "text"
                        }
                    ]
                },
                {
                    "content": [
                        {
                            "textContent": "Image shows economic data",
                            "type": "image"
                        },
                        {
                            "textContent": "Speaker discusses the same data",
                            "type": "audio"
                        }
                    ]
                }
            ]
        }
    )

class AnalysisResponse(BaseModel):
    """Response from fact-checking analysis"""
    message_id: str = Field(..., description="Unique identifier for this analysis")
    rationale: str = Field(..., description="Complete analysis text with context, verdicts, and citations")
    responseWithoutLinks: str = Field(..., description="Analysis text before sources section (before 'Fontes de apoio:')")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_id": "analysis_123",
                "rationale": "O texto contém uma alegação sobre anúncio governamental. A informação foi verificada contra fontes oficiais.\n\nAnálise por alegação:\n• Governo anunciou novas políticas: VERDADEIRO\n\nFontes de apoio:\n- Portal Oficial: \"Novas políticas foram anunciadas ontem\" (https://gov.example.com/news)",
                "responseWithoutLinks": "O texto contém uma alegação sobre anúncio governamental. A informação foi verificada contra fontes oficiais.\n\nAnálise por alegação:\n• Governo anunciou novas políticas: VERDADEIRO",
            }
        }
    )
