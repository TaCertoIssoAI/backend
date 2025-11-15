from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ===== ENUMS FOR TYPE SAFETY =====
class VerdictLabel(str, Enum):
    TRUE = "verdadeiro"
    FALSE = "falso"
    MISLEADING = "enganoso"
    UNVERIFIABLE = "não verificável"


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
class TextRequest(BaseModel):
    """Request for text-only fact checking"""
    text: str = Field(..., description="Text to be fact-checked")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The government announced new policies yesterday"
            }
        }


class ImageRequest(BaseModel):
    """Request for image-based fact checking"""
    chatId: Optional[str] = Field(None, description="Optional chat identifier")
    # Note: image file will be handled as multipart/form-data in FastAPI

    class Config:
        json_schema_extra = {
            "example": {
                "chatId": "5521999999999@g.us"
            }
        }


class MultimodalRequest(BaseModel):
    """Request for multimodal (text + image) fact checking"""
    text: Optional[str] = Field(None, description="Optional text content")
    chatId: Optional[str] = Field(None, description="Optional chat identifier")
    # Note: image file will be handled as multipart/form-data in FastAPI

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Look at this news article",
                "chatId": "5521999999999@g.us"
            }
        }


class AnalysisResponse(BaseModel):
    """Response from fact-checking analysis"""
    message_id: str = Field(..., description="Unique identifier for this analysis")
    verdict: str = Field(..., description="Fact-check verdict")
    rationale: str = Field(..., description="Complete analysis text with context, verdicts, and citations")
    responseWithoutLinks: str = Field(..., description="Analysis text before sources section (before 'Fontes de apoio:')")
    processing_time_ms: int = Field(default=0, description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "analysis_123",
                "verdict": "text_analysis",
                "rationale": "O texto contém uma alegação sobre anúncio governamental. A informação foi verificada contra fontes oficiais.\n\nAnálise por alegação:\n• Governo anunciou novas políticas: VERDADEIRO\n\nFontes de apoio:\n- Portal Oficial: \"Novas políticas foram anunciadas ontem\" (https://gov.example.com/news)",
                "responseWithoutLinks": "O texto contém uma alegação sobre anúncio governamental. A informação foi verificada contra fontes oficiais.\n\nAnálise por alegação:\n• Governo anunciou novas políticas: VERDADEIRO",
                "processing_time_ms": 3500
            }
        }
