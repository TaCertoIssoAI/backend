"""
analytics data models for pipeline execution tracking.

provides comprehensive observability of the fact-checking pipeline,
capturing inputs, intermediate steps, and final outputs.
"""

from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """type of message source."""
    FROM_WHATSAPP_GROUP = "FromWhatsappGroup"
    FROM_DIRECT_MESSAGE = "FromDirectMessage"
    FROM_WEB = "FromWeb"


class ClaimAnalytics(BaseModel):
    """analytics for a single extracted claim."""
    text: str = Field(..., description="the claim text")
    links: List[str] = Field(default_factory=list, description="links mentioned in the claim")
    extraction_source: Optional[str] = Field(default=None, description="which data source this claim came from")


class ClaimResponseAnalytics(BaseModel):
    """analytics for adjudication response for a single claim."""
    result: str = Field(..., description="verdict: Fake, True, Misleading, Unverifiable")
    reasoning_text: str = Field(..., description="detailed reasoning for the verdict")
    reasoning_sources: List[str] = Field(
        default_factory=list,
        description="source URLs used in reasoning"
    )
    confidence: Optional[float] = Field(default=None, description="confidence score if available")


class PipelineAnalytics(BaseModel):
    """
    comprehensive analytics for a single pipeline execution.

    captures all inputs, intermediate results, and final outputs
    for debugging, monitoring, and audit trail purposes.
    """

    # document identification
    document_id: UUID = Field(default_factory=uuid4, description="unique document identifier")
    date: datetime = Field(default_factory=datetime.utcnow, description="pipeline execution timestamp")

    # input content breakdown
    pure_text: str = Field(default="", description="original text input without media")
    final_transcribed_text: str = Field(
        default="",
        description="complete text after transcription (text + audio + image + video)"
    )

    # media flags and extracted text
    had_audio: bool = Field(default=False, description="whether audio was present")
    audio_text: str = Field(default="", description="transcribed text from audio")

    had_image: bool = Field(default=False, description="whether image was present")
    image_text: str = Field(default="", description="OCR text from image")

    had_video: bool = Field(default=False, description="whether video was present")
    video_text: str = Field(default="", description="transcribed text from video")

    # extracted links
    links: List[str] = Field(default_factory=list, description="all links found in the message")

    # message metadata
    message_type: MessageType = Field(
        default=MessageType.FROM_DIRECT_MESSAGE,
        description="source type of the message"
    )

    # pipeline step: claim extraction
    claims: Dict[int, ClaimAnalytics] = Field(
        default_factory=dict,
        description="extracted claims indexed by claim number"
    )

    # categorization
    topics: List[str] = Field(default_factory=list, description="detected topics/categories")

    # pipeline step: adjudication
    response_by_claim: Dict[int, ClaimResponseAnalytics] = Field(
        default_factory=dict,
        description="adjudication responses indexed by claim number"
    )

    # final output
    comment_about_complete_context: str = Field(
        default="",
        description="overall summary considering all claims together"
    )
    final_response_text: str = Field(
        default="",
        description="final formatted response text sent to user"
    )

    # execution metadata
    execution_time_ms: Optional[float] = Field(default=None, description="total pipeline execution time")
    steps_completed: List[str] = Field(
        default_factory=list,
        description="list of pipeline steps that completed successfully"
    )
    errors_encountered: List[str] = Field(
        default_factory=list,
        description="list of errors encountered during execution"
    )

    class Config:
        """pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
