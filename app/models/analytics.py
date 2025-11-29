"""
analytics data models for pipeline execution tracking.

provides comprehensive observability of the fact-checking pipeline,
capturing inputs, intermediate steps, and final outputs.
"""

from typing import Optional
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class MessageType(str, Enum):
    """type of message source."""
    FROM_WHATSAPP_GROUP = "FromWhatsappGroup"
    FROM_DIRECT_MESSAGE = "FromDirectMessage"
    FROM_WEB = "FromWeb"


class ScrapedLink(BaseModel):
    """scraped link data."""
    url: str = Field(..., description="url of the scraped link")
    success: bool = Field(default=False, description="whether the link was successfully scraped")
    text: Optional[str] = Field(default=None, description="extracted content from the page")


class ClaimAnalytics(BaseModel):
    """analytics for a single extracted claim."""
    text: str = Field(..., description="the claim text")
    links: list[str] = Field(default_factory=list, description="links mentioned in the claim")


class ClaimResponseAnalytics(BaseModel):
    """analytics for adjudication response for a single claim."""
    Result: str = Field(..., description="verdict: Fake, True, Misleading, Unverifiable")
    reasoningText: str = Field(..., description="detailed reasoning for the verdict")
    reasoningSources: list[str] = Field(
        default_factory=list,
        description="source URLs used in reasoning"
    )


class PipelineAnalytics(BaseModel):
    """
    comprehensive analytics for a single pipeline execution.

    captures all inputs, intermediate results, and final outputs
    for debugging, monitoring, and audit trail purposes.
    """

    # document identification
    DocumentId: str = Field(default_factory=lambda: str(uuid4()), description="unique document identifier")
    Date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="pipeline execution timestamp")
    message_type: MessageType = Field(
        default=MessageType.FROM_DIRECT_MESSAGE,
        description="source type of the message",
        serialization_alias="MessageType"
    )

    # input content breakdown
    PureText: str = Field(default="", description="original text input without media")
    FinalTranscribedText: str = Field(
        default="",
        description="complete text after transcription (text + audio + image + video)"
    )

    # final output
    FinalResponseText: str = Field(
        default="",
        description="final formatted response text sent to user"
    )
    CommentAboutCompleteContext: str = Field(
        default="",
        description="overall summary considering all claims together"
    )

    # scraped links
    ScrapedLinks: list[ScrapedLink] = Field(
        default_factory=list,
        description="all scraped links with their content"
    )

    # audio media
    HadAudio: bool = Field(default=False, description="whether audio was present")
    AudioUrl: Optional[str] = Field(default=None, description="url to audio file if available")
    AudioText: Optional[str] = Field(default=None, description="transcribed text from audio")

    # image media
    HadImage: bool = Field(default=False, description="whether image was present")
    ImageUrl: Optional[str] = Field(default=None, description="url to image file if available")
    ImageText: Optional[str] = Field(default=None, description="OCR text from image")

    # video media
    HadVideo: bool = Field(default=False, description="whether video was present")
    VideoUrl: Optional[str] = Field(default=None, description="url to video file if available")
    VideoText: Optional[str] = Field(default=None, description="transcribed text from video")

    # pipeline step: claim extraction
    Claims: dict[str, ClaimAnalytics] = Field(
        default_factory=dict,
        description="extracted claims indexed by claim number (string keys: '1', '2', etc.)"
    )

    # pipeline step: adjudication
    ResponseByClaim: dict[str, ClaimResponseAnalytics] = Field(
        default_factory=dict,
        description="adjudication responses indexed by claim number (string keys: '1', '2', etc.)"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
