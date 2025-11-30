"""
analytics data models for pipeline execution tracking.

provides comprehensive observability of the fact-checking pipeline,
capturing inputs, intermediate steps, and final outputs.
"""

from typing import Optional,List
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


class CitationAnalytics(BaseModel):
    """citation data for claim analytics."""
    url: str = Field(..., description="url of the citation")
    title: str = Field(..., description="title of the cited source")
    publisher: str = Field(..., description="publisher of the cited source")
    citation_text: str = Field(..., description="relevant text excerpt from the citation")


class ClaimAnalytics(BaseModel):
    """analytics for a single extracted claim."""
    text: str = Field(..., description="the claim text")
    links: list[CitationAnalytics] = Field(default_factory=list, description="citations for the claim")


class ClaimResponseAnalytics(BaseModel):
    """analytics for adjudication response for a single claim."""
    claim_id: str = Field(..., description="ID of the claim being judged")
    claim_text: str = Field(..., description="text of the claim")
    Result: str = Field(..., description="verdict: Fake, True, Misleading, Unverifiable")
    reasoningText: str = Field(..., description="detailed reasoning for the verdict")
    reasoningSources: list[CitationAnalytics] = Field(
        default_factory=list,
        description="full citation details of sources used in reasoning"
    )

class DataSourceResponseAnalytics(BaseModel):
    """Analytics for the Adjundication output separated by Data source"""
    data_source_id: str = Field(...,description="id of the data source")
    data_source_type: str = Field(...,description="data source type")
    claim_verdicts: List[ClaimResponseAnalytics] = Field(
        default_factory=list,
        description="Verdicts for all claims extracted from this source"
    )

class PipelineAnalytics(BaseModel):
    """
    comprehensive analytics for a single pipeline execution.

    captures all inputs, intermediate results, and final outputs
    for debugging, monitoring, and audit trail purposes.
    """

    # document identification
    DocumentId: str = Field(default_factory=lambda: str(uuid4()), description="unique document identifier")
    Date: str = Field(..., description="pipeline execution timestamp")
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
    ResponseByDataSource: list[DataSourceResponseAnalytics] = Field(default_factory=list,description="Judgment response by Data Source")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
