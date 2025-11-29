"""
analytics collector for tracking pipeline execution.

provides helper methods to populate PipelineAnalytics from pipeline step outputs.
"""

import re
import time
import uuid
from typing import List, Optional
from datetime import datetime, timezone

from app.models.analytics import (
    PipelineAnalytics,
    ClaimAnalytics,
    ClaimResponseAnalytics,
    ScrapedLink,
    MessageType
)
from app.models import (
    ClaimExtractionOutput,
    DataSource,
    FactCheckResult
)


class AnalyticsCollector:
    """
    collects analytics data throughout pipeline execution.

    this is a per-request instance that accumulates data as the pipeline progresses.
    NOT a singleton - create one instance per fact-check request.

    example:
        >>> from app.observability.analytics import AnalyticsCollector
        >>> analytics = AnalyticsCollector(msg_id)
        >>> # populate from data sources (recommended)
        >>> analytics.populate_from_data_sources(data_sources)
        >>> # OR set fields manually
        >>> analytics.set_input_text("Check this claim about vaccines")
        >>> analytics.add_scraped_link("https://example.com", success=True, text="content")
        >>> # ... pipeline runs ...
        >>> analytics.add_claims_from_extraction(claim_outputs)
        >>> analytics.add_verdicts_from_adjudication(fact_check_result)
        >>> analytics.finalize()
        >>> data = analytics.get_analytics()
        >>> print(data.model_dump_json(indent=2))
    """

    id: uuid.UUID #id for the anaytics message

    def __init__(self, msg_id: uuid.UUID, message_type: MessageType = MessageType.FROM_DIRECT_MESSAGE):
        """
        initialize analytics collector.

        args:
            message_type: type of message source
        """
        self.analytics = PipelineAnalytics(MessageType=message_type)
        self.start_time = time.time()
        self.id = msg_id
        self.analytics.Date = datetime.now(timezone.utc)

    # ===== INPUT POPULATION =====

    def populate_from_data_sources(self, data_sources: List[DataSource]) -> None:
        """
        populate analytics fields from a list of DataSource objects.

        automatically extracts and categorizes content by source type,
        populating PureText, media fields, and scraped links.

        args:
            data_sources: list of DataSource objects from the pipeline
        """

        text_parts = []

        for ds in data_sources:
            if ds.source_type == "original_text":
                # this is the pure user text
                self.analytics.PureText = ds.original_text
                text_parts.append(ds.original_text)

            elif ds.source_type == "link_context":
                # scraped link content
                url = ds.metadata.get("url", "")
                success = ds.metadata.get("success", False)
                self.add_scraped_link(
                    url=url,
                    success=success,
                    text=ds.original_text
                )
                text_parts.append(f"[LINK {url}]: {ds.original_text}")

            elif ds.source_type == "image":
                # image OCR content
                self.analytics.HadImage = True
                self.analytics.ImageText = ds.original_text
                self.analytics.ImageUrl = ds.metadata.get("url") or ds.metadata.get("image_url")
                text_parts.append(f"[IMAGE]: {ds.original_text}")

            elif ds.source_type == "audio_transcript":
                # audio transcription
                self.analytics.HadAudio = True
                self.analytics.AudioText = ds.original_text
                self.analytics.AudioUrl = ds.metadata.get("url") or ds.metadata.get("audio_url")
                text_parts.append(f"[AUDIO]: {ds.original_text}")

            elif ds.source_type == "video_transcript":
                # video transcription
                self.analytics.HadVideo = True
                self.analytics.VideoText = ds.original_text
                self.analytics.VideoUrl = ds.metadata.get("url") or ds.metadata.get("video_url")
                text_parts.append(f"[VIDEO]: {ds.original_text}")

        # build the final transcribed text from all parts
        self.analytics.FinalTranscribedText += "\n".join(text_parts)

    def set_input_text(self, text: str):
        """set the original pure text input."""
        self.analytics.PureText = text
        self.analytics.FinalTranscribedText = text

    def set_audio_text(self, text: str, url: Optional[str] = None):
        """add transcribed audio text."""
        self.analytics.HadAudio = True
        self.analytics.AudioText = text
        self.analytics.AudioUrl = url
        self.analytics.FinalTranscribedText += f"\n[AUDIO]: {text}"

    def set_image_text(self, text: str, url: Optional[str] = None):
        """add transcribed text from image."""
        self.analytics.HadImage = True
        self.analytics.ImageText = text
        self.analytics.ImageUrl = url
        self.analytics.FinalTranscribedText += f"\n[IMAGE]: {text}"

    def set_video_text(self, text: str, url: Optional[str] = None):
        """add transcribed video text."""
        self.analytics.HadVideo = True
        self.analytics.VideoText = text
        self.analytics.VideoUrl = url
        self.analytics.FinalTranscribedText += f"\n[VIDEO]: {text}"

    def add_scraped_link(self, url: str, success: bool = False, text: Optional[str] = None):
        """add a scraped link with its content."""
        # check if link already exists
        existing_urls = [link.url for link in self.analytics.ScrapedLinks]
        if url not in existing_urls:
            self.analytics.ScrapedLinks.append(
                ScrapedLink(url=url, success=success, text=text)
            )

    def add_scraped_links(self, links: List[ScrapedLink]):
        """add multiple scraped links."""
        for link in links:
            self.add_scraped_link(link.url, link.success, link.text)

    # ===== CLAIM EXTRACTION =====

    def add_claims_from_extraction(self, claim_outputs: List[ClaimExtractionOutput]):
        """
        populate claims from claim extraction outputs.

        args:
            claim_outputs: list of ClaimExtractionOutput from the pipeline
        """
        claim_index = 1
        for output in claim_outputs:
            for claim in output.claims:
                self.analytics.Claims[str(claim_index)] = ClaimAnalytics(
                    text=claim.text,
                    links=[]  # could extract links from claim text if needed
                )
                claim_index += 1

    def add_claim(self, claim_text: str, links: Optional[List[str]] = None):
        """
        manually add a single claim.

        args:
            claim_text: the claim text
            links: optional list of links in the claim
        """
        claim_index = len(self.analytics.Claims) + 1
        self.analytics.Claims[str(claim_index)] = ClaimAnalytics(
            text=claim_text,
            links=links or []
        )

    # ===== ADJUDICATION =====

    def add_verdicts_from_adjudication(self, fact_check_result: FactCheckResult):
        """
        populate verdicts from adjudication result.

        args:
            fact_check_result: FactCheckResult from adjudication step
        """
        verdict_index = 1

        for result in fact_check_result.results:
            for claim_verdict in result.claim_verdicts:
                # extract source URLs from justification if needed
                sources = self._extract_urls_from_text(claim_verdict.justification)

                self.analytics.ResponseByClaim[str(verdict_index)] = ClaimResponseAnalytics(
                    Result=claim_verdict.verdict,
                    reasoningText=claim_verdict.justification,
                    reasoningSources=sources
                )
                verdict_index += 1

        # set overall summary
        if fact_check_result.overall_summary:
            self.analytics.CommentAboutCompleteContext = fact_check_result.overall_summary

    def set_final_response(self, response_text: str):
        """
        set the final formatted response text.

        args:
            response_text: complete response sent to user
        """
        self.analytics.FinalResponseText = response_text

    # ===== RETRIEVAL =====

    def get_analytics(self) -> PipelineAnalytics:
        """
        get the complete analytics object.

        returns:
            PipelineAnalytics with all collected data
        """
        return self.analytics

    def to_dict(self) -> dict:
        """get analytics as dictionary."""
        return self.analytics.model_dump()

    def to_json(self, **kwargs) -> str:
        """
        get analytics as JSON string.

        args:
            **kwargs: additional arguments passed to model_dump_json
        """
        return self.analytics.model_dump_json(**kwargs)

    # ===== HELPER METHODS =====

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        extract URLs from text.

        basic implementation - could be enhanced with regex.
        """
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return list(set(urls))  # deduplicate
