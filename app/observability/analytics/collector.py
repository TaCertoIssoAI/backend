"""
analytics collector for tracking pipeline execution.

provides helper methods to populate PipelineAnalytics from pipeline step outputs.
"""

import re
import time
import uuid
from typing import List, Optional
from datetime import date, datetime, timezone, timedelta

from app.models.analytics import (
    PipelineAnalytics,
    ClaimAnalytics,
    ClaimResponseAnalytics,
    CitationAnalytics,
    ScrapedLink,
    MessageType,
    DataSourceResponseAnalytics
)
from app.models import (
    ClaimExtractionOutput,
    DataSource,
    FactCheckResult,
    EvidenceRetrievalResult
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
        >>> # ... pipeline runs ...
        >>> # populate claims with enriched evidence
        >>> analytics.populate_claims_from_evidence(evidence_result)
        >>> # populate verdicts and overall summary
        >>> analytics.populate_from_fact_check_result(fact_check_result)
        >>> analytics.finalize()
        >>> data = analytics.get_analytics()
        >>> print(data.model_dump_json(indent=2))
    """

    def __init__(self, msg_id: str, message_type: MessageType = MessageType.FROM_DIRECT_MESSAGE):
        """
        initialize analytics collector.

        args:
            message_type: type of message source
        """
        tz = timezone(timedelta(hours=-3))  # GMT-3
        now_str = datetime.now(tz).isoformat(timespec="seconds")
        self.analytics = PipelineAnalytics(DocumentId=msg_id, message_type=message_type, Date=now_str)
        self.start_time = time.time()

      
        self.analytics.Date = now_str

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

    def add_claim(self, claim_text: str, links: Optional[List[CitationAnalytics]] = None):
        """
        manually add a single claim.

        args:
            claim_text: the claim text
            links: optional list of citation analytics objects
        """
        claim_index = len(self.analytics.Claims) + 1
        self.analytics.Claims[str(claim_index)] = ClaimAnalytics(
            text=claim_text,
            links=links or []
        )

    def populate_claims_from_evidence(self, evidence_result: EvidenceRetrievalResult):
        """
        populate claims from evidence retrieval result.

        extracts enriched claims with their evidence (citations) and populates
        the Claims dict with claim text and full citation details.

        args:
            evidence_result: EvidenceRetrievalResult with enriched claims and citations
        """
        for _claim_id, enriched_claim in evidence_result.claim_evidence_map.items():
            # convert Citations to CitationAnalytics
            citation_analytics = [
                CitationAnalytics(
                    url=citation.url,
                    title=citation.title,
                    publisher=citation.publisher,
                    citation_text=citation.citation_text
                )
                for citation in enriched_claim.citations
            ]

            self.analytics.Claims[enriched_claim.id] = ClaimAnalytics(
                text=enriched_claim.text,
                links=citation_analytics
            )

    # ===== ADJUDICATION =====

    def populate_from_fact_check_result(self, fact_check_result: FactCheckResult):
        """
        populate analytics from complete fact check result.

        extracts verdicts for all claims across all data sources and populates
        the ResponseByClaim dict with verdict, reasoning, and full citation details.
        Also sets the overall summary.

        args:
            fact_check_result: FactCheckResult with verdicts and overall summary
        """
        verdict_index = 1

        # iterate through all data source results
        for data_source_result in fact_check_result.results:
            # iterate through all claim verdicts in this data source
            for claim_verdict in data_source_result.claim_verdicts:
                # convert Citations to CitationAnalytics with full details
                reasoning_sources = [
                    CitationAnalytics(
                        url=citation.url,
                        title=citation.title,
                        publisher=citation.publisher,
                        citation_text=citation.citation_text
                    )
                    for citation in claim_verdict.citations_used
                ]

                # add to ResponseByClaim dict with all fields
                self.analytics.ResponseByClaim[str(verdict_index)] = ClaimResponseAnalytics(
                    claim_id=claim_verdict.claim_id,
                    claim_text=claim_verdict.claim_text,
                    Result=claim_verdict.verdict,
                    reasoningText=claim_verdict.justification,
                    reasoningSources=reasoning_sources
                )
                verdict_index += 1

        # set overall summary
        if fact_check_result.overall_summary:
            self.analytics.CommentAboutCompleteContext = fact_check_result.overall_summary

    def populate_from_adjudication(self, fact_check_result: FactCheckResult):
        """
        populate verdicts from adjudication result.

        args:
            fact_check_result: FactCheckResult from adjudication step
        """
        #aggregate responses for all data sources
        responses_for_data_sources:list[DataSourceResponseAnalytics] = []

        for data_source_result in fact_check_result.results:
            responses_by_claim:list[ClaimResponseAnalytics] = []

            #for each data source, collect the claim veredicts
            for claim_verdict in data_source_result.claim_verdicts:
                # convert Citations to CitationAnalytics from citations_used field
                reasoning_sources = [
                    CitationAnalytics(
                        url=citation.url,
                        title=citation.title,
                        publisher=citation.publisher,
                        citation_text=citation.citation_text
                    )
                    for citation in claim_verdict.citations_used
                ]

                claim_response = ClaimResponseAnalytics(
                    claim_id=claim_verdict.claim_id,
                    claim_text=claim_verdict.claim_text,
                    Result=claim_verdict.verdict,
                    reasoningText=claim_verdict.justification,
                    reasoningSources=reasoning_sources,
                )
                responses_by_claim.append(claim_response)
            
            data_source = DataSourceResponseAnalytics(
                data_source_id=data_source_result.data_source_id,
                data_source_type=data_source_result.source_type,
                claim_verdicts= responses_by_claim,
            )
            responses_for_data_sources.append(data_source)

        
        self.analytics.ResponseByDataSource = responses_for_data_sources
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

    def has_extracted_claims(self) -> bool:
        """
        check if any claims were extracted during the pipeline.

        returns:
            True if at least one claim exists in Claims dict or ResponseByDataSource, False otherwise
        """
        has_claims_dict = len(self.analytics.Claims) > 0
        has_response_data = len(self.analytics.ResponseByDataSource) > 0
        return has_claims_dict or has_response_data

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        extract URLs from text.

        basic implementation - could be enhanced with regex.
        """
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return list(set(urls))  # deduplicate
