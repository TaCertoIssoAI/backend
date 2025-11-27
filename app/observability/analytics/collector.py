"""
analytics collector for tracking pipeline execution.

provides helper methods to populate PipelineAnalytics from pipeline step outputs.
"""

import time
from typing import List, Optional
from datetime import datetime

from app.models.analytics import (
    PipelineAnalytics,
    ClaimAnalytics,
    ClaimResponseAnalytics,
    MessageType
)
from app.models import (
    DataSource,
    ClaimExtractionOutput,
    FactCheckResult,
    DataSourceResult
)


class AnalyticsCollector:
    """
    collects analytics data throughout pipeline execution.

    this is a per-request instance that accumulates data as the pipeline progresses.
    NOT a singleton - create one instance per fact-check request.

    example:
        >>> from app.observability.analytics import AnalyticsCollector
        >>> analytics = AnalyticsCollector()
        >>> analytics.set_input_text("Check this claim about vaccines")
        >>> analytics.add_link("https://example.com")
        >>> # ... pipeline runs ...
        >>> analytics.add_claims_from_extraction(claim_outputs)
        >>> analytics.add_verdicts_from_adjudication(fact_check_result)
        >>> data = analytics.get_analytics()
        >>> print(data.model_dump_json(indent=2))
    """

    def __init__(self, message_type: MessageType = MessageType.FROM_DIRECT_MESSAGE):
        """
        initialize analytics collector.

        args:
            message_type: type of message source
        """
        self.analytics = PipelineAnalytics(message_type=message_type)
        self.start_time = time.time()

    # ===== INPUT POPULATION =====

    def set_input_text(self, text: str):
        """set the original pure text input."""
        self.analytics.pure_text = text
        self.analytics.final_transcribed_text = text

    def set_audio_text(self, text: str):
        """add transcribed audio text."""
        self.analytics.had_audio = True
        self.analytics.audio_text = text
        self.analytics.final_transcribed_text += f"\n[AUDIO]: {text}"

    def set_image_text(self, text: str):
        """add transcribed text from image."""
        self.analytics.had_image = True
        self.analytics.image_text = text
        self.analytics.final_transcribed_text += f"\n[IMAGE]: {text}"

    def set_video_text(self, text: str):
        """add transcribed video text."""
        self.analytics.had_video = True
        self.analytics.video_text = text
        self.analytics.final_transcribed_text += f"\n[VIDEO]: {text}"

    def add_link(self, link: str):
        """add a discovered link."""
        if link not in self.analytics.links:
            self.analytics.links.append(link)

    def add_links(self, links: List[str]):
        """add multiple links."""
        for link in links:
            self.add_link(link)

    def add_topic(self, topic: str):
        """add a detected topic/category."""
        if topic not in self.analytics.topics:
            self.analytics.topics.append(topic)

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
                self.analytics.claims[claim_index] = ClaimAnalytics(
                    text=claim.text,
                    links=[],  # could extract links from claim text if needed
                    extraction_source=output.data_source.id  # fixed: use data_source.id
                )
                claim_index += 1

        self.analytics.steps_completed.append("claim_extraction")

    def add_claim(self, claim_text: str, source: Optional[str] = None):
        """
        manually add a single claim.

        args:
            claim_text: the claim text
            source: optional source identifier
        """
        claim_index = len(self.analytics.claims) + 1
        self.analytics.claims[claim_index] = ClaimAnalytics(
            text=claim_text,
            links=[],
            extraction_source=source
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

                self.analytics.response_by_claim[verdict_index] = ClaimResponseAnalytics(
                    result=claim_verdict.verdict,
                    reasoning_text=claim_verdict.justification,
                    reasoning_sources=sources,
                    confidence=None  # add if available in future
                )
                verdict_index += 1

        # set overall summary
        if fact_check_result.overall_summary:
            self.analytics.comment_about_complete_context = fact_check_result.overall_summary

        self.analytics.steps_completed.append("adjudication")

    def set_final_response(self, response_text: str):
        """
        set the final formatted response text.

        args:
            response_text: complete response sent to user
        """
        self.analytics.final_response_text = response_text

    # ===== EXECUTION METADATA =====

    def mark_step_completed(self, step_name: str):
        """mark a pipeline step as completed."""
        if step_name not in self.analytics.steps_completed:
            self.analytics.steps_completed.append(step_name)

    def add_error(self, error_message: str):
        """record an error encountered during execution."""
        self.analytics.errors_encountered.append(error_message)

    def finalize(self):
        """finalize analytics collection (call at end of pipeline)."""
        elapsed = time.time() - self.start_time
        self.analytics.execution_time_ms = elapsed * 1000
        self.analytics.date = datetime.utcnow()

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
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return list(set(urls))  # deduplicate
