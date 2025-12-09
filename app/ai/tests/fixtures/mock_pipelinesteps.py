"""
Mock pipeline steps implementations for testing.

provides alternative implementations of PipelineSteps that avoid expensive operations
like web browsing, making tests faster and more predictable.
"""

from typing import List

from app.models import (
    DataSource,
    PipelineConfig,
    DataSourceWithExtractedClaims,
    FactCheckResult,
)
from app.ai.context import EvidenceGatherer
from app.ai.context.factcheckapi import GoogleFactCheckGatherer
from app.ai.pipeline.steps import DefaultPipelineSteps
from app.ai.context.web import WebSearchGatherer
from app.ai.pipeline.tests.fixtures.mock_linkexpander import hybrid_expand_link_contexts
from app.config import get_trusted_domains


class WithoutBrowsingPipelineSteps(DefaultPipelineSteps):
    """
    hybrid pipeline steps implementation that minimizes expensive operations.

    mocks social media URLs (Facebook, Instagram, Twitter, TikTok) to avoid Apify API calls,
    while allowing real simple HTTP scraping for generic URLs. only uses GoogleFactCheckGatherer
    for evidence gathering (no web search).

    ideal for:
    - fast unit tests
    - offline development (with some limitations for generic URLs)
    - predictable test results for social media content
    - avoiding Apify API rate limits and costs
    - testing with real HTTP scraping for generic websites

    example:
        >>> from app.ai.tests.fixtures.mock_pipelinesteps import WithoutBrowsingPipelineSteps
        >>> from app.ai.main_pipeline import run_fact_check_pipeline
        >>> steps = WithoutBrowsingPipelineSteps()
        >>> result = await run_fact_check_pipeline(sources, config, steps)
    """

    def get_evidence_gatherers(self) -> List[EvidenceGatherer]:
        """
        get evidence gatherers for the pipeline.

        returns only GoogleFactCheckGatherer to avoid web browsing.

        returns:
            list with only GoogleFactCheckGatherer (no WebSearchGatherer)
        """
        allowed_domains = get_trusted_domains()
        return [
            GoogleFactCheckGatherer(timeout=15.0),WebSearchGatherer(max_results=5, timeout=15.0,allowed_domains=allowed_domains)
        ]

    def _expand_data_sources_with_links(
        self,
        data_sources: List[DataSource],
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        hybrid implementation: mocks social media URLs, uses real scraping for generic URLs.

        social media URLs (Facebook, Instagram, Twitter, TikTok) are mocked to avoid
        Apify API calls. generic URLs use real simple HTTP scraping.

        args:
            data_sources: list of data sources to process
            config: pipeline configuration

        returns:
            list of new 'link_context' data sources (mocked + real)
        """
        expanded_link_sources: List[DataSource] = []

        for source in data_sources:
            if source.source_type == "original_text":
                print(f"\n[HYBRID LINK EXPANSION] Processing original_text source: {source.id}")
                print(f"  Text preview: {source.original_text[:100]}...")

                try:
                    # use hybrid link expander (mocks social media, real scraping for generic)
                    expanded_sources = hybrid_expand_link_contexts(source, config)

                    # handle None return
                    if expanded_sources is None:
                        print("  Warning: hybrid link expansion returned None")
                        continue

                    if expanded_sources:
                        print(f"  Created {len(expanded_sources)} link_context data source(s):")
                        for expanded in expanded_sources:
                            url = expanded.metadata.get("url", "unknown")
                            success = expanded.metadata.get("success", False)
                            is_mock = expanded.metadata.get("mock", False)
                            status = "✓" if success else "✗"
                            source_type = "[MOCK]" if is_mock else "[REAL]"
                            print(f"    {status} {source_type} {url}")

                        expanded_link_sources.extend(expanded_sources)
                    else:
                        print("  No links found or expanded")

                except Exception as e:
                    print(f"  Error in hybrid link expansion for source {source.id}: {e}")
                    import logging
                    logging.getLogger(__name__).error(
                        f"Hybrid link expansion failed for source {source.id}: {e}",
                        exc_info=True
                    )

        return expanded_link_sources

    def adjudicate_claims_with_search(
        self,
        sources_with_claims: List[DataSourceWithExtractedClaims],
        model: str = "gpt-4o-mini"
    ) -> FactCheckResult:
        """
        Implementation using OpenAI web search for adjudication.

        This works the same as DefaultPipelineSteps since it doesn't use browser-based
        scraping - it relies on OpenAI's web search tool.

        Args:
            sources_with_claims: List of data sources with their extracted claims
            model: OpenAI model to use (default: gpt-4o-mini)

        Returns:
            FactCheckResult with verdicts for all claims
        """
        from app.ai.pipeline.adjudication_with_search import adjudicate_claims_with_search

        return adjudicate_claims_with_search(
            sources_with_claims=sources_with_claims,
            model=model
        )

    # note: all other methods (extract_claims_from_all_sources, gather_evidence,
    # handle_no_claims_fallback) are inherited from DefaultPipelineSteps and work as normal
