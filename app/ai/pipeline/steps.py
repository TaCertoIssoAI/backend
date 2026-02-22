"""
Pipeline Steps Interface and Default Implementation.

This module defines the interface for all pipeline steps and provides a default
implementation. This allows for easy testing, mocking, and customization of
individual pipeline steps without modifying the main pipeline orchestration.

Architecture:
- PipelineSteps: Protocol defining the interface for all steps
- DefaultPipelineSteps: Default implementation using the standard step functions
- Dependency injection pattern: main_pipeline receives a PipelineSteps instance
"""

from typing import Protocol, List
from app.models import (
    DataSource,
    PipelineConfig,
    ClaimExtractionInput,
    ClaimExtractionOutput,
    EvidenceRetrievalInput,
    EvidenceRetrievalResult,
    LLMConfig,
    DataSourceWithExtractedClaims,
    FactCheckResult,
    AdjudicationInput,
)
from app.ai.context import EvidenceGatherer
from app.ai.pipeline.no_claims_fallback import NoClaimsFallbackOutput

from app.agentic_ai.context.factcheckapi import (
    GoogleFactCheckGatherer
)

from app.config import get_trusted_domains


class PipelineSteps(Protocol):
    """
    Protocol defining the interface for all fact-checking pipeline steps.

    Each method corresponds to one step in the pipeline:
    1. Link expansion - expand URLs from original text
    2. Claim extraction - extract fact-checkable claims from text
    3. Evidence retrieval - gather supporting/refuting evidence for claims

    This protocol enables:
    - Easy testing with mock implementations
    - Custom implementations for specific use cases
    - Clear separation of concerns
    - Type-safe dependency injection
    """

    def expand_links_from_sources(
        self,
        sources: List[DataSource],
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        Expand links from data sources with enhanced logging.

        Used as a callback function in the fire-and-forget pipeline.
        Processes data sources, identifies those with type 'original_text',
        extracts URLs from them, and creates new 'link_context' data sources for each URL.

        Args:
            sources: List of data sources to expand links from
            config: Pipeline configuration with timeout settings

        Returns:
            List of new 'link_context' data sources created from expanding links
        """
        ...

    async def extract_claims_from_all_sources(
        self,
        data_sources: List[DataSource],
        llm_config: LLMConfig
    ) -> List[ClaimExtractionOutput]:
        """
        Extract claims from all data sources.

        Processes each data source, extracts fact-checkable claims using an LLM,
        and returns all extraction results.

        Args:
            data_sources: List of data sources to extract claims from
            llm_config: LLM configuration (model, temperature, timeout)

        Returns:
            List of ClaimExtractionOutput, one per data source
        """
        ...

    def get_evidence_gatherers(self) -> List[EvidenceGatherer]:
        """
        Get the list of evidence gatherers to use for evidence retrieval.

        Returns:
            List of EvidenceGatherer instances configured for this pipeline
        """
        ...

    async def gather_evidence(
        self,
        retrieval_input: EvidenceRetrievalInput,
        gatherers: List[EvidenceGatherer] | None = None,
        timeout: float = 45.0
    ) -> EvidenceRetrievalResult:
        """
        Gather evidence for claims from multiple sources.

        Runs each claim through evidence gatherers (web search, fact-check APIs, etc.)
        and accumulates citations.

        Args:
            retrieval_input: Input containing claims to gather evidence for
            gatherers: List of evidence gatherers. If None, uses defaults.
            timeout: Timeout in seconds for evidence gathering operations (default: 45.0)

        Returns:
            EvidenceRetrievalResult mapping claim IDs to enriched claims with citations
        """
        ...

    async def handle_no_claims_fallback(
        self,
        data_sources: List[DataSource],
        config: PipelineConfig
    ) -> NoClaimsFallbackOutput:
        """
        Generate friendly explanation when no claims are found.

        Uses an LLM to explain to the user why no verifiable claims could be
        extracted from their input.

        Args:
            data_sources: List of data sources that had no claims extracted
            config: Pipeline configuration with fallback LLM config

        Returns:
            NoClaimsFallbackOutput with explanation and original text
        """
        ...

    def adjudicate_claims(
        self,
        adjudication_input: AdjudicationInput,
        llm_config: LLMConfig
    ) -> FactCheckResult:
        """
        Adjudicate claims using traditional evidence-based adjudication.

        Uses pre-gathered evidence (citations) to make verdicts on claims.
        This is the standard adjudication method that analyzes enriched claims
        with their citations.

        Args:
            adjudication_input: Input with data sources and enriched claims
            llm_config: LLM configuration (model, temperature, timeout)

        Returns:
            FactCheckResult with verdicts for all claims
        """
        ...

    def adjudicate_claims_with_search(
        self,
        sources_with_claims: List[DataSourceWithExtractedClaims],
        model: str = "gpt-4o-mini"
    ) -> FactCheckResult:
        """
        Adjudicate claims using web search in a single API call.

        This is an alternative to the traditional evidence gathering + adjudication flow.
        Instead of pre-gathering evidence, this uses OpenAI's web search tool
        to find evidence and generate verdicts in one LLM call.

        Args:
            sources_with_claims: List of data sources with their extracted claims
            model: OpenAI model to use (default: gpt-4o-mini)

        Returns:
            FactCheckResult with verdicts for all claims
        """
        ...


class DefaultPipelineSteps:
    """
    Default implementation of PipelineSteps using the standard step functions.

    This implementation delegates to the actual step functions in their respective
    modules, providing a convenient way to inject the standard pipeline behavior.

    Example:
        >>> from app.ai.pipeline.steps import DefaultPipelineSteps
        >>> from app.ai.main_pipeline import run_fact_check_pipeline
        >>> steps = DefaultPipelineSteps()
        >>> result = await run_fact_check_pipeline(
        ...     data_sources=[...],
        ...     config=config,
        ...     steps=steps
        ... )
    """

    def get_evidence_gatherers(self) -> List[EvidenceGatherer]:
        """
        Get the default list of evidence gatherers.

        Returns a standard set of evidence gatherers with reasonable timeout values.

        Returns:
            List of EvidenceGatherer instances (WebSearchGatherer, GoogleFactCheckGatherer)
        """
        from app.ai.context.web import WebSearchGatherer
        allowed_domains = get_trusted_domains()
        return [
            GoogleFactCheckGatherer(timeout=15.0),
            WebSearchGatherer(max_results=5, timeout=15.0,allowed_domains=allowed_domains)
        ]

    def expand_links_from_sources(
        self,
        sources: List[DataSource],
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        wrapper for _expand_data_sources_with_links with enhanced logging.

        expands links from sources and returns new DataSource objects with detailed logging.
        used as callback in fire-and-forget pipeline.
        """
        from app.observability.logger import get_logger, PipelineStep

        link_logger = get_logger(__name__, PipelineStep.LINK_EXPANSION)

        link_logger.info(f"expand_links_from_sources called with {len(sources)} sources")
        link_logger.debug(f"source types: {[s.source_type for s in sources]}")

        # run link expansion (synchronous function using ThreadPoolManager internally)
        expanded_sources = self._expand_data_sources_with_links(sources, config)

        # ensure we always return a list
        if expanded_sources is None:
            link_logger.warning("link expansion returned None")
            return []

        link_logger.debug(f"expanded {len(expanded_sources)} link sources")

        for i, source in enumerate(expanded_sources, 1):
            url = source.metadata.get("url", "unknown") if source.metadata else "unknown"
            success = source.metadata.get("success", False) if source.metadata else False
            status = "✓" if success else "✗"
            content_preview = (
                source.original_text[:1000] if source.original_text else "(no content)"
            )

            link_logger.debug(f"{i}. {status} {source.source_type} (id: {source.id})")
            link_logger.debug(f"   URL: {url}")
            link_logger.debug(f"   content preview: {content_preview}...")

        return expanded_sources

    def _expand_data_sources_with_links(
        self,
        data_sources: List[DataSource],
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        Private method: processes all data sources and expands links.

        Iterates through data sources, identifies 'original_text' types,
        and expands their links to create new 'link_context' data sources.
        Returns only the new link_context sources, not the original sources.
        """
        from app.ai.pipeline.link_context_expander import expand_link_contexts
        from app.observability.logger import get_logger, PipelineStep

        link_logger = get_logger(__name__, PipelineStep.LINK_EXPANSION)

        link_logger.info(f"_expand_data_sources_with_links called with {len(data_sources)} sources")
        link_logger.debug(
            f"source types: {[s.source_type for s in data_sources]}"
        )

        expanded_link_sources: List[DataSource] = []

        for source in data_sources:
            if source.source_type == "original_text":
                text_preview = source.original_text[:100] if source.original_text else ""
                link_logger.info(f"processing original_text source: {source.id}")
                link_logger.debug(f"text preview: {text_preview}...")

                try:
                    # expand link contexts for this source
                    expanded_sources = expand_link_contexts(source, config)

                    # handle None return
                    if expanded_sources is None:
                        link_logger.warning("link expansion returned None")
                        continue

                    if expanded_sources:
                        link_logger.info(
                            f"created {len(expanded_sources)} new link_context data source(s)"
                        )
                        for expanded in expanded_sources:
                            url = expanded.metadata.get("url", "unknown")
                            success = expanded.metadata.get("success", False)
                            status = "✓" if success else "✗"
                            link_logger.debug(f"{status} {url}")

                        expanded_link_sources.extend(expanded_sources)
                    else:
                        link_logger.debug("no links found or expanded")

                except Exception as e:
                    link_logger.error(
                        f"link expansion failed for source {source.id}: {e}",
                        exc_info=True
                    )

        return expanded_link_sources


    async def extract_claims_from_all_sources(
        self,
        data_sources: List[DataSource],
        llm_config: LLMConfig
    ) -> List[ClaimExtractionOutput]:
        """
        Default implementation: processes each data source and extracts claims.

        Iterates through all data sources, creates ClaimExtractionInput for each,
        calls extract_claims, and returns all results.
        """
        claim_outputs: List[ClaimExtractionOutput] = []

        for source in data_sources:
            # create input for claim extractor
            extraction_input = ClaimExtractionInput(data_source=source)

            # extract claims using the single-source method
            result = await self.__extract_claims(
                extraction_input=extraction_input,
                llm_config=llm_config
            )

            claim_outputs.append(result)

        return claim_outputs

    async def __extract_claims(
        self,
        extraction_input: ClaimExtractionInput,
        llm_config: LLMConfig
    ) -> ClaimExtractionOutput:
        """
        Default implementation: calls extract_claims_async from claim_extractor.

        See claim_extractor.extract_claims_async for detailed documentation.
        """
        from app.ai.pipeline.claim_extractor import extract_claims_async
        return await extract_claims_async(extraction_input, llm_config)

    async def gather_evidence(
        self,
        retrieval_input: EvidenceRetrievalInput,
        gatherers: List[EvidenceGatherer] | None = None,
        timeout: float = 45.0
    ) -> EvidenceRetrievalResult:
        """
        Default implementation: calls gather_evidence_async from evidence_retrieval.

        Uses timeout from configuration to initialize gatherers with proper timeout values.

        See evidence_retrieval.gather_evidence_async for detailed documentation.
        """
        from app.ai.pipeline.evidence_retrieval import gather_evidence_async
        from app.ai.context.web import WebSearchGatherer

        # if no gatherers provided, create default gatherers with configured timeout
        if gatherers is None:
            gatherers = [
                GoogleFactCheckGatherer(timeout=timeout),
                WebSearchGatherer(max_results=5, timeout=timeout)
            ]

        return await gather_evidence_async(retrieval_input, gatherers)

    async def handle_no_claims_fallback(
        self,
        data_sources: List[DataSource],
        config: PipelineConfig
    ) -> NoClaimsFallbackOutput:
        """
        Default implementation: generates explanation when no claims are found.

        Combines text from all data sources and uses LLM to generate a friendly
        explanation for why no verifiable claims could be extracted.

        See no_claims_fallback.generate_no_claims_explanation_async for details.
        """
        from app.ai.pipeline.no_claims_fallback import (
            generate_no_claims_explanation_async,
            get_combined_text_from_sources
        )

        # combine text from all data sources
        combined_text = get_combined_text_from_sources(data_sources)

        # generate explanation using fallback from config
        return await generate_no_claims_explanation_async(combined_text, config)

    def adjudicate_claims(
        self,
        adjudication_input: AdjudicationInput,
        llm_config: LLMConfig
    ) -> FactCheckResult:
        """
        Default implementation: calls adjudicate_claims from judgement.py.

        Uses the standard evidence-based adjudication with pre-gathered citations.

        See judgement.adjudicate_claims for detailed documentation.
        """
        from app.ai.pipeline.judgement import adjudicate_claims

        return adjudicate_claims(
            adjudication_input=adjudication_input,
            llm_config=llm_config
        )

    def adjudicate_claims_with_search(
        self,
        sources_with_claims: List[DataSourceWithExtractedClaims],
        model: str = "gpt-4o-mini"
    ) -> FactCheckResult:
        """
        Default implementation: calls adjudicate_claims_with_search from adjudication_with_search.

        Uses OpenAI web search to find evidence and generate verdicts in a single LLM call.

        See adjudication_with_search.adjudicate_claims_with_search for detailed documentation.
        """
        from app.ai.pipeline.adjudication_with_search import adjudicate_claims_with_search

        return adjudicate_claims_with_search(
            sources_with_claims=sources_with_claims,
            model=model
        )
