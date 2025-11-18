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
)
from app.ai.context import EvidenceGatherer

from app.ai.context.factcheckapi import (
    GoogleFactCheckGatherer
)


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

    def expand_data_sources_with_links(
        self,
        data_sources: List[DataSource],
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        Expand links from original_text data sources into new link_context data sources.

        Processes a list of data sources, identifies those with type 'original_text',
        extracts URLs from them, and creates new 'link_context' data sources for each URL.

        Args:
            data_sources: List of input data sources to process
            config: Pipeline configuration with timeout settings

        Returns:
            List of new 'link_context' data sources created from expanding links.
            Does NOT include the original input data sources.
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


    def expand_data_sources_with_links(
        self,
        data_sources: List[DataSource],
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        Default implementation: processes all data sources and expands links.

        Iterates through data sources, identifies 'original_text' types,
        and expands their links to create new 'link_context' data sources.
        Returns only the new link_context sources, not the original sources.
        """
        expanded_link_sources: List[DataSource] = []

        for source in data_sources:
            if source.source_type == "original_text":
                print(f"\n[LINK EXPANSION] Processing original_text source: {source.id}")
                print(f"  Text preview: {source.original_text[:100]}...")

                try:
                    # expand link contexts for this source
                    expanded_sources = self._expand_link_contexts(source, config)

                    # handle None return
                    if expanded_sources is None:
                        print("  Warning: link expansion returned None")
                        continue

                    if expanded_sources:
                        print(f"  Created {len(expanded_sources)} new link_context data source(s):")
                        for expanded in expanded_sources:
                            url = expanded.metadata.get("url", "unknown")
                            success = expanded.metadata.get("success", False)
                            status = "✓" if success else "✗"
                            print(f"    {status} {url}")

                        expanded_link_sources.extend(expanded_sources)
                    else:
                        print("  No links found or expanded")

                except Exception as e:
                    print(f"  Error expanding links for source {source.id}: {e}")
                    import logging
                    logging.getLogger(__name__).error(
                        f"Link expansion failed for source {source.id}: {e}",
                        exc_info=True
                    )

        return expanded_link_sources

    def _expand_link_contexts(
        self,
        data_source: DataSource,
        config: PipelineConfig
    ) -> List[DataSource]:
        """
        Default implementation: calls expand_link_contexts from link_context_expander.

        See link_context_expander.expand_link_contexts for detailed documentation.
        """
        from app.ai.pipeline.link_context_expander import (
            expand_link_contexts as _expand_link_contexts
        )
        return _expand_link_contexts(data_source, config)

    async def extract_claims_from_all_sources(
        self,
        data_sources: List[DataSource],
        llm_config: LLMConfig
    ) -> List[ClaimExtractionOutput]:
        """
        Default implementation: processes each data source and extracts claims.

        Iterates through all data sources, creates ClaimExtractionInput for each,
        calls extract_claims, and returns all results with logging.
        """
        claim_outputs: List[ClaimExtractionOutput] = []

        for source in data_sources:
            print(f"\n[CLAIM EXTRACTION] Processing {source.source_type} source: {source.id}")
            print(f"  Text preview: {source.original_text[:100]}...")

            # create input for claim extractor
            extraction_input = ClaimExtractionInput(data_source=source)

            # extract claims using the single-source method
            result = await self.__extract_claims(
                extraction_input=extraction_input,
                llm_config=llm_config
            )

            claim_outputs.append(result)

            # print extracted claims
            if result.claims:
                print(f"  Extracted {len(result.claims)} claim(s):")
                for i, claim in enumerate(result.claims, 1):
                    print(f"    {i}. {claim.text}")
                    if claim.entities:
                        print(f"       entities: {', '.join(claim.entities)}")
                    if claim.llm_comment:
                        print(f"       comment: {claim.llm_comment}")
            else:
                print("  No claims extracted")

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
