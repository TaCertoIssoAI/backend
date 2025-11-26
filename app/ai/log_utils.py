"""
Logging utilities for the fact-checking pipeline.

provides structured logging functions for complex pipeline steps,
particularly for adjudication input and output.
"""

from typing import TYPE_CHECKING

from app.observability.logger import get_logger, PipelineStep

if TYPE_CHECKING:
    from app.models import AdjudicationInput, FactCheckResult, LLMConfig


def log_adjudication_input(
    adjudication_input: "AdjudicationInput",
    llm_config: "LLMConfig"
) -> None:
    """
    log detailed information about adjudication input.

    logs the structure of the adjudication input including data sources,
    enriched claims, citations, and LLM configuration. useful for debugging
    and understanding what data is being passed to the adjudication step.

    args:
        adjudication_input: the input to the adjudication step
        llm_config: LLM configuration for adjudication
    """
    logger = get_logger(__name__, PipelineStep.ADJUDICATION)

    logger.info("=" * 80)
    logger.info("building adjudication input")
    logger.info("=" * 80)

    logger.info(
        f"adjudication input created successfully: "
        f"{len(adjudication_input.sources_with_claims)} data sources"
    )

    # log summary of each data source with claims
    for i, ds_with_claims in enumerate(adjudication_input.sources_with_claims, 1):
        ds = ds_with_claims.data_source
        claims = ds_with_claims.enriched_claims

        logger.info(
            f"{i}. data source: {ds.id} ({ds.source_type}) "
            f"with {len(claims)} enriched claims"
        )

        # log details of each claim (debug level)
        for j, claim in enumerate(claims, 1):
            citations_count = len(claim.citations)
            text_preview = claim.text[:80] if len(claim.text) > 80 else claim.text

            logger.debug(f"  {j}) claim ID: {claim.id}")
            logger.debug(f"     text: {text_preview}...")
            logger.debug(f"     citations: {citations_count}")
            logger.debug(
                f"     source: {claim.source.source_type} ({claim.source.source_id})"
            )

    # debug validation logging
    logger.debug("adjudication input validation:")
    logger.debug(
        f"  sources_with_claims type: {type(adjudication_input.sources_with_claims)}"
    )
    logger.debug(
        f"  sources_with_claims length: {len(adjudication_input.sources_with_claims)}"
    )

    # detailed debug logging for first source
    for i, swc in enumerate(adjudication_input.sources_with_claims):
        logger.debug(f"source {i+1} detailed inspection:")
        logger.debug(f"  data_source.id: {swc.data_source.id}")
        logger.debug(f"  data_source.source_type: {swc.data_source.source_type}")
        logger.debug(f"  enriched_claims length: {len(swc.enriched_claims)}")

        if swc.enriched_claims:
            first_claim = swc.enriched_claims[0]
            logger.debug(f"  first claim ID: {first_claim.id}")
            logger.debug(f"  first claim has 'text' attr: {hasattr(first_claim, 'text')}")
            logger.debug(
                f"  first claim has 'citations' attr: {hasattr(first_claim, 'citations')}"
            )

            if hasattr(first_claim, 'text'):
                text_preview = first_claim.text[:50] if len(first_claim.text) > 50 else first_claim.text
                logger.debug(f"  first claim text preview: {text_preview}...")

            if hasattr(first_claim, 'citations'):
                logger.debug(f"  first claim citations count: {len(first_claim.citations)}")
                if first_claim.citations:
                    first_citation = first_claim.citations[0]
                    logger.debug(f"  first citation type: {type(first_citation)}")
                    logger.debug(
                        f"  first citation has 'citation_text': "
                        f"{hasattr(first_citation, 'citation_text')}"
                    )
                    logger.debug(
                        f"  first citation has 'date': "
                        f"{hasattr(first_citation, 'date')}"
                    )

    # log LLM configuration
    logger.debug("LLM config:")

    # handle both ChatOpenAI (has model_name) and AzureChatOpenAI (has azure_deployment)
    model_identifier = getattr(llm_config.llm, 'model_name', None) or \
                       getattr(llm_config.llm, 'azure_deployment', 'unknown') or \
                       getattr(llm_config.llm, 'model', 'unknown')

    logger.debug(f"  model: {model_identifier}")

    # temperature might be None for o3 models
    temperature = getattr(llm_config.llm, 'temperature', 'N/A')
    logger.debug(f"  temperature: {temperature}")

    # timeout should always be present
    timeout = getattr(llm_config.llm, 'timeout', 'N/A')
    logger.debug(f"  timeout: {timeout}")

    logger.info("=" * 80)
    logger.info("adjudication - making final verdicts")
    logger.info("=" * 80)


def log_adjudication_output(fact_check_result: "FactCheckResult") -> None:
    """
    log detailed information about adjudication output.

    logs the results of the adjudication step including verdicts,
    justifications, and overall summary. useful for understanding
    the final decisions made by the adjudication LLM.

    args:
        fact_check_result: the output from the adjudication step
    """
    logger = get_logger(__name__, PipelineStep.ADJUDICATION)

    logger.info(
        f"adjudication completed: "
        f"{len(fact_check_result.results)} data source results"
    )

    # log summary of each data source result
    for i, ds_result in enumerate(fact_check_result.results, 1):
        logger.info(
            f"{i}. data source: {ds_result.data_source_id} ({ds_result.source_type}) "
            f"with {len(ds_result.claim_verdicts)} verdicts"
        )

        # log details of each verdict (debug level)
        for j, verdict in enumerate(ds_result.claim_verdicts, 1):
            claim_preview = (
                verdict.claim_text[:60] if len(verdict.claim_text) > 60
                else verdict.claim_text
            )
            justification_preview = (
                verdict.justification[:100] if len(verdict.justification) > 100
                else verdict.justification
            )

            logger.debug(f"  {j}) claim: {claim_preview}...")
            logger.debug(f"     verdict: {verdict.verdict}")
            logger.debug(f"     justification: {justification_preview}...")

    # log overall summary if present
    if fact_check_result.overall_summary:
        logger.info(f"overall summary: {fact_check_result.overall_summary}")
