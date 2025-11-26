"""
Test endpoints for development and debugging.

provides alternative endpoints with different configurations for testing purposes.
"""

from fastapi import APIRouter, HTTPException
from app.models.api import Request, AnalysisResponse
from app.api import request_to_data_sources
from app.ai import run_fact_check_pipeline
from app.config.default import get_default_pipeline_config
from app.config.azure_models import get_azure_default_pipeline_config
from app.config.gemini_models import get_gemini_default_pipeline_config
from app.ai.tests.fixtures.mock_pipelinesteps import WithoutBrowsingPipelineSteps

router = APIRouter()


@router.post("/text-without-browser", response_model=AnalysisResponse)
async def analyze_text_without_browser(request: Request) -> AnalysisResponse:
    """
    fact-check content without browser-based scraping (Apify).

    uses hybrid approach:
    - mocks social media URLs (Facebook, Instagram, Twitter, TikTok) to avoid Apify
    - uses real simple HTTP scraping for generic URLs
    - only uses GoogleFactCheckGatherer for evidence (no web search)

    ideal for:
    - development and testing
    - avoiding Apify API costs
    - faster response times
    - offline testing (with limitations)

    accepts an array of content items, each with textContent and type.
    returns detailed analysis with verdict, rationale, and citations.
    """
    try:
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)

        # step 2: get pipeline configuration
        config = get_gemini_default_pipeline_config()

        # step 3: use WithoutBrowsingPipelineSteps (hybrid mock/real)
        pipeline_steps = WithoutBrowsingPipelineSteps()

        # step 4: run the async fact-checking pipeline
        fact_check_result = await run_fact_check_pipeline(
            data_sources,
            config,
            pipeline_steps
        )

        # step 5: process results and build response
        # collect all verdicts from all data sources
        all_verdicts = []
        for ds_result in fact_check_result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        # build rationale text from verdicts
        if all_verdicts:
            rationale_parts = ["Resultado da verificação:\n"]

            for i, verdict_item in enumerate(all_verdicts, 1):
                rationale_parts.append(f"\n{i}. Alegação: {verdict_item.claim_text}")
                rationale_parts.append(f"   Veredito: {verdict_item.verdict}")
                rationale_parts.append(f"   Justificativa: {verdict_item.justification}")

            # add overall summary if present
            if fact_check_result.overall_summary:
                rationale_parts.append(
                    f"\n\nResumo Geral:\n{fact_check_result.overall_summary}"
                )

            rationale = "\n".join(rationale_parts)

            # determine overall verdict based on individual verdicts
            verdict_counts = {}
            for v in all_verdicts:
                verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

            # use the most common verdict as overall verdict
            if verdict_counts:
                overall_verdict = max(verdict_counts, key=verdict_counts.get)
            else:
                overall_verdict = "no_claims"
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."
            overall_verdict = "no_claims"

        return AnalysisResponse(
            message_id=data_sources[0].id if data_sources else "unknown",
            verdict=overall_verdict,
            rationale=rationale,
            responseWithoutLinks=rationale,
            processing_time_ms=0  # TODO: measure actual time
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        ) from e


@router.post("/text-no-browser-azure", response_model=AnalysisResponse)
async def analyze_text_no_browser_azure(request: Request) -> AnalysisResponse:
    """
    fact-check content without browser-based scraping using Azure OpenAI.

    identical to /text-without-browser but uses Azure OpenAI models instead of
    standard OpenAI. requires Azure OpenAI environment variables to be set.

    uses hybrid approach:
    - mocks social media URLs (Facebook, Instagram, Twitter, TikTok) to avoid Apify
    - uses real simple HTTP scraping for generic URLs
    - only uses GoogleFactCheckGatherer for evidence (no web search)

    ideal for:
    - organizations using Azure OpenAI deployments
    - compliance requirements for Azure-hosted AI
    - development and testing with Azure
    - avoiding standard OpenAI API

    required environment variables:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_VERSION
    - AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI
    - AZURE_OPENAI_DEPLOYMENT_O3_MINI

    accepts an array of content items, each with textContent and type.
    returns detailed analysis with verdict, rationale, and citations.
    """
    try:
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)

        # step 2: get azure pipeline configuration
        config = get_azure_default_pipeline_config()

        # step 3: use WithoutBrowsingPipelineSteps (hybrid mock/real)
        pipeline_steps = WithoutBrowsingPipelineSteps()

        # step 4: run the async fact-checking pipeline
        fact_check_result = await run_fact_check_pipeline(
            data_sources,
            config,
            pipeline_steps
        )

        # step 5: process results and build response
        # collect all verdicts from all data sources
        all_verdicts = []
        for ds_result in fact_check_result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        # build rationale text from verdicts
        if all_verdicts:
            rationale_parts = ["Resultado da verificação:\n"]

            for i, verdict_item in enumerate(all_verdicts, 1):
                rationale_parts.append(f"\n{i}. Alegação: {verdict_item.claim_text}")
                rationale_parts.append(f"   Veredito: {verdict_item.verdict}")
                rationale_parts.append(f"   Justificativa: {verdict_item.justification}")

            # add overall summary if present
            if fact_check_result.overall_summary:
                rationale_parts.append(
                    f"\n\nResumo Geral:\n{fact_check_result.overall_summary}"
                )

            rationale = "\n".join(rationale_parts)

            # determine overall verdict based on individual verdicts
            verdict_counts = {}
            for v in all_verdicts:
                verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

            # use the most common verdict as overall verdict
            if verdict_counts:
                overall_verdict = max(verdict_counts, key=verdict_counts.get)
            else:
                overall_verdict = "no_claims"
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."
            overall_verdict = "no_claims"

        return AnalysisResponse(
            message_id=data_sources[0].id if data_sources else "unknown",
            verdict=overall_verdict,
            rationale=rationale,
            responseWithoutLinks=rationale,
            processing_time_ms=0  # TODO: measure actual time
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request with Azure OpenAI: {str(e)}"
        ) from e


@router.post("/text-no-browser-gemini", response_model=AnalysisResponse)
async def analyze_text_no_browser_gemini(request: Request) -> AnalysisResponse:
    """
    fact-check content without browser-based scraping using Google Gemini.

    identical to /text-without-browser but uses Google Gemini with thinking mode
    for adjudication instead of standard OpenAI.

    uses hybrid approach:
    - mocks social media URLs (Facebook, Instagram, Twitter, TikTok) to avoid Apify
    - uses real simple HTTP scraping for generic URLs
    - only uses GoogleFactCheckGatherer for evidence (no web search)

    ideal for:
    - testing Google Gemini with extended thinking capabilities
    - comparing Gemini performance vs OpenAI/Azure
    - development and testing with Google AI
    - leveraging Gemini's thinking mode for complex fact-checking

    required environment variables:
    - GOOGLE_API_KEY: your Google AI API key

    accepts an array of content items, each with textContent and type.
    returns detailed analysis with verdict, rationale, and citations.
    """
    try:
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)

        # step 2: get gemini pipeline configuration
        config = get_gemini_default_pipeline_config()

        # step 3: use WithoutBrowsingPipelineSteps (hybrid mock/real)
        pipeline_steps = WithoutBrowsingPipelineSteps()

        # step 4: run the async fact-checking pipeline
        fact_check_result = await run_fact_check_pipeline(
            data_sources,
            config,
            pipeline_steps
        )

        # step 5: process results and build response
        # collect all verdicts from all data sources
        all_verdicts = []
        for ds_result in fact_check_result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        # build rationale text from verdicts
        if all_verdicts:
            rationale_parts = ["Resultado da verificação:\n"]

            for i, verdict_item in enumerate(all_verdicts, 1):
                rationale_parts.append(f"\n{i}. Alegação: {verdict_item.claim_text}")
                rationale_parts.append(f"   Veredito: {verdict_item.verdict}")
                rationale_parts.append(f"   Justificativa: {verdict_item.justification}")

            # add overall summary if present
            if fact_check_result.overall_summary:
                rationale_parts.append(
                    f"\n\nResumo Geral:\n{fact_check_result.overall_summary}"
                )

            rationale = "\n".join(rationale_parts)

            # determine overall verdict based on individual verdicts
            verdict_counts = {}
            for v in all_verdicts:
                verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

            # use the most common verdict as overall verdict
            if verdict_counts:
                overall_verdict = max(verdict_counts, key=verdict_counts.get)
            else:
                overall_verdict = "no_claims"
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."
            overall_verdict = "no_claims"

        return AnalysisResponse(
            message_id=data_sources[0].id if data_sources else "unknown",
            verdict=overall_verdict,
            rationale=rationale,
            responseWithoutLinks=rationale,
            processing_time_ms=0  # TODO: measure actual time
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request with Google Gemini: {str(e)}"
        ) from e
