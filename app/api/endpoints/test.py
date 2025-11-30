"""
Test endpoints for development and debugging.

provides alternative endpoints with different configurations for testing purposes.
"""
import uuid
import time
import asyncio
import traceback
from fastapi import APIRouter, HTTPException
from app.models.api import Request, AnalysisResponse
from app.clients import send_analytics_payload
from app.api import request_to_data_sources,fact_check_result_to_response
from app.ai import run_fact_check_pipeline
from app.config.default import get_default_pipeline_config
from app.config.azure_models import get_azure_default_pipeline_config
from app.config.gemini_models import get_gemini_default_pipeline_config
from app.ai.tests.fixtures.mock_pipelinesteps import WithoutBrowsingPipelineSteps
from app.observability.logger.logger import get_logger
from app.observability.analytics import AnalyticsCollector


router = APIRouter()
logger = get_logger(__name__)


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
    start_time = time.time()
    msg_id = uuid.uuid4()

    logger.info(f"[{msg_id}] received /text-without-browser request with {len(request.content)} content item(s)")

    try:
        analytics = AnalyticsCollector(msg_id)
        # log request details
        for idx, item in enumerate(request.content):
            content_preview = item.textContent[:100] if item.textContent else "None"
            logger.info(f"[{msg_id}] content[{idx}]: type={item.type}, text_length={len(item.textContent or '')}, preview='{content_preview}...'")

        # step 1: convert API request to internal DataSource format
        logger.info(f"[{msg_id}] converting request to data sources")
        data_sources = request_to_data_sources(request)
        analytics.populate_from_data_sources(data_sources)
        logger.info(f"[{msg_id}] created {len(data_sources)} data source(s)")

        # step 2: get pipeline configuration
        logger.info(f"[{msg_id}] initializing gemini pipeline config (no-browser mode)")
        config = get_gemini_default_pipeline_config()

        # step 3: use WithoutBrowsingPipelineSteps (hybrid mock/real)
        pipeline_steps = WithoutBrowsingPipelineSteps()
        logger.info(f"[{msg_id}] using WithoutBrowsingPipelineSteps (mocks social media, real simple scraping)")

        # step 4: run the async fact-checking pipeline
        logger.info(f"[{msg_id}] starting fact-check pipeline")
        pipeline_start = time.time()
        fact_check_result = await run_fact_check_pipeline(
            data_sources,
            config,
            pipeline_steps,
            analytics,
        )
        pipeline_duration = (time.time() - pipeline_start) * 1000
        logger.info(f"[{msg_id}] pipeline completed in {pipeline_duration:.0f}ms")

        # log pipeline results
        total_claims = sum(len(ds_result.claim_verdicts) for ds_result in fact_check_result.results)
        logger.info(f"[{msg_id}] extracted {total_claims} claim(s) from {len(fact_check_result.results)} data source(s)")

        # build response
        logger.info(f"[{msg_id}] building response")
        response = fact_check_result_to_response(msg_id, fact_check_result)
        analytics.set_final_response(response.rationale)
        asyncio.create_task(send_analytics_payload(analytics))

        total_duration = (time.time() - start_time) * 1000
        logger.info(f"[{msg_id}] request completed successfully in {total_duration:.0f}ms")

        return response

    except Exception as e:
        total_duration = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        logger.error(f"[{msg_id}] request failed after {total_duration:.0f}ms: {error_type}: {str(e)}")
        logger.error(f"[{msg_id}] traceback:\n{traceback.format_exc()}")
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
    start_time = time.time()
    msg_id = uuid.uuid4()

    logger.info(f"[{msg_id}] received /text-no-browser-azure request with {len(request.content)} content item(s)")

    try:
        # log request details
        for idx, item in enumerate(request.content):
            content_preview = item.textContent[:100] if item.textContent else "None"
            logger.info(f"[{msg_id}] content[{idx}]: type={item.type}, text_length={len(item.textContent or '')}, preview='{content_preview}...'")

        # step 1: convert API request to internal DataSource format
        logger.info(f"[{msg_id}] converting request to data sources")
        data_sources = request_to_data_sources(request)
        logger.info(f"[{msg_id}] created {len(data_sources)} data source(s)")

        # step 2: get azure pipeline configuration
        logger.info(f"[{msg_id}] initializing azure openai pipeline config")
        config = get_azure_default_pipeline_config()

        # step 3: use WithoutBrowsingPipelineSteps (hybrid mock/real)
        pipeline_steps = WithoutBrowsingPipelineSteps()
        logger.info(f"[{msg_id}] using WithoutBrowsingPipelineSteps (mocks social media, real simple scraping)")

        # step 4: run the async fact-checking pipeline
        logger.info(f"[{msg_id}] starting fact-check pipeline with azure models")
        pipeline_start = time.time()
        fact_check_result = await run_fact_check_pipeline(
            data_sources,
            config,
            pipeline_steps
        )
        pipeline_duration = (time.time() - pipeline_start) * 1000
        logger.info(f"[{msg_id}] pipeline completed in {pipeline_duration:.0f}ms")

        # step 5: process results and build response
        # collect all verdicts from all data sources
        all_verdicts = []
        for ds_result in fact_check_result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        total_claims = len(all_verdicts)
        logger.info(f"[{msg_id}] extracted {total_claims} claim(s) from {len(fact_check_result.results)} data source(s)")

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

        total_duration = (time.time() - start_time) * 1000
        logger.info(f"[{msg_id}] request completed successfully in {total_duration:.0f}ms, verdict={overall_verdict}")

        return AnalysisResponse(
            message_id=str(msg_id),
            verdict=overall_verdict,
            rationale=rationale,
            responseWithoutLinks=rationale,
            processing_time_ms=int(total_duration)
        )

    except Exception as e:
        total_duration = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        logger.error(f"[{msg_id}] request failed after {total_duration:.0f}ms: {error_type}: {str(e)}")
        logger.error(f"[{msg_id}] traceback:\n{traceback.format_exc()}")
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
    start_time = time.time()
    msg_id = uuid.uuid4()

    logger.info(f"[{msg_id}] received /text-no-browser-gemini request with {len(request.content)} content item(s)")

    try:
        # log request details
        for idx, item in enumerate(request.content):
            content_preview = item.textContent[:100] if item.textContent else "None"
            logger.info(f"[{msg_id}] content[{idx}]: type={item.type}, text_length={len(item.textContent or '')}, preview='{content_preview}...'")

        # step 1: convert API request to internal DataSource format
        logger.info(f"[{msg_id}] converting request to data sources")
        data_sources = request_to_data_sources(request)
        logger.info(f"[{msg_id}] created {len(data_sources)} data source(s)")

        # step 2: get gemini pipeline configuration
        logger.info(f"[{msg_id}] initializing gemini pipeline config with thinking mode")
        config = get_gemini_default_pipeline_config()

        # step 3: use WithoutBrowsingPipelineSteps (hybrid mock/real)
        pipeline_steps = WithoutBrowsingPipelineSteps()
        logger.info(f"[{msg_id}] using WithoutBrowsingPipelineSteps (mocks social media, real simple scraping)")

        # step 4: run the async fact-checking pipeline
        logger.info(f"[{msg_id}] starting fact-check pipeline with gemini models")
        pipeline_start = time.time()
        fact_check_result = await run_fact_check_pipeline(
            data_sources,
            config,
            pipeline_steps
        )
        pipeline_duration = (time.time() - pipeline_start) * 1000
        logger.info(f"[{msg_id}] pipeline completed in {pipeline_duration:.0f}ms")

        # step 5: process results and build response
        # collect all verdicts from all data sources
        all_verdicts = []
        for ds_result in fact_check_result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        total_claims = len(all_verdicts)
        logger.info(f"[{msg_id}] extracted {total_claims} claim(s) from {len(fact_check_result.results)} data source(s)")

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

        total_duration = (time.time() - start_time) * 1000
        logger.info(f"[{msg_id}] request completed successfully in {total_duration:.0f}ms, verdict={overall_verdict}")

        return AnalysisResponse(
            message_id=str(msg_id),
            verdict=overall_verdict,
            rationale=rationale,
            responseWithoutLinks=rationale,
            processing_time_ms=int(total_duration)
        )

    except Exception as e:
        total_duration = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        logger.error(f"[{msg_id}] request failed after {total_duration:.0f}ms: {error_type}: {str(e)}")
        logger.error(f"[{msg_id}] traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request with Google Gemini: {str(e)}"
        ) from e
