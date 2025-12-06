from fastapi import APIRouter, HTTPException
import uuid
import time
import traceback
import asyncio
from app.models.api import Request, AnalysisResponse
from app.clients import send_analytics_payload
from app.observability.analytics import AnalyticsCollector
from app.api.mapper import request_to_data_sources,fact_check_result_to_response
from app.ai import run_fact_check_pipeline
from app.config.gemini_models import get_gemini_default_pipeline_config
from app.ai.pipeline.steps import PipelineSteps, DefaultPipelineSteps
from app.observability.logger.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _log_request_details(msg_id: uuid.UUID, request: Request) -> None:
    """
    log detailed request information for debugging.

    args:
        msg_id: unique message identifier for correlation
        request: the incoming API request to log
    """
    logger.debug(f"[{msg_id}] request = {request}")
    logger.debug(f"[{msg_id}] request.content = {request.content}")
    logger.debug(f"[{msg_id}] Number of content items: {len(request.content)}")

    for idx, item in enumerate(request.content):
        content_preview = item.textContent[:100] if item.textContent else "None"
        logger.debug(f"[{msg_id}] === Content Item {idx} ===")
        logger.debug(f"[{msg_id}]   type: {item.type}")
        logger.debug(f"[{msg_id}]   textContent length: {len(item.textContent or '')}")
        logger.debug(f"[{msg_id}]   textContent preview: '{content_preview}...'")
        logger.debug(f"[{msg_id}]   full textContent:\n{item.textContent}")
        logger.info(f"[{msg_id}] content[{idx}]: type={item.type}, text_length={len(item.textContent or '')}, preview='{content_preview}...')")


@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(request: Request) -> AnalysisResponse:
    """
    Fact-check content from text, image descriptions, or audio transcriptions.

    Accepts an array of content items, each with textContent and type.
    Returns detailed analysis with verdict, rationale, and citations.

    The key to calling async functions:
    1. Make the endpoint async (async def)
    2. Use await to call async functions
    3. FastAPI handles the rest automatically
    """
    start_time = time.time()
    msg_id = uuid.uuid4()
    logger.info(f"[{msg_id}] received /text request with {len(request.content)} content item(s)")

    try:
        # log full request for debugging
        _log_request_details(msg_id, request)

        #init analytics for the pipeline
        analytics = AnalyticsCollector(msg_id)
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)
        analytics.populate_from_data_sources(data_sources)

        logger.info(f"[{msg_id}] created {len(data_sources)} data source(s)")

        # step 2: get pipeline configuration
        config = get_gemini_default_pipeline_config()
        pipeline_step = DefaultPipelineSteps()

        # step 3: run the async fact-checking pipeline
        logger.info(f"[{msg_id}] starting fact-check pipeline")
        pipeline_start = time.time()
        fact_check_result = await run_fact_check_pipeline(data_sources, config, pipeline_step,analytics)
        pipeline_duration = (time.time() - pipeline_start) * 1000
        logger.info(f"[{msg_id}] pipeline completed in {pipeline_duration:.0f}ms")

        # log pipeline results
        total_claims = sum(len(ds_result.claim_verdicts) for ds_result in fact_check_result.results)
        logger.info(f"[{msg_id}] extracted {total_claims} claim(s) from {len(fact_check_result.results)} data source(s)")

        # step 4: build response
        logger.info(f"[{msg_id}] building response")
        response = fact_check_result_to_response(msg_id, fact_check_result)
        analytics.set_final_response(response.rationale)

        # only send analytics if claims were extracted
        if analytics.has_extracted_claims():
            logger.info(f"[{msg_id}] sending analytics payload (claims found)")
            asyncio.create_task(send_analytics_payload(analytics))
        else:
            logger.info(f"[{msg_id}] skipping analytics payload (no claims extracted)")

        total_duration = (time.time() - start_time) * 1000
        logger.info(f"[{msg_id}] request completed successfully in {total_duration:.0f}ms")

        return response

    except Exception as e:
        total_duration = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        logger.error(f"[{msg_id}] request failed after {total_duration:.0f}ms: {error_type}: {str(e)}")
        logger.error(f"[{msg_id}] traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e
