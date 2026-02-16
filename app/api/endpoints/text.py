from fastapi import APIRouter, HTTPException
import time
import traceback
import asyncio
from app.models.api import Request, AnalysisResponse
from app.clients import send_analytics_payload
from app.observability.analytics import AnalyticsCollector
from app.api.mapper import request_to_data_sources,fact_check_result_to_response, sanitize_request, sanitize_response
from app.agentic_ai.run import run_fact_check
from app.observability.logger.logger import get_logger
from app.utils.id_generator import generate_message_id

router = APIRouter()
logger = get_logger(__name__)


def _log_request_details(msg_id: str, request: Request) -> None:
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
    """
    start_time = time.time()
    msg_id = generate_message_id()
    logger.info(f"[{msg_id}] received /text request with {len(request.content)} content item(s)")

    try:
        # step 0: sanitize request to remove PII
        sanitized_request = sanitize_request(request)

        # log full request for debugging
        _log_request_details(msg_id, sanitized_request)

        #init analytics for the pipeline
        analytics = AnalyticsCollector(msg_id)
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(sanitized_request)
        analytics.populate_from_data_sources(data_sources)

        logger.info(f"[{msg_id}] created {len(data_sources)} data source(s)")

        # step 2: run the agentic fact-checking graph
        logger.info(f"[{msg_id}] starting agentic fact-check graph")
        graph_start = time.time()
        fact_check_result = await run_fact_check(data_sources)
        graph_duration = (time.time() - graph_start) * 1000
        logger.info(f"[{msg_id}] graph completed in {graph_duration:.0f}ms")

        # log results
        total_claims = sum(len(ds_result.claim_verdicts) for ds_result in fact_check_result.results)
        logger.info(f"[{msg_id}] extracted {total_claims} claim(s) from {len(fact_check_result.results)} data source(s)")

        # step 3: build response
        logger.info(f"[{msg_id}] building response")
        response = fact_check_result_to_response(msg_id, fact_check_result)

        # step 4: sanitize response to remove PII
        sanitized_response = sanitize_response(response)

        analytics.set_final_response(sanitized_response.rationale)

        # only send analytics if claims were extracted
        if analytics.has_extracted_claims():
            logger.info(f"[{msg_id}] sending analytics payload (claims found)")
            asyncio.create_task(send_analytics_payload(analytics))
        else:
            logger.info(f"[{msg_id}] skipping analytics payload (no claims extracted)")

        total_duration = (time.time() - start_time) * 1000
        logger.info(f"[{msg_id}] request completed successfully in {total_duration:.0f}ms")

        return sanitized_response

    except Exception as e:
        total_duration = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        logger.error(f"[{msg_id}] request failed after {total_duration:.0f}ms: {error_type}: {str(e)}")
        logger.error(f"[{msg_id}] traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e

