from fastapi import APIRouter, HTTPException
import uuid
from app.models.api import Request, AnalysisResponse
from app.api import request_to_data_sources,fact_check_result_to_response
from app.ai import run_fact_check_pipeline
from app.config.gemini_models import get_gemini_default_pipeline_config
from app.ai.pipeline.steps import PipelineSteps, DefaultPipelineSteps

router = APIRouter()


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
    try:
        msg_id =  uuid.uuid4()
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)

        # step 2: get pipeline configuration
        config = get_gemini_default_pipeline_config()
        pipeline_step = DefaultPipelineSteps()

        # step 3: run the async fact-checking pipeline
        # IMPORTANT: use 'await' to get the actual results
        fact_check_result = await run_fact_check_pipeline(data_sources, config, pipeline_step)
        return fact_check_result_to_response(msg_id,fact_check_result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e
