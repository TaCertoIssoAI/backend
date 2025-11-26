from fastapi import APIRouter, HTTPException
from app.models.api import Request, AnalysisResponse
from app.api import request_to_data_sources
from app.ai import run_fact_check_pipeline
from app.config.gemini_models import get_gemini_default_pipeline_config
from app.ai.pipeline.steps import PipelineSteps, DefaultPipelineSteps
import uuid

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
        msg_id = uuid.uuid4()

        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)

        # step 2: get pipeline configuration
        config = get_gemini_default_pipeline_config()
        pipeline_step = DefaultPipelineSteps()

        # step 3: run the async fact-checking pipeline
        # IMPORTANT: use 'await' to get the actual results
        fact_check_result = await run_fact_check_pipeline(data_sources, config, pipeline_step)

        # step 4: process results and build response
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
                rationale_parts.append(f"\n\nResumo Geral:\n{fact_check_result.overall_summary}")

            rationale = "\n".join(rationale_parts)

            # determine overall verdict based on individual verdicts
            verdict_counts = {}
            for v in all_verdicts:
                verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."

        return AnalysisResponse(
            message_id=data_sources[0].id if data_sources else "unknown",
            rationale=rationale,
            processing_time_ms=0  # TODO: measure actual time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e
