from fastapi import APIRouter, HTTPException
from app.models.api import Request, AnalysisResponse
from app.api import request_to_data_sources
from app.ai import run_fact_check_pipeline
from app.config.default import get_default_pipeline_config
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
        # step 1: convert API request to internal DataSource format
        data_sources = request_to_data_sources(request)

        # step 2: get pipeline configuration
        config = get_default_pipeline_config()
        pipeline_step = DefaultPipelineSteps()

        # step 3: run the async fact-checking pipeline
        # IMPORTANT: use 'await' to get the actual results
        claim_outputs = await run_fact_check_pipeline(data_sources, config,pipeline_step)

        # step 4: process results and build response
        # collect all claims from all sources
        all_claims = []
        for output in claim_outputs:
            all_claims.extend(output.claims)
        
        # build rationale text from extracted claims
        if all_claims:
            rationale_parts = ["Análise das alegações extraídas:\n"]
            for i, claim in enumerate(all_claims, 1):
                rationale_parts.append(f"{i}. {claim.text}")
                if claim.entities:
                    rationale_parts.append(f"   Entidades: {', '.join(claim.entities)}")
            rationale = "\n".join(rationale_parts)
            verdict = f"{len(all_claims)} claim(s) extracted"
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido."
            verdict = "no_claims"
        
        return AnalysisResponse(
            message_id=data_sources[0].id if data_sources else "unknown",
            verdict=verdict,
            rationale=rationale,
            responseWithoutLinks=rationale,
            processing_time_ms=0  # TODO: measure actual time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e
