from fastapi import APIRouter, HTTPException
from app.models.api import Request, AnalysisResponse

router = APIRouter()


@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(request: Request) -> AnalysisResponse:
    """
    Fact-check content from text, image descriptions, or audio transcriptions.
    
    Accepts an array of content items, each with textContent and type.
    Returns detailed analysis with verdict, rationale, and citations.
    """
    try:
        # TODO: implement actual fact-checking pipeline
        # placeholder response for now
        return AnalysisResponse(
            message_id="placeholder_id",
            verdict="pending",
            rationale="Analysis not yet implemented",
            responseWithoutLinks="Analysis not yet implemented",
            processing_time_ms=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e
