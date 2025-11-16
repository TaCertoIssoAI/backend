import time
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ai.context.apify_utils import searchGoogleClaim, scrapeGenericSimple

router = APIRouter()
logger = logging.getLogger(__name__)


class ClaimSearchRequest(BaseModel):
    """request model for claim research"""
    claim: str = Field(..., description="claim or statement to research", min_length=3)
    max_results: Optional[int] = Field(default=10, description="maximum number of search results", ge=1, le=50)
    enrich_with_content: Optional[bool] = Field(
        default=False, 
        description="try to fetch full content from top result (no apify credits used)"
    )


class SearchResult(BaseModel):
    """individual search result"""
    title: str
    url: str
    description: str
    position: int
    domain: str
    full_content: Optional[str] = None  # enriched content from simple scraping
    content_length: Optional[int] = None  # length of full_content if available
    scraping_success: Optional[bool] = None  # whether content enrichment worked


class ClaimSearchResponse(BaseModel):
    """response model for claim research"""
    success: bool
    claim: str
    results: list[SearchResult]
    total_results: int
    processing_time_ms: int
    metadata: Optional[dict] = None
    error: Optional[str] = None


@router.post("/search-claim", response_model=ClaimSearchResponse)
async def search_claim(request: ClaimSearchRequest) -> ClaimSearchResponse:
    """
    search google for information about a claim or statement.
    helps with fact-checking by providing relevant search results.
    
    if enrich_with_content=true, tries to scrape full content from top result
    (uses simple http scraping, no browser, no apify credits).
    
    examples:
    - "vaccines cause autism"
    - "earth is flat"
    - "coffee prevents cancer"
    """
    start_time = time.time()
    
    try:
        result = await searchGoogleClaim(claim=request.claim, maxResults=request.max_results)
        processing_time = int((time.time() - start_time) * 1000)
        
        if result["success"]:
            search_results = result["results"]
            
            # enrich first result with full content if requested
            if request.enrich_with_content and len(search_results) > 0:
                logger.info(f"enriching top result with full content for claim: {request.claim}")
                top_result = search_results[0]
                top_url = top_result.get("url", "")
                
                if top_url:
                    # try simple scraping (no browser, no apify)
                    scrape_result = await scrapeGenericSimple(url=top_url, maxChars=10000)
                    
                    if scrape_result["success"]:
                        top_result["full_content"] = scrape_result["content"]
                        top_result["content_length"] = len(scrape_result["content"])
                        top_result["scraping_success"] = True
                        logger.info(f"successfully enriched with {top_result['content_length']} chars")
                    else:
                        top_result["full_content"] = None
                        top_result["content_length"] = 0
                        top_result["scraping_success"] = False
                        logger.warning(f"failed to enrich content: {scrape_result.get('error')}")
            
            return ClaimSearchResponse(
                success=True,
                claim=result["claim"],
                results=[
                    SearchResult(**r) for r in search_results
                ],
                total_results=result["total_results"],
                processing_time_ms=processing_time,
                metadata=result.get("metadata"),
                error=None
            )
        else:
            return ClaimSearchResponse(
                success=False,
                claim=request.claim,
                results=[],
                total_results=0,
                processing_time_ms=processing_time,
                metadata=None,
                error=result.get("error", "unknown error")
            )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"claim search failed for '{request.claim}': {e}")
        
        return ClaimSearchResponse(
            success=False,
            claim=request.claim,
            results=[],
            total_results=0,
            processing_time_ms=processing_time,
            metadata=None,
            error=str(e)
        )


@router.get("/research-status")
async def research_status():
    """check google search configuration and availability"""
    import os
    apify_token = os.getenv("APIFY_TOKEN")
    token_configured = bool(apify_token)
    
    return {
        "research_available": token_configured,
        "apify_configured": token_configured,
        "apify_token_status": "configured" if token_configured else "missing",
        "search_engine": "google",
        "supported_features": {
            "claim_search": "✅ available",
            "fact_checking_support": "✅ provides search results for verification",
            "multi_language": "✅ supports portuguese (pt) and other languages"
        },
        "actor": "apify/google-search-scraper",
        "note": "set APIFY_TOKEN in environment" if not token_configured else "google search ready"
    }

