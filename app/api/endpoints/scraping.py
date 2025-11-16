import os
import time
import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ai.context.apify_utils import scrapeGenericUrl

router = APIRouter()
logger = logging.getLogger(__name__)


class ScrapingRequest(BaseModel):
    """request model for web scraping"""
    url: str = Field(..., description="URL to scrape")
    max_chars: Optional[int] = Field(default=128000, description="max characters to extract")


class ScrapingResponse(BaseModel):
    """response model for web scraping"""
    success: bool
    url: str
    content: str
    content_length: int
    processing_time_ms: int
    metadata: Optional[dict] = None
    error: Optional[str] = None


@router.post("/scrape", response_model=ScrapingResponse)
async def scrape_url(request: ScrapingRequest) -> ScrapingResponse:
    """
    scrape web content using apify.
    automatically detects platform and uses appropriate actor.
    supports: facebook, instagram, twitter, tiktok, generic websites.
    """
    start_time = time.time()
    
    try:
        result = await scrapeGenericUrl(url=request.url, maxChars=request.max_chars)
        processing_time = int((time.time() - start_time) * 1000)
        
        if result["success"]:
            return ScrapingResponse(
                success=True,
                url=request.url,
                content=result["content"],
                content_length=len(result["content"]),
                processing_time_ms=processing_time,
                metadata=result.get("metadata"),
                error=None
            )
        else:
            return ScrapingResponse(
                success=False,
                url=request.url,
                content="",
                content_length=0,
                processing_time_ms=processing_time,
                metadata=None,
                error=result.get("error", "unknown error")
            )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"scraping failed for {request.url}: {e}")
        
        return ScrapingResponse(
            success=False,
            url=request.url,
            content="",
            content_length=0,
            processing_time_ms=processing_time,
            metadata=None,
            error=str(e)
        )


@router.get("/scrape-test")
async def scrape_test():
    """quick test endpoint for apify integration"""
    try:
        test_url = "https://www.facebook.com/"
        result = await scrapeGenericUrl(test_url, maxChars=500)
        
        return {
            "success": result["success"],
            "message": "apify integration working" if result["success"] else "scraping failed",
            "test_url": test_url,
            "content_length": len(result.get("content", "")),
            "error": result.get("error")
        }
    except Exception as e:
        return {"success": False, "message": "test failed", "error": str(e)}


@router.get("/scraping-status")
async def scraping_status():
    """check apify configuration and available actors"""
    apify_token = os.getenv("APIFY_TOKEN")
    token_configured = bool(apify_token)
    
    return {
        "scraping_available": token_configured,
        "apify_configured": token_configured,
        "apify_token_status": "configured" if token_configured else "missing",
        "supported_platforms": {
            "facebook": "✅ available (apify/facebook-posts-scraper)",
            "instagram": "✅ available (apify/instagram-scraper)",
            "twitter": "✅ available (apidojo/tweet-scraper)",
            "tiktok": "✅ available (clockworks/tiktok-scraper)",
            "generic": "✅ available (apify/website-content-crawler)"
        },
        "smart_detection": "✅ automatic platform detection via regex",
        "note": "set APIFY_TOKEN in environment" if not token_configured else "all actors ready"
    }
