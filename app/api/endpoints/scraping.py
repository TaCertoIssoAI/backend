from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging

from app.ai.context.apify_utils import scrapeGenericUrl

router = APIRouter()
logger = logging.getLogger(__name__)


class ScrapingRequest(BaseModel):
    """Request model for web scraping"""
    url: str = Field(..., description="URL to scrape", example="https://example.com")
    max_chars: Optional[int] = Field(
        default=128000,
        description="Maximum characters to extract"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.facebook.com/share/p/17hnyGF3hB/",
                "max_chars": 128000
            }
        }


class ScrapingResponse(BaseModel):
    """Response model for web scraping"""
    success: bool = Field(..., description="Whether scraping was successful")
    url: str = Field(..., description="The URL that was scraped")
    content: str = Field(..., description="Extracted content from the page")
    content_length: int = Field(..., description="Length of extracted content in characters")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata from scraping")
    error: Optional[str] = Field(default=None, description="Error message if scraping failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "url": "https://www.facebook.com/example/posts/123",
                "content": "This is the content of the Facebook post...",
                "content_length": 1256,
                "processing_time_ms": 15234,
                "metadata": {
                    "postUrl": "https://www.facebook.com/example/posts/123",
                    "author": "Example Page",
                    "likes": 42,
                    "shares": 10,
                    "comments": 5
                },
                "error": None
            }
        }


@router.post("/scrape", response_model=ScrapingResponse)
async def scrape_url(request: ScrapingRequest) -> ScrapingResponse:
    """
    Extract content from a web page using Apify scraping service.
    
    **Currently Supported:**
    - Facebook posts (public posts only)
    
    **Future Support:**
    - Twitter/X posts
    - Instagram posts
    - Generic web pages
    
    **Strategy:**
    - Uses Apify actors for robust, cloud-based scraping
    - Handles JavaScript-rendered content automatically
    - Returns structured data with metadata
    
    **Requirements:**
    - APIFY_TOKEN must be set in environment variables
    - URL must be publicly accessible
    """
    start_time = time.time()
    
    try:
        logger.info(f"scraping request for url: {request.url}")
        
        # call apify scraping utility
        result = await scrapeGenericUrl(
            url=request.url,
            maxChars=request.max_chars
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        if result["success"]:
            logger.info(f"scraping successful. content length: {len(result['content'])} chars")
            
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
            logger.warning(f"scraping failed for {request.url}: {result.get('error')}")
            
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
        error_msg = str(e)
        
        logger.error(f"scraping failed for {request.url}: {error_msg}")
        
        return ScrapingResponse(
            success=False,
            url=request.url,
            content="",
            content_length=0,
            processing_time_ms=processing_time,
            metadata=None,
            error=error_msg
        )


@router.get("/scrape-test")
async def scrape_test():
    """
    Quick test endpoint to verify Apify integration is working.
    
    Tests with a simple Facebook post URL (if available).
    """
    try:
        # test url - will need to be updated with actual test post
        test_url = "https://www.facebook.com/"
        
        result = await scrapeGenericUrl(test_url, maxChars=500)
        
        return {
            "success": result["success"],
            "message": "apify scraping integration is working!" if result["success"] else "scraping test failed",
            "test_url": test_url,
            "content_length": len(result.get("content", "")),
            "content_preview": result.get("content", "")[:200] if result.get("content") else None,
            "error": result.get("error")
        }
    except Exception as e:
        return {
            "success": False,
            "message": "apify scraping test failed",
            "error": str(e)
        }


@router.get("/scraping-status")
async def scraping_status():
    """
    Check the status of web scraping capabilities.
    
    Verifies:
    - Apify token configuration
    - Available scraping methods
    """
    import os
    
    apify_token = os.getenv("APIFY_TOKEN")
    token_configured = bool(apify_token)
    
    return {
        "scraping_available": token_configured,
        "apify_configured": token_configured,
        "apify_token_status": "configured" if token_configured else "missing",
        "supported_platforms": {
            "facebook": "✅ available",
            "twitter": "⏳ coming soon",
            "instagram": "⏳ coming soon",
            "generic": "⏳ coming soon"
        },
        "note": "set APIFY_TOKEN in environment to enable scraping"
    }
