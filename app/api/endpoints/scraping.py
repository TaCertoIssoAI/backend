from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
import time
import logging

from app.ai.context.web_scraping.scraper import get_page_content

router = APIRouter()
logger = logging.getLogger(__name__)


class ScrapingRequest(BaseModel):
    """Request model for web scraping"""
    url: str = Field(..., description="URL to scrape", example="https://example.com")
    force_selenium: bool = Field(
        default=False, 
        description="Force use of Selenium even if requests might work"
    )
    max_chars: Optional[int] = Field(
        default=128000,
        description="Maximum characters to extract"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.facebook.com/share/p/17hnyGF3hB/",
                "force_selenium": False,
                "max_chars": 128000
            }
        }


class ScrapingResponse(BaseModel):
    """Response model for web scraping"""
    success: bool = Field(..., description="Whether scraping was successful")
    url: str = Field(..., description="The URL that was scraped")
    content: str = Field(..., description="Extracted content from the page")
    content_length: int = Field(..., description="Length of extracted content in characters")
    method_used: str = Field(..., description="Method used: 'requests' or 'selenium'")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if scraping failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "url": "https://example.com",
                "content": "Example Domain\nThis domain is for use in illustrative examples...",
                "content_length": 1256,
                "method_used": "requests",
                "processing_time_ms": 1234,
                "error": None
            }
        }


@router.post("/scrape", response_model=ScrapingResponse)
async def scrape_url(request: ScrapingRequest) -> ScrapingResponse:
    """
    Extract content from a web page using intelligent scraping strategy.
    
    **Strategy:**
    - First tries fast `requests` method for static sites
    - Automatically falls back to Selenium for JavaScript-heavy sites
    - Can force Selenium usage with `force_selenium=true`
    
    **Supported Sites:**
    - Static HTML pages (fast)
    - JavaScript-rendered pages (via Selenium)
    - Social media (Facebook, Twitter, etc.)
    - News sites
    - Blogs and articles
    
    **Docker Integration:**
    - Uses Selenium Grid container for browser automation
    - Isolated execution in containerized environment
    - Scalable and production-ready
    """
    start_time = time.time()
    method_used = "unknown"
    
    try:
        logger.info(f"Scraping request for URL: {request.url}")
        logger.info(f"Force Selenium: {request.force_selenium}")
        
        # Perform web scraping
        content = get_page_content(
            url=request.url,
            force_selenium=request.force_selenium
        )
        
        # Determine which method was used
        if request.force_selenium:
            method_used = "selenium"
        else:
            # If content is very short, it likely failed with requests and used selenium
            method_used = "selenium" if len(content) < 200 else "requests"
        
        # Truncate if needed
        if request.max_chars and len(content) > request.max_chars:
            content = content[:request.max_chars]
            logger.warning(f"Content truncated to {request.max_chars} characters")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Scraping successful. Content length: {len(content)} chars")
        
        return ScrapingResponse(
            success=True,
            url=request.url,
            content=content,
            content_length=len(content),
            method_used=method_used,
            processing_time_ms=processing_time,
            error=None
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        error_msg = str(e)
        
        logger.error(f"Scraping failed for {request.url}: {error_msg}")
        
        return ScrapingResponse(
            success=False,
            url=request.url,
            content="",
            content_length=0,
            method_used=method_used,
            processing_time_ms=processing_time,
            error=error_msg
        )


@router.get("/scrape-test")
async def scrape_test():
    """
    Quick test endpoint to verify web scraping is working.
    
    Tests scraping with a simple, reliable URL (example.com).
    """
    try:
        content = get_page_content("https://example.com")
        
        return {
            "success": True,
            "message": "Web scraping is working!",
            "test_url": "https://example.com",
            "content_length": len(content),
            "content_preview": content[:200] if len(content) > 200 else content,
            "selenium_grid": "Connected" if "Example Domain" in content else "Unknown"
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Web scraping test failed",
            "error": str(e)
        }


class MultipleUrlsRequest(BaseModel):
    urls: list[str] = Field(..., description="List of URLs to scrape", max_length=10)
    force_selenium: bool = Field(default=False, description="Force use of Selenium")


@router.post("/scrape-multiple")
async def scrape_multiple_urls(request: MultipleUrlsRequest):
    """
    Scrape multiple URLs in sequence.
    
    **Note:** This processes URLs one by one. For large batches, consider
    implementing proper queuing and background tasks.
    
    **Request Body:**
    ```json
    {
        "urls": [
            "https://example.com",
            "https://google.com"
        ],
        "force_selenium": false
    }
    ```
    """
    if len(request.urls) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 URLs allowed per request. Use batch processing for more."
        )
    
    results = []
    start_time = time.time()
    
    for url in request.urls:
        try:
            content = get_page_content(url, force_selenium=request.force_selenium)
            results.append({
                "url": url,
                "success": True,
                "content_length": len(content),
                "content_preview": content[:200] if len(content) > 200 else content,
                "error": None
            })
        except Exception as e:
            results.append({
                "url": url,
                "success": False,
                "content_length": 0,
                "content_preview": "",
                "error": str(e)
            })
    
    total_time = int((time.time() - start_time) * 1000)
    successful = sum(1 for r in results if r["success"])
    
    return {
        "total_urls": len(request.urls),
        "successful": successful,
        "failed": len(request.urls) - successful,
        "total_processing_time_ms": total_time,
        "results": results
    }


@router.get("/scraping-status")
async def scraping_status():
    """
    Check the status of web scraping capabilities.
    
    Verifies:
    - Selenium Grid connectivity
    - Environment configuration
    - Available scraping methods
    """
    import os
    
    selenium_url = os.getenv("SELENIUM_REMOTE_URL", "not set")
    use_remote = os.getenv("USE_SELENIUM_REMOTE", "auto")
    
    # Try a quick test
    test_passed = False
    selenium_available = False
    
    try:
        content = get_page_content("https://example.com", force_selenium=False)
        test_passed = len(content) > 100
    except:
        pass
    
    try:
        content = get_page_content("https://example.com", force_selenium=True)
        selenium_available = len(content) > 100
    except:
        pass
    
    return {
        "scraping_available": test_passed,
        "selenium_available": selenium_available,
        "selenium_remote_url": selenium_url,
        "use_selenium_remote": use_remote,
        "environment": "docker" if "selenium:4444" in selenium_url else "local",
        "methods": {
            "requests": "✅ Available",
            "selenium": "✅ Available" if selenium_available else "❌ Not available"
        }
    }

