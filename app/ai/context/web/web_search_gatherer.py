from typing import List, Optional, Any, Dict
import asyncio
import os

import httpx

from app.models import (
    ExtractedClaim,
    Citation,
)

from app.observability.logger import time_profile, PipelineStep

# ===== WEB SEARCH EVIDENCE GATHERER =====

class WebSearchGatherer:
    """
    Evidence gatherer that uses Google Custom Search API to find relevant information.

    Searches Google for the claim text and converts top results into citations.
    """

    def __init__(self, max_results: int = 5, timeout: float = 45.0):
        """
        Initialize web search gatherer.

        Args:
            max_results: Maximum number of search results to retrieve per claim
            timeout: Timeout in seconds for web search operations (default: 45.0)
        """
        self.max_results = min(max_results, 10)  # google api max is 10
        self.timeout = timeout
        self.api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
        self.cse_cx = os.environ.get("GOOGLE_CSE_CX", "")
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    @property
    def source_name(self) -> str:
        return "google_web_search"

    @time_profile(PipelineStep.EVIDENCE_RETRIEVAL)
    async def gather(self, claim: ExtractedClaim) -> List[Citation]:
        """
        Search the web for information about the claim using Google Custom Search API.

        Args:
            claim: The claim to search for

        Returns:
            List of citations from search results
        """
        try:
            print(f"\n[WEB SEARCH] searching for: {claim.text[:80]}...")
            print(f"[WEB SEARCH] timeout: {self.timeout}s, max results: {self.max_results}")

            # validate api credentials
            if not self.api_key or not self.cse_cx:
                print("[WEB SEARCH ERROR] missing GOOGLE_SEARCH_API_KEY or GOOGLE_CSE_CX")
                return []

            # build request parameters
            params: Dict[str, Any] = {
                "key": self.api_key,
                "cx": self.cse_cx,
                "q": claim.text,
                "num": self.max_results,
            }

            # perform search with timeout
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)

            # check response status
            if response.status_code != 200:
                print(f"[WEB SEARCH ERROR] api returned {response.status_code}: {response.text[:100]}")
                return []

            # parse response
            data = response.json()
            items = data.get("items", [])

            # convert search results to citations
            citations: List[Citation] = []
            for item in items:
                # extract fields from search result
                url = item.get("link", "")
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                display_link = item.get("displayLink", "")

                # skip results with missing critical fields
                if not url or not title:
                    continue

                # create citation
                citation = Citation(
                    url=url,
                    title=title,
                    publisher=display_link if display_link else url.split("/")[2] if len(url.split("/")) > 2 else "unknown",
                    citation_text=snippet if snippet else title,
                    source="google_web_search",
                    rating=None,  # web search doesn't provide ratings
                    date=None,  # web search results don't include publication date
                )
                citations.append(citation)

            print(f"[WEB SEARCH] found {len(citations)} citation(s)")
            return citations

        except httpx.TimeoutException:
            print(f"\n[WEB SEARCH ERROR] TIMEOUT after {self.timeout}s")
            print(f"[WEB SEARCH ERROR] claim was: {claim.text[:100]}...")
            return []
        except Exception as e:
            print(f"\n[WEB SEARCH ERROR] unexpected error: {type(e).__name__}: {str(e)[:100]}")
            return []

    @time_profile(PipelineStep.EVIDENCE_RETRIEVAL)
    def gather_sync(self, claim: ExtractedClaim) -> List[Citation]:
        """synchronous version - creates new event loop and runs async gather"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.gather(claim))
        finally:
            loop.close()
            asyncio.set_event_loop(None)