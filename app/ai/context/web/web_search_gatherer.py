from typing import List, Optional, Any, Dict
import asyncio
import logging
import os

import httpx

from app.models import (
    ExtractedClaim,
    Citation,
)

from app.observability.logger import time_profile, PipelineStep
from app.ai.context.web.serper_search import (
    serper_search,
    _is_serper_configured,
)

logger = logging.getLogger(__name__)

# ===== WEB SEARCH EVIDENCE GATHERER =====

class WebSearchGatherer:
    """
    Evidence gatherer that uses Google Custom Search API to find relevant information.

    Searches Google for the claim text and converts top results into citations.
    """

    def __init__(self, max_results: int = 5, timeout: float = 45.0, allowed_domains:list[str] | None = None):
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
        if allowed_domains is not None:
            self.allowed_domains = allowed_domains
        else:
            self.allowed_domains = []

    @property
    def source_name(self) -> str:
        return "google_web_search"

    @time_profile(PipelineStep.EVIDENCE_RETRIEVAL)
    async def gather(self, claim: ExtractedClaim) -> List[Citation]:
        """
        Search the web for information about the claim using Google Custom Search API.
        Falls back to serper.dev when google fails.

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
                return await self._serper_fallback_gather(claim)

            query_with_domains = self._build_search_query_with_domains(claim.text)
            # build request parameters
            params: Dict[str, Any] = {
                "key": self.api_key,
                "cx": self.cse_cx,
                "q": query_with_domains,
                "num": self.max_results,
            }

            # perform search with timeout
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)

            # check response status
            if response.status_code != 200:
                print(f"[WEB SEARCH ERROR] api returned {response.status_code}: {response.text[:100]}")
                return await self._serper_fallback_gather(claim)

            # parse response
            data = response.json()
            items = data.get("items", [])

            # convert search results to citations
            citations = self._items_to_citations(items, source="google_web_search")

            print(f"[WEB SEARCH] found {len(citations)} citation(s)")
            return citations

        except httpx.TimeoutException:
            print(f"\n[WEB SEARCH ERROR] TIMEOUT after {self.timeout}s")
            print(f"[WEB SEARCH ERROR] claim was: {claim.text[:100]}...")
            return await self._serper_fallback_gather(claim)
        except Exception as e:
            print(f"\n[WEB SEARCH ERROR] unexpected error: {type(e).__name__}: {str(e)[:100]}")
            return await self._serper_fallback_gather(claim)

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
    
    def _items_to_citations(self, items: list, source: str = "google_web_search") -> List[Citation]:
        """convert search result items (google or serper format) to Citation objects."""
        citations: List[Citation] = []
        for item in items:
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            display_link = item.get("displayLink", "")

            if not url or not title:
                continue

            citation = Citation(
                url=url,
                title=title,
                publisher=display_link if display_link else url.split("/")[2] if len(url.split("/")) > 2 else "unknown",
                citation_text=snippet if snippet else title,
                source=source,
                rating=None,
                date=None,
            )
            citations.append(citation)
        return citations

    async def _serper_fallback_gather(self, claim: ExtractedClaim) -> List[Citation]:
        """fallback to serper.dev when google search fails."""
        if not _is_serper_configured():
            logger.warning("serper not configured, returning empty results")
            return []

        try:
            print("[WEB SEARCH] trying serper.dev fallback...")
            query_with_domains = self._build_search_query_with_domains(claim.text)
            items = await serper_search(
                query=query_with_domains,
                num=self.max_results,
                timeout=self.timeout,
            )
            citations = self._items_to_citations(items, source="serper_web_search_fallback")
            print(f"[WEB SEARCH] serper fallback found {len(citations)} citation(s)")
            return citations
        except Exception as e:
            logger.error(f"serper fallback also failed: {e}")
            print(f"[WEB SEARCH ERROR] serper fallback failed: {type(e).__name__}: {str(e)[:100]}")
            return []

    def _build_search_query_with_domains(self,original_query:str)->str:
        if not self.allowed_domains:
            return original_query
        
        valid_domains = [d.strip() for d in self.allowed_domains if d and d.strip()]
        if not valid_domains:
            return original_query

        domain_filters = " OR ".join([f"site:{domain}" for domain in valid_domains])
        return  f"{original_query} ({domain_filters})"