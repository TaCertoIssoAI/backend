from typing import List, Optional
import asyncio

from app.models import (
    ExtractedClaim,
    Citation,
)

from app.ai.context.web import (
    searchGoogleClaim
)

# ===== WEB SEARCH EVIDENCE GATHERER =====

class WebSearchGatherer:
    """
    Evidence gatherer that uses Apify web search to find relevant information.

    Searches Google for the claim text and converts top results into citations.
    """

    def __init__(self, max_results: int = 5, timeout: float = 45.0):
        """
        Initialize web search gatherer.

        Args:
            max_results: Maximum number of search results to retrieve per claim
            timeout: Timeout in seconds for web search operations (default: 45.0)
        """
        self.max_results = max_results
        self.timeout = timeout

    @property
    def source_name(self) -> str:
        return "apify_web_search"

    async def gather(self, claim: ExtractedClaim) -> List[Citation]:
        """
        Search the web for information about the claim.

        Args:
            claim: The claim to search for

        Returns:
            List of citations from search results
        """
        try:
            print(f"\n[WEB SEARCH] searching for: {claim.text[:80]}...")
            print(f"[WEB SEARCH] timeout: {self.timeout}s, max results: {self.max_results}")

            # search google for the claim with configured timeout
            search_result = await asyncio.wait_for(searchGoogleClaim(
                claim=claim.text,
                maxResults=self.max_results,
                timeout=self.timeout
            ), self.timeout)

            # if search failed, return empty list
            if not search_result.get("success", False):
                error_msg = search_result.get("error", "unknown error")
                print(f"[WEB SEARCH] search failed: {error_msg}")
                return []

            # convert search results to citations
            citations: List[Citation] = []
            for result in search_result.get("results", []):
                # extract fields from search result
                url = result.get("url", "")
                title = result.get("title", "")
                description = result.get("description", "")
                domain = result.get("domain", "")

                # skip results with missing critical fields
                if not url or not title:
                    continue

                # create citation
                citation = Citation(
                    url=url,
                    title=title,
                    publisher=domain if domain else url.split("/")[2] if len(url.split("/")) > 2 else "unknown",
                    citation_text=description if description else title,
                    source="apify_web_search",
                    rating=None,  # web search doesn't provide ratings
                    date=None,  # web search results don't include publication date
                )
                citations.append(citation)

            print(f"[WEB SEARCH] found {len(citations)} citation(s)")
            return citations

        except TimeoutError as e:
            print(f"\n[WEB SEARCH ERROR] TIMEOUT after {self.timeout}s")
            print(f"[WEB SEARCH ERROR] claim was: {claim.text[:100]}...")
            return []
        except Exception as e:
            print(f"\n[WEB SEARCH ERROR] unexpected error: {type(e).__name__}: {str(e)[:100]}")
            return []

    def gather_sync(self, claim: ExtractedClaim) -> List[Citation]:
        """synchronous version - creates new event loop and runs async gather"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.gather(claim))
        finally:
            loop.close()
            asyncio.set_event_loop(None)