from typing import List

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

    def __init__(self, max_results: int = 5):
        """
        Initialize web search gatherer.

        Args:
            max_results: Maximum number of search results to retrieve per claim
        """
        self.max_results = max_results

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
        # search google for the claim
        search_result = await searchGoogleClaim(
            claim=claim.text,
            maxResults=self.max_results
        )

        # if search failed, return empty list
        if not search_result.get("success", False):
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

        return citations