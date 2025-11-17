"""
Google Fact-Check API Evidence Gatherer

This module implements an EvidenceGatherer for the Google Fact-Check Tools API.
It searches for existing fact-checks from reputable fact-checking organizations
and returns them as citations.

Architecture:
- Implements the EvidenceGatherer protocol from evidence_retrieval.py
- Uses Google Fact-Check Tools API v1alpha1
- Returns Citation objects with rating and review_date metadata
- Handles API errors gracefully with empty results

API Documentation:
https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search
"""

import os
import logging
from typing import List, Optional
import httpx

from app.models import ExtractedClaim, Citation

logger = logging.getLogger(__name__)


class GoogleFactCheckGatherer:
    """
    Evidence gatherer that searches the Google Fact-Check Tools API.

    The Google Fact-Check API aggregates fact-checks from credible organizations
    worldwide. It returns structured claim reviews with ratings like "True",
    "False", "Misleading", etc.

    Example:
        >>> gatherer = GoogleFactCheckGatherer(api_key="your_key", max_results=5)
        >>> claim = ExtractedClaim(...)
        >>> citations = await gatherer.gather(claim)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        timeout: float = 30.0
    ):
        """
        Initialize the Google Fact-Check gatherer.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            max_results: Maximum number of fact-check results to return per claim.
            timeout: Request timeout in seconds.

        Raises:
            RuntimeError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    @property
    def source_name(self) -> str:
        """Return the source identifier for this gatherer."""
        return "google_fact_checking_api"

    async def gather(self, claim: ExtractedClaim) -> List[Citation]:
        """
        Search Google Fact-Check API for fact-checks about this claim.

        Makes an async HTTP request to the Google Fact-Check Tools API and
        parses the results into Citation objects.

        Args:
            claim: The claim to search fact-checks for.

        Returns:
            List of Citation objects from fact-checking organizations.
            Returns empty list if API call fails or no results found.

        Example API Response Structure:
            {
              "claims": [
                {
                  "text": "The claim text...",
                  "claimant": "Person who made the claim",
                  "claimDate": "2024-01-15",
                  "claimReview": [
                    {
                      "publisher": {"name": "FactChecker.org", "site": "factchecker.org"},
                      "url": "https://factchecker.org/...",
                      "title": "Fact-check title",
                      "textualRating": "False",
                      "languageCode": "en",
                      "reviewDate": "2024-01-16"
                    }
                  ]
                }
              ]
            }
        """
        try:
            logger.info(f"searching google fact-check api for: {claim.text[:80]}...")

            # build request parameters
            params = {
                "query": claim.text,
                "key": self.api_key,
            }

            # make async HTTP request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

            # parse JSON response
            data = response.json()

            # extract citations from response
            citations = self._parse_response(data)

            # limit results
            citations = citations[:self.max_results]

            logger.info(
                f"found {len(citations)} fact-check(s) for claim: {claim.text[:50]}..."
            )
            return citations

        except httpx.HTTPStatusError as e:
            logger.error(
                f"google fact-check api http error: {e.response.status_code} - {e}"
            )
            return []
        except httpx.RequestError as e:
            logger.error(f"google fact-check api request error: {e}")
            return []
        except Exception as e:
            logger.error(f"unexpected error in google fact-check api: {e}")
            return []

    def _parse_response(self, data: dict) -> List[Citation]:
        """
        Parse Google Fact-Check API response into Citation objects.

        Args:
            data: JSON response from Google API.

        Returns:
            List of Citation objects parsed from the response.
        """
        citations: List[Citation] = []

        # check if response has claims
        if "claims" not in data or not data["claims"]:
            logger.debug("no claims found in google api response")
            return citations

        # process each claim in the response
        for claim_data in data["claims"]:
            # each claim can have multiple reviews from different fact-checkers
            claim_reviews = claim_data.get("claimReview", [])

            for review in claim_reviews:
                citation = self._parse_claim_review(claim_data, review)
                if citation:
                    citations.append(citation)

        return citations

    def _parse_claim_review(
        self,
        claim_data: dict,
        review: dict
    ) -> Optional[Citation]:
        """
        Parse a single claimReview into a Citation object.

        Args:
            claim_data: The claim data from the API response.
            review: A single claimReview object.

        Returns:
            Citation object or None if required fields are missing.
        """
        try:
            # extract required fields
            url = review.get("url", "")
            title = review.get("title", "")

            # skip if missing critical fields
            if not url or not title:
                logger.debug("skipping review with missing url or title")
                return None

            # extract publisher info
            publisher_data = review.get("publisher", {})
            publisher = publisher_data.get("name", "Unknown Publisher")

            # extract rating (textualRating like "True", "False", "Misleading")
            rating = review.get("textualRating")

            # extract review date
            review_date = review.get("reviewDate")

            # build citation text from claim and rating
            claim_text = claim_data.get("text", "")
            if rating and claim_text:
                citation_text = (
                    f"Fact-check verdict: {rating}. "
                    f"Original claim: {claim_text[:200]}"
                )
            elif claim_text:
                citation_text = f"Fact-check: {claim_text[:250]}"
            else:
                citation_text = title

            return Citation(
                url=url,
                title=title,
                publisher=publisher,
                citation_text=citation_text,
                source="google_fact_checking_api",
                rating=rating,
                date=review_date,
            )

        except Exception as e:
            logger.error(f"error parsing claim review: {e}")
            return None
