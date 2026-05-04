"""
fact-check search tool — wraps the Google Fact-Check API.

reuses parsing logic from app.agentic_ai.context.factcheckapi.google_factcheck_gatherer.
"""

import asyncio
import logging
import os
from uuid import uuid4
from typing import Optional

import httpx

from app.models.agenticai import FactCheckApiContext, SourceReliability
from app.agentic_ai.context.factcheckapi.google_factcheck_gatherer import (
    map_english_rating_to_portuguese,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class FactCheckSearchTool:
    """searches the Google Fact-Check API for existing fact-checks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.max_results = max_results
        self.timeout = timeout

    async def search(self, queries: list[str]) -> list[FactCheckApiContext]:
        """run all queries concurrently and return merged results."""
        tasks = [self._search_single(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[FactCheckApiContext] = []
        for r in results:
            if isinstance(r, list):
                merged.extend(r)
            elif isinstance(r, Exception):
                logger.error(f"fact-check search error: {r}")

        # dedup by URL across queries — keeps first occurrence
        seen_urls: set[str] = set()
        unique: list[FactCheckApiContext] = []
        for entry in merged:
            if entry.url not in seen_urls:
                seen_urls.add(entry.url)
                unique.append(entry)

        dropped = len(merged) - len(unique)
        logger.debug(f"fact-check search: {len(unique)} result(s) (dedup removed {dropped})")

        return unique

    async def _search_single(self, query: str) -> list[FactCheckApiContext]:
        """search a single query against the fact-check API."""
        if not self.api_key:
            logger.warning("missing GOOGLE_API_KEY, skipping fact-check search")
            return []

        params = {"query": query, "key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(BASE_URL, params=params)
                response.raise_for_status()

            data = response.json()
            return self._parse_response(data)

        except httpx.TimeoutException:
            logger.error(f"fact-check api timeout for query: {query[:80]}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"fact-check api http error {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"fact-check api unexpected error: {e}")
            return []

    def _parse_response(self, data: dict) -> list[FactCheckApiContext]:
        """parse API response into FactCheckApiContext objects."""
        results: list[FactCheckApiContext] = []

        claims = data.get("claims", [])
        if not claims:
            return results

        for claim_data in claims:
            claim_text = claim_data.get("text", "")

            for review in claim_data.get("claimReview", []):
                url = review.get("url", "")
                title = review.get("title", "")
                if not url or not title:
                    continue

                publisher_data = review.get("publisher", {})
                publisher = publisher_data.get("name", "")

                # parse rating
                textual_rating = review.get("textualRating", "")
                rating = ""
                rating_comment = None
                if textual_rating:
                    parts = textual_rating.split(".", 1)
                    english_rating = parts[0].strip()
                    mapped = map_english_rating_to_portuguese(english_rating)
                    rating = mapped or ""
                    if len(parts) > 1 and parts[1].strip():
                        rating_comment = parts[1].strip()

                review_date = review.get("reviewDate")

                results.append(
                    FactCheckApiContext(
                        id=str(uuid4()),
                        url=url,
                        parent_id=None,
                        reliability=SourceReliability.MUITO_CONFIAVEL,
                        title=title,
                        publisher=publisher,
                        rating=rating,
                        rating_comment=rating_comment,
                        claim_text=claim_text,
                        review_date=review_date,
                    )
                )

        return results[: self.max_results]
