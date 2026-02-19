"""
serper.dev search api integration — used as fallback when google custom search fails.
"""

import os
import logging
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"

# language code mapping: google lr format → serper hl/gl
_LANGUAGE_MAP: Dict[str, Dict[str, str]] = {
    "lang_pt": {"hl": "pt", "gl": "br"},
    "lang_en": {"hl": "en", "gl": "us"},
    "lang_es": {"hl": "es", "gl": "es"},
    "lang_fr": {"hl": "fr", "gl": "fr"},
    "lang_de": {"hl": "de", "gl": "de"},
}


class SerperSearchError(Exception):
    """exception raised when serper.dev search api fails"""
    pass


def _is_serper_configured() -> bool:
    """check whether serper api key is available in the environment."""
    return bool(os.environ.get("SERPER_API_KEY", ""))


def _build_serper_query(
    query: str,
    site_search: str | None = None,
    site_search_filter: str | None = None,
) -> str:
    """prepend site: or -site: operator to query based on google-style params."""
    if not site_search:
        return query

    if site_search_filter == "e":
        return f"-site:{site_search} {query}"
    # default to include
    return f"site:{site_search} {query}"


async def serper_search(
    query: str,
    *,
    num: int = 10,
    site_search: str | None = None,
    site_search_filter: str | None = None,
    date_restrict: str | None = None,
    language: str | None = None,
    timeout: float = 15.0,
) -> list[Dict[str, Any]]:
    """
    performs a search using serper.dev api and returns results in the same
    format as google custom search (items with title, link, snippet, displayLink).

    args:
        query: search query string
        num: number of results to return (1-10)
        site_search: domain to filter by (e.g., "who.int")
        site_search_filter: "i" to include only site_search, "e" to exclude
        date_restrict: relative date filter (e.g., "d7" for last 7 days)
        language: language restriction in google format (e.g., "lang_pt")
        timeout: request timeout in seconds

    returns:
        list of search result items matching google cse item format
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        raise SerperSearchError("missing SERPER_API_KEY")

    effective_query = _build_serper_query(query, site_search, site_search_filter)

    payload: Dict[str, Any] = {
        "q": effective_query,
        "num": min(num, 10),
    }

    # map date_restrict (e.g. "d7") → tbs (e.g. "qdr:d7")
    if date_restrict:
        payload["tbs"] = f"qdr:{date_restrict}"

    # map language
    if language and language in _LANGUAGE_MAP:
        lang_cfg = _LANGUAGE_MAP[language]
        payload["hl"] = lang_cfg["hl"]
        payload["gl"] = lang_cfg["gl"]

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(SERPER_API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        raise SerperSearchError(
            f"serper search error: {response.status_code} {response.text[:200]}"
        )

    data = response.json()
    organic = data.get("organic", [])

    # map serper organic results → google cse item format
    items: list[Dict[str, Any]] = []
    for result in organic:
        items.append({
            "title": result.get("title", ""),
            "link": result.get("link", ""),
            "snippet": result.get("snippet", ""),
            "displayLink": result.get("domain", ""),
        })

    return items
