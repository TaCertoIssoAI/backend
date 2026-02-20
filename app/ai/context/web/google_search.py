"""
google custom search api integration for web search.
falls back in cascade: vertex ai search → google custom search → serper.dev.
"""

import os
import logging
from typing import Any, Dict

import httpx

from app.ai.context.web.serper_search import (
    serper_search,
    _is_serper_configured,
)
from app.ai.context.web.vertex_search import (
    vertex_search,
    VertexSearchError,
    _is_vertex_configured,
)

logger = logging.getLogger(__name__)


class GoogleSearchError(Exception):
    """exception raised when google search api fails"""
    pass


async def searchGoogleClaim(claim: str, maxResults: int = 10, timeout: float = 45.0) -> dict:
    """
    search google for information about a claim using Google Custom Search API.
    falls back in cascade: vertex ai search → google custom search → serper.dev.

    args:
        claim: the claim text to search for
        maxResults: maximum number of search results to return (max 10)
        timeout: timeout in seconds for the search operation (default: 45.0)

    returns:
        dict with search results and metadata
    """
    result = await _searchGoogleClaimInternal(claim, maxResults, timeout)

    if not result["success"] and _is_serper_configured():
        logger.warning(f"google claim search failed ({result.get('error')}), trying serper fallback")
        fallback = await _searchSerperClaimFallback(claim, maxResults, timeout)
        if fallback["success"]:
            return fallback

    return result


async def _searchGoogleClaimInternal(claim: str, maxResults: int = 10, timeout: float = 45.0) -> dict:
    """internal google claim search — original logic without fallback."""
    try:
        logger.info(f"searching google for claim: {claim[:100]}...")
        logger.info(f"search timeout: {timeout}s, max results: {maxResults}")

        api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
        cse_cx = os.environ.get("GOOGLE_CSE_CX", "")

        if not api_key or not cse_cx:
            logger.error("missing GOOGLE_SEARCH_API_KEY or GOOGLE_CSE_CX environment variables")
            return {
                "success": False,
                "claim": claim,
                "results": [],
                "total_results": 0,
                "error": "missing google search api credentials"
            }

        params = {
            "key": api_key,
            "cx": cse_cx,
            "q": claim,
            "num": min(maxResults, 10),
            "lr": "lang_pt",
        }

        base_url = "https://www.googleapis.com/customsearch/v1"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(base_url, params=params)

        if response.status_code != 200:
            error_msg = f"google api returned {response.status_code}: {response.text[:100]}"
            logger.error(error_msg)
            return {
                "success": False,
                "claim": claim,
                "results": [],
                "total_results": 0,
                "error": error_msg
            }

        data = response.json()
        items = data.get("items", [])

        if not items:
            logger.info("google search completed: no results found")
            return {
                "success": False,
                "claim": claim,
                "results": [],
                "total_results": 0,
                "error": "no search results found"
            }

        searchResults = []
        for position, item in enumerate(items, start=1):
            searchResults.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": item.get("snippet", ""),
                "position": position,
                "domain": item.get("displayLink", "")
            })

        logger.info(f"google search completed: {len(searchResults)} results found")

        return {
            "success": True,
            "claim": claim,
            "results": searchResults,
            "total_results": len(searchResults),
            "metadata": {
                "search_engine": "google",
                "language": "pt",
                "api": "google-custom-search"
            },
            "error": None
        }

    except httpx.TimeoutException:
        logger.error(f"google search timeout after {timeout}s")
        print(f"\n[GOOGLE SEARCH] TIMEOUT after {timeout}s")
        print(f"[GOOGLE SEARCH] claim was: {claim[:100]}...")
        return {
            "success": False,
            "claim": claim,
            "results": [],
            "total_results": 0,
            "error": f"timeout after {timeout}s"
        }
    except Exception as e:
        logger.error(f"google search error: {e}")
        print(f"\n[GOOGLE SEARCH] ERROR: {type(e).__name__}: {str(e)[:100]}")
        return {
            "success": False,
            "claim": claim,
            "results": [],
            "total_results": 0,
            "error": str(e)
        }


async def _searchSerperClaimFallback(claim: str, maxResults: int = 10, timeout: float = 45.0) -> dict:
    """fallback claim search using serper.dev, returns same dict format as google."""
    try:
        logger.info(f"serper fallback: searching for claim: {claim[:100]}...")
        items = await serper_search(
            query=claim,
            num=min(maxResults, 10),
            language="lang_pt",
            timeout=timeout,
        )

        if not items:
            return {
                "success": False,
                "claim": claim,
                "results": [],
                "total_results": 0,
                "error": "no search results found (serper fallback)"
            }

        searchResults = []
        for position, item in enumerate(items, start=1):
            searchResults.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": item.get("snippet", ""),
                "position": position,
                "domain": item.get("displayLink", "")
            })

        logger.info(f"serper fallback completed: {len(searchResults)} results found")

        return {
            "success": True,
            "claim": claim,
            "results": searchResults,
            "total_results": len(searchResults),
            "metadata": {
                "search_engine": "serper",
                "language": "pt",
                "api": "serper-dev-fallback"
            },
            "error": None
        }

    except Exception as e:
        logger.error(f"serper fallback also failed: {e}")
        return {
            "success": False,
            "claim": claim,
            "results": [],
            "total_results": 0,
            "error": f"serper fallback failed: {e}"
        }


async def google_search(
    query: str,
    *,
    num: int = 10,
    start: int = 1,
    site_search: str | None = None,
    site_search_filter: str | None = None,  # "i" include, "e" exclude
    date_restrict: str | None = None,       # e.g., "d7", "m1"
    sort: str | None = None,                # e.g., "date:r:20240101:20241231"
    file_type: str | None = None,           # e.g., "pdf"
    safe: str | None = None,                # "active" or "off"
    language: str | None = None,            # e.g., "lang_pt"
    timeout: float = 15.0,
) -> list[Dict[str, Any]]:
    """
    performs a search using fallback cascade and returns the list of result items.
    fallback order: vertex ai search → google custom search → serper.dev.

    args:
        query: search query string
        num: number of results to return (1-10)
        start: index of first result (for pagination)
        site_search: domain to filter by (e.g., "who.int")
        site_search_filter: "i" to include only site_search, "e" to exclude
        date_restrict: relative date filter (e.g., "d7" for last 7 days)
        sort: date sorting (e.g., "date:r:20240101:20241231")
        file_type: filter by file type (e.g., "pdf")
        safe: safe search setting ("active" or "off")
        language: language restriction (e.g., "lang_pt")
        timeout: request timeout in seconds

    returns:
        list of search result items from vertex/google/serper fallback chain
    """
    try:
        return await _vertex_search_internal(
            query,
            num=num,
            site_search=site_search,
            site_search_filter=site_search_filter,
        )
    except (VertexSearchError, Exception) as vertex_err:
        logger.warning(f"vertex search failed, trying google fallback: {vertex_err}")

    try:
        return await _google_search_internal(
            query,
            num=num, start=start,
            site_search=site_search, site_search_filter=site_search_filter,
            date_restrict=date_restrict, sort=sort, file_type=file_type,
            safe=safe, language=language, timeout=timeout,
        )
    except (GoogleSearchError, httpx.TimeoutException, Exception) as google_err:
        logger.warning(f"google search failed, trying serper fallback: {google_err}")
        return await _serper_fallback(
            google_err,
            query,
            num=num,
            site_search=site_search,
            site_search_filter=site_search_filter,
            date_restrict=date_restrict,
            language=language,
            timeout=timeout,
        )


async def _vertex_search_internal(
    query: str,
    *,
    num: int = 10,
    site_search: str | None = None,
    site_search_filter: str | None = None,
) -> list[Dict[str, Any]]:
    """vertex search integration for google_search() fallback cascade."""
    if not _is_vertex_configured():
        raise VertexSearchError("vertex search is not configured")

    allowed_domains: list[str] | None = None
    if site_search and site_search_filter != "e":
        allowed_domains = [site_search]

    return await vertex_search(
        query=query,
        num=num,
        allowed_domains=allowed_domains,
    )


async def _google_search_internal(
    query: str,
    *,
    num: int = 10,
    start: int = 1,
    site_search: str | None = None,
    site_search_filter: str | None = None,
    date_restrict: str | None = None,
    sort: str | None = None,
    file_type: str | None = None,
    safe: str | None = None,
    language: str | None = None,
    timeout: float = 15.0,
) -> list[Dict[str, Any]]:
    """original google search logic without fallback."""
    api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
    cse_cx = os.environ.get("GOOGLE_CSE_CX", "")

    if not api_key or not cse_cx:
        raise GoogleSearchError("missing GOOGLE_SEARCH_API_KEY or GOOGLE_CSE_CX")

    params: Dict[str, Any] = {
        "key": api_key,
        "cx": cse_cx,
        "q": query,
        "num": num,
        "start": start,
    }

    if site_search:
        params["siteSearch"] = site_search
    if site_search_filter:
        params["siteSearchFilter"] = site_search_filter
    if date_restrict:
        params["dateRestrict"] = date_restrict
    if sort:
        params["sort"] = sort
    if file_type:
        params["fileType"] = file_type
    if safe:
        params["safe"] = safe
    if language:
        params["lr"] = language

    base_url = "https://www.googleapis.com/customsearch/v1"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(base_url, params=params)

    if response.status_code != 200:
        raise GoogleSearchError(
            f"google search error: {response.status_code} {response.text}"
        )

    data = response.json()
    return data.get("items", [])


async def _serper_fallback(
    original_error: Exception,
    query: str,
    *,
    num: int = 10,
    site_search: str | None = None,
    site_search_filter: str | None = None,
    date_restrict: str | None = None,
    language: str | None = None,
    timeout: float = 15.0,
) -> list[Dict[str, Any]]:
    """attempt serper.dev as fallback; re-raise original error if serper is not configured or also fails."""
    if not _is_serper_configured():
        logger.warning("serper not configured, re-raising original error")
        raise original_error

    try:
        items = await serper_search(
            query=query,
            num=num,
            site_search=site_search,
            site_search_filter=site_search_filter,
            date_restrict=date_restrict,
            language=language,
            timeout=timeout,
        )
        logger.info(f"serper fallback succeeded: {len(items)} result(s)")
        return items
    except Exception as serper_err:
        logger.error(f"serper fallback also failed: {serper_err}")
        raise original_error from serper_err
