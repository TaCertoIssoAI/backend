"""
vertex ai search (discovery engine) integration — uses google cloud python sdk.
"""

import os
import logging
from typing import Any, Dict, Sequence
from urllib.parse import urlparse
import asyncio

logger = logging.getLogger(__name__)


class VertexSearchError(Exception):
    """exception raised when vertex ai search api fails"""
    pass


def _is_vertex_configured() -> bool:
    """check whether all required vertex search env vars are set."""
    return all(
        os.environ.get(v)
        for v in (
            "VERTEX_SEARCH_PROJECT_ID",
            "VERTEX_SEARCH_LOCATION",
            "VERTEX_SEARCH_DATA_STORE_ID",
            "GOOGLE_APPLICATION_CREDENTIALS",
        )
    )


def _build_vertex_filter(allowed_domains: Sequence[str] | None = None) -> str | None:
    if not allowed_domains:
        return None

    normalized_domains = [d.strip() for d in allowed_domains if d and d.strip()]
    if not normalized_domains:
        return None

    filter_parts = [f'siteSearch:"https://{domain}/"' for domain in normalized_domains]
    return " OR ".join(filter_parts)


def _sync_vertex_search(query: str, num: int, allowed_domains: Sequence[str] | None = None):
    from google.cloud import discoveryengine_v1 as discoveryengine

    project_id = os.environ.get("VERTEX_SEARCH_PROJECT_ID", "")
    location = os.environ.get("VERTEX_SEARCH_LOCATION", "global")
    data_store_id = os.environ.get("VERTEX_SEARCH_DATA_STORE_ID", "")

    if not project_id or not data_store_id:
        raise VertexSearchError("missing VERTEX_SEARCH_PROJECT_ID or VERTEX_SEARCH_DATA_STORE_ID")

    client = discoveryengine.SearchServiceClient()

    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection"
        f"/dataStores/{data_store_id}/servingConfigs/default_serving_config"
    )

    request_kwargs: Dict[str, Any] = {
        "serving_config": serving_config,
        "query": query,
        "page_size": min(num, 100),
        "content_search_spec": discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            )
        ),
    }

    search_filter = _build_vertex_filter(allowed_domains)
    if search_filter:
        request_kwargs["filter"] = search_filter

    request = discoveryengine.SearchRequest(
        **request_kwargs,
    )

    response = client.search(request=request)
    return response


async def vertex_search(
    query: str,
    *,
    num: int = 10,
    allowed_domains: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:

    response = await asyncio.to_thread(_sync_vertex_search, query, num, allowed_domains)

    items: list[Dict[str, Any]] = []

    for result in response.results:
        doc = result.document
        derived = dict(doc.derived_struct_data or {})

        link = derived.get("link", "")
        title = derived.get("title") or derived.get("htmlTitle", "")

        snippet = ""
        snippets = derived.get("snippets", [])
        if snippets:
            snippet = snippets[0].get("snippet", "")
        if not snippet:
            snippet = derived.get("snippet", "")

        display_link = derived.get("displayLink", "")
        if not display_link and link:
            parsed = urlparse(link)
            display_link = parsed.netloc

        if not link:
            continue

        items.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "displayLink": display_link,
        })

    return items
