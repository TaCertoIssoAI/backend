"""
vertex ai search (discovery engine) integration — uses google cloud python sdk.
"""

import os
import logging
from typing import Any, Dict
from urllib.parse import urlparse
from google.cloud import discoveryengine_v1 as discoveryengine
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


def _sync_vertex_search(query: str, num: int):
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

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        filter='siteSearch:"https://g1.globo.com/"',
        page_size=min(num, 100),
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            )
        ),
    )

    response = client.search(request=request)
    r0 = next(iter(response.results), None)
    print(dict(r0.document.derived_struct_data).keys())
    print(dict(r0.document.derived_struct_data))
    return response


async def vertex_search(
    query: str,
    *,
    num: int = 10,
) -> list[Dict[str, Any]]:

    response = await asyncio.to_thread(_sync_vertex_search, query, num)

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
