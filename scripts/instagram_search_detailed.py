#!/usr/bin/env python3
"""
instagram search script using google custom search api.

searches for instagram posts and prints detailed API response fields.
uses the same API configuration as the web_search_gatherer module.
"""

import os
import sys
import asyncio
import json
from typing import Any, Dict, List

import httpx

# add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


async def search_instagram(
    query: str,
    max_results: int = 10,
    timeout: float = 45.0
) -> List[Dict[str, Any]]:
    """
    search instagram using google custom search api.

    args:
        query: search query (will be automatically filtered to instagram.com)
        max_results: maximum number of results (1-10)
        timeout: timeout in seconds for the request

    returns:
        list of search result items with all available fields
    """
    # get api credentials from environment
    api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
    cse_cx = os.environ.get("GOOGLE_CSE_CX", "")

    if not api_key or not cse_cx:
        raise ValueError("missing GOOGLE_SEARCH_API_KEY or GOOGLE_CSE_CX environment variables")

    # build search query with instagram domain filter
    instagram_query = f"{query} site:instagram.com"

    # build request parameters
    params: Dict[str, Any] = {
        "key": api_key,
        "cx": cse_cx,
        "q": instagram_query,
        "num": min(max_results, 10),  # google api max is 10
    }

    print(f"\n[SEARCH] query: {instagram_query}")
    print(f"[SEARCH] max results: {params['num']}")
    print(f"[SEARCH] timeout: {timeout}s\n")

    # perform search
    base_url = "https://www.googleapis.com/customsearch/v1"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(base_url, params=params)

    # check response status
    if response.status_code != 200:
        raise Exception(f"google api error {response.status_code}: {response.text[:200]}")

    # parse response
    data = response.json()
    items = data.get("items", [])

    print(f"[SEARCH] found {len(items)} result(s)\n")
    print("=" * 100)

    return items


def extract_author_info(item: Dict[str, Any]) -> Dict[str, str]:
    """
    extract instagram author information from search result.

    args:
        item: search result item dictionary

    returns:
        dict with author information (username, display_name, user_id)
    """
    import re

    author_info = {
        'username': None,
        'display_name': None,
        'user_id': None,
    }

    # try to get from pagemap metatags
    if 'pagemap' in item and 'metatags' in item['pagemap']:
        metatags = item['pagemap']['metatags'][0] if item['pagemap']['metatags'] else {}

        # extract user_id
        author_info['user_id'] = metatags.get('instapp:owner_user_id')

        # extract username and display name from og:title
        # format: "Display Name on Instagram: ..." or "Display Name (@username) • Instagram photo"
        og_title = metatags.get('og:title', '')
        if og_title:
            # try pattern: "Display Name on Instagram:"
            match = re.match(r'^(.+?)\s+on Instagram:', og_title)
            if match:
                author_info['display_name'] = match.group(1).strip()

            # try pattern: "Display Name (@username)"
            match = re.search(r'\(@(\w+)\)', og_title)
            if match:
                author_info['username'] = match.group(1)

        # extract username from og:description if not found yet
        # format: "X likes, Y comments - username on Date: ..."
        if not author_info['username']:
            og_desc = metatags.get('og:description', '')
            match = re.search(r'- (\w+) on \w+ \d+, \d{4}:', og_desc)
            if match:
                author_info['username'] = match.group(1)

    return author_info


def print_item_details(item: Dict[str, Any], index: int) -> None:
    """
    print detailed fields from a single search result item.

    args:
        item: search result item dictionary
        index: result number for display
    """
    print(f"\n{'=' * 100}")
    print(f"RESULT #{index}")
    print(f"{'=' * 100}\n")

    # extract and display author information first
    author_info = extract_author_info(item)
    print("INSTAGRAM AUTHOR INFO:")
    print(f"  username: {author_info['username'] or 'N/A'}")
    print(f"  display_name: {author_info['display_name'] or 'N/A'}")
    print(f"  user_id: {author_info['user_id'] or 'N/A'}")
    print()

    # all possible top-level fields
    fields_to_print = [
        'kind',
        'title',
        'htmlTitle',
        'link',
        'displayLink',
        'snippet',
        'htmlSnippet',
        'formattedUrl',
        'htmlFormattedUrl',
        'cacheId',
        'mime',
        'fileFormat',
    ]

    for field in fields_to_print:
        value = item.get(field)
        if value:
            print(f"{field}:")
            if isinstance(value, str) and len(value) > 200:
                # truncate very long strings
                print(f"  {value[:200]}...")
            else:
                print(f"  {value}")
            print()

    # print pagemap if present (contains structured data)
    if 'pagemap' in item:
        print("pagemap:")
        print(json.dumps(item['pagemap'], indent=2, ensure_ascii=False))
        print()

    # print any other fields not in our list
    other_fields = set(item.keys()) - set(fields_to_print) - {'pagemap'}
    if other_fields:
        print("other fields:")
        for field in sorted(other_fields):
            print(f"  {field}: {item[field]}")
        print()


async def main():
    """
    main execution function.
    """
    # default search query - you can modify this
    default_query = "bolsonaro autismo"

    # check for command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = default_query
        print(f"[INFO] using default query: '{query}'")
        print(f"[INFO] usage: python {sys.argv[0]} <your search query>\n")

    try:
        # perform search
        items = await search_instagram(query, max_results=10)

        # collect all authors
        all_authors = []

        # print detailed info for each result
        for idx, item in enumerate(items, start=1):
            print_item_details(item, idx)
            author_info = extract_author_info(item)
            all_authors.append(author_info)

        # print summary
        print(f"\n{'=' * 100}")
        print(f"SUMMARY: {len(items)} Instagram result(s) found")
        print(f"{'=' * 100}\n")

        # print authors summary
        print("AUTHORS FOUND:")
        for idx, author in enumerate(all_authors, start=1):
            username = author['username'] or 'N/A'
            display_name = author['display_name'] or 'N/A'
            user_id = author['user_id'] or 'N/A'
            print(f"  [{idx}] @{username} ({display_name}) - ID: {user_id}")
        print()

        # save full raw response to JSON file for inspection
        output_file = "instagram_search_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"[INFO] full raw results saved to: {output_file}")

        # save authors to separate file
        authors_file = "instagram_authors.json"
        with open(authors_file, "w", encoding="utf-8") as f:
            json.dump(all_authors, f, indent=2, ensure_ascii=False)
        print(f"[INFO] extracted authors saved to: {authors_file}\n")

    except ValueError as e:
        print(f"\n[ERROR] configuration error: {e}")
        print("[ERROR] make sure GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_CX are set in .env file")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
