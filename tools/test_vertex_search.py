#!/usr/bin/env python3
"""
simple CLI to test vertex ai search integration.

usage:
    python tools/test_vertex_search.py "vacina causa autismo"
    python tools/test_vertex_search.py "lula preso" --num 5
    python tools/test_vertex_search.py "mudança climática" --raw

requires env vars:
    GOOGLE_APPLICATION_CREDENTIALS  (path to service account JSON)
    VERTEX_SEARCH_PROJECT_ID
    VERTEX_SEARCH_LOCATION          (default: global)
    VERTEX_SEARCH_DATA_STORE_ID
"""

import argparse
import asyncio
import json
import os
import sys

# add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ai.context.web.vertex_search import (
    vertex_search,
    _is_vertex_configured,
    VertexSearchError,
)


def _check_env():
    """print env var status and exit if not configured."""
    vars_needed = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_SEARCH_PROJECT_ID",
        "VERTEX_SEARCH_LOCATION",
        "VERTEX_SEARCH_DATA_STORE_ID",
    ]
    print("--- env vars ---")
    all_ok = True
    for v in vars_needed:
        val = os.environ.get(v, "")
        status = "OK" if val else "MISSING"
        preview = val[:8] + "..." if len(val) > 8 else val
        print(f"  {v}: {status} ({preview})")
        if not val:
            all_ok = False

    if not all_ok:
        print("\nset the missing env vars and try again.")
        sys.exit(1)
    print()


async def main():
    parser = argparse.ArgumentParser(description="test vertex ai search")
    parser.add_argument("query", help="search query")
    parser.add_argument("--num", type=int, default=10, help="max results (default: 10)")
    parser.add_argument("--timeout", type=float, default=15.0, help="timeout in seconds")
    parser.add_argument("--raw", action="store_true", help="print raw JSON response")
    args = parser.parse_args()

    _check_env()

    print(f"query: {args.query}")
    print(f"num: {args.num}, timeout: {args.timeout}s")
    print("---")

    try:
        items = await vertex_search(
            args.query,
            num=args.num,
        )
    except VertexSearchError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if args.raw:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return

    if not items:
        print("no results found.")
        return

    print(f"\n{len(items)} result(s):\n")
    for i, item in enumerate(items, 1):
        print(f"[{i}] {item.get('title', '(no title)')}")
        print(f"    {item.get('link', '')}")
        print(f"    {item.get('displayLink', '')}")
        snippet = item.get("snippet", "")
        if snippet:
            print(f"    {snippet[:150]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
