#!/usr/bin/env python3
"""
google_search_cli.py — interactive CLI for Google Custom Search API.

usage:
    python scripts/playground/google/google_search_cli.py

configuration (edit in code below):
    SITE_FILTER   — restrict results to a domain, e.g. "g1.globo.com"
                    set to None to search the open web
    NUM_RESULTS   — number of results per query (max 10)
    DATE_RESTRICT — relative date window, e.g. "m3" (last 3 months) or None
"""

import asyncio
import os
import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.playground.common import (
    Colors,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    with_spinner,
)
from app.ai.context.web.google_search import google_search, GoogleSearchError


# ─── configuration (edit here) ────────────────────────────────────────────────

# estadao.com.br
# folha.uol.com.br
# g1.globo.com
# aosfatos.org
SITE_FILTER: str | None = "aosfatos.org"   # e.g. "g1.globo.com" or None for open web
NUM_RESULTS: int = 10                        # 1–10
DATE_RESTRICT: str | None = None            # e.g. "d7", "m3", "y1" or None


# ─── helpers ──────────────────────────────────────────────────────────────────

def _check_env() -> bool:
    missing = [v for v in ("GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_CX") if not os.environ.get(v)]
    if missing:
        print_error(f"missing environment variables: {', '.join(missing)}")
        print_info("set them before running:\n  export GOOGLE_SEARCH_API_KEY=...\n  export GOOGLE_CSE_CX=...")
        return False
    return True


def _print_config() -> None:
    print_section("active configuration")
    rows = {
        "site filter":   SITE_FILTER or "(none — open web)",
        "results":       NUM_RESULTS,
        "date restrict": DATE_RESTRICT or "(none)",
    }
    max_k = max(len(k) for k in rows)
    for k, v in rows.items():
        print(f"  {Colors.BOLD}{k.ljust(max_k)}{Colors.END}  {v}")


def _print_results(items: list, query: str) -> None:
    if not items:
        print_warning("no results found")
        return

    print_success(f"{len(items)} result(s) for: {Colors.BOLD}{query}{Colors.END}")

    for i, item in enumerate(items, 1):
        title   = item.get("title", "")
        link    = item.get("link", "")
        snippet = item.get("snippet", "").replace("\n", " ")
        domain  = item.get("displayLink", "")
        date    = item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", "")

        print(f"\n  {Colors.BOLD}{Colors.CYAN}{i}.{Colors.END} {Colors.BOLD}{title}{Colors.END}")
        if domain:
            print(f"     {Colors.YELLOW}{domain}{Colors.END}", end="")
            if date:
                print(f"  ·  {date[:10]}", end="")
            print()
        if snippet:
            # wrap snippet at ~80 chars
            words, line, lines = snippet.split(), "", []
            for word in words:
                if len(line) + len(word) + 1 > 78:
                    lines.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                lines.append(line)
            for l in lines:
                print(f"     {l}")
        print(f"     {Colors.CYAN}{link}{Colors.END}")


async def _run_query(query: str) -> None:
    try:
        items = await google_search(
            query,
            num=NUM_RESULTS,
            site_search=SITE_FILTER,
            site_search_filter="i" if SITE_FILTER else None,
            date_restrict=DATE_RESTRICT,
            sort="date",
        )
        _print_results(items, query)
    except GoogleSearchError as e:
        print_error(str(e))


# ─── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print_header("Google Custom Search — interactive CLI")

    if not _check_env():
        sys.exit(1)

    _print_config()

    print_info("\ntype a query and press Enter  ·  empty line to quit\n")

    while True:
        try:
            raw = input(f"{Colors.BOLD}search>{Colors.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            break

        with_spinner(lambda: asyncio.run(_run_query(raw)), "searching...")
        print()

    print_info("bye")


if __name__ == "__main__":
    main()
