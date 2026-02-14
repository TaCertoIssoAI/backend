#!/usr/bin/env python3
"""
folha_article_cli.py — extract and display content from a Folha de S.Paulo article URL.

usage:
    python scripts/playground/folha/folha_article_cli.py

paste any folha.uol.com.br article link, press Enter, and the full content is shown.
empty line or Ctrl+C to quit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.playground.common import (
    Colors,
    print_header,
    print_section,
    print_error,
    print_warning,
    print_info,
    with_spinner,
)
from experiments.folha_scraper.folha_explorer import scrape_article


def _print_article(data: dict) -> None:
    print_section(f"article — HTTP {data['http_status']}  ·  {data['html_chars']:,} chars")

    if data["title"]:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{data['title']}{Colors.END}")

    if data["subtitle"]:
        print(f"{Colors.BLUE}{data['subtitle']}{Colors.END}")

    meta_parts = []
    if data["author"]:
        meta_parts.append(data["author"])
    if data["published"]:
        meta_parts.append(data["published"][:10])
    if meta_parts:
        print(f"\n{Colors.YELLOW}{' · '.join(meta_parts)}{Colors.END}")

    if data["description"]:
        print(f"\n{Colors.BOLD}summary:{Colors.END}")
        print(f"  {data['description']}")

    if data["body_text"]:
        print(f"\n{Colors.BOLD}body  ({data['body_paragraphs']} paragraphs · {data['body_chars']} chars):{Colors.END}")
        print("─" * 70)
        for para in data["body_text"].split("\n\n"):
            para = para.strip()
            if not para:
                continue
            words, line, lines = para.split(), "", []
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    lines.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                lines.append(line)
            print("\n".join(f"  {l}" for l in lines))
            print()
        print("─" * 70)
    elif data["trafilatura_text"]:
        print(f"\n{Colors.BOLD}body (trafilatura fallback · {data['trafilatura_chars']} chars):{Colors.END}")
        print("─" * 70)
        print(f"  {data['trafilatura_text'][:1500]}")
        if data["trafilatura_chars"] > 1500:
            print(f"  {Colors.YELLOW}... ({data['trafilatura_chars'] - 1500} more chars){Colors.END}")
        print("─" * 70)
    else:
        print_warning("no body text extracted")


def main() -> None:
    print_header("Folha de S.Paulo Article Extractor")
    print_info("paste a folha.uol.com.br URL and press Enter  ·  empty line to quit\n")

    while True:
        try:
            raw = input(f"{Colors.BOLD}url>{Colors.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            break

        if "folha.uol.com.br" not in raw:
            print_warning("URL does not look like a Folha link — trying anyway")

        try:
            data = with_spinner(lambda: scrape_article(raw), "fetching article...")
            _print_article(data)
        except Exception as e:
            print_error(str(e))

        print()

    print_info("bye")


if __name__ == "__main__":
    main()
