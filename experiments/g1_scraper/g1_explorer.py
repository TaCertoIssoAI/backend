#!/usr/bin/env python3
"""
g1_explorer.py — exploratory HTTP scraper for g1.globo.com

CreateFindings:
  Q1 (home page)    — extractable: 20-30 .ghtml links, headlines from h1/h2/h3
  Q2 (search page)  — NOT extractable via HTTP: results are JS-rendered client-side;
                       window.__CONTEXT__ has search config but no article data
  Q3 (article page) — fully extractable: title, author, date, full body via .content-text

Notes:
  - Use `requests`, NOT `httpx` — httpx triggers SSL fingerprint rejection on article pages
  - Article pages are ~1MB but most of that is two large inline JS bundles (410KB + 155KB)
  - The actual article text sits in 13 `.content-text` divs as plain <p> tags
  - trafilatura also works well as a fallback extractor
  - RSS feeds are all 404 — G1 removed them
  - Search results require JavaScript execution (out of scope for HTTP-only approach)
"""

import json
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import requests
import trafilatura
from bs4 import BeautifulSoup

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)

TIMEOUT = 20
SECTION_SEP = "\n" + "=" * 70 + "\n"


# ---------------------------------------------------------------------------
# Q1 — home page extraction
# ---------------------------------------------------------------------------


def scrape_home(url: str = "https://g1.globo.com/") -> dict:
    """
    Extract article links and headlines from a G1 index/home page.
    Works on any listing page (section pages, topic pages, etc.).
    Returns .ghtml article links and h1/h2/h3 headlines found in the HTML.
    """
    resp = SESSION.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # collect unique .ghtml article links
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".ghtml" in href and href not in seen:
            seen.add(href)
            title_text = a.get_text(strip=True)
            links.append({"url": href, "link_text": title_text[:120]})

    # headlines from heading tags
    headlines = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        txt = tag.get_text(strip=True)
        if len(txt) > 20:
            headlines.append(txt)

    return {
        "url": url,
        "http_status": resp.status_code,
        "html_chars": len(resp.text),
        "article_links": links,
        "headlines": headlines,
    }


# ---------------------------------------------------------------------------
# Q2 — search page (HTTP limitation documented)
# ---------------------------------------------------------------------------


def scrape_search(query: str, url: Optional[str] = None) -> dict:
    """
    Attempt to extract search results from g1.globo.com/busca/.

    FINDING: Search results are entirely JS-rendered. The initial HTML response
    contains only site config in window.__CONTEXT__ but zero article data.
    To get real results you would need a headless browser (Selenium/Playwright).

    This function returns what IS available from the static HTML:
    - The search config (queryId, searchProfile)
    - Confirmation that 0 articles are in the initial HTML
    """
    if url is None:
        url = f"https://g1.globo.com/busca/?q={quote_plus(query)}"

    resp = SESSION.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # article links in initial HTML (expected: 0)
    article_links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if ".ghtml" in a["href"]
    ]

    # extract search config from window.__CONTEXT__
    search_config = {}
    script_text = ""
    for script in soup.find_all("script"):
        if script.string and "window.__CONTEXT__" in script.string:
            script_text = script.string
            break

    match = re.search(r"window\.__CONTEXT__\s*=\s*(\{.*?\});?\s*(?:window\.|</script>)", script_text, re.DOTALL)
    if match:
        try:
            ctx = json.loads(match.group(1))
            config = ctx.get("api_content", {}).get("resource", {}).get("config", {})
            search_config = {
                "queryId": config.get("queryId", {}),
                "searchProfile": config.get("searchProfile", ""),
                "filters": [f["slug"] for f in config.get("filters", [])],
            }
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "url": url,
        "query": query,
        "http_status": resp.status_code,
        "html_chars": len(resp.text),
        "article_links_in_html": len(article_links),
        "js_rendered": True,
        "note": "Results require JavaScript execution. HTTP-only gives 0 articles.",
        "search_config_from_context": search_config,
    }


# ---------------------------------------------------------------------------
# Q3 — full article extraction
# ---------------------------------------------------------------------------


def scrape_article(url: str) -> dict:
    """
    Extract the full content of a G1 article page.

    Reliable selectors (confirmed working 2026-02):
    - Title:     h1 (first match)
    - Author:    <meta name="author">
    - Date:      <meta property="article:published_time">
    - Body:      all .content-text divs — handles both regular articles (<p>)
                 and Fato ou Fake layout (<ul>, <blockquote>, <div.content-intertitle>)

    trafilatura is provided as an alternative/fallback extractor.

    IMPORTANT: use `requests`, NOT `httpx` — httpx gets SSL-rejected by G1.
    """
    resp = SESSION.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    # title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # subtitle
    subtitle = ""
    for sel in [".content-head__subtitle", "h2"]:
        tag = soup.select_one(sel)
        if tag:
            subtitle = tag.get_text(strip=True)
            break

    # author from meta
    author = ""
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author:
        author = meta_author.get("content", "")

    # published date
    published = ""
    for prop in ["article:published_time", "og:updated_time"]:
        meta = soup.find("meta", attrs={"property": prop})
        if meta:
            published = meta.get("content", "")
            break

    # og:description — useful as a summary
    description = ""
    meta_desc = soup.find("meta", attrs={"property": "og:description"})
    if meta_desc:
        description = meta_desc.get("content", "")

    # body: iterate direct children of each .content-text div in document order.
    # handles both regular articles (text in <p>) and fato-ou-fake layout
    # (text in <ul.content-unordered-list>, <blockquote.content-blockquote>,
    # and section headers in <div.content-intertitle>).
    # stops when a navigation intertitle ("Veja também", "VÍDEOS") is found.
    _NAV_MARKERS = {"veja também", "vídeos:", "veja mais", "assine"}
    body_paragraphs = []
    stop = False

    for div in soup.select(".content-text"):
        if stop:
            break
        for child in div.children:
            if not hasattr(child, "name") or not child.name:
                continue
            child_classes = set(child.get("class") or [])
            txt = child.get_text(separator=" ", strip=True)

            if child.name == "div" and "content-intertitle" in child_classes:
                if any(m in txt.lower() for m in _NAV_MARKERS):
                    stop = True
                    break
                if len(txt) > 5:
                    body_paragraphs.append(txt)

            elif child.name == "p":
                if len(txt) > 30:
                    body_paragraphs.append(txt)

            elif child.name == "ul" and "content-unordered-list" in child_classes:
                if len(txt) > 30:
                    body_paragraphs.append(txt)

            elif child.name == "blockquote" and "content-blockquote" in child_classes:
                if len(txt) > 30:
                    body_paragraphs.append(txt)

    body_text = "\n\n".join(body_paragraphs)

    # trafilatura as fallback
    traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

    return {
        "url": url,
        "http_status": resp.status_code,
        "html_chars": len(html),
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "published": published,
        "description": description,
        "body_paragraphs": len(body_paragraphs),
        "body_chars": len(body_text),
        "body_text": body_text,
        "trafilatura_chars": len(traf_text),
        "trafilatura_text": traf_text,
    }


# ---------------------------------------------------------------------------
# main demo
# ---------------------------------------------------------------------------


def pretty(obj: object, label: str = "") -> None:
    if label:
        print(SECTION_SEP + f"  {label}" + SECTION_SEP)

    # for long text fields, print summary + preview
    if isinstance(obj, dict):
        printable = {}
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 400:
                printable[k] = f"[{len(v)} chars] " + v[:400] + "..."
            elif isinstance(v, list) and len(v) > 8:
                printable[k] = v[:8] + [f"... (+{len(v) - 8} more)"]
            else:
                printable[k] = v
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    print(SECTION_SEP + "  G1 GLOBO — HTTP EXPLORER" + SECTION_SEP)

    # Q1
    print("[Q1] home page")
    home = scrape_home("https://g1.globo.com/")
    pretty(home, "Q1 — home page")

    # Q2
    print("[Q2] search page")
    search = scrape_search("vacinas fake news")
    pretty(search, "Q2 — search page")

    # Q3 — use first article found in Q1, fallback to hardcoded url
    article_url = None
    for link in home.get("article_links", []):
        if "g1.globo.com" in link["url"]:
            article_url = link["url"]
            break
    if not article_url:
        article_url = (
            "https://g1.globo.com/sp/vale-do-paraiba-regiao/noticia/2026/02/13/"
            "campos-do-jordao-realiza-teste-de-sirene-de-alerta-para-risco-de-"
            "deslizamento-neste-sabado-14.ghtml"
        )

    print(f"[Q3] article: {article_url}")
    article = scrape_article(article_url)
    pretty(article, "Q3 — article extraction")

    print(SECTION_SEP + "  done" + SECTION_SEP)


if __name__ == "__main__":
    main()
