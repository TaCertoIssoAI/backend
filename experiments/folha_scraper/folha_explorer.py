"""
folha_explorer.py — HTTP scraper for folha.uol.com.br articles.

Findings (confirmed 2026-02):
  - Host:        must be www1.folha.uol.com.br — other variants (www., no
                 subdomain) redirect to section pages instead of the article.
  - Encoding:    page declares utf-8 but requests mis-detects as ISO-8859-1;
                 always decode from resp.content as UTF-8.
  - Title:       h1 (or JSON-LD headline)
  - Subtitle:    JSON-LD description field (ReportageNewsArticle)
  - Author/Date: ReportageNewsArticle JSON-LD — author[0].name, datePublished
  - Description: <meta property="og:description">
  - Body:        <p> with NO class inside div.c-news__body; noise paragraphs
                 (gallery captions, newsletter widgets, subscription UI) always
                 carry at least one CSS class.
  - Fallback:    trafilatura

Notes:
  - `requests` works fine — no SSL issues.
  - JSON-LD @type is a list ["CreativeWork", "ReportageNewsArticle"].
  - The article has a soft paywall; most body content is server-rendered and
    accessible without a subscription via direct HTTP fetch.
"""

import json
from typing import Optional
from urllib.parse import urlparse, urlunparse

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


def _normalize_url(url: str) -> str:
    """Rewrite www./bare folha.uol.com.br to www1 to avoid section-page redirects.
    Other subdomains (e.g. piaui.folha.uol.com.br) are kept as-is."""
    parsed = urlparse(url)
    if parsed.netloc in ("folha.uol.com.br", "www.folha.uol.com.br"):
        parsed = parsed._replace(netloc="www1.folha.uol.com.br")
    return urlunparse(parsed)


def _type_matches(node: dict, schema_type: str) -> bool:
    t = node.get("@type", "")
    return schema_type in t if isinstance(t, list) else t == schema_type


def _get_jsonld(soup: BeautifulSoup, schema_type: str) -> Optional[dict]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                for item in data:
                    if _type_matches(item, schema_type):
                        return item
            elif _type_matches(data, schema_type):
                return data
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def scrape_article(url: str) -> dict:
    """
    Extract the full content of a Folha de S.Paulo article page.

    Returns a dict with: url, canonical_url, http_status, html_chars, title,
    subtitle, author, published, description, body_paragraphs, body_chars,
    body_text, trafilatura_chars, trafilatura_text.
    """
    canonical_url = _normalize_url(url)
    resp = SESSION.get(canonical_url, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    # always decode from bytes — requests mis-detects encoding as ISO-8859-1
    html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # title — prefer JSON-LD headline, fall back to h1
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # subtitle, author, date from JSON-LD — try ReportageNewsArticle (main site)
    # then NewsArticle (piaui and other folha sub-sites)
    subtitle = ""
    author = ""
    published = ""
    article_ld = (
        _get_jsonld(soup, "ReportageNewsArticle")
        or _get_jsonld(soup, "NewsArticle")
    )
    if article_ld:
        subtitle = article_ld.get("description", "")
        published = article_ld.get("datePublished", "")
        author_field = article_ld.get("author", [])
        if isinstance(author_field, list) and author_field:
            author = author_field[0].get("name", "")
        elif isinstance(author_field, dict):
            author = author_field.get("name", "")

    # description from og meta
    description = ""
    meta_desc = soup.find("meta", attrs={"property": "og:description"})
    if meta_desc:
        description = meta_desc.get("content", "")

    # body: classless <p> inside known body containers, tried in order:
    #   .c-news__body          — main folha (www1.folha.uol.com.br)
    #   .noticia__main--materia — piaui.folha.uol.com.br and other sub-sites
    # noise elements always carry at least one CSS class, so classless <p> only.
    _BODY_SELECTORS = [".c-news__body", ".noticia__main--materia"]
    body_paragraphs = []
    for sel in _BODY_SELECTORS:
        news_body = soup.select_one(sel)
        if not news_body:
            continue
        for p in news_body.find_all("p"):
            if p.get("class"):
                continue
            txt = p.get_text(strip=True)
            if len(txt) > 40:
                body_paragraphs.append(txt)
        if body_paragraphs:
            break

    body_text = "\n\n".join(body_paragraphs)

    # trafilatura fallback
    traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

    return {
        "url": url,
        "canonical_url": canonical_url,
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
