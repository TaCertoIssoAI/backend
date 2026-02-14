"""
estadao_explorer.py — HTTP scraper for estadao.com.br articles.

Findings (confirmed 2026-02):
  - Title:       h1
  - Subtitle:    h2 (first one, which is the article hat/deck)
  - Author/Date: NewsArticle JSON-LD (most reliable, paywall-stable)
  - Description: <meta property="og:description">
  - Body:        all <p> inside div.news-body, filtering out noise classes
                 (headlines, li-title, ads-placeholder-label, loading-text)
  - Fallback:    trafilatura

Notes:
  - `requests` works fine — no SSL issues like G1's httpx problem
  - Estadão has a soft JS paywall; the full article text is in the HTML
  - Body paragraphs use generated CSS class names (unstable) — filtering
    by noise exclusion is more robust than selecting by generated class
"""

import json
from typing import Optional

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

# p tag classes that belong to UI chrome, not article body
_NOISE_CLASSES = {
    "headlines",
    "description",
    "loading-text",
    "ads-placeholder-label",
    "li-title",
}


def _get_jsonld(soup: BeautifulSoup, schema_type: str) -> Optional[dict]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if data.get("@type") == schema_type:
                return data
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def scrape_article(url: str) -> dict:
    """
    Extract the full content of an Estadão article page.

    Returns a dict with: url, http_status, html_chars, title, subtitle,
    author, published, description, body_paragraphs, body_chars,
    body_text, trafilatura_chars, trafilatura_text.
    """
    resp = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    # title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # subtitle — first h2 (article deck/hat), skip nav h2s by length
    subtitle = ""
    for h2 in soup.find_all("h2"):
        txt = h2.get_text(strip=True)
        if len(txt) > 30:
            subtitle = txt
            break

    # author and date from NewsArticle JSON-LD (most reliable)
    author = ""
    published = ""
    article_ld = _get_jsonld(soup, "NewsArticle")
    if article_ld:
        author_field = article_ld.get("author", {})
        if isinstance(author_field, dict):
            author = author_field.get("name", "")
        elif isinstance(author_field, list) and author_field:
            author = author_field[0].get("name", "")
        published = article_ld.get("datePublished", "")

    # description from og meta
    description = ""
    meta_desc = soup.find("meta", attrs={"property": "og:description"})
    if meta_desc:
        description = meta_desc.get("content", "")

    # body: all <p> inside div.news-body, excluding noise classes
    body_paragraphs = []
    news_body = soup.select_one("div.news-body")
    if news_body:
        for p in news_body.find_all("p"):
            p_classes = set(p.get("class") or [])
            # skip any p whose classes overlap with known noise
            if p_classes & _NOISE_CLASSES:
                continue
            txt = p.get_text(strip=True)
            if len(txt) > 40:
                body_paragraphs.append(txt)

    body_text = "\n\n".join(body_paragraphs)

    # trafilatura fallback
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
