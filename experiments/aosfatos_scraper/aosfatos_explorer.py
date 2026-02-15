"""
aosfatos_explorer.py — HTTP scraper for aosfatos.org articles.

Findings (confirmed 2026-02):
  - Title:         h1
  - Description:   <meta property="og:description">
  - Author/Date:   ClaimReview JSON-LD (author[0].name, datePublished)
  - Verdict:       ClaimReview JSON-LD → reviewRating.alternateName
  - Claim:         ClaimReview JSON-LD → claimReviewed
  - Review body:   ClaimReview JSON-LD → reviewBody
  - Body:          <p> with no class attribute inside div.prose, filtering
                   out noise by excluding <p> that have any CSS class
  - Fallback:      trafilatura

Notes:
  - Both `requests` and `httpx` fail with UNEXPECTED_EOF_WHILE_READING
    (Python 3.14 TLS strictness). Must use urllib with permissive SSL ctx.
  - Site is Next.js based; full content is server-rendered in the HTML.
  - ClaimReview JSON-LD is rich — author, date, verdict, claim, review body.
  - Body <p> tags inside div.prose have no class attribute; noise elements
    (related article titles, copyright) always carry Tailwind utility classes.
"""

import json
import ssl
import urllib.request
from typing import Optional

import trafilatura
from bs4 import BeautifulSoup

TIMEOUT = 20

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# permissive SSL context — needed because Python 3.14 TLS strictness causes
# UNEXPECTED_EOF_WHILE_READING on aosfatos.org with requests/httpx
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE
_SSL_CTX.set_ciphers("DEFAULT:@SECLEVEL=1")


def _fetch(url: str) -> tuple[str, int]:
    """Return (html, http_status) for the given URL."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=TIMEOUT) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        return html, resp.status


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
    Extract the full content of an Aos Fatos article page.

    Returns a dict with: url, http_status, html_chars, title, author,
    published, description, claim_reviewed, verdict, review_body,
    body_paragraphs, body_chars, body_text, trafilatura_chars, trafilatura_text.
    """
    html, http_status = _fetch(url)
    soup = BeautifulSoup(html, "lxml")

    # title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # description
    description = ""
    for prop in ["og:description", "description"]:
        meta = soup.find("meta", attrs={"property": prop}) or soup.find(
            "meta", attrs={"name": prop}
        )
        if meta:
            description = meta.get("content", "")
            if description:
                break

    # author, date, verdict, claim, review body from ClaimReview JSON-LD
    author = ""
    published = ""
    verdict = ""
    claim_reviewed = ""
    review_body = ""

    claim_ld = _get_jsonld(soup, "ClaimReview")
    if claim_ld:
        author_field = claim_ld.get("author", {})
        if isinstance(author_field, list) and author_field:
            author = author_field[0].get("name", "")
        elif isinstance(author_field, dict):
            author = author_field.get("name", "")

        published = claim_ld.get("datePublished", "")
        claim_reviewed = claim_ld.get("claimReviewed", "")
        review_body = claim_ld.get("reviewBody", "")

        rating = claim_ld.get("reviewRating", {})
        verdict = rating.get("alternateName", "") or rating.get("ratingValue", "")

    # body: <p> with no class inside div.prose (noise elements always have classes)
    body_paragraphs = []
    prose = soup.select_one("div.prose")
    if prose:
        for p in prose.find_all("p"):
            if p.get("class"):
                continue  # skip noise: related titles, footer text, etc.
            txt = p.get_text(strip=True)
            if len(txt) > 30:
                body_paragraphs.append(txt)

    body_text = "\n\n".join(body_paragraphs)

    # trafilatura fallback
    traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

    return {
        "url": url,
        "http_status": http_status,
        "html_chars": len(html),
        "title": title,
        "author": author,
        "published": published,
        "description": description,
        "claim_reviewed": claim_reviewed,
        "verdict": verdict,
        "review_body": review_body,
        "body_paragraphs": len(body_paragraphs),
        "body_chars": len(body_text),
        "body_text": body_text,
        "trafilatura_chars": len(traf_text),
        "trafilatura_text": traf_text,
    }
