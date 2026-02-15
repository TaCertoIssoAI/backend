"""
domain-specific scrapers for brazilian news outlets.

each scraper uses tailored CSS selectors and JSON-LD extraction
for better content quality than generic BeautifulSoup fallback.
all functions are synchronous and return the standard apify_utils dict:
  {"success": bool, "content": str, "metadata": dict, "error": str|None}
"""

import json
import logging
import ssl
import urllib.request
from typing import Optional

import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

_TIMEOUT = 20


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _type_matches(node: dict, schema_type: str) -> bool:
    """check if a JSON-LD node matches a schema type (handles list or string)"""
    t = node.get("@type", "")
    return schema_type in t if isinstance(t, list) else t == schema_type


def _get_jsonld(soup: BeautifulSoup, schema_type: str) -> Optional[dict]:
    """extract the first JSON-LD block matching the given schema type"""
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


def _build_result(body_text: str, traf_text: str, title: str, tool_name: str) -> dict:
    """build the standard result dict, preferring body_text over trafilatura"""
    content = body_text if body_text.strip() else traf_text
    if not content or len(content) < 50:
        return {
            "success": False,
            "content": "",
            "metadata": {"extraction_tool": tool_name},
            "error": "extracted content too short or empty",
        }
    return {
        "success": True,
        "content": content,
        "metadata": {
            "extraction_tool": tool_name,
            "title": title,
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# g1.globo.com
# ---------------------------------------------------------------------------

# navigation markers that signal end of article body
_G1_NAV_MARKERS = {"veja também", "vídeos:", "veja mais", "assine"}


def scrape_g1_article(url: str) -> dict:
    """
    scrape a g1.globo.com article using .content-text selectors.
    uses requests (NOT httpx — httpx triggers SSL rejection on G1).
    """
    try:
        logger.info(f"scraping g1 article: {url}")
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        # title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # body: iterate .content-text divs, stop at navigation markers
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
                    if any(m in txt.lower() for m in _G1_NAV_MARKERS):
                        stop = True
                        break
                    if len(txt) > 5:
                        body_paragraphs.append(txt)
                elif child.name == "p" and len(txt) > 30:
                    body_paragraphs.append(txt)
                elif child.name == "ul" and "content-unordered-list" in child_classes and len(txt) > 30:
                    body_paragraphs.append(txt)
                elif child.name == "blockquote" and "content-blockquote" in child_classes and len(txt) > 30:
                    body_paragraphs.append(txt)

        body_text = "\n\n".join(body_paragraphs)
        traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

        return _build_result(body_text, traf_text, title, "g1_scraper")

    except Exception as e:
        logger.error(f"g1 scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# estadao.com.br
# ---------------------------------------------------------------------------

# p tag classes that belong to UI chrome, not article body
_ESTADAO_NOISE_CLASSES = {
    "headlines", "description", "loading-text",
    "ads-placeholder-label", "li-title",
}


def scrape_estadao_article(url: str) -> dict:
    """scrape an estadao.com.br article using div.news-body with noise-class exclusion."""
    try:
        logger.info(f"scraping estadao article: {url}")
        resp = _SESSION.get(url, timeout=_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        # title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # body: all <p> inside div.news-body, excluding noise classes
        body_paragraphs = []
        news_body = soup.select_one("div.news-body")
        if news_body:
            for p in news_body.find_all("p"):
                p_classes = set(p.get("class") or [])
                if p_classes & _ESTADAO_NOISE_CLASSES:
                    continue
                txt = p.get_text(strip=True)
                if len(txt) > 40:
                    body_paragraphs.append(txt)

        body_text = "\n\n".join(body_paragraphs)
        traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

        return _build_result(body_text, traf_text, title, "estadao_scraper")

    except Exception as e:
        logger.error(f"estadao scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# folha.uol.com.br
# ---------------------------------------------------------------------------

def _normalize_folha_url(url: str) -> str:
    """rewrite www./bare folha.uol.com.br to www1 to avoid section-page redirects."""
    parsed = urlparse(url)
    if parsed.netloc in ("folha.uol.com.br", "www.folha.uol.com.br"):
        parsed = parsed._replace(netloc="www1.folha.uol.com.br")
    return urlunparse(parsed)


_FOLHA_BODY_SELECTORS = [".c-news__body", ".noticia__main--materia"]


def scrape_folha_article(url: str) -> dict:
    """scrape a folha.uol.com.br article using .c-news__body, with URL normalization."""
    try:
        logger.info(f"scraping folha article: {url}")
        canonical_url = _normalize_folha_url(url)
        resp = _SESSION.get(canonical_url, timeout=_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        # always decode from bytes — requests mis-detects encoding as ISO-8859-1
        html = resp.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        # title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # body: classless <p> inside body containers
        body_paragraphs = []
        for sel in _FOLHA_BODY_SELECTORS:
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
        traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

        return _build_result(body_text, traf_text, title, "folha_scraper")

    except Exception as e:
        logger.error(f"folha scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# aosfatos.org
# ---------------------------------------------------------------------------

# permissive SSL context — needed because Python 3.14 TLS strictness causes
# UNEXPECTED_EOF_WHILE_READING on aosfatos.org with requests/httpx
_AOSFATOS_SSL_CTX = ssl.create_default_context()
_AOSFATOS_SSL_CTX.check_hostname = False
_AOSFATOS_SSL_CTX.verify_mode = ssl.CERT_NONE
_AOSFATOS_SSL_CTX.set_ciphers("DEFAULT:@SECLEVEL=1")

_AOSFATOS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _fetch_aosfatos(url: str) -> tuple[str, int]:
    """fetch aosfatos.org using urllib with permissive SSL context."""
    req = urllib.request.Request(url, headers=_AOSFATOS_HEADERS)
    with urllib.request.urlopen(req, context=_AOSFATOS_SSL_CTX, timeout=_TIMEOUT) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        return html, resp.status


def scrape_aosfatos_article(url: str) -> dict:
    """scrape an aosfatos.org article using div.prose selectors."""
    try:
        logger.info(f"scraping aosfatos article: {url}")
        html, _ = _fetch_aosfatos(url)
        soup = BeautifulSoup(html, "lxml")

        # title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # body: <p> with no class inside div.prose
        body_paragraphs = []
        prose = soup.select_one("div.prose")
        if prose:
            for p in prose.find_all("p"):
                if p.get("class"):
                    continue
                txt = p.get_text(strip=True)
                if len(txt) > 30:
                    body_paragraphs.append(txt)

        body_text = "\n\n".join(body_paragraphs)
        traf_text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""

        return _build_result(body_text, traf_text, title, "aosfatos_scraper")

    except Exception as e:
        logger.error(f"aosfatos scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}
