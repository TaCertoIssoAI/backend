"""
apify integration utilities for web scraping
"""

import asyncio
import os
import re
import logging
import time
from typing import Optional
from enum import Enum

import httpx
from bs4 import BeautifulSoup
from apify_client import ApifyClientAsync

from app.agentic_ai.context.web.news_scrapers import (
    scrape_g1_article,
    scrape_estadao_article,
    scrape_folha_article,
    scrape_aosfatos_article,
)

logger = logging.getLogger(__name__)

# check if brotli decompression is available
try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    try:
        import brotlicffi
        BROTLI_AVAILABLE = True
    except ImportError:
        BROTLI_AVAILABLE = False
        logger.warning("brotli decompression not available - install 'brotli' or 'brotlicffi' for better compression support")

APIFY_TOKEN_ENV_KEY = "APIFY_TOKEN"


class PlatformType(Enum):
    """supported scraping platforms"""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    G1 = "g1"
    ESTADAO = "estadao"
    FOLHA = "folha"
    AOSFATOS = "aosfatos"
    GENERIC = "generic"


ACTOR_MAP = {
    PlatformType.FACEBOOK: "apify/facebook-posts-scraper",
    PlatformType.INSTAGRAM: "apify/instagram-scraper",
    PlatformType.TWITTER: "apidojo/tweet-scraper",
    PlatformType.TIKTOK: "clockworks/tiktok-scraper",
    PlatformType.GENERIC: "apify/website-content-crawler"
}

# memory limits for apify actors (in MB)
# with 4 max concurrent workers and 8GB free tier:
# 4 workers × 2048MB = 8GB (max capacity)
ACTOR_MEMORY_LIMITS = {
    PlatformType.FACEBOOK: 512,      # increased from 256 - facebook needs more for anti-scraping
    PlatformType.INSTAGRAM: 256,     # social media - lightweight
    PlatformType.TWITTER: 256,       # social media - lightweight
    PlatformType.TIKTOK: 512,        # tiktok needs more (video platform)
    PlatformType.GENERIC: 2048,      # websites with anti-scraping need full browser (2GB)
}

# compiled regex for fast non-printable character removal
# matches any character that is NOT: printable, newline, carriage return, tab, or space
NON_PRINTABLE_PATTERN = re.compile(r'[^\x20-\x7E\x0A\x0D\x09\u0080-\uFFFF]')


def has_corruption(text: str, sample_size: int = 1000, threshold: float = 0.1) -> bool:
    """
    quickly check if text contains corrupted/binary content.
    samples first N characters and checks for non-printable ratio.

    args:
        text: text to check
        sample_size: number of characters to sample
        threshold: ratio of non-printable chars that indicates corruption (0.0-1.0)

    returns:
        True if text appears corrupted
    """
    if not text:
        return False

    start_time = time.perf_counter()

    sample = text[:sample_size]
    # count non-printable chars (excluding whitespace)
    non_printable = sum(1 for c in sample if not (c.isprintable() or c in '\n\r\t '))

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    is_corrupt = (non_printable / len(sample)) > threshold

    logger.info(
        f"[BENCHMARK] corruption check: {elapsed_ms:.2f}ms | "
        f"sample={len(sample)} chars | "
        f"non_printable={non_printable} ({non_printable/len(sample)*100:.1f}%) | "
        f"corrupt={is_corrupt}"
    )

    return is_corrupt


def clean_non_printable(text: str) -> str:
    """
    remove non-printable characters from text using fast regex.
    much faster than character-by-character iteration.
    preserves: printable ASCII, whitespace, and unicode characters.

    args:
        text: text to clean

    returns:
        cleaned text
    """
    start_time = time.perf_counter()
    original_length = len(text)

    cleaned = NON_PRINTABLE_PATTERN.sub('', text)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    removed = original_length - len(cleaned)

    logger.info(
        f"[BENCHMARK] cleaned non-printable chars: {elapsed_ms:.2f}ms | "
        f"original={original_length} chars | "
        f"removed={removed} chars ({removed/original_length*100:.2f}%)"
    )

    return cleaned


def detectPlatform(url: str) -> PlatformType:
    """detect platform type from url using regex"""
    url_lower = url.lower()
    
    facebook_patterns = [r'facebook\.com', r'fb\.com', r'fb\.watch', r'm\.facebook\.com']
    instagram_patterns = [r'instagram\.com', r'instagr\.am']
    twitter_patterns = [r'twitter\.com', r'x\.com', r't\.co', r'mobile\.twitter\.com']
    tiktok_patterns = [r'tiktok\.com', r'vm\.tiktok\.com', r'vt\.tiktok\.com']
    
    for pattern in facebook_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.FACEBOOK
    
    for pattern in instagram_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.INSTAGRAM
    
    for pattern in twitter_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.TWITTER
    
    for pattern in tiktok_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.TIKTOK

    g1_patterns = [r'g1\.globo\.com']
    estadao_patterns = [r'estadao\.com\.br']
    folha_patterns = [r'folha\.uol\.com\.br']
    aosfatos_patterns = [r'aosfatos\.org']

    for pattern in g1_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.G1

    for pattern in estadao_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.ESTADAO

    for pattern in folha_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.FOLHA

    for pattern in aosfatos_patterns:
        if re.search(pattern, url_lower):
            return PlatformType.AOSFATOS

    return PlatformType.GENERIC


def getApifyClientAsync() -> ApifyClientAsync:
    """create async apify client from environment token"""
    apifyToken = os.getenv(APIFY_TOKEN_ENV_KEY)
    if not apifyToken:
        raise RuntimeError("missing APIFY_TOKEN in environment")
    
    return ApifyClientAsync(apifyToken)


async def scrapeFacebookPost(url: str, maxChars: Optional[int] = None) -> dict:
    """scrape facebook post using apify actor"""
    try:
        logger.info(f"scraping facebook post: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.FACEBOOK])
        
        runInput = {
            "startUrls": [{"url": url}],
            "resultsLimit": 1,
            "maxPostCount": 1
        }

        # limit memory to reduce RAM usage (8GB free tier optimization)
        callResult = await actorClient.call(
            run_input=runInput,
            timeout_secs=120,
            memory_mbytes=ACTOR_MEMORY_LIMITS[PlatformType.FACEBOOK]
        )
        
        if callResult is None:
            return {"success": False, "content": "", "metadata": {}, "error": "actor run failed"}
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {"success": False, "content": "", "metadata": {}, "error": "no content extracted"}
        
        item = listItemsResult.items[0]
        content = item.get("text", "") or item.get("content", "") or item.get("postText", "")
        
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        return {
            "success": True,
            "content": content,
            "metadata": {
                "platform": "facebook",
                "postUrl": item.get("postUrl", url),
                "timestamp": item.get("time", ""),
                "author": item.get("from", ""),
                "likes": item.get("likes", 0),
                "shares": item.get("shares", 0),
                "comments": item.get("comments", 0)
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"facebook scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


async def scrapeInstagramPost(url: str, maxChars: Optional[int] = None) -> dict:
    """scrape instagram post using apify actor"""
    try:
        logger.info(f"scraping instagram: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.INSTAGRAM])
        
        runInput = {"directUrls": [url], "resultsLimit": 1}

        # limit memory to reduce RAM usage (8GB free tier optimization)
        callResult = await actorClient.call(
            run_input=runInput,
            timeout_secs=120,
            memory_mbytes=ACTOR_MEMORY_LIMITS[PlatformType.INSTAGRAM]
        )
        
        if callResult is None:
            return {"success": False, "content": "", "metadata": {}, "error": "actor run failed"}
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {"success": False, "content": "", "metadata": {}, "error": "no content extracted"}
        
        item = listItemsResult.items[0]
        content = item.get("caption", "") or item.get("text", "")
        
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        return {
            "success": True,
            "content": content,
            "metadata": {
                "platform": "instagram",
                "postUrl": item.get("url", url),
                "timestamp": item.get("timestamp", ""),
                "author": item.get("ownerUsername", ""),
                "likes": item.get("likesCount", 0),
                "comments": item.get("commentsCount", 0)
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"instagram scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


async def scrapeTwitterPost(url: str, maxChars: Optional[int] = None) -> dict:
    """scrape twitter/x post using apify actor"""
    try:
        logger.info(f"scraping twitter: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.TWITTER])
        
        runInput = {"startUrls": [url], "maxItems": 1}

        # limit memory to reduce RAM usage (8GB free tier optimization)
        callResult = await actorClient.call(
            run_input=runInput,
            timeout_secs=120,
            memory_mbytes=ACTOR_MEMORY_LIMITS[PlatformType.TWITTER]
        )
        
        if callResult is None:
            return {"success": False, "content": "", "metadata": {}, "error": "actor run failed"}
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {"success": False, "content": "", "metadata": {}, "error": "no content extracted"}
        
        item = listItemsResult.items[0]
        content = item.get("text", "") or item.get("full_text", "")
        
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        return {
            "success": True,
            "content": content,
            "metadata": {
                "platform": "twitter",
                "postUrl": item.get("url", url),
                "timestamp": item.get("created_at", ""),
                "author": item.get("author", {}).get("userName", ""),
                "likes": item.get("likeCount", 0),
                "retweets": item.get("retweetCount", 0),
                "replies": item.get("replyCount", 0)
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"twitter scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


async def scrapeTikTokPost(url: str, maxChars: Optional[int] = None) -> dict:
    """scrape tiktok video using apify actor"""
    try:
        logger.info(f"scraping tiktok: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.TIKTOK])
        
        runInput = {"postURLs": [url], "resultsPerPage": 1}

        # limit memory to reduce RAM usage (8GB free tier optimization)
        callResult = await actorClient.call(
            run_input=runInput,
            timeout_secs=120,
            memory_mbytes=ACTOR_MEMORY_LIMITS[PlatformType.TIKTOK]
        )
        
        if callResult is None:
            return {"success": False, "content": "", "metadata": {}, "error": "actor run failed"}
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {"success": False, "content": "", "metadata": {}, "error": "no content extracted"}
        
        item = listItemsResult.items[0]
        content = item.get("text", "") or item.get("description", "")
        
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        return {
            "success": True,
            "content": content,
            "metadata": {
                "platform": "tiktok",
                "postUrl": item.get("webVideoUrl", url),
                "timestamp": item.get("createTime", ""),
                "author": item.get("authorMeta", {}).get("name", ""),
                "likes": item.get("diggCount", 0),
                "shares": item.get("shareCount", 0),
                "comments": item.get("commentCount", 0),
                "plays": item.get("playCount", 0)
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"tiktok scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


async def scrapeGenericSimple(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape generic website using simple http request (no browser).
    lightweight approach for static content - tries first before using apify.
    """
    try:
        logger.info(f"attempting simple scraping (no browser) for: {url}")

        # configure accept-encoding based on brotli availability
        # only request brotli if we can decompress it
        accept_encoding = "gzip, deflate"
        if BROTLI_AVAILABLE:
            accept_encoding += ", br"

        # configure httpx client with realistic headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": accept_encoding,
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # check if content is html
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logger.warning(f"non-html content type: {content_type}")
                return {
                    "success": False,
                    "content": "",
                    "metadata": {},
                    "error": f"unsupported content type: {content_type}"
                }
            
            html_content = response.text

            # detect if we got corrupted/binary content (decompression failure)
            if has_corruption(html_content, sample_size=1000, threshold=0.2):
                logger.warning("detected binary/corrupted content (decompression failure?)")
                return {
                    "success": False,
                    "content": "",
                    "metadata": {},
                    "error": "received corrupted content - possible decompression failure"
                }

            # parse with beautifulsoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            # remove script and style elements
            for script in soup(["script", "style", "iframe", "noscript"]):
                script.decompose()
            
            # extract metadata
            title = ""
            if soup.title:
                title = soup.title.string or ""
            
            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"}) or \
                       soup.find("meta", attrs={"property": "og:description"})
            if meta_desc:
                description = meta_desc.get("content", "")
            
            # try to extract main content (prioritize article/main tags)
            main_content = soup.find("main") or soup.find("article") or soup.find("body")
            
            # use BeautifulSoup's get_text() instead of html2text
            # this preserves encoding better and avoids corruption
            if main_content:
                text_content = main_content.get_text(separator='\n', strip=True)
            else:
                text_content = soup.get_text(separator='\n', strip=True)
            
            logger.info(f"extracted {len(text_content)} chars using BeautifulSoup get_text()")
            
            # clean up excessive whitespace
            text_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_content)
            text_content = text_content.strip()

            # only clean non-printable chars if we detect issues (faster)
            if has_corruption(text_content, sample_size=500, threshold=0.05):
                logger.info("[BENCHMARK] corruption detected - cleaning non-printable characters")
                text_content = clean_non_printable(text_content)
            else:
                logger.info("[BENCHMARK] content is clean - skipping character cleaning (performance optimization)")
            
            if not text_content or len(text_content) < 50:
                logger.warning(f"extracted content too short: {len(text_content)} chars")
                return {
                    "success": False,
                    "content": "",
                    "metadata": {},
                    "error": "extracted content too short or empty"
                }
            
            # apply max chars limit
            if maxChars and len(text_content) > maxChars:
                text_content = text_content[:maxChars]
            
            logger.info(f"simple scraping successful: {len(text_content)} chars extracted")
            
            return {
                "success": True,
                "content": text_content,
                "metadata": {
                    "platform": "generic_simple",
                    "url": str(response.url),
                    "title": title.strip(),
                    "description": description.strip(),
                    "scraping_method": "simple_http"
                },
                "error": None
            }
            
    except httpx.HTTPStatusError as e:
        logger.warning(f"http error during simple scraping: {e.response.status_code}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": f"http error {e.response.status_code}"
        }
    except httpx.TimeoutException:
        logger.warning(f"timeout during simple scraping")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": "request timeout"
        }
    except Exception as e:
        logger.warning(f"simple scraping failed: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeGenericWebsite(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape any website using apify actor (with browser).
    used as fallback when simple scraping fails.
    """
    try:
        logger.info(f"scraping generic website with apify: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.GENERIC])
        
        runInput = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": 1,
            "crawlerType": "playwright:adaptive"  # adaptive: uses cheerio when possible, playwright when needed
        }

        # limit memory to reduce RAM usage (8GB free tier optimization)
        callResult = await actorClient.call(
            run_input=runInput,
            timeout_secs=120,
            memory_mbytes=ACTOR_MEMORY_LIMITS[PlatformType.GENERIC]
        )
        
        if callResult is None:
            return {"success": False, "content": "", "metadata": {}, "error": "actor run failed"}
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {"success": False, "content": "", "metadata": {}, "error": "no content extracted"}
        
        item = listItemsResult.items[0]
        raw_content = item.get("text", "") or item.get("markdown", "") or item.get("html", "")

        # only clean non-printable chars if we detect issues (faster)
        if has_corruption(raw_content, sample_size=500, threshold=0.05):
            logger.info("[BENCHMARK] corruption detected in apify content - cleaning non-printable characters")
            content = clean_non_printable(raw_content)
        else:
            logger.info("[BENCHMARK] apify content is clean - skipping character cleaning (performance optimization)")
            content = raw_content
        
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        return {
            "success": True,
            "content": content,
            "metadata": {
                "platform": "generic_web",
                "url": item.get("url", url),
                "title": item.get("metadata", {}).get("title", ""),
                "description": item.get("metadata", {}).get("description", "")
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"generic scraping error: {e}")
        return {"success": False, "content": "", "metadata": {}, "error": str(e)}


async def scrapeGenericUrl(url: str, maxChars: Optional[int] = None) -> dict:
    """
    main scraping function with automatic platform detection.
    detects platform via regex and routes to appropriate actor.
    for generic websites: tries simple http first, falls back to apify if needed.
    """
    platform = detectPlatform(url)
    logger.info(f"detected platform: {platform.value} for {url}")
    
    if platform == PlatformType.FACEBOOK:
        return await scrapeFacebookPost(url, maxChars)
    elif platform == PlatformType.INSTAGRAM:
        return await scrapeInstagramPost(url, maxChars)
    elif platform == PlatformType.TWITTER:
        return await scrapeTwitterPost(url, maxChars)
    elif platform == PlatformType.TIKTOK:
        return await scrapeTikTokPost(url, maxChars)
    elif platform in (PlatformType.G1, PlatformType.ESTADAO, PlatformType.FOLHA, PlatformType.AOSFATOS):
        # domain-specific news scrapers (sync, run in thread)
        scraper_map = {
            PlatformType.G1: scrape_g1_article,
            PlatformType.ESTADAO: scrape_estadao_article,
            PlatformType.FOLHA: scrape_folha_article,
            PlatformType.AOSFATOS: scrape_aosfatos_article,
        }
        scraper_fn = scraper_map[platform]
        logger.info(f"using domain-specific scraper for {platform.value}")
        result = await asyncio.to_thread(scraper_fn, url)

        if result["success"]:
            logger.info(f"{platform.value} scraper succeeded")
            return result

        # fallback to generic path if domain scraper fails
        logger.info(f"{platform.value} scraper failed ({result.get('error')}), falling back to generic")

    # for generic websites: try simple scraping first (no browser, no apify credits)
    logger.info("trying simple scraping first for generic website")
    result = await scrapeGenericSimple(url, maxChars)

    if result["success"]:
        logger.info("simple scraping succeeded - no apify credits used")
        return result

    # if simple scraping failed, fallback to apify actor with browser
    logger.info(f"simple scraping failed ({result.get('error')}), falling back to apify actor")
    return await scrapeGenericWebsite(url, maxChars)
