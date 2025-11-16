"""
apify integration utilities for web scraping
"""

import os
import re
import logging
import time
from typing import Optional
from enum import Enum

import httpx
from bs4 import BeautifulSoup
import html2text
from apify_client import ApifyClientAsync

logger = logging.getLogger(__name__)

APIFY_TOKEN_ENV_KEY = "APIFY_TOKEN"


class PlatformType(Enum):
    """supported scraping platforms"""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    GENERIC = "generic"


ACTOR_MAP = {
    PlatformType.FACEBOOK: "apify/facebook-posts-scraper",
    PlatformType.INSTAGRAM: "apify/instagram-scraper",
    PlatformType.TWITTER: "apidojo/tweet-scraper",
    PlatformType.TIKTOK: "clockworks/tiktok-scraper",
    PlatformType.GENERIC: "apify/website-content-crawler"
}


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
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
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
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
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
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
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
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
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
        
        # configure httpx client with realistic headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
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
            
            # convert html to clean text using html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_emphasis = False
            h.body_width = 0  # no line wrapping
            
            # try to extract main content (prioritize article/main tags)
            main_content = soup.find("main") or soup.find("article") or soup.find("body")
            if main_content:
                text_content = h.handle(str(main_content))
            else:
                text_content = h.handle(html_content)
            
            # clean up excessive whitespace
            text_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_content)
            text_content = text_content.strip()
            
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
            "crawlerType": "playwright:firefox"
        }
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            return {"success": False, "content": "", "metadata": {}, "error": "actor run failed"}
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {"success": False, "content": "", "metadata": {}, "error": "no content extracted"}
        
        item = listItemsResult.items[0]
        content = item.get("text", "") or item.get("markdown", "") or item.get("html", "")
        
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


async def searchGoogleClaim(claim: str, maxResults: int = 10) -> dict:
    """
    search google for information about a claim using apify actor.
    returns structured search results to help with fact-checking.
    """
    try:
        logger.info(f"searching google for claim: {claim[:100]}...")
        
        apifyClient = getApifyClientAsync()
        # using google search results scraper
        actorClient = apifyClient.actor("apify/google-search-scraper")
        
        runInput = {
            "queries": claim,
            "maxPagesPerQuery": 1,
            "resultsPerPage": maxResults,
            "languageCode": "pt-BR",  # portuguese brazil results preferred
            "mobileResults": False,
            "includeUnfilteredResults": False
        }
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            return {
                "success": False,
                "claim": claim,
                "results": [],
                "total_results": 0,
                "error": "actor run failed"
            }
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {
                "success": False,
                "claim": claim,
                "results": [],
                "total_results": 0,
                "error": "no search results found"
            }
        
        # extract and structure search results
        searchResults = []
        for item in listItemsResult.items:
            organicResults = item.get("organicResults", [])
            
            for result in organicResults[:maxResults]:
                searchResults.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("description", ""),
                    "position": result.get("rank", 0),
                    "domain": result.get("displayedUrl", "")
                })
        
        logger.info(f"google search completed: {len(searchResults)} results found")
        
        return {
            "success": True,
            "claim": claim,
            "results": searchResults,
            "total_results": len(searchResults),
            "metadata": {
                "search_engine": "google",
                "language": "pt",
                "actor": "apify/google-search-scraper"
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"google search error: {e}")
        return {
            "success": False,
            "claim": claim,
            "results": [],
            "total_results": 0,
            "error": str(e)
        }


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
    else:
        # for generic websites: try simple scraping first (no browser, no apify credits)
        logger.info("trying simple scraping first for generic website")
        result = await scrapeGenericSimple(url, maxChars)
        
        if result["success"]:
            logger.info("simple scraping succeeded - no apify credits used")
            return result
        
        # if simple scraping failed, fallback to apify actor with browser
        logger.info(f"simple scraping failed ({result.get('error')}), falling back to apify actor")
        return await scrapeGenericWebsite(url, maxChars)
