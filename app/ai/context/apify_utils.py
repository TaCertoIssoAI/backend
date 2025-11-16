"""
apify integration utilities

provides helper functions to interact with apify actors for web scraping.
follows project rules: token from env, async-first, lowercase names, short comments.
"""

import os
import re
import logging
from typing import Optional
from enum import Enum

from apify_client import ApifyClientAsync

logger = logging.getLogger(__name__)

APIFY_TOKEN_ENV_KEY = "APIFY_TOKEN"


class PlatformType(Enum):
    """supported platforms for scraping"""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    GENERIC = "generic"


# actor ids mapping
ACTOR_MAP = {
    PlatformType.FACEBOOK: "apify/facebook-posts-scraper",
    PlatformType.INSTAGRAM: "apify/instagram-scraper",
    PlatformType.TWITTER: "apidojo/tweet-scraper",
    PlatformType.TIKTOK: "clockworks/tiktok-scraper",
    PlatformType.GENERIC: "apify/website-content-crawler"
}


def detectPlatform(url: str) -> PlatformType:
    """
    detect platform type from url using regex patterns.
    
    args:
        url: url to analyze
        
    returns:
        PlatformType enum
    """
    url_lower = url.lower()
    
    # facebook patterns
    facebook_patterns = [
        r'facebook\.com',
        r'fb\.com',
        r'fb\.watch',
        r'm\.facebook\.com'
    ]
    
    # instagram patterns
    instagram_patterns = [
        r'instagram\.com',
        r'instagr\.am'
    ]
    
    # twitter/x patterns
    twitter_patterns = [
        r'twitter\.com',
        r'x\.com',
        r't\.co',
        r'mobile\.twitter\.com'
    ]
    
    # tiktok patterns
    tiktok_patterns = [
        r'tiktok\.com',
        r'vm\.tiktok\.com',
        r'vt\.tiktok\.com'
    ]
    
    # check each platform
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
    
    # default to generic web crawler
    return PlatformType.GENERIC


def getApifyClientAsync() -> ApifyClientAsync:
    """
    create async apify client using token from environment.
    raises RuntimeError if token is missing.
    """
    apifyToken = os.getenv(APIFY_TOKEN_ENV_KEY)
    if not apifyToken:
        raise RuntimeError("missing APIFY_TOKEN in environment")
    
    return ApifyClientAsync(apifyToken)


async def scrapeFacebookPost(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape facebook post using apify actor.
    
    args:
        url: facebook post url
        maxChars: optional limit for content length
        
    returns:
        dict with scraping results
    """
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
            logger.error(f"actor run failed for {url}")
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "actor run failed or returned no data"
            }
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            logger.warning(f"no items extracted from {url}")
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "no content extracted from post"
            }
        
        item = listItemsResult.items[0]
        content = item.get("text", "") or item.get("content", "") or item.get("postText", "")
        
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        logger.info(f"successfully scraped {len(content)} chars from {url}")
        
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
        logger.error(f"error scraping facebook post {url}: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeInstagramPost(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape instagram post using apify actor.
    """
    try:
        logger.info(f"scraping instagram post: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.INSTAGRAM])
        
        runInput = {
            "directUrls": [url],
            "resultsLimit": 1
        }
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "actor run failed or returned no data"
            }
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "no content extracted from post"
            }
        
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
        logger.error(f"error scraping instagram post {url}: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeTwitterPost(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape twitter/x post using apify actor.
    """
    try:
        logger.info(f"scraping twitter post: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.TWITTER])
        
        runInput = {
            "startUrls": [url],
            "maxItems": 1
        }
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "actor run failed or returned no data"
            }
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "no content extracted from post"
            }
        
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
        logger.error(f"error scraping twitter post {url}: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeTikTokPost(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape tiktok post using apify actor.
    """
    try:
        logger.info(f"scraping tiktok post: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.TIKTOK])
        
        runInput = {
            "postURLs": [url],
            "resultsPerPage": 1
        }
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "actor run failed or returned no data"
            }
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "no content extracted from post"
            }
        
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
        logger.error(f"error scraping tiktok post {url}: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeGenericWebsite(url: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape generic website using apify website content crawler.
    """
    try:
        logger.info(f"scraping generic website: {url}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(ACTOR_MAP[PlatformType.GENERIC])
        
        runInput = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": 1,
            "crawlerType": "playwright:firefox"
        }
        
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "actor run failed or returned no data"
            }
        
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "no content extracted from website"
            }
        
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
        logger.error(f"error scraping generic website {url}: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeGenericUrl(url: str, maxChars: Optional[int] = None) -> dict:
    """
    smart scraping function that detects platform and uses appropriate actor.
    
    args:
        url: url to scrape
        maxChars: optional limit for content length
        
    returns:
        dict with scraping results
    """
    # detect platform type
    platform = detectPlatform(url)
    
    logger.info(f"detected platform: {platform.value} for url: {url}")
    
    # route to appropriate scraper
    if platform == PlatformType.FACEBOOK:
        return await scrapeFacebookPost(url, maxChars)
    elif platform == PlatformType.INSTAGRAM:
        return await scrapeInstagramPost(url, maxChars)
    elif platform == PlatformType.TWITTER:
        return await scrapeTwitterPost(url, maxChars)
    elif platform == PlatformType.TIKTOK:
        return await scrapeTikTokPost(url, maxChars)
    else:
        # fallback to generic website crawler
        return await scrapeGenericWebsite(url, maxChars)
