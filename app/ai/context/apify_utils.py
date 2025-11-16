"""
apify integration utilities

provides helper functions to interact with apify actors for web scraping.
follows project rules: token from env, async-first, lowercase names, short comments.
"""

import os
import logging
from typing import Optional

from apify_client import ApifyClient, ApifyClientAsync

logger = logging.getLogger(__name__)

APIFY_TOKEN_ENV_KEY = "APIFY_TOKEN"

# actor ids for common scraping tasks
FACEBOOK_POSTS_ACTOR = "apify/facebook-posts-scraper"


def getApifyClient() -> ApifyClient:
    """
    create sync apify client using token from environment.
    raises RuntimeError if token is missing.
    """
    apifyToken = os.getenv(APIFY_TOKEN_ENV_KEY)
    if not apifyToken:
        raise RuntimeError("missing APIFY_TOKEN in environment")
    
    return ApifyClient(apifyToken)


def getApifyClientAsync() -> ApifyClientAsync:
    """
    create async apify client using token from environment.
    raises RuntimeError if token is missing.
    """
    apifyToken = os.getenv(APIFY_TOKEN_ENV_KEY)
    if not apifyToken:
        raise RuntimeError("missing APIFY_TOKEN in environment")
    
    return ApifyClientAsync(apifyToken)


async def scrapeFacebookPost(postUrl: str, maxChars: Optional[int] = None) -> dict:
    """
    scrape facebook post using apify actor.
    
    args:
        postUrl: facebook post url
        maxChars: optional limit for content length
        
    returns:
        dict with keys:
            - success: bool
            - content: str (extracted text)
            - metadata: dict (additional info from apify)
            - error: str (if failed)
    """
    try:
        logger.info(f"scraping facebook post: {postUrl}")
        
        apifyClient = getApifyClientAsync()
        actorClient = apifyClient.actor(FACEBOOK_POSTS_ACTOR)
        
        # configure input for facebook posts actor
        runInput = {
            "startUrls": [{"url": postUrl}],
            "resultsLimit": 1,
            "maxPostCount": 1
        }
        
        # start actor and wait for completion
        callResult = await actorClient.call(run_input=runInput, timeout_secs=120)
        
        if callResult is None:
            logger.error(f"actor run failed for {postUrl}")
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "actor run failed or returned no data"
            }
        
        # fetch items from dataset
        datasetClient = apifyClient.dataset(callResult["defaultDatasetId"])
        listItemsResult = await datasetClient.list_items()
        
        if not listItemsResult or not listItemsResult.items:
            logger.warning(f"no items extracted from {postUrl}")
            return {
                "success": False,
                "content": "",
                "metadata": {},
                "error": "no content extracted from post"
            }
        
        # extract content from first item
        item = listItemsResult.items[0]
        content = item.get("text", "") or item.get("content", "") or item.get("postText", "")
        
        # apply character limit if specified
        if maxChars and len(content) > maxChars:
            content = content[:maxChars]
        
        logger.info(f"successfully scraped {len(content)} chars from {postUrl}")
        
        return {
            "success": True,
            "content": content,
            "metadata": {
                "postUrl": item.get("postUrl", postUrl),
                "timestamp": item.get("time", ""),
                "author": item.get("from", ""),
                "likes": item.get("likes", 0),
                "shares": item.get("shares", 0),
                "comments": item.get("comments", 0)
            },
            "error": None
        }
        
    except Exception as e:
        logger.error(f"error scraping facebook post {postUrl}: {e}")
        return {
            "success": False,
            "content": "",
            "metadata": {},
            "error": str(e)
        }


async def scrapeGenericUrl(url: str, maxChars: Optional[int] = None) -> dict:
    """
    placeholder for generic url scraping.
    currently only supports facebook posts.
    
    args:
        url: url to scrape
        maxChars: optional limit for content length
        
    returns:
        dict with scraping results
    """
    # detect if url is facebook post
    if "facebook.com" in url or "fb.com" in url:
        return await scrapeFacebookPost(url, maxChars)
    
    # for other urls, return not implemented
    logger.warning(f"generic scraping not yet implemented for {url}")
    return {
        "success": False,
        "content": "",
        "metadata": {},
        "error": "only facebook posts are currently supported"
    }

