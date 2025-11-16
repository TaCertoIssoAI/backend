"""
re-export apify utilities from web module for backwards compatibility
"""

from app.ai.context.web.apify_utils import (
    # enums and constants
    PlatformType,
    ACTOR_MAP,
    APIFY_TOKEN_ENV_KEY,
    
    # core functions
    detectPlatform,
    getApifyClientAsync,
    
    # social media scraping
    scrapeFacebookPost,
    scrapeInstagramPost,
    scrapeTwitterPost,
    scrapeTikTokPost,
    
    # generic scraping
    scrapeGenericSimple,
    scrapeGenericWebsite,
    scrapeGenericUrl,
    
    # google search
    searchGoogleClaim,
)

__all__ = [
    "PlatformType",
    "ACTOR_MAP",
    "APIFY_TOKEN_ENV_KEY",
    "detectPlatform",
    "getApifyClientAsync",
    "scrapeFacebookPost",
    "scrapeInstagramPost",
    "scrapeTwitterPost",
    "scrapeTikTokPost",
    "scrapeGenericSimple",
    "scrapeGenericWebsite",
    "scrapeGenericUrl",
    "searchGoogleClaim",
]

