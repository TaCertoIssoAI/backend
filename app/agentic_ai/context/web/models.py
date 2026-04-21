from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SocialMediaMetadata(BaseModel):
    """metadata for social media content scraped from platforms like facebook, instagram, etc"""
    
    platform: str = Field(..., description="social media platform name")
    postUrl: str = Field(..., description="URL of the post")
    timestamp: str = Field(..., description="ISO timestamp when the post was created")
    author: str = Field(default="", description="author username or name")
    likes: int = Field(default=0, description="number of likes on the post")
    shares: int = Field(default=0, description="number of shares")
    comments: int = Field(default=0, description="number of comments")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "platform": "facebook",
                "postUrl": "https://www.facebook.com/share/p/16k5YVoKc1/",
                "timestamp": "2025-11-16T01:07:44.000Z",
                "author": "user123",
                "likes": 32851,
                "shares": 1891,
                "comments": 1768
            }
        }
    )


class WebContentResult(BaseModel):
    """result from web content scraping operation"""

    success: bool = Field(..., description="whether the scraping was successful")
    url: str = Field(..., description="URL that was scraped")
    content: str = Field(default="", description="extracted text content from the page")
    content_length: int = Field(default=0, description="length of extracted content in characters")
    metadata: Optional[SocialMediaMetadata] = Field(None, description="platform-specific metadata if available")
    error: Optional[str] = Field(None, description="error message if scraping failed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "url": "https://www.facebook.com/share/p/16k5YVoKc1/",
                "content": "Neymar está dando tudo de si para tentar tirar o Santos da zona de rebaixamento",
                "content_length": 79,
                "metadata": {
                    "platform": "facebook",
                    "postUrl": "https://www.facebook.com/share/p/16k5YVoKc1/",
                    "timestamp": "2025-11-16T01:07:44.000Z",
                    "author": "",
                    "likes": 32851,
                    "shares": 1891,
                    "comments": 1768
                },
                "error": None
            }
        }
    )

    @classmethod
    def from_dict(cls, data: dict, url: str) -> "WebContentResult":
        """
        Create WebContentResult from the dict returned by scrapeGenericUrl.

        Args:
            data: The dict returned from scrapeGenericUrl function
            url: The original URL that was scraped

        Returns:
            WebContentResult instance parsed from the dict

        Example:
            >>> result_dict = {
            ...     "success": True,
            ...     "content": "Article text...",
            ...     "metadata": {"platform": "facebook", "postUrl": "..."},
            ...     "error": None
            ... }
            >>> result = WebContentResult.from_dict(result_dict, "https://example.com")
        """
        content = data.get("content", "")
        content_length = len(content) if content else 0

        # extract metadata if available
        metadata_dict = data.get("metadata", {})
        social_metadata = None

        # check if this is social media content (has platform field)
        if metadata_dict and "platform" in metadata_dict:
            # only create SocialMediaMetadata if we have the required fields
            platform = metadata_dict.get("platform", "")
            if platform in ["facebook", "instagram", "twitter", "tiktok"]:
                social_metadata = SocialMediaMetadata(
                    platform=platform,
                    postUrl=metadata_dict.get("postUrl", url),
                    timestamp=metadata_dict.get("timestamp", ""),
                    author=metadata_dict.get("author", ""),
                    likes=metadata_dict.get("likes", 0),
                    shares=metadata_dict.get("shares", 0),
                    comments=metadata_dict.get("comments", 0),
                )

        return cls(
            success=data.get("success", False),
            url=url,
            content=content,
            content_length=content_length,
            metadata=social_metadata,
            error=data.get("error"),
        )

