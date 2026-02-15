"""
data models for the agentic AI context search loop.

defines source reliability levels, context dataclasses for each tool output,
and the output contract for the adjudication agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SourceReliability(str, Enum):
    """reliability level assigned to each context source."""
    MUITO_CONFIAVEL = "Muito confiável"
    POUCO_CONFIAVEL = "Pouco Confiável"
    NEUTRO = "Neutro"


@dataclass
class BaseContext:
    """base fields shared by all context entries."""
    id: str
    url: str
    parent_id: Optional[str]
    reliability: SourceReliability


@dataclass
class FactCheckApiContext(BaseContext):
    """result from the Google Fact-Check API."""
    title: str = ""
    publisher: str = ""
    rating: str = ""
    rating_comment: Optional[str] = None
    claim_text: str = ""
    review_date: Optional[str] = None


@dataclass
class GoogleSearchContext(BaseContext):
    """result from Google Custom Search API."""
    title: str = ""
    snippet: str = ""
    domain: str = ""
    position: int = 0


@dataclass
class WebScrapeContext(BaseContext):
    """result from scraping a web page."""
    title: str = ""
    content: str = ""
    extraction_status: str = ""
    extraction_tool: str = ""


class ScrapeTarget(BaseModel):
    """target for the scrape_pages tool."""
    url: str
    title: str


@dataclass
class ContextNodeOutput:
    """output contract for the context agent, consumed by the adjudication agent."""
    fact_check_results: list[FactCheckApiContext] = field(default_factory=list)
    search_results: dict[str, list[GoogleSearchContext]] = field(default_factory=dict)
    scraped_pages: list[WebScrapeContext] = field(default_factory=list)
