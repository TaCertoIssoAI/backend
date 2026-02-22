"""
tool protocols for dependency injection and testing.

all IO operations in the context agent are behind these protocols
so they can be swapped with mocks in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    ScrapeTarget,
    WebScrapeContext,
)


@runtime_checkable
class FactCheckSearchProtocol(Protocol):
    async def search(self, queries: list[str]) -> list[FactCheckApiContext]: ...


@runtime_checkable
class WebSearchProtocol(Protocol):
    async def search(
        self,
        queries: list[str],
        max_results_specific_search: int = 5,
        max_results_general: int = 5,
    ) -> dict[str, list[GoogleSearchContext]]: ...


@runtime_checkable
class PageScraperProtocol(Protocol):
    async def scrape(self, targets: list[ScrapeTarget]) -> list[WebScrapeContext]: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def bind_tools(self, tools: list[Any]) -> Any: ...
    async def ainvoke(self, messages: list[Any]) -> Any: ...
