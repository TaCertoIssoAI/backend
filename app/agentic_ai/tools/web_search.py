"""
web search tool — runs parallel searches across general + domain-specific sources.

reuses google_search() from app.ai.context.web.google_search
and trusted domains from app.config.trusted_domains.
"""

import asyncio
import logging
from uuid import uuid4

from app.models.agenticai import GoogleSearchContext, SourceReliability
from app.ai.context.web.google_search import google_search, GoogleSearchError
from app.config.trusted_domains import get_trusted_domains

from app.agentic_ai.config import DOMAIN_SEARCHES, SEARCH_TIMEOUT_PER_QUERY

logger = logging.getLogger(__name__)


def _build_query_with_trusted_domains(query: str) -> str:
    """append trusted domain site: filters to the general search query."""
    domains = get_trusted_domains()
    if not domains:
        return query
    valid = [d.strip() for d in domains if d and d.strip()]
    if not valid:
        return query
    domain_filters = " OR ".join(f"site:{d}" for d in valid)
    return f"{query} ({domain_filters})"


class WebSearchTool:
    """runs parallel searches per query: general + domain-specific (g1, aosfatos, folha)."""

    def __init__(self, timeout: float = SEARCH_TIMEOUT_PER_QUERY):
        self.timeout = timeout

    async def search(
        self,
        queries: list[str],
        max_results_per_domain: int = 4,
        max_results_general: int = 4,
    ) -> dict[str, list[GoogleSearchContext]]:
        """search all queries across all domain groups concurrently."""
        merged: dict[str, list[GoogleSearchContext]] = {
            key: [] for key in DOMAIN_SEARCHES
        }

        tasks = []
        task_keys = []

        for query in queries:
            for domain_key, domain_cfg in DOMAIN_SEARCHES.items():
                # pick base max: general vs domain-specific
                base_max = max_results_general if domain_key == "geral" else max_results_per_domain

                # config override still takes priority when present
                max_cfg = domain_cfg.get("max_results_per_call")
                final_max_results = max_cfg if max_cfg else base_max

                tasks.append(
                    self._search_single(
                        query=query,
                        domain_key=domain_key,
                        site_search=domain_cfg["site_search"],
                        site_search_filter=domain_cfg["site_search_filter"],
                        query_suffix=domain_cfg.get("query_suffix"),
                        reliability=domain_cfg["reliability"],
                        max_results=final_max_results,
                    )
                )
                task_keys.append(domain_key)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for key, result in zip(task_keys, results):
            if isinstance(result, list):
                merged[key].extend(result)
            elif isinstance(result, Exception):
                logger.error(f"web search error for {key}: {result}")

        # dedup by URL within each domain key — keeps first occurrence
        total_before = sum(len(v) for v in merged.values())
        for key in merged:
            seen_urls: set[str] = set()
            unique: list[GoogleSearchContext] = []
            for entry in merged[key]:
                if entry.url not in seen_urls:
                    seen_urls.add(entry.url)
                    unique.append(entry)
            merged[key] = unique
        total_after = sum(len(v) for v in merged.values())

        per_key = {k: len(v) for k, v in merged.items() if v}
        logger.debug(
            f"web search: {total_after} result(s) across {len(per_key)} domain(s) "
            f"(dedup removed {total_before - total_after}): {per_key}"
        )

        return merged

    async def _search_single(
        self,
        query: str,
        domain_key: str,
        site_search: str | None,
        site_search_filter: str | None,
        query_suffix: str | None,
        reliability: SourceReliability,
        max_results: int,
    ) -> list[GoogleSearchContext]:
        """execute a single search against one domain configuration."""
        try:
            # for general search, use trusted domain filters in query
            effective_query = query
            if domain_key == "geral":
                effective_query = _build_query_with_trusted_domains(query)

            # append query_suffix (e.g. multi-domain site: filters)
            if query_suffix:
                effective_query = f"{effective_query} {query_suffix}"

            items = await google_search(
                query=effective_query,
                num=min(max_results, 10),
                site_search=site_search,
                site_search_filter=site_search_filter,
               # language="lang_pt" this leads to better sources but they sometimes come in english, however if there are portuguese sources they will come first due to the query language
                timeout=self.timeout,
            )

            results: list[GoogleSearchContext] = []
            for position, item in enumerate(items, 1):
                url = item.get("link", "")
                title = item.get("title", "")
                if not url:
                    continue

                results.append(
                    GoogleSearchContext(
                        id=str(uuid4()),
                        url=url,
                        parent_id=None,
                        reliability=reliability,
                        title=title,
                        snippet=item.get("snippet", ""),
                        domain=item.get("displayLink", ""),
                        position=position,
                    )
                )

            return results

        except GoogleSearchError as e:
            logger.error(f"google search error ({domain_key}): {e}")
            return []
        except Exception as e:
            logger.error(f"web search unexpected error ({domain_key}): {e}")
            return []
