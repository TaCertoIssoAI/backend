"""
configuration constants for the agentic context search loop.
"""

from app.models.agenticai import SourceReliability

# iteration limits
MAX_ITERATIONS = 5

# timeouts (seconds)
SEARCH_TIMEOUT_PER_QUERY = 15.0
SCRAPE_TIMEOUT_PER_PAGE = 30.0
FACT_CHECK_TIMEOUT = 30.0

# web search domain configuration
DOMAIN_SEARCHES: dict[str, dict] = {
    "geral": {
        "site_search": None,
        "site_search_filter": None,
        "reliability": SourceReliability.NEUTRO,
    },
    "g1": {
        "site_search": "g1.globo.com",
        "site_search_filter": "i",
        "reliability": SourceReliability.MUITO_CONFIAVEL,
    },
    "estadao": {
        "site_search": "estadao.com.br",
        "site_search_filter": "i",
        "reliability": SourceReliability.MUITO_CONFIAVEL,
    },
    "aosfatos": {
        "site_search": "aosfatos.org",
        "site_search_filter": "i",
        "reliability": SourceReliability.MUITO_CONFIAVEL,
    },
    "folha": {
        "site_search": "folha.uol.com.br",
        "site_search_filter": "i",
        "reliability": SourceReliability.MUITO_CONFIAVEL,
    },
}

# default LLM model for the context agent
DEFAULT_MODEL = "gpt-4o-mini"
