"""
configuration constants for the agentic context search loop.
"""

from app.models.agenticai import SourceReliability
import logging

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
        "query_suffix": "(site:g1.globo.com OR site:ge.globo.com)", #globo esporte e G1
        "site_search": None,
        "site_search_filter": None,
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

# link expansion settings
LINK_SCRAPE_TIMEOUT_PER_URL = 30.0
MAX_LINKS_TO_EXPAND = 5

# default LLM model for the context agent
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# model for the adjudication step (can be stronger than the context agent model)
ADJUDICATION_MODEL = "gemini-2.5-flash"


# suppress verbose debug logs from trafilatura (HTML processing library)
logging.getLogger("trafilatura").setLevel(logging.WARNING)
logging.getLogger("trafilatura.htmlprocessing").setLevel(logging.WARNING)
logging.getLogger("trafilatura.main_extractor").setLevel(logging.WARNING)

logging.getLogger("grpc._cython.cygrpc").setLevel(logging.WARNING)
