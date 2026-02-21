"""
configuration constants for the agentic context search loop.
"""

from app.models.agenticai import SourceReliability
import logging

# iteration limits
MAX_ITERATIONS = 5

# retry on insufficient sources
MAX_RETRY_COUNT = 1
MAX_RETRY_ITERATIONS = 3

# timeouts (seconds)
SEARCH_TIMEOUT_PER_QUERY = 30.0
SCRAPE_TIMEOUT_PER_PAGE = 30.0
FACT_CHECK_TIMEOUT = 30.0

# web search domain configuration
DOMAIN_SEARCHES: dict[str, dict] = {
    "geral": {
        "site_search": None,
        "site_search_filter": None,
        "reliability": SourceReliability.NEUTRO,
    },
    "especifico": {
        # unified specific search across trusted Brazilian news sources
        "domains": [
            "g1.globo.com",
            "ge.globo.com",
            "estadao.com.br",
            "folha.uol.com.br",
            "aosfatos.org",
        ],
        "site_search": None,
        "site_search_filter": None,
        "reliability": SourceReliability.MUITO_CONFIAVEL,
    },
}

# link expansion settings
LINK_SCRAPE_TIMEOUT_PER_URL = 30.0
MAX_LINKS_TO_EXPAND = 5

# default LLM model for the context agent
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# model for the adjudication step
ADJUDICATION_MODEL = "gemini-2.5-flash-lite"
ADJUDICATION_THINKING_BUDGET = 1024

# adjudication timeout retry policy
ADJUDICATION_TIMEOUT = 20.0        # seconds per attempt
ADJUDICATION_MAX_RETRIES = 2       # max retry attempts on timeout


# suppress verbose debug logs from trafilatura (HTML processing library)
logging.getLogger("trafilatura").setLevel(logging.WARNING)
logging.getLogger("trafilatura.htmlprocessing").setLevel(logging.WARNING)
logging.getLogger("trafilatura.main_extractor").setLevel(logging.WARNING)

logging.getLogger("grpc._cython.cygrpc").setLevel(logging.WARNING)
