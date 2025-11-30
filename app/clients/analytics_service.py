import httpx
import os
import logging
from app.observability.analytics import AnalyticsCollector

_URL_ENV_VAR  = os.getenv("ANALYTICS_SERVICE_URL") 
ANALYTICS_SERVICE_URL = _URL_ENV_VAR if _URL_ENV_VAR is not None else ""

_ENDPOINT_ENV_VAR = os.getenv("ANALYTICS_SERVICE_ENDPOINT")
ANALYTICS_SERVICE_ENDPOINT  = _ENDPOINT_ENV_VAR if _ENDPOINT_ENV_VAR is not None else "/analises"

_TIMEOUT = 20
logger = logging.getLogger(__name__)


async def send_analytics_payload(collector: AnalyticsCollector)->None:
    """
    Best effort, fire-and-forget call to the analytics service endpoint to collect the results
    from a fact-checking pipeline run
    """
    try:
        full_path = ANALYTICS_SERVICE_URL + ANALYTICS_SERVICE_ENDPOINT
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                full_path,
                json=collector.to_dict(),
            )
            logger.info("Analytics status: %s", resp.status_code)
    except Exception:
        pass
        # swallow or log only
        logger.exception("Failed to send analytics payload")
    
