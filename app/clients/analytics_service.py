import httpx
import os
import logging,json
from app.observability.analytics import AnalyticsCollector

_URL_ENV_VAR  = os.getenv("ANALYTICS_SERVICE_URL") 
ANALYTICS_SERVICE_URL = _URL_ENV_VAR if _URL_ENV_VAR is not None else ""

_ENDPOINT_ENV_VAR = os.getenv("ANALYTICS_SERVICE_ENDPOINT")
ANALYTICS_SERVICE_ENDPOINT  = _ENDPOINT_ENV_VAR if _ENDPOINT_ENV_VAR is not None else "/analises"

_ANALYTICS_WEBSITE_VAR = os.getenv("ANALYTICS_WEBSITE_VAR")
ANALYTICS_WEBSITE_URL = _ANALYTICS_WEBSITE_VAR if _ANALYTICS_WEBSITE_VAR is not None else "https://tacertoissoai.com.br"

_TIMEOUT = 20
logger = logging.getLogger(__name__)


def get_analytics_url_for_fact_check(msg_id:str)->str:
    """
    Returns the full URL where the fact-checking report will be hosted at in the analytics web page
    """
    return f"{ANALYTICS_WEBSITE_URL}/verificacao/{msg_id}"

async def send_analytics_payload(collector: AnalyticsCollector)->None:
    """
    Best effort, fire-and-forget call to the analytics service endpoint to collect the results
    from a fact-checking pipeline run
    """
    try:
        full_path = ANALYTICS_SERVICE_URL + ANALYTICS_SERVICE_ENDPOINT
        json_val = collector.to_dict()

        logger.info("Analytics output keys %s",json_val.keys())
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                full_path,
                json=json_val,
            )
            logger.info("Analytics status: %s", resp.status_code)
    except Exception:
        pass
        # swallow or log only
        logger.exception("Failed to send analytics payload")
    
