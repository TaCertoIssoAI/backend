"""
utility to load trusted domains for web search filtering.

reads from trusted_domains.json in the same directory and returns a list of domains.
fail-open: returns empty list if file doesn't exist or can't be parsed.
"""

import json
from pathlib import Path
from typing import List

from app.observability.logger import get_logger

logger = get_logger(__name__)


def get_trusted_domains() -> List[str]:
    """
    load trusted domains from trusted_domains.json.

    returns:
        list of trusted domain strings, or empty list if file doesn't exist or parsing fails

    example:
        >>> domains = get_trusted_domains()
        >>> print(domains)
        ['who.int', 'cdc.gov', 'saude.gov.br']
    """
    try:
        # get the path to trusted_domains.json in the same directory as this file
        config_dir = Path(__file__).parent
        json_path = config_dir / "trusted_domains.json"

        # check if file exists
        if not json_path.exists():
            logger.warning(f"trusted_domains.json not found at {json_path}, returning empty list")
            return []

        # read and parse json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # extract domains list
        if isinstance(data, list):
            # if json is directly a list of domains
            domains = [str(domain) for domain in data if domain]
        elif isinstance(data, dict) and "domains" in data:
            # if json has {"domains": [...]} structure
            domains = [str(domain) for domain in data["domains"] if domain]
        else:
            logger.warning(f"unexpected json structure in {json_path}, returning empty list")
            return []

        logger.info(f"loaded {len(domains)} trusted domain(s) from {json_path}")
        return domains

    except json.JSONDecodeError as e:
        logger.error(f"failed to parse trusted_domains.json: {e}, returning empty list")
        return []
    except Exception as e:
        logger.error(f"unexpected error loading trusted domains: {e}, returning empty list")
        return []
