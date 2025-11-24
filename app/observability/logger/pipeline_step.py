"""
pipeline step enumeration for logging context.

defines all pipeline steps to track where logs originate from in the fact-checking pipeline.
"""

from enum import Enum


class PipelineStep(str, Enum):
    """
    enumeration of all pipeline steps in the fact-checking system.

    used to tag log messages with their originating pipeline step for better
    observability and debugging.
    """

    # core pipeline steps
    LINK_EXPANSION = "link_expansion"
    CLAIM_EXTRACTION = "claim_extraction"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    ADJUDICATION = "adjudication"

    # api and preprocessing
    API_INTAKE = "api_intake"
    PREPROCESSING = "preprocessing"

    # supporting services
    WEB_SCRAPING = "web_scraping"

    # system level
    SYSTEM = "system"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """return the enum value as string"""
        return self.value
