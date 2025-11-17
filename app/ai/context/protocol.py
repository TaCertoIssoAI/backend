
from typing import List, Protocol
from abc import abstractmethod

from app.models import (
    ExtractedClaim,
    Citation,
)

# ===== EVIDENCE GATHERER PROTOCOL =====

class EvidenceGatherer(Protocol):
    """
    Protocol defining the interface for evidence gatherers.

    Any evidence source (web search, fact-check API, database, etc.)
    must implement this interface to be pluggable into the pipeline.
    """

    @abstractmethod
    async def gather(self, claim: ExtractedClaim) -> List[Citation]:
        """
        Gather evidence citations for a given claim.

        Args:
            claim: The claim to gather evidence for

        Returns:
            List of citations found (can be empty if no evidence found)
        """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Returns the name of this evidence source.
        Used for citation.source field.
        """