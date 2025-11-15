"""
Evidence Retrieval Module - Step 3 of Fact-Checking Pipeline

This module implements evidence retrieval using the Google Fact-Check Tools API.
Takes ClaimExtractionOutput from Step 2 and retrieves fact-check evidence for each claim.

Outputs structured data models for Step 4 (Adjudication).
"""

import requests
from typing import List, Optional
import logging

from app.models.factchecking import (
    LinkEnrichmentOutput,
    Citation,
    EnrichedClaim,
    EvidenceRetrievalResult
)
from app.core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()


class GoogleFactCheckRetriever:
    """
    Retrieves fact-check evidence using Google Fact-Check Tools API
    """
    
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        
    async def search_claim(self, claim_text: str) -> List[Citation]:
        """
        Search for fact-check evidence for a single claim
        
        Args:
            claim_text: The claim to search for
            
        Returns:
            List of Citation objects from fact-checkers
        """
        if not self.api_key:
            logger.warning("Google API key not configured")
            return []
            
        try:
            # Build request URL
            url = f"{self.base_url}?query={claim_text}&key={self.api_key}"
            
            # Make API request
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract citations
            citations = []
            if 'claims' in data and data['claims']:
                for claim in data['claims']:
                    if 'claimReview' in claim and claim['claimReview']:
                        for review in claim['claimReview']:
                            citation = self._parse_claim_review(claim, review)
                            if citation:
                                citations.append(citation)
            
            logger.info(f"Found {len(citations)} fact-check results for claim: {claim_text[:50]}...")
            return citations
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Google API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing Google API response: {e}")
            return []
    
    def _parse_claim_review(self, claim: dict, review: dict) -> Optional[Citation]:
        """
        Parse a single claimReview into a Citation object
        
        Args:
            claim: The claim data from Google API
            review: The claimReview data from Google API
            
        Returns:
            Citation object or None if parsing fails
        """
        try:
            # Extract required fields
            url = review.get('url', '')
            title = review.get('title', f"Fact-check: {claim.get('text', 'Unknown claim')[:50]}...")
            publisher = review.get('publisher', {}).get('name', 'Unknown Publisher')
            rating = review.get('textualRating', 'Unknown')
            review_date = review.get('reviewDate')
            
            # Create quoted text with better formatting
            claim_text = claim.get('text', 'No claim text available')
            quoted = f"Fact-check verdict: {rating}. Original claim: {claim_text[:150]}..."
            
            return Citation(
                url=url,
                title=title,
                publisher=publisher,
                quoted=quoted,
                rating=rating,
                review_date=review_date
            )
            
        except Exception as e:
            logger.error(f"Error parsing claim review: {e}")
            return None


async def retrieve_evidence_from_enriched(enrichment_result: LinkEnrichmentOutput) -> EvidenceRetrievalResult:
    """
    Evidence retrieval function that works with enriched claims from Step 2.5
    
    Simply propagates enriched link data forward while doing normal Google API evidence retrieval.
    
    Args:
        enrichment_result: Output from Step 2.5 (Link Enrichment)
        
    Returns:
        EvidenceRetrievalResult: External evidence + enriched link content for Step 4 (Adjudication)
    """
    retriever = GoogleFactCheckRetriever()
    evidence_map = {}
    total_sources_found = 0
    
    # Process each enriched claim - same logic as before, just different input structure
    for enriched_claim in enrichment_result.enriched_claims:
        logger.info(f"Retrieving evidence for claim: {enriched_claim.text}")
        
        # Use the SAME Google Fact-Check search logic as before
        citations = await retriever.search_claim(enriched_claim.text)
        
        # Create EnrichedClaim that includes BOTH:
        # 1. External evidence (Google Fact-Check citations)
        # 2. Enriched links (user-provided URL content from Step 2.5)
        claim_evidence = EnrichedClaim(
            claim_text=enriched_claim.text,
            citations=citations,  # External evidence from Google API
            search_queries=[f"Google Fact-Check for: {enriched_claim.text}"],
            enriched_links=enriched_claim.enriched_links,  # Propagated enriched content
            retrieval_notes=f"Google API: {len(citations)} external sources. User links: {len(enriched_claim.enriched_links)} enriched."
        )
        
        evidence_map[enriched_claim.text] = claim_evidence
        total_sources_found += len(citations)
    
    return EvidenceRetrievalResult(
        claim_evidence_map=evidence_map,
        total_sources_found=total_sources_found,
        retrieval_time_ms=0  # TODO: Add timing
    )


