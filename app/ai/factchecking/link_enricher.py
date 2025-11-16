"""
Link Enrichment Module - Step 2.5 of Fact-Checking Pipeline

This module takes claims with URLs and enriches them by extracting content from those URLs.
Prepared to integrate with external scraping API service.

Follows LangChain best practices:
- Async-first design
- Structured outputs with Pydantic models
- Error handling and fallback mechanisms
- Detailed logging and processing notes
"""

import asyncio
import time
import logging

from app.models.factchecking import (
    ClaimExtractionOutput,
    ExtractedClaim,
    EnrichedLink,
    EnrichedClaim,
    LinkEnrichmentOutput
)

logger = logging.getLogger(__name__)


class LinkEnricher:
    """
    Enriches claims by extracting content from their associated URLs.
    
    Uses external scraping API service for web content extraction.
    """
    
    def __init__(self, content_limit: int = 5000):
        """Initialize the link enricher with web scraping capabilities."""
        
        # Content limit for extracted text
        self.content_limit = content_limit

    async def enrich_links(self, claims_result: ClaimExtractionOutput) -> LinkEnrichmentOutput:
        """
        Main method to enrich all claims with link content.
        
        Args:
            claims_result: Result from claim extraction step
            
        Returns:
            LinkEnrichmentOutput with enriched claims
        """
        start_time = time.time()
        
        enriched_claims = []
        total_links = 0
        successful_extractions = 0
        
        for claim in claims_result.claims:
            if claim.links:
                total_links += len(claim.links)
                enriched_claim = await self._enrich_single_claim(claim)
                
                # Count successful extractions
                for enriched_link in enriched_claim.enriched_links:
                    if enriched_link.extraction_status == "success":
                        successful_extractions += 1
                        
                enriched_claims.append(enriched_claim)
            else:
                # No links to enrich, convert to EnrichedClaim as-is
                enriched_claim = EnrichedClaim(
                    text=claim.text,
                    original_links=[],
                    enriched_links=[],
                    llm_comment=claim.llm_comment,
                    entities=claim.entities
                )
                enriched_claims.append(enriched_claim)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        processing_notes = (
            f"Processados {total_links} links. "
            f"{successful_extractions} extrações bem-sucedidas. "
            f"{total_links - successful_extractions} falhas."
        )
        
        return LinkEnrichmentOutput(
            original_claims=claims_result.claims,
            enriched_claims=enriched_claims,
            total_links_processed=total_links,
            successful_extractions=successful_extractions,
            processing_time_ms=processing_time,
            processing_notes=processing_notes
        )

    async def _enrich_single_claim(self, claim: ExtractedClaim) -> EnrichedClaim:
        """Enrich a single claim by extracting content from its links."""
        
        enriched_links = []
        
        # Process each link in the claim
        for url in claim.links:
            enriched_link = await self._extract_link_content(url)
            enriched_links.append(enriched_link)
        
        return EnrichedClaim(
            text=claim.text,
            original_links=claim.links,
            enriched_links=enriched_links,
            llm_comment=claim.llm_comment,
            entities=claim.entities
        )

    async def _extract_link_content(self, url: str) -> EnrichedLink:
        """
        Extract content from a single URL using external scraping API.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            EnrichedLink with extracted content and summary
        """
        enriched_link = EnrichedLink(
            url=url,
            extraction_status="pending"
        )
        
        try:
            # TODO: Implement external scraping API call here
            # Example structure:
            # extraction_result = await self._call_external_scraping_api(url)
            # 
            # if extraction_result and extraction_result.get("success"):
            #     enriched_link.title = extraction_result.get("title", "")
            #     
            #     # Apply content limit
            #     full_content = extraction_result.get("content", "")
            #     enriched_link.content = full_content[:self.content_limit] if full_content else ""
            #     
            #     # Create a simple summary
            #     enriched_link.summary = self._create_simple_summary(
            #         enriched_link.title, 
            #         enriched_link.content
            #     )
            #     
            #     enriched_link.extraction_status = "success"
            #     enriched_link.extraction_notes = f"Content extracted successfully. Size: {len(full_content)} chars"
            # else:
            #     enriched_link.extraction_status = "failed"
            #     enriched_link.extraction_notes = "External API extraction failed"
            
            # Placeholder until external API is integrated
            logger.warning(f"Link enrichment not implemented for {url}")
            enriched_link.extraction_status = "failed"
            enriched_link.extraction_notes = "Scraping functionality not yet implemented. Waiting for external API integration."
                
        except Exception as e:
            logger.error(f"Link enrichment failed for {url}: {e}")
            enriched_link.extraction_status = "failed"
            enriched_link.extraction_notes = f"Error during extraction: {str(e)[:100]}"
        
        return enriched_link

    # async def _call_external_scraping_api(self, url: str) -> dict:
    #     """
    #     Call external scraping API to extract content.
    #     
    #     TODO: Implement actual external API call
    #     
    #     Args:
    #         url: URL to scrape
    #         
    #     Returns:
    #         dict with extraction results
    #     """
    #     pass

    def _create_simple_summary(self, title: str, content: str) -> str:
        """
        Create a simple summary without LLM, using basic text processing.
        """
        if not content:
            return "Conteúdo não disponível"
        
        # Use title and first paragraph as summary
        summary_parts = []
        
        if title:
            summary_parts.append(f"Título: {title}")
        
        # Get first paragraph (up to first double newline or first 200 chars)
        first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content
        first_paragraph = first_paragraph[:200] + "..." if len(first_paragraph) > 200 else first_paragraph
        
        if first_paragraph.strip():
            summary_parts.append(f"Resumo: {first_paragraph.strip()}")
        
        return " | ".join(summary_parts) if summary_parts else "Conteúdo extraído mas sem informações resumíveis"


# Factory function for creating enricher
def create_link_enricher(content_limit: int = 5000) -> LinkEnricher:
    """
    Factory function to create a LinkEnricher instance.
    """
    return LinkEnricher(content_limit=content_limit)


# Async helper function for direct usage
async def enrich_claim_links(claims_result: ClaimExtractionOutput) -> LinkEnrichmentOutput:
    """
    Convenience function to enrich links from claim extraction result.
    
    Args:
        claims_result: ClaimExtractionOutput from step 2
        
    Returns:
        LinkEnrichmentOutput with enriched claims
    """
    enricher = create_link_enricher()
    return await enricher.enrich_links(claims_result)
