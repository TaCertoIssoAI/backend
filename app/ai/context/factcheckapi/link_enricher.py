"""
link enrichment module for fact-checking pipeline.
extracts content from urls using external scraping api.
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
    """enriches claims by extracting content from urls using external scraping api"""
    
    def __init__(self, content_limit: int = 5000):
        self.content_limit = content_limit

    async def enrich_links(self, claims_result: ClaimExtractionOutput) -> LinkEnrichmentOutput:
        """enrich all claims with link content"""
        start_time = time.time()
        
        enriched_claims = []
        total_links = 0
        successful_extractions = 0
        
        for claim in claims_result.claims:
            if claim.links:
                total_links += len(claim.links)
                enriched_claim = await self._enrich_single_claim(claim)
                
                for enriched_link in enriched_claim.enriched_links:
                    if enriched_link.extraction_status == "success":
                        successful_extractions += 1
                        
                enriched_claims.append(enriched_claim)
            else:
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
        """enrich single claim by extracting content from its links"""
        enriched_links = []
        
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
        """extract content from url using external scraping api"""
        enriched_link = EnrichedLink(url=url, extraction_status="pending")
        
        try:
            # TODO: integrate with scrapeGenericUrl from apify_utils
            # from app.ai.context.apify_utils import scrapeGenericUrl
            # result = await scrapeGenericUrl(url, maxChars=self.content_limit)
            # 
            # if result["success"]:
            #     enriched_link.content = result["content"]
            #     enriched_link.title = result.get("metadata", {}).get("title", "")
            #     enriched_link.summary = self._create_simple_summary(
            #         enriched_link.title,
            #         enriched_link.content
            #     )
            #     enriched_link.extraction_status = "success"
            #     enriched_link.extraction_notes = f"extracted {len(result['content'])} chars"
            # else:
            #     enriched_link.extraction_status = "failed"
            #     enriched_link.extraction_notes = result.get("error", "unknown error")
            
            enriched_link.extraction_status = "failed"
            enriched_link.extraction_notes = "scraping not yet integrated with link enricher"
                
        except Exception as e:
            logger.error(f"link enrichment failed for {url}: {e}")
            enriched_link.extraction_status = "failed"
            enriched_link.extraction_notes = f"error: {str(e)[:100]}"
        
        return enriched_link

    def _create_simple_summary(self, title: str, content: str) -> str:
        """create simple summary from title and first paragraph"""
        if not content:
            return "conteúdo não disponível"
        
        summary_parts = []
        
        if title:
            summary_parts.append(f"título: {title}")
        
        first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content
        first_paragraph = first_paragraph[:200] + "..." if len(first_paragraph) > 200 else first_paragraph
        
        if first_paragraph.strip():
            summary_parts.append(f"resumo: {first_paragraph.strip()}")
        
        return " | ".join(summary_parts) if summary_parts else "conteúdo sem resumo"


def create_link_enricher(content_limit: int = 5000) -> LinkEnricher:
    """factory function to create link enricher instance"""
    return LinkEnricher(content_limit=content_limit)


async def enrich_claim_links(claims_result: ClaimExtractionOutput) -> LinkEnrichmentOutput:
    """convenience function to enrich links from claim extraction result"""
    enricher = create_link_enricher()
    return await enricher.enrich_links(claims_result)
