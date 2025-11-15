from fastapi import APIRouter, HTTPException
import time
import json
import os
from datetime import datetime
from app.models.api import TextRequest, AnalysisResponse
from app.ai.main_pipeline import process_text_request, test_adjudicator, test_evidence_retrieval, test_full_pipeline_steps_1_3_4

router = APIRouter()


@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest) -> AnalysisResponse:
    """
    Analyze text-only messages for fact-checking
    """
    try:
        return await process_text_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/test-adjudicator")
async def test_adjudicator_endpoint():
    """
    Test endpoint for the adjudicator with hard-coded realistic data
    """
    try:
        result = await test_adjudicator()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")


@router.get("/test-evidence-retrieval")
async def test_evidence_retrieval_endpoint():
    """
    Test endpoint for evidence retrieval (Step 3) using Google Fact-Check API
    """
    try:
        result = await test_evidence_retrieval()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evidence retrieval test error: {str(e)}")


@router.get("/test-full-pipeline")
async def test_full_pipeline_endpoint():
    """
    Test endpoint for complete pipeline: Claim Extraction -> Evidence Retrieval -> Adjudication
    This shows what the adjudicator LLM actually returns when given real Google API evidence.
    """
    try:
        result = await test_full_pipeline_steps_1_3_4()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full pipeline test error: {str(e)}")


@router.get("/test-claims-with-urls")
async def test_claims_with_urls_endpoint():
    """
    Test endpoint for claim extraction with URLs to verify link extraction
    """
    try:
        from app.models.factchecking import UserInput
        from app.ai.claim_extractor import create_claim_extractor
        
        # Test with URLs
        user_input = UserInput(
            text="vacina causa autismo segundo este link https://example.com/fake-vaccine-study e pessoas com olhos azuis sao mais inteligente como mostra https://fakesite.com/blue-eyes-iq",
            locale="pt-BR"
        )
        
        claim_extractor = create_claim_extractor()
        claims_result = await claim_extractor.extract_claims(user_input)
        
        return {
            "success": True,
            "original_text": claims_result.original_text,
            "claims_extracted": len(claims_result.claims),
            "claims": [
                {
                    "text": claim.text,
                    "links": claim.links,
                    "entities": claim.entities,
                    "llm_comment": claim.llm_comment
                } for claim in claims_result.claims
            ],
            "processing_notes": claims_result.processing_notes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claims extraction test error: {str(e)}")


@router.get("/test-link-enrichment")
async def test_link_enrichment_endpoint(url: str = None):
    """
    Test endpoint for link enrichment (Step 2.5) using real web scraping
    Saves output to timestamped file like other pipeline steps
    
    Args:
        url: Optional URL to test. If not provided, uses default test URLs
    """
    try:
        from app.models.factchecking import ClaimExtractionOutput, ExtractedClaim
        from app.ai.factchecking.link_enricher import create_link_enricher
        
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        output_dir = os.path.join(project_root, "testoutput")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test claims with real URLs for testing
        if url:
            # Test only the provided URL
            test_claims = [
                ExtractedClaim(
                    text=f"Teste de link enricher para: {url}",
                    links=[url],
                    llm_comment=f"Teste específico do link enricher com URL: {url}",
                    entities=["teste", "link", "enricher"]
                )
            ]
        else:
            # Default test URLs
            test_claims = [
                ExtractedClaim(
                    text="Post do X/Twitter sobre teste de link enricher",
                    links=["https://x.com/lfdinizcosta/status/1969468124559716570"],
                    llm_comment="Teste específico do link enricher com X/Twitter",
                    entities=["X", "Twitter", "teste"]
                ),
                ExtractedClaim(
                    text="Informações sobre políticas públicas no Brasil",
                    links=["https://www.gov.br/pt-br"],
                    llm_comment="Alegação sobre políticas que pode ser verificada através de fontes oficiais",
                    entities=["políticas públicas", "Brasil"]
                ),
                ExtractedClaim(
                    text="Notícias sobre tecnologia",
                    links=["https://g1.globo.com/tecnologia/"],
                    llm_comment="Alegação sobre tecnologia que pode ter fontes verificáveis",
                    entities=["tecnologia", "notícias"]
                )
            ]
        
        claims_result = ClaimExtractionOutput(
            original_text="Test text with multiple URLs for link enrichment",
            claims=test_claims,
            processing_notes="Teste de enriquecimento de links"
        )
        
        # Test link enrichment
        link_enricher = create_link_enricher()
        enrichment_result = await link_enricher.enrich_links(claims_result)
        
        # Save Step 2.5 output to file
        step25_output = {
            "timestamp": timestamp,
            "step": "2.5_link_enrichment",
            "input": claims_result.dict(),
            "output": enrichment_result.dict(),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
        with open(f"{output_dir}/step25_link_enrichment_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(step25_output, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "timestamp": timestamp,
            "file_saved": f"step25_link_enrichment_{timestamp}.json",
            "total_links_processed": enrichment_result.total_links_processed,
            "successful_extractions": enrichment_result.successful_extractions,
            "processing_time_ms": enrichment_result.processing_time_ms,
            "processing_notes": enrichment_result.processing_notes,
            "enriched_claims": [
                {
                    "text": claim.text,
                    "original_links": claim.original_links,
                    "enriched_links": [
                        {
                            "url": link.url,
                            "title": link.title,
                            "content_length": len(link.content),
                            "summary": link.summary,
                            "extraction_status": link.extraction_status,
                            "extraction_notes": link.extraction_notes
                        } for link in claim.enriched_links
                    ],
                    "llm_comment": claim.llm_comment,
                    "entities": claim.entities
                } for claim in enrichment_result.enriched_claims
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Link enrichment test error: {str(e)}")