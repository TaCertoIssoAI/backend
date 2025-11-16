"""
Fact-Checking Pipeline - Main Entry Point

This module orchestrates the complete 4-step fact-checking pipeline:
1. User Input Processing
2. Claim Extraction
3. Evidence Retrieval  
4. LLM Adjudication

Follows LangChain best practices with structured inputs/outputs and async processing.
"""

import time
from typing import Optional
from app.models.api import TextRequest, AnalysisResponse
from app.models.factchecking import (
    UserInput, 
    ClaimExtractionOutput, 
    ExtractedClaim,
    AdjudicationInput,
    EnrichedClaim,
    Citation
)
from app.ai.claim_extractor import create_claim_extractor
from app.ai.adjudicator import adjudicate_claims
from app.ai.factchecking.evidence_retrieval import retrieve_evidence_from_enriched
from app.ai.factchecking.link_enricher import create_link_enricher
from app.core.config import get_settings


def save_pipeline_step_json(step_name: str, step_data: dict, timestamp: str, prefix: str = "") -> Optional[str]:
    """
    Common function to save pipeline step data to JSON files.
    Only saves if DEBUG environment variable is True.
    
    Args:
        step_name: Name of the pipeline step (e.g., "1_claim_extraction")
        step_data: Dictionary containing the step data to save
        timestamp: Timestamp string for the filename
        prefix: Optional prefix for the filename (e.g., "prod_", "test_")
        
    Returns:
        Filename if saved, None if DEBUG is False or save failed
    """
    import json
    import os
    
    # Check if DEBUG mode is enabled
    settings = get_settings()
    print(f"🔧 DEBUG mode: {settings.DEBUG}")  # Temporary debug print
    if not settings.DEBUG:
        print("❌ DEBUG is False, not saving JSON files")  # Temporary debug print
        return None
    
    try:
        # Get the project root directory for saving files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        output_dir = os.path.join(project_root, "testoutput")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = f"{prefix}{step_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save the data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)
        
        return filename
        
    except Exception as e:
        # Log error but don't fail the pipeline
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save JSON dump for {step_name}: {e}")
        return None


def save_final_result_json(final_data: dict, timestamp: str) -> Optional[str]:
    """
    Save the final pipeline result to both timestamped and latest result.json files.
    Only saves if DEBUG environment variable is True.
    
    Args:
        final_data: Complete pipeline result data
        timestamp: Timestamp string for the filename
        
    Returns:
        Filename if saved, None if DEBUG is False or save failed
    """
    import json
    import os
    
    # Check if DEBUG mode is enabled
    settings = get_settings()
    if not settings.DEBUG:
        return None
    
    try:
        # Get the project root directory for saving files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        output_dir = os.path.join(project_root, "testoutput")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save timestamped version
        timestamped_filename = f"result_{timestamp}.json"
        timestamped_filepath = os.path.join(output_dir, timestamped_filename)
        
        with open(timestamped_filepath, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Save latest version
        latest_filepath = os.path.join(output_dir, "result.json")
        with open(latest_filepath, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        return timestamped_filename
        
    except Exception as e:
        # Log error but don't fail the pipeline
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save final result JSON: {e}")
        return None


async def process_text_request(request: TextRequest) -> AnalysisResponse:
    """
    Main pipeline entry point for text-only fact-checking.
    Runs the complete 5-step pipeline: Claim Extraction -> Link Enrichment -> Evidence Retrieval -> Adjudication
    
    Args:
        request: TextRequest containing the text to fact-check
        
    Returns:
        AnalysisResponse with fact-check results
    """
    from datetime import datetime
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Step 1: Convert API request to UserInput
        user_input = UserInput(
            text=request.text,
            locale="pt-BR",  # Default to Portuguese
            timestamp=None,
            context=None
        )
        
        # Step 2: Extract claims
        claim_extractor = create_claim_extractor()
        claims_result: ClaimExtractionOutput = await claim_extractor.extract_claims(user_input)
        
        # Save Step 1 output using common function
        step1_output = {
            "timestamp": timestamp,
            "step": "1_claim_extraction",
            "input": user_input.dict(),
            "output": claims_result.dict(),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        save_pipeline_step_json("step1_claims", step1_output, timestamp, "prod_")
        
        # Step 2.5: Link Enrichment
        step25_start = time.time()
        link_enricher = create_link_enricher()
        enrichment_result = await link_enricher.enrich_links(claims_result)
        
        # Save Step 2.5 output using common function
        step25_output = {
            "timestamp": timestamp,
            "step": "2.5_link_enrichment",
            "input": claims_result.dict(),
            "output": enrichment_result.dict(),
            "processing_time_ms": int((time.time() - step25_start) * 1000)
        }
        save_pipeline_step_json("step25_link_enrichment", step25_output, timestamp, "prod_")
        
        # Step 3: Evidence Retrieval
        step3_start = time.time()
        evidence_result = await retrieve_evidence_from_enriched(enrichment_result)
        
        # Save Step 3 output using common function
        step3_output = {
            "timestamp": timestamp,
            "step": "3_evidence_retrieval",
            "input": enrichment_result.dict(),
            "output": evidence_result.dict(),
            "processing_time_ms": int((time.time() - step3_start) * 1000)
        }
        save_pipeline_step_json("step3_evidence", step3_output, timestamp, "prod_")
        
        # Step 4: Adjudication
        step4_start = time.time()
        adjudication_input = AdjudicationInput(
            original_user_text=user_input.text,
            enriched_claims=enrichment_result.enriched_claims,
            evidence_map=evidence_result.claim_evidence_map,
            additional_context="Production pipeline execution"
        )
        
        final_result = await adjudicate_claims(adjudication_input)
        
        # Save Step 4 output using common function
        step4_output = {
            "timestamp": timestamp,
            "step": "4_adjudication",
            "input": adjudication_input.dict(),
            "output": final_result.dict(),
            "processing_time_ms": int((time.time() - step4_start) * 1000)
        }
        save_pipeline_step_json("step4_adjudication", step4_output, timestamp, "prod_")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Convert final result to AnalysisResponse format
        # Extract text before "Fontes de apoio:" for responseWithoutLinks
        analysis_text = final_result.analysis_text
        split_marker = "Fontes de apoio:"
        if split_marker in analysis_text:
            response_without_links = analysis_text.split(split_marker)[0].strip()
            # If responseWithoutLinks is empty after splitting, use full rationale
            if not response_without_links:
                response_without_links = analysis_text
        else:
            response_without_links = analysis_text  # If no sources section, use full text
        
        api_response = AnalysisResponse(
            message_id=f"prod_{hash(request.text)}",
            verdict="text_analysis",  # Simple indicator that this is text-based
            rationale=analysis_text,
            responseWithoutLinks=response_without_links,
            processing_time_ms=processing_time
        )
        
        # Save final result using common function
        final_output = {
            "timestamp": timestamp,
            "request": request.dict(),
            "response": api_response.dict(),
            "pipeline_summary": {
                "step1_claims_extracted": len(claims_result.claims),
                "step25_links_processed": enrichment_result.total_links_processed,
                "step25_successful_extractions": enrichment_result.successful_extractions,
                "step3_total_sources": evidence_result.total_sources_found,
                "step4_analysis_text_length": len(final_result.analysis_text),
                "total_processing_time_ms": processing_time
            },
            "files_created": [
                f"prod_step1_claims_{timestamp}.json",
                f"prod_step25_link_enrichment_{timestamp}.json",
                f"prod_step3_evidence_{timestamp}.json",
                f"prod_step4_adjudication_{timestamp}.json",
                f"result_{timestamp}.json"
            ]
        }
        save_final_result_json(final_output, timestamp)
        
        return api_response
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        
        # Return error response
        error_message = f"Erro durante processamento: {str(e)}. Não foi possível completar a análise."
        return AnalysisResponse(
            message_id=f"error_{hash(request.text)}",
            verdict="error",
            rationale=error_message,
            responseWithoutLinks=error_message,  # Same as rationale for errors
            processing_time_ms=processing_time
        )


async def test_adjudicator() -> dict:
    """
    Test function for the adjudicator with hard-coded realistic input.
    
    Returns:
        Dict with test results and timing information
    """
    start_time = time.time()
    
    # Create realistic extracted claims
    claims = [
        ExtractedClaim(
            text="Vacinas causam autismo",
            links=[],
            llm_comment="Alegação médica sobre efeitos adversos de vacinas que requer verificação científica",
            entities=["vacinas", "autismo"]
        ),
        ExtractedClaim(
            text="Pessoas com olhos azuis são mais inteligentes",
            links=[],
            llm_comment="Alegação sobre características físicas e inteligência que pode ser verificada com estudos científicos",
            entities=["olhos azuis", "inteligência"]
        )
    ]
    
    # Create realistic evidence with citations
    evidence_map = {
        "Vacinas causam autismo": EnrichedClaim(
            claim_text="Vacinas causam autismo",
            citations=[
                Citation(
                    url="https://www.cdc.gov/vaccinesafety/concerns/autism.html",
                    title="Vaccines Do Not Cause Autism",
                    publisher="Centers for Disease Control and Prevention",
                    quoted="Multiple studies have found no link between vaccines and autism spectrum disorders."
                ),
                Citation(
                    url="https://www.who.int/news-room/feature-stories/detail/autism-spectrum-disorders",
                    title="Autism spectrum disorders",
                    publisher="World Health Organization",
                    quoted="There is no evidence of a causal association between measles, mumps and rubella vaccine and autism."
                ),
                Citation(
                    url="https://pediatrics.aappublications.org/content/113/2/259",
                    title="Age at first measles-mumps-rubella vaccination in children with autism",
                    publisher="American Academy of Pediatrics",
                    quoted="No significant association was found between the timing of MMR vaccination and the onset of autistic symptoms."
                )
            ],
            search_queries=["vaccines autism link", "MMR autism studies", "vaccine safety autism"],
            retrieval_notes="Encontradas múltiplas fontes autoritativas contradizendo a alegação"
        ),
        "Pessoas com olhos azuis são mais inteligentes": EnrichedClaim(
            claim_text="Pessoas com olhos azuis são mais inteligentes",
            citations=[
                Citation(
                    url="https://www.nature.com/articles/s41562-018-0362-x",
                    title="No association between eye color and intelligence",
                    publisher="Nature Human Behaviour",
                    quoted="Large-scale genetic studies show no causal relationship between eye pigmentation and cognitive ability."
                ),
                Citation(
                    url="https://psycnet.apa.org/record/2007-10421-003",
                    title="Intelligence and physical attractiveness",
                    publisher="American Psychological Association",
                    quoted="Eye color shows no significant correlation with measured IQ in controlled studies."
                )
            ],
            search_queries=["eye color intelligence correlation", "blue eyes IQ studies"],
            retrieval_notes="Estudos científicos não encontram correlação entre cor dos olhos e inteligência"
        )
    }
    
    # Create adjudication input
    adjudication_input = AdjudicationInput(
        original_user_text="vacina causa autismo e pessoas com olhos azuis sao mais inteligente",
        claims=claims,
        evidence_map=evidence_map,
        additional_context="Teste do sistema de adjudicação com alegações comuns"
    )
    
    try:
        # Test the adjudicator
        result = await adjudicate_claims(adjudication_input)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "original_query": result.original_query,
            "overall_verdict": result.overall_verdict,
            "rationale": result.rationale,
            "citations_count": len(result.supporting_citations),
            "citations": [{"title": c.title, "publisher": c.publisher, "url": c.url, "quoted": c.quoted} for c in result.supporting_citations],
            "processing_time_ms": processing_time,
            "error": None
        }
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time_ms": processing_time,
            "full_error": repr(e)
        }


async def test_full_pipeline_steps_1_3_4() -> dict:
    """
    Test the complete pipeline: Claim Extraction -> Evidence Retrieval -> Adjudication
    This will show what the adjudicator LLM actually returns when given real Google API evidence.
    Saves output of each step to timestamped files.
    
    Returns:
        Dict with results from all 3 steps
    """
    import time
    import json
    import os
    from datetime import datetime
    from app.models.factchecking import ClaimExtractionOutput, UserInput, AdjudicationInput
    from app.ai.claim_extractor import create_claim_extractor
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the project root directory (where this script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from app/ai/
    output_dir = os.path.join(project_root, "testoutput")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Claim Extraction with real input (including URLs)
        user_input = UserInput(
            text="Flavio bolsonaro amo o PT segundo esse link: https://noticias.uol.com.br/politica/ultimas-noticias/2025/09/20/flavio-bolsonaro-defende-anistia-pec-da-blindagem-de-pec-da-sobrevivencia.htm e as vacinas causam autismo segundo esse link https://familia.sbim.org.br/duvidas/mitos/o-mercurio-presente-nas-vacinas-causa-autismo",
            locale="pt-BR"
        )
        
        claim_extractor = create_claim_extractor()
        claims_result = await claim_extractor.extract_claims(user_input)
        
        # Save Step 1 output
        step1_output = {
            "timestamp": timestamp,
            "step": "1_claim_extraction",
            "input": user_input.dict(),
            "output": claims_result.dict(),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
        save_pipeline_step_json("step1_claims", step1_output, timestamp)
        
        # Step 2.5: Link Enrichment
        step25_start = time.time()
        link_enricher = create_link_enricher()
        enrichment_result = await link_enricher.enrich_links(claims_result)
        
        # Save Step 2.5 output
        step25_output = {
            "timestamp": timestamp,
            "step": "2.5_link_enrichment",
            "input": claims_result.dict(),
            "output": enrichment_result.dict(),
            "processing_time_ms": int((time.time() - step25_start) * 1000)
        }
        
        save_pipeline_step_json("step25_link_enrichment", step25_output, timestamp)
        
        # Step 3: Evidence Retrieval with enriched claims
        step3_start = time.time()
        evidence_result = await retrieve_evidence_from_enriched(enrichment_result)
        
        # Save Step 3 output
        step3_output = {
            "timestamp": timestamp,
            "step": "3_evidence_retrieval",
            "input": enrichment_result.dict(),
            "output": evidence_result.dict(),
            "processing_time_ms": int((time.time() - step3_start) * 1000)
        }
        
        save_pipeline_step_json("step3_evidence", step3_output, timestamp)
        
        # Step 4: Adjudication with enriched claims and evidence
        step4_start = time.time()
        adjudication_input = AdjudicationInput(
            original_user_text=user_input.text,
            enriched_claims=enrichment_result.enriched_claims,
            evidence_map=evidence_result.claim_evidence_map,
            additional_context="Pipeline completo: extração -> enriquecimento -> evidências -> adjudicação"
        )
        
        final_result = await adjudicate_claims(adjudication_input)
        
        # Save Step 4 output
        step4_output = {
            "timestamp": timestamp,
            "step": "4_adjudication",
            "input": adjudication_input.dict(),
            "output": final_result.dict(),
            "processing_time_ms": int((time.time() - step4_start) * 1000)
        }
        
        save_pipeline_step_json("step4_adjudication", step4_output, timestamp)
        
        # Save complete pipeline summary
        total_processing_time = int((time.time() - start_time) * 1000)
        pipeline_summary = {
            "timestamp": timestamp,
            "complete_pipeline_summary": {
                "user_input": user_input.text,
                "step1_claims_extracted": len(claims_result.claims),
                "step25_links_processed": enrichment_result.total_links_processed,
                "step25_successful_extractions": enrichment_result.successful_extractions,
                "step3_total_sources": evidence_result.total_sources_found,
                "step4_final_verdict": final_result.overall_verdict,
                "step4_rationale": final_result.rationale,
                "step4_citations_count": len(final_result.supporting_citations),
                "total_processing_time_ms": total_processing_time
            },
            "files_created": [
                f"step1_claims_{timestamp}.json",
                f"step25_link_enrichment_{timestamp}.json",
                f"step3_evidence_{timestamp}.json", 
                f"step4_adjudication_{timestamp}.json",
                f"pipeline_summary_{timestamp}.json"
            ]
        }
        
        save_pipeline_step_json("pipeline_summary", pipeline_summary, timestamp)
        
        # Return API response
        return {
            "success": True,
            "timestamp": timestamp,
            "files_saved": pipeline_summary["files_created"],
            "step1_claims": [{"text": c.text, "llm_comment": c.llm_comment} for c in claims_result.claims],
            "step3_evidence_summary": {
                "total_sources": evidence_result.total_sources_found,
                "claims_with_evidence": len(evidence_result.claim_evidence_map)
            },
            "step4_adjudication_result": {
                "original_query": final_result.original_query,
                "overall_verdict": final_result.overall_verdict,
                "rationale": final_result.rationale,
                "citations_count": len(final_result.supporting_citations),
                "citations": [
                    {
                        "title": c.title,
                        "publisher": c.publisher,
                        "url": c.url,
                        "quoted": c.quoted,
                        "rating": c.rating,
                        "review_date": c.review_date
                    } for c in final_result.supporting_citations
                ]
            },
            "processing_time_ms": total_processing_time,
            "error": None
        }
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        
        # Save error to file
        error_output = {
            "timestamp": timestamp,
            "step": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time_ms": processing_time
        }
        
        save_pipeline_step_json("error", error_output, timestamp)
        
        return {
            "success": False,
            "timestamp": timestamp,
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time_ms": processing_time,
            "error_file": f"error_{timestamp}.json"
        }


async def test_evidence_retrieval() -> dict:
    """
    Test function for evidence retrieval (Step 3) with real claims
    
    Returns:
        Dict with test results and evidence found
    """
    import time
    from app.models.factchecking import ClaimExtractionOutput
    
    start_time = time.time()
    
    # Create realistic extracted claims (same as adjudicator test)
    claims = [
        ExtractedClaim(
            text="Vacinas causam autismo",
            links=[],
            llm_comment="Alegação médica sobre efeitos adversos de vacinas que requer verificação científica",
            entities=["vacinas", "autismo"]
        ),
        ExtractedClaim(
            text="Pessoas com olhos azuis são mais inteligentes",
            links=[],
            llm_comment="Alegação sobre características físicas e inteligência que pode ser verificada com estudos científicos",
            entities=["olhos azuis", "inteligência"]
        )
    ]
    
    # Create ClaimExtractionOutput
    claims_result = ClaimExtractionOutput(
        original_text="vacina causa autismo e pessoas com olhos azuis sao mais inteligente",
        claims=claims,
        processing_notes="Teste do sistema de recuperação de evidências"
    )
    
    try:
        # Step 2.5: Link Enrichment first
        link_enricher = create_link_enricher()
        enrichment_result = await link_enricher.enrich_links(claims_result)
        
        # Test evidence retrieval with enriched claims
        evidence_result = await retrieve_evidence_from_enriched(enrichment_result)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Format response with full citation details
        evidence_summary = {}
        for claim_text, evidence in evidence_result.claim_evidence_map.items():
            evidence_summary[claim_text] = {
                "citations_found": len(evidence.citations),
                "publishers": [c.publisher for c in evidence.citations],
                "search_queries": evidence.search_queries,
                "retrieval_notes": evidence.retrieval_notes,
                "full_citations": [
                    {
                        "url": c.url,
                        "title": c.title,
                        "publisher": c.publisher,
                        "quoted": c.quoted,
                        "rating": c.rating,
                        "review_date": c.review_date
                    } for c in evidence.citations
                ]
            }
        
        return {
            "success": True,
            "total_sources_found": evidence_result.total_sources_found,
            "claims_processed": len(evidence_result.claim_evidence_map),
            "evidence_summary": evidence_summary,
            "processing_time_ms": processing_time,
            "error": None
        }
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time_ms": processing_time
        }
