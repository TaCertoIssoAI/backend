from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.models.factchecking import (
    AdjudicationInput, 
    FactCheckResult, 
    EnrichedClaim,
    Citation,
    EnrichedClaim,
)
from app.core.config import get_settings

settings = get_settings()

# Initialize OpenAI model for reasoning
reasoning_model = ChatOpenAI(
    model="o4-mini",  # Use reasoning model
    # Note: o4-mini only supports default temperature (1)
    max_tokens=4000,  # Sufficient for detailed analysis
    timeout=30,  # 30 second timeout
)

# Portuguese adjudication prompt
adjudication_prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um especialista em verificação de fatos. Analise as alegações contra as evidências e forneça uma análise em texto simples e claro.

FORMATO DE RESPOSTA:
Retorne apenas um texto simples (não JSON) seguindo este formato:

{{2-3 frases descrevendo o texto de entrada, as alegações e o contexto geral}}

         Análise por alegação:
         • [Alegação 1]: [VERDICT em maiúsculo - VERDADEIRO/FALSO/ENGANOSO/NÃO VERIFICÁVEL]
         • [Alegação 2]: [VERDICT em maiúsculo]

Fontes de apoio:
- [Publisher]: "[Citação]" ([URL])
- [Publisher]: "[Citação]" ([URL])

REGRAS:
    1. Use APENAS as evidências fornecidas
    2. VERDICTS: VERDADEIRO, FALSO, ENGANOSO, NÃO VERIFICÁVEL
    3. Inclua as fontes mais relevantes
    4. Seja claro e objetivo"""),
    
    ("user", """CONSULTA ORIGINAL DO USUÁRIO:
{original_query}

ALEGAÇÕES EXTRAÍDAS:
{claims_text}

EVIDÊNCIAS COLETADAS:
{evidence_text}

ANÁLISE SOLICITADA:
Forneça uma análise em texto simples seguindo o formato especificado no sistema. Use apenas as evidências fornecidas.""")
])

# Create the adjudication chain for text output
adjudication_chain = (
    adjudication_prompt 
    | reasoning_model
)


async def adjudicate_claims(adjudication_input: AdjudicationInput) -> FactCheckResult:
    """
    Main adjudication function that analyzes claims against evidence.
    
    Args:
        adjudication_input: Contains original text, claims, and evidence
        
    Returns:
        FactCheckResult with analysis text
    """
    try:
        # Format claims for the prompt
        claims_text = _format_claims_for_prompt(adjudication_input.enriched_claims)
        
        # Format evidence for the prompt
        evidence_text = _format_evidence_for_prompt(adjudication_input.evidence_map)
        
        # Invoke the adjudication chain
        response = await adjudication_chain.ainvoke({
            "original_query": adjudication_input.original_user_text,
            "claims_text": claims_text,
            "evidence_text": evidence_text
        })
        
        # Extract text content from response
        analysis_text = response.content if hasattr(response, 'content') else str(response)
        
        return FactCheckResult(
            original_query=adjudication_input.original_user_text,
            analysis_text=analysis_text
        )
        
    except Exception as e:
        # Fallback to unverifiable with error explanation
        return FactCheckResult(
            original_query=adjudication_input.original_user_text,
            analysis_text=f"Erro durante processamento: {str(e)}. Não foi possível completar a análise."
        )


def _format_claims_for_prompt(claims: List[EnrichedClaim]) -> str:
    """Format enriched claims for the LLM prompt"""
    if not claims:
        return "Nenhuma alegação específica foi extraída."
    
    formatted_claims = []
    for i, claim in enumerate(claims, 1):
        claim_text = f"{i}. **Alegação**: {claim.text}\n"
        
        if claim.entities:
            claim_text += f"   **Entidades**: {', '.join(claim.entities)}\n"
        
        if claim.original_links:
            claim_text += f"   **Links originais**: {', '.join(claim.original_links)}\n"
        
        if claim.enriched_links:
            claim_text += f"   **Conteúdo dos links**: \n"
            for link in claim.enriched_links:
                if link.extraction_status == "success":
                    link_content = link.summary if link.summary else link.content[:200] + "..."
                    link_text = f"     - URL: {link.url}\n       Título: {link.title}\n       Resumo: {link_content}\n"
                    claim_text += link_text
                else:
                    claim_text += f"     - URL: {link.url} (Falha na extração: {link.extraction_notes})\n"
        
        claim_text += f"   **Análise LLM**: {claim.llm_comment}\n"
        
        formatted_claims.append(claim_text)
    
    return "\n".join(formatted_claims)


def _format_evidence_for_prompt(evidence_map: Dict[str, EnrichedClaim]) -> str:
    """Format evidence for the LLM prompt"""
    if not evidence_map:
        return "Nenhuma evidência foi coletada."
    
    formatted_evidence = []
    
    for claim_text, evidence in evidence_map.items():
        evidence_text = f"\n**EVIDÊNCIAS PARA**: {claim_text}\n"
        evidence_text += f"**Consultas utilizadas**: {', '.join(evidence.search_queries)}\n"
        
        if evidence.citations:
            evidence_text += "**Fontes encontradas**:\n"
            for i, citation in enumerate(evidence.citations, 1):
                evidence_text += f"  {i}. **{citation.title}** ({citation.publisher})\n"
                evidence_text += f"     URL: {citation.url}\n"
                if citation.quoted:
                    evidence_text += f"     Trecho: \"{citation.quoted}\"\n"
                evidence_text += "\n"
        else:
            evidence_text += "**Nenhuma fonte relevante encontrada**\n"
        
        if evidence.retrieval_notes:
            evidence_text += f"**Notas**: {evidence.retrieval_notes}\n"
        
        formatted_evidence.append(evidence_text)
    
    return "\n".join(formatted_evidence)

def _create_fallback_result(input_data: AdjudicationInput, error_msg: str) -> FactCheckResult:
    """Create a fallback result when adjudication fails"""
    
    # Extract any available citations
    citations = []
    for evidence in input_data.evidence_map.values():
        citations.extend(evidence.citations[:2])  # Max 2 per claim
    
    if not citations:
        # Create a minimal citation to satisfy the constraint
        citations = [Citation(
            url="https://example.com/error",
            title="Erro no processamento",
            publisher="Sistema",
            quoted="Não foi possível processar as evidências adequadamente."
        )]
    
    return FactCheckResult(
        original_query=input_data.original_user_text,
        overall_verdict="unverifiable",
        rationale=f"Não foi possível completar a análise devido a um erro técnico. "
                 f"Recomendamos verificar manualmente as fontes disponíveis. Erro: {error_msg[:100]}",
        supporting_citations=citations[:3]  # Max 3 citations
    )
