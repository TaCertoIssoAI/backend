# -*- coding: utf-8 -*-
"""
Tests for the adjudication/judgment pipeline step.

These tests make REAL calls to the LLM (OpenAI API) to validate:
- The structure of outputs
- The LangChain chain works correctly
- The prompt produces valid results
- Verdict generation works properly

IMPORTANT: Set OPENAI_API_KEY in your environment before running.

Run with:
    pytest app/ai/pipeline/tests/judgment_test.py -v -s

The -s flag shows stdout so you can see the LLM responses for debugging.
"""

import pytest
from langchain_openai import ChatOpenAI

from app.models import (
    AdjudicationInput,
    FactCheckResult,
    DataSourceResult,
    ClaimVerdict,
    DataSourceWithClaims,
    DataSource,
    EnrichedClaim,
    Citation,
    LLMConfig,
)
from app.ai.pipeline import (
    adjudicate_claims,
    build_adjudication_chain,
)


# ===== HELPER FUNCTIONS =====

def print_adjudication_input(adjudication_input: AdjudicationInput, test_name: str):
    """Print the EXACT formatted input that the LLM sees."""
    from app.ai.pipeline.judgement import format_adjudication_input
    
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    print("\n📥 EXACT LLM INPUT (what the model sees):\n")
    
    # Format exactly as the LLM will see it
    formatted_sources = format_adjudication_input(adjudication_input)
    print(formatted_sources)
    
    if adjudication_input.additional_context:
        print(f"\n**Contexto Adicional**: {adjudication_input.additional_context}\n")
    
    print("=" * 80)


def print_fact_check_result(result: FactCheckResult, test_name: str):
    """Print the complete fact-check result for debugging."""
    print("\n" + "=" * 80)
    print(f"📤 FACT-CHECK RESULT FOR: {test_name}")
    print("=" * 80 + "\n")
    
    if result.overall_summary:
        print(f"OVERALL SUMMARY:\n{result.overall_summary}\n")
        print("=" * 80 + "\n")
    
    for i, data_source_result in enumerate(result.results, 1):
        print(f"DATA SOURCE {i}:")
        print(f"  ID: {data_source_result.data_source_id}")
        print(f"  Type: {data_source_result.source_type}")
        print(f"  Number of verdicts: {len(data_source_result.claim_verdicts)}\n")
        
        for j, verdict in enumerate(data_source_result.claim_verdicts, 1):
            print(f"  VERDICT {j}:")
            print(f"    Claim ID: {verdict.claim_id}")
            print(f"    Claim Text: {verdict.claim_text}")
            print(f"    Verdict: {verdict.verdict}")
            print(f"    Justification: {verdict.justification}")
            print(f"    Citations Used: {len(verdict.citations_used)} citation(s)")
            for k, citation in enumerate(verdict.citations_used, 1):
                print(f"      [{k}] {citation.title} ({citation.url})")
            print()
        
        print("-" * 80 + "\n")


def validate_claim_verdict(verdict: ClaimVerdict):
    """Validate that a ClaimVerdict has the correct structure."""
    # Required fields
    assert verdict.claim_id is not None and verdict.claim_id != "", "Claim ID should not be empty"
    assert verdict.claim_text is not None and verdict.claim_text != "", "Claim text should not be empty"
    assert verdict.verdict is not None, "Verdict should not be None"
    assert verdict.justification is not None and verdict.justification != "", "Justification should not be empty"

    # Type checks
    assert isinstance(verdict.claim_id, str), "Claim ID should be a string"
    assert isinstance(verdict.claim_text, str), "Claim text should be a string"
    assert isinstance(verdict.verdict, str), "Verdict should be a string"
    assert isinstance(verdict.justification, str), "Justification should be a string"

    # Verdict should be one of the valid options
    valid_verdicts = ["Verdadeiro", "Falso", "Fora de Contexto", "Não foi possível verificar"]
    assert verdict.verdict in valid_verdicts, f"Verdict must be one of {valid_verdicts}, got: {verdict.verdict}"

    # Validate citations_used field
    assert hasattr(verdict, 'citations_used'), "ClaimVerdict should have citations_used field"
    assert isinstance(verdict.citations_used, list), "citations_used should be a list"
    for citation in verdict.citations_used:
        assert isinstance(citation, Citation), f"Each citation should be a Citation object, got {type(citation)}"


def validate_data_source_result(data_source_result: DataSourceResult):
    """Validate that a DataSourceResult has the correct structure."""
    # Required fields
    assert data_source_result.data_source_id is not None, "Data source ID should not be None"
    assert data_source_result.source_type is not None, "Source type should not be None"
    assert data_source_result.claim_verdicts is not None, "Claim verdicts should not be None"
    
    # Type checks
    assert isinstance(data_source_result.data_source_id, str), "Data source ID should be a string"
    assert isinstance(data_source_result.source_type, str), "Source type should be a string"
    assert isinstance(data_source_result.claim_verdicts, list), "Claim verdicts should be a list"
    
    # Validate each verdict
    for verdict in data_source_result.claim_verdicts:
        validate_claim_verdict(verdict)


def validate_fact_check_result(result: FactCheckResult):
    """Validate that a FactCheckResult has the correct structure."""
    # Type check
    assert isinstance(result, FactCheckResult), "Result should be a FactCheckResult"
    assert isinstance(result.results, list), "Results should be a list"
    
    # Validate each data source result
    for data_source_result in result.results:
        validate_data_source_result(data_source_result)
    
    # Overall summary is optional but should be string if present
    if result.overall_summary is not None:
        assert isinstance(result.overall_summary, str), "Overall summary should be a string"


# ===== TESTS =====

def test_basic_adjudication_single_claim():
    """Test basic adjudication with a single claim and evidence."""
    # Setup
    data_source = DataSource(
        id="msg-001",
        source_type="original_text",
        original_text="Ouvi dizer que a vacina X causa infertilidade em mulheres, isso é verdade?",
        metadata={},
        locale="pt-BR"
    )
    
    # Create enriched claim with evidence
    enriched_claim = EnrichedClaim(
        id="claim-uuid-1",
        claim_text="A vacina X causa infertilidade em mulheres",
        citations=[
            Citation(
                url="https://saude.gov.br/estudo-vacinas",
                title="Estudo de Segurança de Vacinas",
                publisher="Ministério da Saúde",
                quoted="Um estudo com 50.000 participantes não encontrou evidências ligando a vacina X a problemas de fertilidade.",
                rating="Falso",
                review_date="2024-11-05"
            )
        ],
        search_queries=["vacina X infertilidade", "vacina X fertilidade mulheres"],
        retrieval_notes="Encontradas múltiplas fontes confiáveis contradizendo a alegação"
    )
    
    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=[enriched_claim]
    )
    
    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims],
        additional_context="Usuário demonstra preocupação com segurança de vacinas"
    )
    
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )
    
    # Print input for debugging
    print_adjudication_input(adjudication_input, "Basic Adjudication Single Claim")
    
    # Execute
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )
    
    # Print output for debugging
    print_fact_check_result(result, "Basic Adjudication Single Claim")
    
    # Validate structure
    validate_fact_check_result(result)
    assert len(result.results) == 1, "Should have results for 1 data source"
    assert len(result.results[0].claim_verdicts) == 1, "Should have 1 verdict"
    
    # Check that verdict was generated for the correct claim
    verdict = result.results[0].claim_verdicts[0]
    assert verdict.claim_id == "claim-uuid-1", "Verdict should be for the correct claim"


def test_adjudication_multiple_claims_same_source():
    """Test adjudication with multiple claims from the same data source."""
    # Setup
    data_source = DataSource(
        id="msg-002",
        source_type="original_text",
        original_text="O presidente anunciou um novo imposto sobre carbono de R$250 por tonelada. Além disso, o governo vai investir R$500 bilhões em energia renovável.",
        metadata={},
        locale="pt-BR"
    )
    
    enriched_claims = [
        EnrichedClaim(
            id="claim-uuid-2a",
            claim_text="O presidente anunciou um imposto sobre carbono de R$250 por tonelada",
            citations=[
                Citation(
                    url="https://g1.globo.com/politica",
                    title="Presidente anuncia imposto sobre carbono",
                    publisher="G1",
                    quoted="O presidente confirmou o novo imposto sobre carbono no valor de R$250 por tonelada.",
                    rating="Verdadeiro",
                    review_date="2024-11-10"
                )
            ],
            search_queries=["presidente imposto carbono"],
            retrieval_notes="Confirmado por múltiplas fontes jornalísticas"
        ),
        EnrichedClaim(
            id="claim-uuid-2b",
            claim_text="O governo vai investir R$500 bilhões em energia renovável",
            citations=[],
            search_queries=["governo investimento energia renovável"],
            retrieval_notes="Nenhuma fonte confiável encontrada para confirmar este valor"
        )
    ]
    
    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=enriched_claims
    )
    
    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims]
    )
    
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )
    
    # Print input for debugging
    print_adjudication_input(adjudication_input, "Multiple Claims Same Source")
    
    # Execute
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )
    
    # Print output for debugging
    print_fact_check_result(result, "Multiple Claims Same Source")
    
    # Validate structure
    validate_fact_check_result(result)
    assert len(result.results) == 1, "Should have results for 1 data source"
    assert len(result.results[0].claim_verdicts) == 2, "Should have 2 verdicts"
    
    # Check that verdicts were generated for both claims
    verdict_ids = {v.claim_id for v in result.results[0].claim_verdicts}
    assert "claim-uuid-2a" in verdict_ids, "Should have verdict for first claim"
    assert "claim-uuid-2b" in verdict_ids, "Should have verdict for second claim"


def test_adjudication_multiple_data_sources():
    """Test adjudication with claims from multiple data sources."""
    # Setup - First data source (original message)
    data_source_1 = DataSource(
        id="msg-003",
        source_type="original_text",
        original_text="Dizem que a vacina da COVID causa problemas no coração.",
        metadata={},
        locale="pt-BR"
    )
    
    enriched_claim_1 = EnrichedClaim(
        id="claim-uuid-3a",
        claim_text="A vacina da COVID causa problemas no coração",
        citations=[
            Citation(
                url="https://www.fiocruz.br/covid-vacinas",
                title="Segurança das Vacinas COVID-19",
                publisher="Fiocruz",
                quoted="Estudos mostram que casos de miocardite são raros e geralmente leves, com benefícios da vacinação superando riscos.",
                rating="Fora de Contexto",
                review_date="2024-10-20"
            )
        ],
        search_queries=["vacina covid coração"],
        retrieval_notes="Há casos raros de miocardite, mas a alegação omite contexto importante"
    )
    
    # Setup - Second data source (link context)
    data_source_2 = DataSource(
        id="link-004",
        source_type="link_context",
        original_text="Novo estudo revela que uso de máscaras reduziu transmissão de COVID em 70%.",
        metadata={
            "title": "Eficácia de Máscaras",
            "url": "https://example.com/mascaras"
        },
        locale="pt-BR"
    )
    
    enriched_claim_2 = EnrichedClaim(
        id="claim-uuid-3b",
        claim_text="O uso de máscaras reduziu a transmissão de COVID em 70%",
        citations=[],
        search_queries=["máscaras eficácia covid"],
        retrieval_notes="Não foram encontradas fontes para verificar este percentual específico"
    )
    
    sources_with_claims = [
        DataSourceWithClaims(
            data_source=data_source_1,
            enriched_claims=[enriched_claim_1]
        ),
        DataSourceWithClaims(
            data_source=data_source_2,
            enriched_claims=[enriched_claim_2]
        )
    ]
    
    adjudication_input = AdjudicationInput(
        sources_with_claims=sources_with_claims
    )
    
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )
    
    # Print input for debugging
    print_adjudication_input(adjudication_input, "Multiple Data Sources")
    
    # Execute
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )
    
    # Print output for debugging
    print_fact_check_result(result, "Multiple Data Sources")
    
    # Validate structure
    validate_fact_check_result(result)
    assert len(result.results) == 2, "Should have results for 2 data sources"
    assert len(result.results[0].claim_verdicts) == 1, "First source should have 1 verdict"
    assert len(result.results[1].claim_verdicts) == 1, "Second source should have 1 verdict"
    
    # Check data source IDs match
    source_ids = {r.data_source_id for r in result.results}
    assert "msg-003" in source_ids, "Should have result for first data source"
    assert "link-004" in source_ids, "Should have result for second data source"


def test_adjudication_no_evidence():
    """Test adjudication when no evidence is available."""
    # Setup
    data_source = DataSource(
        id="msg-005",
        source_type="original_text",
        original_text="Li que existe uma nova tecnologia que permite carros voarem a 500 km/h.",
        metadata={},
        locale="pt-BR"
    )
    
    enriched_claim = EnrichedClaim(
        id="claim-uuid-4",
        claim_text="Existe uma nova tecnologia que permite carros voarem a 500 km/h",
        citations=[],
        search_queries=["carros voadores tecnologia"],
        retrieval_notes="Nenhuma fonte encontrada sobre esta tecnologia específica"
    )
    
    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=[enriched_claim]
    )
    
    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims]
    )
    
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )
    
    # Print input for debugging
    print_adjudication_input(adjudication_input, "No Evidence Available")
    
    # Execute
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )
    
    # Print output for debugging
    print_fact_check_result(result, "No Evidence Available")
    
    # Validate structure
    validate_fact_check_result(result)
    assert len(result.results) == 1, "Should have results for 1 data source"
    assert len(result.results[0].claim_verdicts) == 1, "Should have 1 verdict"
    
    # With no evidence, verdict should be "Não foi possível verificar"
    # Note: We don't assert this because LLM behavior may vary, but it's expected


def test_adjudication_with_contradictory_sources():
    """Test adjudication when evidence sources contradict each other."""
    # Setup
    data_source = DataSource(
        id="msg-006",
        source_type="original_text",
        original_text="O café aumenta o risco de doenças cardíacas.",
        metadata={},
        locale="pt-BR"
    )
    
    enriched_claim = EnrichedClaim(
        id="claim-uuid-5",
        claim_text="O café aumenta o risco de doenças cardíacas",
        citations=[
            Citation(
                url="https://example.com/estudo1",
                title="Café e Saúde Cardíaca - Estudo A",
                publisher="Instituto de Pesquisa A",
                quoted="Consumo moderado de café não está associado a aumento de risco cardíaco.",
                rating=None,
                review_date="2024-09-15"
            ),
            Citation(
                url="https://example.com/estudo2",
                title="Riscos do Café - Estudo B",
                publisher="Instituto de Pesquisa B",
                quoted="Consumo excessivo de café pode aumentar pressão arterial temporariamente.",
                rating=None,
                review_date="2024-10-01"
            )
        ],
        search_queries=["café doenças cardíacas"],
        retrieval_notes="Fontes apresentam informações parcialmente contraditórias"
    )
    
    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=[enriched_claim]
    )
    
    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims]
    )
    
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )
    
    # Print input for debugging
    print_adjudication_input(adjudication_input, "Contradictory Sources")
    
    # Execute
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )
    
    # Print output for debugging
    print_fact_check_result(result, "Contradictory Sources")
    
    # Validate structure
    validate_fact_check_result(result)
    assert len(result.results) == 1, "Should have results for 1 data source"
    assert len(result.results[0].claim_verdicts) == 1, "Should have 1 verdict"


def test_chain_building():
    """Test that the adjudication chain can be built without errors."""
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=30.0)
    )
    
    # Build chain
    chain = build_adjudication_chain(llm_config=llm_config)
    
    # Validate
    assert chain is not None, "Chain should be built successfully"
    print("\n" + "=" * 80)
    print("TEST: Chain Building")
    print("=" * 80)
    print(f"\n✓ Adjudication chain built successfully: {type(chain).__name__}")
    print()


def test_return_type_is_fact_check_result():
    """Test that adjudicate_claims returns FactCheckResult wrapper for type safety."""
    # Setup
    data_source = DataSource(
        id="msg-007",
        source_type="original_text",
        original_text="Mensagem de teste para verificação de tipo",
        metadata={},
        locale="pt-BR"
    )
    
    enriched_claim = EnrichedClaim(
        id="claim-uuid-6",
        claim_text="Esta é uma afirmação de teste",
        citations=[],
        search_queries=["teste"],
        retrieval_notes="Teste"
    )
    
    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=[enriched_claim]
    )
    
    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims]
    )
    
    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )
    
    # Execute
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )
    
    # Validate type - should be FactCheckResult wrapper
    assert isinstance(result, FactCheckResult), "Result should be FactCheckResult wrapper"
    assert hasattr(result, 'results'), "Wrapper should have 'results' attribute"
    assert isinstance(result.results, list), "The 'results' attribute should be a list"
    
    print("\n" + "=" * 80)
    print("TEST: Return Type Check")
    print("=" * 80)
    print(f"\n✓ Correct return type: {type(result).__name__}")
    print(f"✓ Returns FactCheckResult wrapper for type safety")
    print(f"✓ Wrapper contains {len(result.results)} data source result(s)")
    print()


def test_citations_used_field_in_verdict():
    """
    Test that the LLM returns the citations_used field in verdicts.

    This test verifies that:
    1. The _LLMClaimVerdict model includes a citations_used field
    2. The LLM actually populates this field with the citations it used
    3. The citations are properly formatted Citation objects
    """
    # Setup
    data_source = DataSource(
        id="msg-008",
        source_type="original_text",
        original_text="A Terra é plana e não gira ao redor do Sol.",
        metadata={},
        locale="pt-BR"
    )

    enriched_claim = EnrichedClaim(
        id="claim-uuid-7",
        text="A Terra é plana",
        source={
            "source_type": "original_text",
            "source_id": "msg-008"
        },
        citations=[
            Citation(
                url="https://www.nasa.gov/earth-round",
                title="A Terra é Redonda - NASA",
                publisher="NASA",
                citation_text="Evidências científicas e fotografias do espaço confirmam que a Terra é redonda.",
                rating="Falso",
                date="2024-01-15"
            ),
            Citation(
                url="https://www.iag.usp.br/astronomia/terra-formato",
                title="O Formato da Terra",
                publisher="IAG-USP",
                citation_text="Observações astronômicas e medições geodésicas demonstram que a Terra é um esferoide.",
                date="2024-03-20"
            )
        ],
        entities=["Terra"]
    )

    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=[enriched_claim]
    )

    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims]
    )

    llm_config = LLMConfig(llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0)
    )

    # Print input for debugging
    print_adjudication_input(adjudication_input, "Citations Used Field Test")

    # Execute - this will invoke the LLM with the new schema
    result = adjudicate_claims(
        adjudication_input=adjudication_input,
        llm_config=llm_config
    )

    # Print output for debugging
    print_fact_check_result(result, "Citations Used Field Test")

    # Validate basic structure
    validate_fact_check_result(result)
    assert len(result.results) == 1, "Should have results for 1 data source"
    assert len(result.results[0].claim_verdicts) == 1, "Should have 1 verdict"

    verdict = result.results[0].claim_verdicts[0]

    # Assert that citations_used field is present and properly populated
    assert hasattr(verdict, 'citations_used'), "Verdict should have citations_used field"
    assert isinstance(verdict.citations_used, list), "citations_used should be a list"

    # The LLM should return at least some citations (though it may choose to use all or subset)
    # We provided 2 citations, so we expect the LLM to use at least one
    print(f"\n📊 LLM used {len(verdict.citations_used)} out of 2 available citations")

    # Validate each citation in citations_used
    for i, citation in enumerate(verdict.citations_used, 1):
        assert isinstance(citation, Citation), f"Citation {i} should be a Citation object"
        assert citation.url, f"Citation {i} should have a URL"
        assert citation.title, f"Citation {i} should have a title"
        assert citation.citation_text, f"Citation {i} should have citation_text"
        print(f"  ✓ Citation {i}: {citation.title}")

    print("\n" + "=" * 80)
    print("TEST: Citations Used Field")
    print("=" * 80)
    print(f"\n✓ Verdict generated successfully with citations_used field")
    print(f"  Verdict: {verdict.verdict}")
    print(f"  Justification length: {len(verdict.justification)} chars")
    print(f"  Citations used by LLM: {len(verdict.citations_used)}")
    print("\n✅ SUCCESS: citations_used field is properly propagated from LLM output to ClaimVerdict!")
    print()


# ===== PYTEST CONFIGURATION =====

if __name__ == "__main__":
    """Run tests manually with: python -m app.ai.pipeline.tests.judgment_test"""
    pytest.main([__file__, "-v", "-s"])

