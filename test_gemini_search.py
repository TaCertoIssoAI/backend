#!/usr/bin/env python3
"""
Test script for Gemini 3 with Google Search Grounding
Tests the claim: "Federal ganhou o Tusca 2025"
"""

import os
import sys
from datetime import datetime

# Disable file logging to avoid permission issues
os.environ["DISABLE_FILE_LOGGING"] = "true"

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config.gemini_models import get_gemini_default_pipeline_config
from app.models import (
    AdjudicationInput,
    DataSourceWithClaims,
    DataSource,
    EnrichedClaim,
    Citation,
    SourceType
)
from app.ai.pipeline.judgement import adjudicate_claims


def create_test_claim():
    """
    Create a test claim about "Federal ganhou o Tusca 2025"
    with minimal evidence to force Google Search usage
    """

    # Create a data source (simulated user message)
    data_source = DataSource(
        id="test-source-001",
        source_type=SourceType.TEXT,
        text="Federal ganhou o Tusca 2025",
        metadata={"timestamp": datetime.now().isoformat()}
    )

    # Create an enriched claim with minimal/no evidence
    # This should trigger Google Search because evidence is insufficient
    claim = EnrichedClaim(
        id="claim-001",
        text="A Universidade Federal do Rio Grande do Sul (UFRGS) ganhou o Tusca 2025",
        source="test-source-001",
        entities=["Federal", "UFRGS", "Tusca", "2025"],
        citations=[]  # NO CITATIONS - should trigger web search
    )

    # Create the adjudication input
    source_with_claims = DataSourceWithClaims(
        data_source=data_source,
        enriched_claims=[claim]
    )

    adjudication_input = AdjudicationInput(
        sources_with_claims=[source_with_claims],
        additional_context="Tusca é uma competição universitária esportiva tradicional."
    )

    return adjudication_input


def main():
    """
    Main test function
    """
    print("=" * 80)
    print("🧪 TESTE: Gemini 3 Pro Preview + Google Search Grounding")
    print("=" * 80)
    print()

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERRO: GOOGLE_API_KEY não configurada!")
        print("   Configure com: export GOOGLE_API_KEY='sua-chave'")
        sys.exit(1)

    print(f"✅ GOOGLE_API_KEY configurada: {api_key[:10]}...")
    print()

    # Get configuration
    print("📋 Carregando configuração do Gemini...")
    try:
        config = get_gemini_default_pipeline_config()
        model_name = config.adjudication_llm_config.llm.model
        has_tools = hasattr(config.adjudication_llm_config.llm, 'tools') and config.adjudication_llm_config.llm.tools

        print(f"   Modelo: {model_name}")
        print(f"   Google Search: {'✅ HABILITADO' if has_tools else '❌ NÃO HABILITADO'}")

        if not has_tools:
            print()
            print("⚠️  AVISO: Google Search não está habilitado!")
            print("   O teste vai rodar, mas sem busca na web.")

        print()
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create test input
    print("📝 Criando claim de teste...")
    print("   Claim: 'Federal ganhou o Tusca 2025'")
    print("   Evidências fornecidas: NENHUMA (para forçar Google Search)")
    print()

    adjudication_input = create_test_claim()

    # Run adjudication
    print("🚀 Iniciando julgamento com Gemini 3 + Google Search...")
    print("   (Isso pode levar 10-30 segundos se o Google Search for usado)")
    print()
    print("-" * 80)

    try:
        result = adjudicate_claims(
            adjudication_input=adjudication_input,
            llm_config=config.adjudication_llm_config
        )

        print("-" * 80)
        print()
        print("=" * 80)
        print("📊 RESULTADO DO JULGAMENTO")
        print("=" * 80)
        print()

        # Display results
        for source_result in result.results:
            print(f"🔍 Data Source ID: {source_result.data_source_id}")
            print(f"📂 Source Type: {source_result.source_type}")
            print()

            for verdict in source_result.claim_verdicts:
                print(f"📌 Claim: {verdict.claim_text}")
                print(f"⚖️  Veredito: {verdict.verdict}")
                print()
                print(f"📄 Justificativa:")
                print(f"{verdict.justification}")
                print()

                # Check for [WEB_SEARCH] tag
                if "[WEB_SEARCH]" in verdict.justification:
                    print("✅ Google Search FOI USADO! (tag [WEB_SEARCH] encontrada)")
                else:
                    print("ℹ️  Não foi possível confirmar uso do Google Search pela justificativa")

                print()

        if result.overall_summary:
            print("=" * 80)
            print("📋 RESUMO GERAL")
            print("=" * 80)
            print(result.overall_summary)
            print()

        print("=" * 80)
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERRO NO JULGAMENTO")
        print("=" * 80)
        print(f"Erro: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
