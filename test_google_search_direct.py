#!/usr/bin/env python3
"""
Direct test of Google Gemini 3 with Google Search Grounding
No dependencies on the project's code - pure Google API test
"""

import os
import json
from datetime import datetime
from google import genai
from google.genai import types

# Check API key
api_key = "AIzaSyB4qu5TqTU6bAta-391QFWfZGr__rqUunU"  # Hardcoded for testing

print("=" * 80)
print("🧪 TESTE DIRETO: Google Gemini 3 Pro + Google Search")
print("=" * 80)
print()
print(f"✅ API Key: {api_key[:10]}...")
print()

# Create client
print("🔧 Criando cliente Google AI...")
client = genai.Client(api_key=api_key)
print("✅ Cliente criado")
print()

# Create Google Search tool
print("🔍 Criando Google Search tool...")
search_tool = types.Tool(google_search=types.GoogleSearch())
print("✅ Tool criada")
print()

# Create generation config with Google Search
print("⚙️  Configurando modelo...")
config = types.GenerateContentConfig(
    temperature=0.0,
    tools=[search_tool],
    system_instruction="""Você é um fact-checker expert brasileiro.

Analise alegações e determine se são verdadeiras ou falsas.

Você tem acesso a busca no Google. Use-a quando as informações fornecidas forem insuficientes.

Se você usar busca na web, cite as fontes com [WEB_SEARCH] no início da citação."""
)

model_name = "gemini-2.0-flash-exp"  # ou "gemini-1.5-pro"
print(f"✅ Modelo: {model_name}")
print(f"✅ Google Search: HABILITADO")
print()

# Test message
test_claim = "Federal ganhou o Tusca 2025"

prompt = f"""Analise esta alegação e me diga se é verdadeira ou falsa:

"{test_claim}"

Contexto: O Tusca é uma competição universitária esportiva tradicional em São Paulo, Brasil.

Forneça:
1. **Veredito**: Verdadeiro, Falso, ou Não foi possível verificar
2. **Justificativa**: Explique seu raciocínio citando fontes

Se você precisar buscar informações na web para verificar, use a ferramenta de busca."""

print("📝 Pergunta:")
print(f"   '{test_claim}'")
print()
print("🚀 Enviando para o Gemini...")
print("   (Aguarde 10-60 segundos - busca na web pode demorar)")
print()
print("-" * 80)

try:
    # Generate response
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config
    )

    print("-" * 80)
    print()
    print("=" * 80)
    print("📊 RESPOSTA DO GEMINI")
    print("=" * 80)
    print()
    print(response.text)
    print()

    # Check grounding metadata - DETAILED OUTPUT
    print("=" * 80)
    print("🔍 GROUNDING METADATA (PROVA DE QUE PESQUISOU NA WEB)")
    print("=" * 80)
    print()

    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]

        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            metadata = candidate.grounding_metadata

            print("✅ GOOGLE SEARCH FOI USADO! Aqui está a prova:")
            print()

            # Web search queries
            if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                queries = metadata.web_search_queries
                print(f"📌 QUERIES EXECUTADAS PELO GEMINI ({len(queries)}):")
                for i, query in enumerate(queries, 1):
                    print(f"   {i}. \"{query}\"")
                print()

            # Grounding chunks (sources) - SHOW ALL
            if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                chunks = metadata.grounding_chunks
                print(f"📚 FONTES DA WEB ENCONTRADAS ({len(chunks)} URLs):")
                print()
                for i, chunk in enumerate(chunks, 1):
                    if hasattr(chunk, 'web') and chunk.web:
                        print(f"   [{i}]")
                        print(f"   Título: {chunk.web.title}")
                        print(f"   URL: {chunk.web.uri}")
                        print()
                print()

            # Grounding supports - DETAILED
            if hasattr(metadata, 'grounding_supports') and metadata.grounding_supports:
                supports = metadata.grounding_supports
                print(f"🔗 TRECHOS DA RESPOSTA FUNDAMENTADOS EM FONTES WEB ({len(supports)} trechos):")
                print()
                for i, support in enumerate(supports, 1):
                    if hasattr(support, 'segment') and support.segment:
                        print(f"   [{i}] Trecho da resposta:")
                        print(f"       \"{support.segment.text}\"")

                    if hasattr(support, 'grounding_chunk_indices') and support.grounding_chunk_indices:
                        indices = support.grounding_chunk_indices
                        print(f"       Baseado nas fontes: {list(indices)}")
                    print()

            # Show raw metadata as JSON for debugging
            print("=" * 80)
            print("📋 METADATA COMPLETA (JSON):")
            print("=" * 80)
            import json
            try:
                # Try to convert to dict
                metadata_dict = {
                    'web_search_queries': list(metadata.web_search_queries) if hasattr(metadata, 'web_search_queries') else [],
                    'grounding_chunks_count': len(metadata.grounding_chunks) if hasattr(metadata, 'grounding_chunks') else 0,
                    'grounding_supports_count': len(metadata.grounding_supports) if hasattr(metadata, 'grounding_supports') else 0,
                }
                print(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
            except:
                print(metadata)
            print()

        else:
            print("❌ GROUNDING METADATA NÃO ENCONTRADA!")
            print("   Isso significa que o Gemini NÃO usou Google Search")
            print("   Possíveis razões:")
            print("   - A tool não está configurada corretamente")
            print("   - O Gemini decidiu não usar (informação já conhecida)")
            print("   - Erro na API")
            print()
    else:
        print("❌ Nenhum candidate na resposta!")
        print()

    # Check for [WEB_SEARCH] tag
    if "[WEB_SEARCH]" in response.text:
        print("✅ Tag [WEB_SEARCH] encontrada na resposta!")
    else:
        print("ℹ️  Tag [WEB_SEARCH] não encontrada")

    # Save complete result to JSON file
    print("=" * 80)
    print("💾 SALVANDO RESULTADO EM JSON")
    print("=" * 80)
    print()

    # Build complete result object
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "test_claim": test_claim,
        "model": model_name,
        "google_search_enabled": True,
        "response": {
            "text": response.text,
            "verdict": "Não foi possível verificar"  # Extract from response
        },
        "grounding_metadata": None
    }

    # Add grounding metadata if available
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            metadata = candidate.grounding_metadata

            # Extract all metadata
            grounding_data = {
                "web_search_queries": [],
                "grounding_chunks": [],
                "grounding_supports": []
            }

            # Queries
            if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                grounding_data["web_search_queries"] = list(metadata.web_search_queries)

            # Chunks (sources)
            if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        grounding_data["grounding_chunks"].append({
                            "title": chunk.web.title,
                            "uri": chunk.web.uri
                        })

            # Supports (which parts of response came from which sources)
            if hasattr(metadata, 'grounding_supports') and metadata.grounding_supports:
                for support in metadata.grounding_supports:
                    support_data = {}
                    if hasattr(support, 'segment') and support.segment:
                        support_data["segment_text"] = support.segment.text
                    if hasattr(support, 'grounding_chunk_indices') and support.grounding_chunk_indices:
                        support_data["source_indices"] = list(support.grounding_chunk_indices)
                    grounding_data["grounding_supports"].append(support_data)

            result_data["grounding_metadata"] = grounding_data

    # Save to file
    output_file = "gemini_test_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Resultado salvo em: {output_file}")
    print()
    print("📋 Conteúdo do arquivo:")
    print(json.dumps(result_data, ensure_ascii=False, indent=2))
    print()
    print("=" * 80)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 80)

except Exception as e:
    print()
    print("=" * 80)
    print("❌ ERRO")
    print("=" * 80)
    print(f"{type(e).__name__}: {e}")
    print()
    import traceback
    traceback.print_exc()
    exit(1)
