#!/usr/bin/env python3
"""
Simple test for Gemini 3 with Google Search - bypasses pipeline
Tests direct model invocation
"""

import os
import sys

# Check API key first
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ ERRO: GOOGLE_API_KEY não configurada!")
    print("   Configure com: export GOOGLE_API_KEY='sua-chave'")
    sys.exit(1)

print("=" * 80)
print("🧪 TESTE SIMPLES: Gemini 3 Pro Preview + Google Search")
print("=" * 80)
print()
print(f"✅ GOOGLE_API_KEY configurada: {api_key[:10]}...")
print()

# Import after API key check
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage
from app.llms.gemini import GeminiChatModel
from google.genai import types

print("📋 Criando modelo Gemini 3 Pro Preview com Google Search...")

# Create the tool
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# Create model
model = GeminiChatModel(
    model="gemini-3-pro-preview",
    google_api_key=api_key,
    temperature=0.0,
    tools=[google_search_tool]
)

print(f"   Modelo: {model.model}")
print(f"   Tools: {len(model.tools) if model.tools else 0} tool(s)")
print()

# Create test messages
print("📝 Criando mensagem de teste...")
print("   Claim: 'Federal ganhou o Tusca 2025'")
print("   Contexto: NENHUM (para forçar busca na web)")
print()

system_msg = SystemMessage(content="""Você é um fact-checker expert.

Analise a alegação fornecida e determine se é verdadeira ou falsa.

Você tem acesso a uma ferramenta de busca do Google. Use-a quando necessário para verificar informações.

Se você usar a busca na web, cite as fontes com [WEB_SEARCH] no início.""")

user_msg = HumanMessage(content="""Analise esta alegação:

"Federal ganhou o Tusca 2025"

Forneça:
1. Veredito (Verdadeiro, Falso, ou Não foi possível verificar)
2. Justificativa detalhada com fontes

Tusca é uma competição universitária esportiva tradicional no Brasil.""")

messages = [system_msg, user_msg]

# Invoke model
print("🚀 Invocando Gemini 3 + Google Search...")
print("   (Isso pode levar 10-60 segundos)")
print()
print("-" * 80)

try:
    response = model.invoke(messages)

    print("-" * 80)
    print()
    print("=" * 80)
    print("📊 RESPOSTA DO GEMINI")
    print("=" * 80)
    print()
    print(response.content)
    print()

    # Check for grounding metadata
    if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
        grounding = response.additional_kwargs.get('grounding_metadata')
        if grounding:
            print("=" * 80)
            print("🔍 GROUNDING METADATA (Google Search foi usado!)")
            print("=" * 80)
            print()

            queries = grounding.get('web_search_queries', [])
            if queries:
                print(f"📌 Queries executadas ({len(queries)}):")
                for i, q in enumerate(queries, 1):
                    print(f"   {i}. {q}")
                print()

            chunks = grounding.get('grounding_chunks', [])
            if chunks:
                print(f"📚 Fontes encontradas ({len(chunks)}):")
                for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
                    uri = chunk.get('uri', 'N/A')
                    title = chunk.get('title', 'N/A')
                    print(f"   {i}. {title}")
                    print(f"      {uri}")
                print()

                if len(chunks) > 5:
                    print(f"   ... e mais {len(chunks) - 5} fonte(s)")
                    print()
        else:
            print("ℹ️  Grounding metadata não encontrada")
            print("   (Google Search pode não ter sido usado)")
            print()

    # Check for [WEB_SEARCH] tag
    if "[WEB_SEARCH]" in response.content:
        print("✅ Tag [WEB_SEARCH] encontrada na resposta!")
    else:
        print("ℹ️  Tag [WEB_SEARCH] não encontrada na resposta")

    print()
    print("=" * 80)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

except Exception as e:
    print()
    print("=" * 80)
    print("❌ ERRO")
    print("=" * 80)
    print(f"Erro: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)
