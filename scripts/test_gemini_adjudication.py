"""
test script for google gemini api with adjudication prompts.

tests the gemini model with the same prompts used in the adjudication step
of the fact-checking pipeline.
"""

import time
from google import genai
from google.genai import types

# adjudication system prompt (from app/ai/pipeline/prompts.py)
ADJUDICATION_SYSTEM_PROMPT = """Você é um especialista em verificação de fatos (fact-checking) para um sistema de checagem de notícias e alegações.

Sua tarefa é analisar alegações extraídas de diferentes fontes de dados e emitir um veredito fundamentado para cada uma, baseando-se estritamente nas evidências e citações fornecidas.

## Categorias de Veredito:

Você deve classificar cada alegação em UMA das seguintes categorias:

1. **Verdadeiro**: A alegação é comprovadamente verdadeira com base nas evidências apresentadas. As fontes são confiáveis e concordam que a alegação é factual.

2. **Falso**: A alegação é comprovadamente falsa com base nas evidências apresentadas. As fontes confiáveis contradizem diretamente a alegação.

3. **Fora de Contexto**: A alegação contém elementos verdadeiros, mas foi apresentada de forma enganosa, omitindo contexto importante, ou misturando fatos verdadeiros com interpretações falsas.

4. **Fontes insuficientes para verificar**: Não há evidências suficientes nas fontes fornecidas para confirmar ou refutar a alegação. As fontes são insuficientes, contraditórias demais, ou a alegação requer informação que não está disponível.

## Diretrizes para Julgamento:

1. **Baseie-se APENAS nas evidências fornecidas**: Use exclusivamente as citações, fontes e contexto apresentados. Não use conhecimento externo.

2. **Avalie a qualidade das fontes**: Considere a confiabilidade do publicador (órgãos governamentais, instituições científicas, veículos de imprensa estabelecidos vs. sites desconhecidos).

3. **Cite suas fontes**: Sempre mencione explicitamente as URLs e trechos das fontes que fundamentam seu veredito. Use aspas para citações diretas.

4. **Seja claro e objetivo**: Explique seu raciocínio de forma concisa mas completa. O usuário precisa entender POR QUE você chegou àquela conclusão.

5. **Identifique contexto faltante**: Se uma alegação é tecnicamente verdadeira mas apresentada de forma enganosa, classifique como "Fora de Contexto" e explique o que está faltando.

6. **Reconheça limitações**: Se as evidências são insuficientes ou contraditórias demais, seja honesto e classifique como "Fontes insuficientes para verificar".

7. **Atenção a datas**: Verifique se as informações nas fontes são recentes e relevantes para a alegação sendo analisada.

## Formato de Resposta:

Para cada fonte de dados (data source), você receberá:
- As informações da fonte (tipo, id, texto original, metadados)
- Uma ou mais alegações extraídas dessa fonte
- Para cada alegação, as citações e evidências coletadas (URLs, títulos, trechos, avaliações prévias)

Você deve retornar um objeto JSON estruturado contendo:
- Para cada fonte de dados, um objeto com:
  - data_source_id: o ID da fonte de dados (você verá no cabeçalho "Source: ... (ID: xxx)")
  - claim_verdicts: lista de vereditos para alegações desta fonte
- Cada veredito contém:
  - claim_id: o ID da alegação (você verá em "Afirmação ID: xxx")
  - claim_text: o texto da alegação (exatamente como foi apresentado)
  - verdict: uma das quatro categorias ("Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar")
  - justification: sua explicação detalhada, citando as fontes

IMPORTANTE:
- Inclua o data_source_id e claim_id quando possível para identificar cada grupo de vereditos
- mantenha os resultados NA MESMA ORDEM das fontes apresentadas

## Exemplo de Justificação:

BOM:
"Segundo o Ministério da Saúde (https://saude.gov.br/estudo-vacinas), um estudo com 50.000 participantes não encontrou evidências ligando a vacina X a problemas de fertilidade. A alegação é contradita por múltiplas fontes científicas confiáveis."

RUIM:
"Esta alegação é falsa." (Falta fundamentação e citação de fontes)

## Importante:

- Seja rigoroso mas justo
- Prefira "Fontes insuficientes para verificar" a fazer suposições
- Contexto importa: "Fora de Contexto" é tão importante quanto "Falso"
- Sempre cite URLs completas nas justificativas
- Mantenha um tom profissional e imparcial"""

# sample claim data based on the logs provided
FORMATTED_SOURCES_AND_CLAIMS = """
================================================================================
NOVA FONTE DE DADOS

Source: link_context (ID: link-46a21fc4)
URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
Link Title: Cúpula dos Povos encerra com propostas para COP30

Afirmações extraídas da fonte e as evidências de cada uma

Afirmação 1:
  Afirmação ID: claim-001
  Texto: A Cúpula dos Povos foi realizada em Belém do Pará entre 12 e 16 de novembro de 2024

  Citações e Evidências (2 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "Realizada em Belém do Pará entre 12 e 16 de novembro de 2024, a Cúpula dos Povos reuniu movimentos sociais, povos indígenas e organizações da sociedade civil."
        Data da revisão: 2024-11-16

    [2] Evento paralelo à COP29 termina em Belém
        Fonte: G1
        URL: https://g1.globo.com/pa/para/noticia/2024/11/16/cupula-dos-povos-belem.ghtml
        Trecho: "O evento aconteceu de 12 a 16 de novembro na capital paraense, como preparação para a COP30 que será realizada em 2025."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 2:
  Afirmação ID: claim-002
  Texto: A Cúpula dos Povos reuniu mais de 70 mil participantes de movimentos sociais, povos indígenas e organizações

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "a Cúpula dos Povos reuniu mais de 70 mil participantes de movimentos sociais, povos indígenas, quilombolas, ribeirinhos e organizações da sociedade civil de todo o Brasil e outros países."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 3:
  Afirmação ID: claim-003
  Texto: Representantes da Cúpula entregaram ao presidente da COP30, o embaixador André Corrêa do Lago, um documento com propostas

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "representantes da Cúpula entregaram ao presidente da COP30, o embaixador André Corrêa do Lago, um documento com propostas e demandas dos povos para a conferência climática."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 4:
  Afirmação ID: claim-004
  Texto: O documento da Cúpula defende a maior proteção a territórios indígenas e comunidades locais

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "O documento defende a maior proteção a territórios indígenas e comunidades locais, reconhecendo seu papel fundamental na preservação ambiental."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 5:
  Afirmação ID: claim-005
  Texto: A Cúpula propõe mecanismos de transição energética justa com foco em direitos humanos

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "A Cúpula propõe ainda mecanismos de transição energética justa, com foco em direitos humanos, soberania alimentar e combate às desigualdades."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 6:
  Afirmação ID: claim-006
  Texto: A COP30 continua seus trabalhos na próxima semana, com a chegada de chefes de Estado e governo

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "Apesar de a Cúpula dos Povos se encerrar neste domingo, a COP30 continua seus trabalhos na próxima semana, com a chegada de chefes de Estado e governo para a fase de negociações de alto nível."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 7:
  Afirmação ID: claim-007
  Texto: Na semana passada, muitas diferenças não foram resolvidas, resultando em mais de 900 trechos entre colchetes no texto da conferência

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "Na semana passada, muitas diferenças não foram resolvidas, de forma que se encerrou com mais de 900 trechos entre colchetes no texto, evidenciando os pontos de discordância entre os países."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------

Afirmação 8:
  Afirmação ID: claim-008
  Texto: Os quatro principais pontos de divergência entre países ricos e pobres são: financiamento climático, proteção comercial, transição energética e responsabilidade histórica

  Citações e Evidências (1 fonte(s)):

    [1] Cúpula dos Povos encerra com propostas para COP30
        Fonte: CNN Brasil
        URL: https://www.cnnbrasil.com.br/politica/cupula-dos-povos-encerra-com-propostas-para-cop30/
        Trecho: "Os quatro principais pontos de divergência entre países ricos e pobres são: 1. financiamento climático; 2. proteção comercial com argumentos ambientais; 3. transição energética; 4. responsabilidade histórica pelas emissões."
        Data da revisão: 2024-11-16

--------------------------------------------------------------------------------
"""

ADDITIONAL_CONTEXT = ""

# adjudication user prompt
USER_PROMPT = f"""Analise as alegações abaixo e forneça um veredito fundamentado para cada uma.

{FORMATTED_SOURCES_AND_CLAIMS}

{ADDITIONAL_CONTEXT}

Para cada alegação, forneça:
1. O veredito (Verdadeiro, Falso, Fora de Contexto, ou Fontes insuficientes para verificar)
2. Uma justificativa detalhada citando as fontes fornecidas

Retorne sua análise como um objeto JSON estruturado conforme especificado."""


def test_gemini_adjudication():
    """test gemini with adjudication prompts and sample claim data."""
    print("=" * 80)
    print("testing google gemini with adjudication prompts")
    print("=" * 80)

    # initialize gemini client
    client = genai.Client()

    # create the content with system instruction and user message
    print("\nsending request to gemini-2.0-flash-exp...")
    print(f"system prompt length: {len(ADJUDICATION_SYSTEM_PROMPT)} chars")
    print(f"user prompt length: {len(USER_PROMPT)} chars")
    print(f"total input length: ~{(len(ADJUDICATION_SYSTEM_PROMPT) + len(USER_PROMPT)) // 4} tokens (estimated)")

    try:
        # start timing
        start_time = time.time()

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=USER_PROMPT,
            config=types.GenerateContentConfig(
                system_instruction=ADJUDICATION_SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )

        # end timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n" + "=" * 80)
        print("gemini response:")
        print("=" * 80)
        print(response.text)
        print("\n" + "=" * 80)

        # print timing information
        print(f"\nLLM call completed in {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")

        # print usage metadata if available
        if hasattr(response, 'usage_metadata'):
            print("\nusage metadata:")
            print(f"  prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"  response tokens: {response.usage_metadata.candidates_token_count}")
            print(f"  total tokens: {response.usage_metadata.total_token_count}")

        return response

    except Exception as e:
        print(f"\nerror calling gemini: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_gemini_adjudication()

    if result:
        print("\n✓ test completed successfully")
    else:
        print("\n✗ test failed")
