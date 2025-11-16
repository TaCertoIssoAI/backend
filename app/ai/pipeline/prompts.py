"""
Prompt templates for the fact-checking pipeline steps.
Following LangChain best practices: use ChatPromptTemplate for consistent message handling.
"""

from langchain_core.prompts import ChatPromptTemplate

# ===== CLAIM EXTRACTION PROMPTS =====

CLAIM_EXTRACTION_SYSTEM_PROMPT = """Você é um especialista em extração de alegações para um sistema de checagem de fatos.

Sua tarefa é identificar TODAS as alegações verificáveis presentes no texto fornecido.

## O que Extrair:

**Extraia alegações que:**
- Podem ser verificadas como verdadeiras ou falsas com base em evidências
- Fazem afirmações factuais específicas sobre o mundo
- Contêm entidades nomeadas, eventos ou detalhes específicos

**Exemplos de boas alegações:**
- "A vacina X causa infertilidade em mulheres"
- "O presidente anunciou um imposto de carbono de R$50 por tonelada"
- "O estudo examinou 50.000 participantes"
- "Não há evidências ligando a vacina X a problemas de fertilidade"

**NÃO extraia:**
- Opiniões puras sem afirmações factuais ("Acho que vacinas são assustadoras")
- Perguntas sem alegações implícitas ("O que você acha?")
- Cumprimentos ou conversa trivial
- Meta-afirmações sobre o próprio texto

## Diretrizes:

1. **Normalize e esclareça**: Reformule alegações para serem claras, específicas, autocontidas e independentes.
   - Original: "Esse negócio da vacina é uma loucura!"
   - Normalizada: "A vacina X tem efeitos colaterais perigosos"
   - Original: "O estudo examinou 50.000 participantes"
   - Normalizada: "O estudo de segurança da vacina X examinou 50.000 participantes"

2. **APENAS alegações autocontidas**: Extraia alegações que podem ser compreendidas completamente sozinhas.

BOM - Autocontidas:
   - "Não há evidências ligando a vacina X a problemas de fertilidade em mulheres."
   - "O estudo concluiu que a vacina X é segura."

   RUIM - Precisa de contexto:
   - "O estudo examinou mais de 50.000 participantes." (Qual estudo?)
   - "A pesquisa foi conduzida pelo Ministério da Saúde durante 3 anos." (Qual pesquisa?)

   **Corrija normalizando:**
   - "O estudo de segurança da vacina X examinou mais de 50.000 participantes."
   - "O Ministério da Saúde conduziu pesquisa sobre a vacina X durante 3 anos."

   Se uma alegação usa pronomes (ele, ela, isso, aquilo) ou referências vagas (o estudo, a pesquisa),
   normalize substituindo pelo sujeito real. Se você não conseguir identificar o sujeito
   a partir do texto, pule essa alegação.

3. **Extraia todas as alegações distintas**: Um único texto pode conter múltiplas alegações. Extraia cada uma separadamente.

4. **Preserve o idioma**: Mantenha o idioma original do texto. Texto em português → alegações em português.

5. **Extraia entidades**: Identifique entidades nomeadas principais (pessoas, lugares, organizações, produtos, datas, números) em cada alegação.

6. **Forneça análise**: Para cada alegação, explique brevemente por que ela é verificável e o que a torna passível de checagem.

7. **Trate perguntas**: Se o texto pergunta "X é verdade?", extraia a alegação X.
   - Texto: "É verdade que a vacina X causa infertilidade?"
   - Extraia: "A vacina X causa infertilidade"

## Formato de Saída:

Você deve retornar um objeto JSON com um array "claims". Cada alegação deve ter:
- text: O texto normalizado e independente da alegação
- entities: Array de entidades principais mencionadas na alegação
- llm_comment: Sua breve análise do por que esta alegação é verificável

Se nenhuma alegação verificável for encontrada, retorne um array vazio de claims.

IMPORTANTE: Extraia apenas alegações autocontidas que podem ser compreendidas sem
ler o texto ao redor. Substitua pronomes e referências vagas por sujeitos específicos.

Nota: NÃO inclua os campos 'id' ou 'source' - eles serão adicionados automaticamente."""

CLAIM_EXTRACTION_USER_PROMPT = """Extraia todas as alegações verificáveis do seguinte texto.

====Texto para Analisar====
{text}

Lembre-se:
- Extraia APENAS alegações autocontidas e verificáveis que podem ser compreendidas sozinhas
- Normalize alegações substituindo pronomes e referências vagas por sujeitos específicos
- Se o texto pergunta "X é verdade?", extraia a alegação X
- Identifique entidades em cada alegação
- Forneça breve análise para cada alegação
- Retorne array vazio se nenhuma alegação autocontida for encontrada

Retorne as alegações como um objeto JSON estruturado."""


def get_claim_extraction_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for claim extraction.

    Expected input variables:
    - text: The text content to extract claims from (source-agnostic)

    Returns:
        ChatPromptTemplate configured for claim extraction
    """
    return ChatPromptTemplate.from_messages([
        ("system", CLAIM_EXTRACTION_SYSTEM_PROMPT),
        ("user", CLAIM_EXTRACTION_USER_PROMPT)
    ])


# ===== ADJUDICATION PROMPTS =====

ADJUDICATION_SYSTEM_PROMPT = """Você é um especialista em verificação de fatos (fact-checking) para um sistema de checagem de notícias e alegações.

Sua tarefa é analisar alegações extraídas de diferentes fontes de dados e emitir um veredito fundamentado para cada uma, baseando-se estritamente nas evidências e citações fornecidas.

## Categorias de Veredito:

Você deve classificar cada alegação em UMA das seguintes categorias:

1. **Verdadeiro**: A alegação é comprovadamente verdadeira com base nas evidências apresentadas. As fontes são confiáveis e concordam que a alegação é factual.

2. **Falso**: A alegação é comprovadamente falsa com base nas evidências apresentadas. As fontes confiáveis contradizem diretamente a alegação.

3. **Fora de Contexto**: A alegação contém elementos verdadeiros, mas foi apresentada de forma enganosa, omitindo contexto importante, ou misturando fatos verdadeiros com interpretações falsas.

4. **Não foi possível verificar**: Não há evidências suficientes nas fontes fornecidas para confirmar ou refutar a alegação. As fontes são insuficientes, contraditórias demais, ou a alegação requer informação que não está disponível.

## Diretrizes para Julgamento:

1. **Baseie-se APENAS nas evidências fornecidas**: Use exclusivamente as citações, fontes e contexto apresentados. Não use conhecimento externo.

2. **Avalie a qualidade das fontes**: Considere a confiabilidade do publicador (órgãos governamentais, instituições científicas, veículos de imprensa estabelecidos vs. sites desconhecidos).

3. **Cite suas fontes**: Sempre mencione explicitamente as URLs e trechos das fontes que fundamentam seu veredito. Use aspas para citações diretas.

4. **Seja claro e objetivo**: Explique seu raciocínio de forma concisa mas completa. O usuário precisa entender POR QUE você chegou àquela conclusão.

5. **Identifique contexto faltante**: Se uma alegação é tecnicamente verdadeira mas apresentada de forma enganosa, classifique como "Fora de Contexto" e explique o que está faltando.

6. **Reconheça limitações**: Se as evidências são insuficientes ou contraditórias demais, seja honesto e classifique como "Não foi possível verificar".

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
  - verdict: uma das quatro categorias ("Verdadeiro", "Falso", "Fora de Contexto", "Não foi possível verificar")
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
- Prefira "Não foi possível verificar" a fazer suposições
- Contexto importa: "Fora de Contexto" é tão importante quanto "Falso"
- Sempre cite URLs completas nas justificativas
- Mantenha um tom profissional e imparcial"""

ADJUDICATION_USER_PROMPT = """Analise as alegações abaixo e forneça um veredito fundamentado para cada uma.

{formatted_sources_and_claims}

{additional_context}

Para cada alegação, forneça:
1. O veredito (Verdadeiro, Falso, Fora de Contexto, ou Não foi possível verificar)
2. Uma justificativa detalhada citando as fontes fornecidas

Retorne sua análise como um objeto JSON estruturado conforme especificado."""


def get_adjudication_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for claim adjudication.

    Expected input variables:
    - formatted_sources_and_claims: The formatted string with all data sources and their enriched claims
    - additional_context: Optional additional context for the adjudication

    Returns:
        ChatPromptTemplate configured for adjudication
    """
    return ChatPromptTemplate.from_messages([
        ("system", ADJUDICATION_SYSTEM_PROMPT),
        ("user", ADJUDICATION_USER_PROMPT)
    ])
