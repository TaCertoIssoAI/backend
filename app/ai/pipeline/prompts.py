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
