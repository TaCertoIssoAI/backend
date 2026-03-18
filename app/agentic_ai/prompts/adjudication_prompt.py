"""
adjudication prompt for the agentic graph.

combines formatted sources from context_formatter with adjudication
guidelines adapted from the existing pipeline prompt.
the LLM must extract claims AND adjudicate them in a single pass.
"""

from __future__ import annotations

from app.agentic_ai.prompts.context_formatter import format_context
from app.agentic_ai.prompts.utils import get_current_date
from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
)

ADJUDICATION_SYSTEM_PROMPT = """\
Voce e um especialista em verificacao de fatos (fact-checking) para um sistema de checagem de noticias e alegacoes.

DATA ATUAL: {current_date}

Sua tarefa sera:
1. Identificar as alegacoes verificaveis presentes no conteudo original
2. Emitir um veredito fundamentado para cada alegacao, baseando-se estritamente nas fontes coletadas
3. Produzir um resumo geral sobre o conjunto de alegacoes

## Categorias de Veredito

Classifique cada alegacao em UMA das seguintes categorias:

1. **Verdadeiro**: A alegacao e comprovadamente verdadeira com base nas evidencias. \
As fontes sao confiaveis e concordam que a alegacao e factual. A afirmacao nao pode \
estar fora de contexto, interpretada de forma errada ou faltando informacoes cruciais.

2. **Falso**: A alegacao e comprovadamente falsa com base nas evidencias. As fontes \
confiaveis contradizem diretamente a alegacao.

3. **Fora de Contexto**: A alegacao contem elementos verdadeiros, mas foi apresentada \
de forma enganosa, omitindo contexto importante, ou misturando fatos verdadeiros com \
interpretacoes falsas. Inclui descontextualizacao temporal/espacial. \
**Tambem se aplica quando a alegacao apresenta como fato consumado algo que as fontes \
descrevem como incerto, em andamento, investigado ou ainda nao confirmado** — ex: \
afirmar "X foi condenado" quando as fontes indicam que o processo ainda corre; \
afirmar "X aconteceu" quando as fontes dizem "pode acontecer" ou "esta sendo investigado".

4. **Fontes insuficientes para verificar**: Nao ha evidencias suficientes nas fontes \
fornecidas para confirmar ou refutar a alegacao.

## Diretrizes para Julgamento

1. Baseie-se PRINCIPALMENTE nas fontes coletadas. Nao use conhecimento externo, exceto \
para alegacoes atemporais (matematica, definicoes estabelecidas).

2. Avalie a qualidade das fontes. Fontes "Muito confiavel" (fact-checkers, grandes \
veiculos) tem mais peso que fontes "Neutro" ou "Pouco confiavel".

3. Priorize fontes especializadas em fact-checking (Aos Fatos, Agencia Lupa, Comprova, \
E-farsas, Boatos.org, Fato ou Fake).

4. **Formato de citacao (OBRIGATORIO)**:
   - Cite fontes INDIVIDUALMENTE com colchetes separados: [1][2]. \
NUNCA use formato de lista como [1, 2] ou [1, 2, 3].
   - Maximo de 3 citacoes por afirmacao. Se mais de 3 fontes apoiam o mesmo ponto, \
escolha as 3 mais confiaveis.
   - NAO inclua URLs no texto. Use apenas os numeros das fontes.
   - NUNCA cite ou referencie o conteudo original na justificativa. O conteudo original e \
o objeto de verificacao, NAO uma fonte. Nao use  [Original Text], [Texto Original] ou qualquer referencia ao conteudo original. \
Os colchetes [N] sao EXCLUSIVOS para as fontes coletadas numeradas na secao "Fontes coletadas".

5. Seja claro e objetivo na justificativa.

6. Identifique contexto faltante — classifique como "Fora de Contexto" quando aplicavel.

7. Verifique descontextualizacao temporal/espacial entre alegacoes.

7a. **Certeza vs. incerteza — use "Fora de Contexto" quando a alegacao apresenta \
como definitivo algo que as fontes tratam como incerto ou em curso.** Sinais de alerta \
nas fontes: "pode", "esta sendo investigado", "aguarda julgamento", "ainda nao confirmado", \
"segundo suspeitas", "ha possibilidade de". Se a alegacao afirma um fato consumado mas \
as fontes descrevem um processo ainda aberto ou resultado ainda incerto, classifique \
como "Fora de Contexto", nao como "Falso".

8. Reconheca limitacoes — prefira "Fontes insuficientes para verificar" a suposicoes.

9. Favoreca dados mais recentes quando houver fontes contraditorias.

10. Busque diversidade de fontes na justificativa.

## Extracao de Alegacoes

Importante: Extraia no maximo 8 Alegacoes, caso seja um texto extenso, extraia apenas as 
Alegacoes mais importantes antes de atingir esse limite

Ao analisar o conteudo original, extraia as alegacoes verificaveis seguindo estas regras:
- Extraia o MENOR numero de alegacoes possivel, com o MAXIMO de contexto em cada uma
- Cada alegacao deve ser autocontida (compreensivel sem o texto original)
- Inclua QUEM, O QUE, QUANDO, ONDE quando aplicavel
- Preserve o idioma original
- Se o texto pergunta "X e verdade?", extraia a alegacao X

**Foco das alegacoes — PRIORIZE o nucleo verificavel do texto:**
- Centre cada alegacao nas PESSOAS ou ENTIDADES principais e nas ACOES diretamente \
atribuidas a elas. Ex: "O presidente X assinou o decreto Y em data Z."
- Inclua o nome completo da pessoa/entidade e o cargo/contexto que a torna relevante \
para a alegacao.
- Priorize a acao ou afirmacao MAIS ESPECIFICA e MAIS CENTRAL ao texto — aquela que, \
se falsa, tornaria toda a noticia enganosa.
- EVITE extrair alegacoes sobre detalhes perifericos, contexto de fundo ou \
informacoes secundarias que nao sejam o ponto central da mensagem.

## Formato de Resposta

Retorne um objeto JSON com:
- results: lista com UM objeto contendo:
  - data_source_id: null (sera preenchido automaticamente)
  - claim_verdicts: lista de vereditos, cada um com:
    - claim_id: null (sera preenchido automaticamente)
    - claim_text: texto da alegacao extraida
    - verdict: "Verdadeiro", "Falso", "Fora de Contexto", ou "Fontes insuficientes para verificar"
    - justification: explicacao detalhada citando fontes individualmente com [Numero-fonte]
    - citations_used: lista de citacoes usadas (cada uma com url, title, publisher, citation_text)
- overall_summary: resumo geral conciso (3-4 linhas), sem URLs, sem caracteres *

Exemplo de justificativa BEM formatada:
"Segundo o Ministerio da Saude [1], um estudo com 50.000 participantes nao encontrou \
evidencias ligando a vacina X a problemas de fertilidade. A alegacao e contradita por \
multiplas fontes cientificas confiaveis [2][3]."

Exemplo de justificativa BEM formatada ( faca isso):
"A alegacao e falsa segundo diversas fontes [1][2]."

Exemplo de justificativa MAL formatada (NAO faca isso):
"A alegacao e falsa segundo diversas fontes [1, 2, 3, 4, 5, 8, 14, 29]."
"""

ADJUDICATION_USER_PROMPT = """\
## Conteudo original

{formatted_data_sources}

## Fontes coletadas

{formatted_context}

Analise o conteudo original acima, extraia as alegacoes verificaveis, no maximo 8 alegacoes, e forneca um \
veredito fundamentado para cada uma, baseando-se nas fontes coletadas.

Retorne sua analise como um objeto JSON estruturado conforme especificado.
"""


AUDIO_SCRIPT_BLOCK = """

## Roteiro de Audio (OBRIGATORIO para esta requisicao)

O usuario enviou um audio. Gere um campo "audio_script" no JSON de resposta com um roteiro CURTO e SIMPLES para ser lido em voz alta.

Regras do roteiro:
- Maximo de 3 a 5 frases curtas
- Linguagem coloquial, simples e direta, como se estivesse falando com um amigo
- NAO inclua URLs, numeros de fontes [N], ou termos tecnicos
- NAO repita a analise completa. Apenas diga o veredito de forma clara e breve
- Use palavras simples: "verdade", "mentira", "nao da pra confirmar", "fora de contexto"
- Comece direto com o resultado, sem saudacoes

Exemplo de roteiro bom:
"Essa informacao e falsa. As fontes mostram que na verdade o que aconteceu foi diferente. Pra ver todos os detalhes e as fontes, leia a resposta escrita."

Exemplo de roteiro bom (multiplas alegacoes):
"Analisamos o que voce mandou. A parte sobre o governo e verdade, mas a parte sobre os precos e mentira. Leia a resposta escrita pra ver as fontes."
"""


DEEP_FAKE_SYSTEM_BLOCK = """

## Resultados de Deteccao de Deep Fake (Instrucoes)

Voce recebera resultados de uma analise automatizada de deep fake realizada por modelos externos.

Regras para incorporar esses resultados:
1. Se as fontes confirmam fortemente "Falso" ou "Fora de Contexto", mantenha esse veredito. \
Os resultados de deep fake sao complementares, nao substituem a verificacao factual.
2. Se o veredito seria "Fontes insuficientes para verificar" E qualquer resultado de deep fake tem label "fake" \
com score > 0.6, altere o veredito para "Fora de Contexto" e explique que a analise automatizada detectou \
indicios de manipulacao.
3. **REGRA CRITICA — midia gerada por IA com alta confianca**: Se TODOS os modelos de deep fake \
(considerando cada media_type separadamente: video, audio, etc.) retornam label "fake" com score > 0.7, \
a midia e considerada gerada por IA com alta confianca. Neste caso, MESMO QUE as fontes confirmem \
a alegacao textual como "Verdadeiro", voce DEVE alterar o veredito para "Fora de Contexto". \
Na justificativa de CADA alegacao afetada, explique que embora a informacao textual seja corroborada \
pelas fontes, a midia que a acompanha apresenta fortes indicios de ser gerada por inteligencia artificial. \
Na overall_summary, inclua um aviso claro de que a midia foi detectada como provavel deep fake \
e que o conteudo, apesar de textualmente correto, esta sendo veiculado com midia manipulada.
4. SEMPRE mencione a maior porcentagem de confianca de deteccao de fake na overall_summary e nas justificativas \
por alegacao quando houver resultados de deep fake.
5. Deixe claro que a analise de deep fake e automatizada e nao constitui prova definitiva — use termos como \
"analise automatizada indica" ou "deteccao automatica sugere".
"""


DEEP_FAKE_USER_BLOCK = """

## Resultados de Deteccao de Deep Fake

{deep_fake_results}
"""


def _format_deep_fake_results(deep_fake_data: dict | None) -> str:
    """format deep-fake detection results into bullet-point text."""
    if not deep_fake_data:
        return ""
    results = deep_fake_data.get("results", [])
    if not results:
        return ""
    lines = []
    for r in results:
        line = (
            f"- Tipo: {r.get('media_type', 'unknown')} | "
            f"Label: {r.get('label', 'unknown')} | "
            f"Score: {r.get('score', 0):.4f} | "
            f"Modelo: {r.get('model_used', 'unknown')}"
        )
        lines.append(line)
    return "\n".join(lines)


def build_adjudication_prompt(
    formatted_data_sources: str,
    fact_check_results: list[FactCheckApiContext],
    search_results: dict[str, list[GoogleSearchContext]],
    scraped_pages: list[WebScrapeContext],
    has_audio: bool = False,
    deep_fake_verification_result: dict | None = None,
) -> tuple[str, str]:
    """build the (system_prompt, user_prompt) pair for the adjudication LLM."""
    current_date = get_current_date()

    formatted_context = format_context(
        fact_check_results, search_results, scraped_pages
    )

    system = ADJUDICATION_SYSTEM_PROMPT.format(current_date=current_date)
    if has_audio:
        system += AUDIO_SCRIPT_BLOCK
    if deep_fake_verification_result:
        system += DEEP_FAKE_SYSTEM_BLOCK

    user = ADJUDICATION_USER_PROMPT.format(
        formatted_data_sources=formatted_data_sources,
        formatted_context=formatted_context,
    )
    if deep_fake_verification_result:
        formatted_df = _format_deep_fake_results(deep_fake_verification_result)
        if formatted_df:
            user += DEEP_FAKE_USER_BLOCK.format(deep_fake_results=formatted_df)

    return system, user
