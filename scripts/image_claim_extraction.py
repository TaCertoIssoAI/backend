# -*- coding: utf-8 -*-
"""
test script for comparing two approaches to image claim extraction:
1. direct claim extraction with image input
2. image transcription followed by text-based claim extraction
"""

import os
import base64
import json
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# import claim extractor
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.ai.pipeline.claim_extractor import extract_claims
from app.ai.pipeline.prompts import IMAGE_CLAIM_EXTRACTION_SYSTEM_PROMPT, IMAGE_CLAIM_EXTRACTION_USER_PROMPT
from app.models import ClaimExtractionInput, DataSource, LLMConfig


# ===== IMAGE TRANSCRIPTION PROMPT =====

IMAGE_TRANSCRIPTION_PROMPT = """Você receberá uma imagem enviada pelo usuário, seu objetivo é transcrever a imagem enviada para o fact-checking de fake news seguindo as tarefas adiantes:

TAREFA 1: Você deve transcrever todo o texto de uma imagem, focando não apenas no texto mas em como ele está visualmente disposto (letras grandes, pequenas, CAPS LOCK, negrito, itálico, cores). Ex: A imagem tem um título "Político perdeu tudo" em negrito e CAPS LOCK com letras grandes.

TAREFA 2: Foque em transcrever elementos visuais/não-textuais da imagem de forma a explicitar pessoas, especialmente figuras famosas, históricas, importantes, políticos ou celebridades, caso essas figuras estejam presentes, apenas mencione o NOME delas, não qualquer status dela como sua posição, emprego, se está vivo ou não. Também busque descrever entidades humanas e não humanas centrais à imagem.

Não dê tanto importância a detalhes cotidianos e comuns da paisagem, apenas em detalhes anormais que possam auxiliar no processo de fact-checking.

Exemplos de descrições detalhadas que ajudam no fact-checking de fake news:

"A imagem mostra o político Abraham Lincoln numa pose constrangedora, sendo zombado por uma multidão"

Exemplo de uma descrição que não ajuda no fact-checking:

"A imagem mostra um homem de terno e cabelo branco, numa festa, com convidados de smoking."

Retornar no seguinte formato:

"Descrição da imagem: [sua descrição detalhada aqui]"""


# ===== DIRECT IMAGE CLAIM EXTRACTION PROMPT =====

IMAGE_VISION_CLAIM_EXTRACTION_SYSTEM_PROMPT = """Você é um especialista em extração de alegações para um sistema de checagem de fatos.

IMPORTANTE: Você receberá uma IMAGEM como input visual. Sua tarefa é analisar a imagem de forma holística e identificar TODAS as alegações verificáveis presentes nela.

## Como Analisar a Imagem:

**PASSO 1 - Identifique Pessoas e Personagens na imagem:**
- Procure por pessoas famosas, políticos, celebridades, figuras históricas ou personagens conhecidos (incluindo personagens fictícios de filmes, séries, livros, etc.) presentes na imagem
- Identifique-os pelo NOME sempre que possível
- NÃO mencione status, cargo, posição ou se estão vivos/mortos - apenas o NOME

**PASSO 2 - Leia Todo o Texto:**
- Leia todo o texto presente na imagem
- Preste atenção em títulos, legendas, manchetes, citações
- Note a formatação visual: texto em CAPS LOCK, negrito, tamanhos diferentes, cores destacadas

**PASSO 3 - Interpretação Holística:**
- NÃO analise apenas o texto OU apenas a imagem separadamente
- CONECTE os elementos visuais + texto + pessoas/personagens identificados
- Interprete a MENSAGEM COMPLETA que a mídia está comunicando
- Considere o contexto: é uma notícia? Um meme? Uma charge? Uma montagem?
- Identifique a NARRATIVA ou ALEGAÇÃO que a combinação de todos esses elementos está fazendo sobre o mundo real

## O que Extrair:

**Extraia alegações que:**
- Podem ser verificadas como verdadeiras ou falsas com base em evidências
- Fazem afirmações sobre o mundo real (fatos, eventos, pessoas, sociedade)
- Resultam da interpretação HOLÍSTICA da imagem (visual + texto + contexto)
- Contêm entidades nomeadas, eventos ou detalhes específicos
- São opiniões que contêm alegações verificáveis sobre fatos do mundo
- São perguntas que contêm alegações implícitas verificáveis

**Exemplos de boas alegações extraídas holisticamente (com NOMES explícitos):**
- Imagem mostra Joãozinho com texto "Roubou milhões": extrair "Joãozinho roubou milhões"
- Foto de Michael Jackson com legenda "Morreu ontem": extrair "Michael Jackson morreu ontem"
- Meme com Homer Simpson dizendo "Vacinas causam autismo": extrair "Vacinas causam autismo" (atribuindo ao contexto do meme, não ao personagem)
- Charge mostrando Pedro esmagando trabalhadores com legenda sobre reforma trabalhista: extrair "Pedro implementou políticas trabalhistas prejudiciais aos trabalhadores"

**Exemplos ERRADOS (sem nomes explícitos):**
- ❌ "O político roubou milhões" (falta o nome - deve ser "Joãozinho roubou milhões")
- ❌ "A celebridade morreu ontem" (falta o nome - deve ser "Michael Jackson morreu ontem")
- ❌ "Um presidente fez declaração polêmica" (falta o nome específico)

**NÃO extraia:**
- Descrições puramente visuais sem alegação factual ("A imagem mostra uma pessoa")
- Perguntas sem alegações implícitas ("O que você acha?")
- Cumprimentos ou conversa trivial
- Elementos visuais sem conexão com alegações do mundo real
- Alegações genéricas sobre "um político" ou "uma celebridade" quando você consegue identificar a pessoa

## Casos Especiais - Memes, Charges e Montagens:

Quando a imagem for um meme, charge, quadrinho ou montagem:

1. **Identifique a mensagem central**: O que a imagem está AFIRMANDO sobre o mundo?
2. **Use pistas contextuais**:
   - Texto sobreposto ou legendas
   - Personagens ou pessoas identificáveis
   - Símbolos ou metáforas visuais
   - Referências a eventos conhecidos
3. **Extraia a alegação factual implícita**: Se a charge "critica a corrupção do governo X", extraia "O governo X é corrupto"
4. **Detecte manipulação visual**: Se a imagem sugere ou afirma ser real mas parece editada/manipulada, extraia alegação sobre autenticidade

## Diretrizes de Normalização:

1. **Alegações autocontidas**: Cada alegação deve ser compreensível sem ver a imagem
   - Original na imagem: "Ele roubou tudo"
   - Normalizada: "O político [nome] roubou dinheiro público"

2. **NOMEIE pessoas famosas explicitamente**: SEMPRE que identificar uma pessoa famosa, político, celebridade ou personagem na imagem, você DEVE incluir o NOME COMPLETO dessa pessoa na alegação extraída
   - ERRADO: "O político roubou dinheiro público"
   - CORRETO: "O político Joãozinho roubou dinheiro público"
   - ERRADO: "A celebridade morreu ontem"
   - CORRETO: "A celebridade Michael Jackson morreu ontem"
   - Se você NÃO conseguir identificar o nome da pessoa, NÃO extraia alegações genéricas sobre "um político" ou "uma celebridade"

3. **Substitua pronomes**: Use nomes específicos das pessoas/entidades identificadas
   - Original: "Essa vacina causa problemas"
   - Normalizada: "A vacina COVID-19 causa problemas de saúde"

4. **Preserve contexto crítico**: Se a imagem mostra data, local, números específicos, inclua na alegação

5. **Múltiplas alegações**: Uma imagem pode conter várias alegações - extraia cada uma separadamente

6. **Preserve o idioma**: Mantenha o idioma do texto na imagem (português → alegações em português)

7. **Entidades**: Identifique e liste entidades principais (pessoas, lugares, organizações, produtos, datas, números)

8. **Análise LLM**: Para cada alegação, explique brevemente por que ela é verificável e como foi extraída da imagem

## Formato de Saída:

Retorne um objeto JSON com array "claims". Cada alegação deve ter:
- text: O texto normalizado e autocontido da alegação
- entities: Array de entidades principais mencionadas
- llm_comment: Análise de por que é verificável e como foi extraída da imagem

Se nenhuma alegação verificável for encontrada, retorne array vazio.

IMPORTANTE: Você está recebendo uma IMAGEM VISUAL. Analise-a completamente:
1. Identifique pessoas/personagens famosos PELO NOME
2. Leia todo o texto
3. Conecte visual + texto + pessoas para extrair alegações holísticas sobre o mundo real

CRÍTICO: Qualquer pessoa famosa, político, celebridade ou personagem identificado na imagem DEVE ser NOMEADO EXPLICITAMENTE nas alegações extraídas. Não use termos genéricos como "o político" ou "a celebridade" - use o NOME da pessoa.

Nota: NÃO inclua campos 'id' ou 'source' - serão adicionados automaticamente."""


# ===== UTILITY FUNCTIONS =====

def load_images_from_folder(folder_path: str) -> List[tuple[str, str]]:
    """
    load all image files from folder and convert to base64.

    args:
        folder_path: path to folder containing images

    returns:
        list of tuples (filename, base64_data)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    images = []

    folder = Path(folder_path)
    if not folder.exists():
        print(f"folder {folder_path} does not exist")
        return images

    for file_path in folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            try:
                with open(file_path, 'rb') as img_file:
                    base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                    images.append((file_path.name, base64_data))
                    print(f"loaded image: {file_path.name}")
            except Exception as e:
                print(f"error loading {file_path.name}: {e}")

    return images


def call_gpt4o_with_image(image_base64: str, prompt: str, model_name: str = "gpt-5-nano") -> str:
    """
    call OpenAI gpt-4o model with image input.

    args:
        image_base64: base64 encoded image data
        prompt: text prompt to send with image
        model_name: model name to use (default: gpt-4o)

    returns:
        model response text
    """
    model = ChatOpenAI(model=model_name, temperature=0.0)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    )

    response = model.invoke([message])
    return response.content


def call_gpt4o_for_claim_extraction_with_image(
    image_base64: str,
    model_name: str = "gpt-5-nano"
) -> dict:
    """
    call gpt-4o with image using the holistic vision-based claim extraction prompt.

    args:
        image_base64: base64 encoded image data
        model_name: model name to use

    returns:
        dict with extracted claims
    """
    # use the new holistic vision prompt that explicitly handles image input
    full_prompt = IMAGE_VISION_CLAIM_EXTRACTION_SYSTEM_PROMPT

    # create model with structured output
    model = ChatOpenAI(model=model_name, temperature=0.0)

    # define schema for structured output
    from pydantic import BaseModel, Field

    class ExtractedClaim(BaseModel):
        text: str = Field(..., description="The normalized claim text")
        entities: List[str] = Field(default_factory=list, description="Named entities in the claim")
        llm_comment: str = Field(None, description="LLM's analysis of why this is fact-checkable")

    class ClaimOutput(BaseModel):
        claims: List[ExtractedClaim] = Field(
            default_factory=list,
            description="List of extracted claims"
        )

    structured_model = model.with_structured_output(ClaimOutput, method="json_mode")

    message = HumanMessage(
        content=[
            {"type": "text", "text": full_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    )

    response = structured_model.invoke([message])

    # convert to dict
    return {
        "claims": [
            {
                "text": claim.text,
                "entities": claim.entities,
                "llm_comment": claim.llm_comment
            }
            for claim in response.claims
        ]
    }


# ===== MAIN TEST FUNCTIONS =====

def test_approach_1_direct_extraction(image_base64: str, image_name: str):
    """
    approach 1: direct claim extraction with image input.
    uses holistic vision-based prompt that:
    - identifies famous people, politicians, celebrities, and fictional characters
    - reads all text in the image
    - interprets the image holistically (visual + text + people)
    - extracts claims about the media combining all elements
    """
    print(f"\n{'='*80}")
    print(f"APPROACH 1: Direct Image Claim Extraction (Holistic Vision)")
    print(f"Image: {image_name}")
    print(f"{'='*80}\n")

    try:
        result = call_gpt4o_for_claim_extraction_with_image(image_base64)

        print(f"Extracted {len(result['claims'])} claims:\n")
        for i, claim in enumerate(result['claims'], 1):
            print(f"Claim {i}:")
            print(f"  Text: {claim['text']}")
            print(f"  Entities: {', '.join(claim['entities']) if claim['entities'] else 'None'}")
            print(f"  LLM Comment: {claim['llm_comment']}")
            print()

    except Exception as e:
        print(f"Error in approach 1: {e}")
        import traceback
        traceback.print_exc()


def test_approach_2_transcribe_then_extract(image_base64: str, image_name: str):
    """
    approach 2: first transcribe image, then extract claims from transcription.
    step 1: use transcription prompt with image
    step 2: use text-based claim extraction (without image)
    """
    print(f"\n{'='*80}")
    print(f"APPROACH 2: Transcribe Image Then Extract Claims")
    print(f"Image: {image_name}")
    print(f"{'='*80}\n")

    try:
        # step 1: transcribe image
        print("Step 1: Transcribing image...")
        transcription = call_gpt4o_with_image(image_base64, IMAGE_TRANSCRIPTION_PROMPT)
        print(f"Transcription:\n{transcription}\n")

        # step 2: extract claims from transcription
        print("Step 2: Extracting claims from transcription...")

        # create data source with transcribed text
        data_source = DataSource(
            id="test-image-transcription",
            source_type="image",
            original_text=transcription,
            locale="pt-BR"
        )

        extraction_input = ClaimExtractionInput(data_source=data_source)

        # configure llm with gpt-4o
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm_config = LLMConfig(llm=llm)

        # extract claims
        result = extract_claims(extraction_input, llm_config)

        print(f"Extracted {len(result.claims)} claims:\n")
        for i, claim in enumerate(result.claims, 1):
            print(f"Claim {i}:")
            print(f"  ID: {claim.id}")
            print(f"  Text: {claim.text}")
            print(f"  Entities: {', '.join(claim.entities) if claim.entities else 'None'}")
            print(f"  LLM Comment: {claim.llm_comment}")
            print(f"  Source: {claim.source.source_type} (ID: {claim.source.source_id})")
            print()

    except Exception as e:
        print(f"Error in approach 2: {e}")
        import traceback
        traceback.print_exc()


# ===== MAIN SCRIPT =====

def main():
    """
    main function to run both test approaches on all images.
    """
    print("="*80)
    print("IMAGE CLAIM EXTRACTION TEST SCRIPT")
    print("="*80)

    # get script directory
    script_dir = Path(__file__).parent
    images_folder = script_dir / "images"

    print(f"\nLoading images from: {images_folder}")

    # load images
    images = load_images_from_folder(str(images_folder))

    if not images:
        print("\nNo images found in the images folder.")
        print("Please add some test images to scripts/images/ and run again.")
        return

    print(f"\nFound {len(images)} image(s)\n")

    # test each image with both approaches
    for image_name, image_base64 in images:
        print(f"\n{'#'*80}")
        print(f"# Processing: {image_name}")
        print(f"{'#'*80}")

        # approach 1: direct extraction
        test_approach_1_direct_extraction(image_base64, image_name)

        # approach 2: transcribe then extract
        test_approach_2_transcribe_then_extract(image_base64, image_name)

        print(f"\n{'#'*80}")
        print(f"# Finished processing: {image_name}")
        print(f"{'#'*80}\n")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
