# Experimento: TaCertoIssoAI vs Meta AI - Fact-Checking

## Contexto

Este experimento foi conduzido como parte do programa AI4Good em 2025 e busca comparar nossa solução (TaCertoIssoAI) contra o Meta AI no WhatsApp para fact-checking com fontes confiáveis.

O objetivo é avaliar a capacidade de ambos os sistemas em verificar desinformação de forma precisa e fundamentada em fontes verificáveis.

## Dataset

Utilizamos fact-checkings do site G1 Fato ou Fake: https://g1.globo.com/fato-ou-fake/

- **Total de fact-checks**: 30 casos
- **Fonte**: Todos os artigos estão linkados no CSV de resultados

## Metodologia

1. **Procedimento de teste**: Enviamos o mesmo input/prompt para nosso modelo e para o Meta AI. No caso do nosso modelo, utilizamos o script experiments/run_against_dataset.py para automatizar o teste contra o serviço de fact-checking rodando localmente de forma equivalente ao envio de uma mensagem no Whatsapp, porém sem a camada de orquestração da API no N8N. Para a MetaAI, executamos os testes manualmente no Whatsapp devido à falta de uma API.
2. **Repetição**: Executamos o dataset completo duas vezes para ambos os modelos
3. **Resultados**: Os valores reportados são a média das duas execuções

### Critérios de Sucesso

Consideramos um fact-checking **bem-sucedido** quando:

- O resultado está em conformidade com o label "Natureza da notícia"
- **Fontes confiáveis foram incluídas na resposta**

**Justificativa**: Não acreditamos que confiar apenas na saída de um LLM sem fontes confiáveis conte como bom fact-checking. A transparência e verificabilidade são essenciais.

### Tratamento de Labels "Falso"

Quando o label é "Fake", consideramos bem-sucedidas **tanto respostas "Falso" quanto "Fora de Contexto"** de AMBOS os modelos, pois ambas categorias indicam que a alegação é enganosa.

## Resultados

### Performance Geral

| Modelo | Run 1 | Run 2 | Taxa de Sucesso |
|--------|-------|-------|-----------------|
| Meta AI | 12/30 | 12/30 | **40,0%** |
| TaCertoIssoAI | 25/30 | 23/30 | **79,9%** |

### 4.1 Problema Crítico: Meta AI Gerando Imagens de Fake News

Em **4 dos 30 casos**, o Meta AI não apenas falhou em fazer o fact-checking corretamente, mas **criou imagens que representavam as fake news**, potencialmente contribuindo para a disseminação de desinformação ao invés de combatê-la. É relevante mencionar que esses casos foram comuns, com prompts como:
*  'Imagem viral traz Ronaldo, Ronaldinho Gaúcho, Roberto Carlos e Rivaldo segurando uma faixa com a frase 'Somos todos Bolsonaro - Anistia já!', sugerindo apoio público dos ex-jogadores.', '
* Imagem de post  mostra Daniel Vorcaro  em registro com outros políticos feito em 7 de agosto.' 
*  'Imagem mostra torcedores do São Paulo no estádio segurando uma faixa pedindo para que o time perca para o Internacional.' 

resultando na geração de imagens em ambos os testes, mostrando um padrão de comportamento da MetaAI que promove a desinformação.

Estes casos são extremamente problemáticos, pois:
- O usuário esperava verificação da informação
- Recebeu conteúdo visual que reforça a desinformação
- Pode compartilhar essas imagens geradas, amplificando o problema
- Texto de resposta é em inglês (Your Image is Ready), então brasileiros que não são fluentes nessa língua não tem nenhum contexto além da imagem gerada

**Casos documentados:**

#### Caso 1

![Imagem gerada pelo Meta AI - Caso 2](images/metaAI_generated_image2.jpeg)


#### Caso 2
![Imagem gerada pelo Meta AI - Caso 3](images/metaAI_generated_image3.jpeg)


#### Caso 3
![Imagem gerada pelo Meta AI - Caso 1](images/metaAI_generated_image1.jpeg)

#### Caso 4
![Imagem gerada pelo Meta AI - Caso 4](images/metaAI_generated_image4.jpeg)

#### Caso 5
![Imagem gerada pelo Meta AI - Caso 5](images/metaAI_generated_image5.jpeg)

#### Caso 6
![Imagem gerada pelo Meta AI - Caso 6](images/metaAI_generated_image6.jpeg)

#### Caso 7
![Imagem gerada pelo Meta AI - Caso 7](images/metaAI_generated_image7.jpeg)

#### Caso 8
![Imagem gerada pelo Meta AI - Caso 8](images/metaAI_generated_image8.jpeg)

### 4.2 Recusa em Responder Questões Eleitorais

O Meta AI demonstrou um padrão de recusar responder prompts relacionados a eleições, retornando uma resposta pré-fabricada direcionando usuários ao site do TSE.

**Problemas identificados:**

1. **Fricção excessiva**: Representa uma quantidade significativa de fricção que os usuários precisam enfrentar para obter informação precisa sobre eventos cruciais
2. **Limitação de escopo**: Assume que a informação desejada está disponível no site do TSE, o que nem sempre é o caso
3. **Timing crítico**: Especialmente problemático considerando que o Brasil terá eleições nacionais em 2026
4. **Experiência fragmentada**: Usuários que buscam fact-checking rápido são forçados a navegar para outro site

Esta abordagem reduz significativamente a utilidade do Meta AI para combater desinformação eleitoral, que é uma das categorias mais críticas de fake news.

**Casos documentados:**

#### Caso Eleitoral 1
![Recusa Meta AI - Eleições 1](images/metaAI_tse_response1.jpeg)

#### Caso Eleitoral 2
![Recusa Meta AI - Eleições 2](images/metaAI_tse_response2.jpeg)

#### Caso Eleitoral 3
![Recusa Meta AI - Eleições 3](images/metaAI_tse_response3.jpeg)

## Conclusões

Os resultados demonstram que:

1. **TaCertoIssoAI teve desempenho 2x superior** ao Meta AI (79,9% vs 40,0%)
2. **Meta AI apresenta riscos de amplificação de desinformação** através da geração de imagens relacionadas a fake news
3. **Meta AI limita utilidade em contextos eleitorais**, que são críticos para a democracia

Estes resultados sugerem que soluções especializadas em fact-checking com fundamentação em fontes (como TaCertoIssoAI) são mais adequadas para combater desinformação do que modelos de propósito geral.
