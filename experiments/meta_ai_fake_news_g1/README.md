# Experimento: TaCertoIssoAI vs Meta AI - Fact-Checking

## Contexto

Este experimento foi conduzido como parte do programa AI4Good em 2025 e busca comparar nossa solução (TaCertoIssoAI) contra o Meta AI no WhatsApp para fact-checking com fontes confiáveis.

O objetivo é avaliar a capacidade de ambos os sistemas em verificar desinformação de forma precisa e fundamentada em fontes verificáveis.

## Dataset

Utilizamos fact-checkings do site G1 Fato ou Fake: https://g1.globo.com/fato-ou-fake/

- **Total de fact-checks**: 31 casos
- **Fonte**: Todos os artigos estão linkados no CSV de resultados

## Metodologia

1. **Procedimento de teste**: Enviamos o mesmo input/prompt para nosso modelo e para o Meta AI
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
| Meta AI | 12/31 | 13/31 | **40,31%** |
| TaCertoIssoAI | 26/31 | 24/31 | **80,63%** |

### 4.1 Problema Crítico: Meta AI Gerando Imagens de Fake News

Em **4 dos 31 casos**, o Meta AI não apenas falhou em fazer o fact-checking corretamente, mas **criou imagens que representavam as fake news**, potencialmente contribuindo para a disseminação de desinformação ao invés de combatê-la.

Estes casos são extremamente problemáticos, pois:
- O usuário esperava verificação da informação
- Recebeu conteúdo visual que reforça a desinformação
- Pode compartilhar essas imagens geradas, amplificando o problema

**Casos documentados:**

#### Caso 1
![Imagem gerada pelo Meta AI - Caso 1](path/to/image1.png)

#### Caso 2
![Imagem gerada pelo Meta AI - Caso 2](path/to/image2.png)

#### Caso 3
![Imagem gerada pelo Meta AI - Caso 3](path/to/image3.png)

#### Caso 4
![Imagem gerada pelo Meta AI - Caso 4](path/to/image4.png)

#### Caso 5
![Imagem gerada pelo Meta AI - Caso 5](path/to/image5.png)

#### Caso 6
![Imagem gerada pelo Meta AI - Caso 6](path/to/image6.png)

#### Caso 7
![Imagem gerada pelo Meta AI - Caso 7](path/to/image7.png)

### 4.2 Recusa em Responder Questões Eleitorais

O Meta AI demonstrou um padrão de recusar responder prompts relacionados a eleições, retornando uma resposta pré-fabricada direcionando usuários ao site do TSE (com um link quebrado).

**Problemas identificados:**

1. **Fricção excessiva**: Representa uma quantidade significativa de fricção que os usuários precisam enfrentar para obter informação precisa sobre eventos cruciais
2. **Limitação de escopo**: Assume que a informação desejada está disponível no site do TSE, o que nem sempre é o caso
3. **Timing crítico**: Especialmente problemático considerando que o Brasil terá eleições nacionais em 2026
4. **Experiência fragmentada**: Usuários que buscam fact-checking rápido são forçados a navegar para outro site

Esta abordagem reduz significativamente a utilidade do Meta AI para combater desinformação eleitoral, que é uma das categorias mais críticas de fake news.

**Casos documentados:**

#### Caso Eleitoral 1
![Recusa Meta AI - Eleições 1](path/to/election1.png)

#### Caso Eleitoral 2
![Recusa Meta AI - Eleições 2](path/to/election2.png)

#### Caso Eleitoral 3
![Recusa Meta AI - Eleições 3](path/to/election3.png)

## Conclusões

Os resultados demonstram que:

1. **TaCertoIssoAI teve desempenho 2x superior** ao Meta AI (80,63% vs 40,31%)
2. **Meta AI apresenta riscos de amplificação de desinformação** através da geração de imagens relacionadas a fake news
3. **Meta AI limita utilidade em contextos eleitorais**, que são críticos para a democracia

Estes resultados sugerem que soluções especializadas em fact-checking com fundamentação em fontes (como TaCertoIssoAI) são mais adequadas para combater desinformação do que modelos de propósito geral.
