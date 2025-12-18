[⬅️ Voltar ao README Principal](../README.md)



# Estrutura do Projeto

Este documento descreve a organização de diretórios e arquivos do **Fake News Detector - Backend**.

## Visão Geral

```
backend/
├── app/                    # Código principal da aplicação
│   ├── ai/                 # Lógica de AI e fact-checking
│   │   ├── context/        # Apify e enrichment de contexto
│   │   ├── factchecking/   # Evidence retrieval e verificação
│   │   └── pipeline/       # Claim extraction e judgment
|   |   └── threads/        # Threadpool and concurrent job queue system
│   ├── api/                # Endpoints FastAPI
│   │   └── endpoints/      # Definição de rotas da API
│   ├── core/               # Configuração central
│   └── models/             # Schemas Pydantic
├── scripts/                # Scripts de automação
├── logs/                   # Arquivos de log persistidos
├── docker-compose.yml      # Configuração Docker
├── Dockerfile              # Imagem Docker
├── requirements.txt        # Dependências Python
└── README.md               # Documentação principal
```



## Detalhamento dos Diretórios

### 📁 `app/`
Diretório principal contendo todo o código da aplicação.

#### `app/ai/` - Inteligência Artificial
Módulos relacionados ao processamento de IA e verificação de fatos:

- **`context/`** - Coleta e enriquecimento de contexto
  - Integração com Apify para scraping
  - Enrichment de informações de fontes externas
  
- **`factchecking/`** - Verificação de Fatos
  - Evidence retrieval (busca de evidências)
  - Análise e validação de fontes
  
- **`pipeline/`** - Pipeline de Processamento
  - Claim extraction (extração de afirmações)
  - Judgment e classificação de veracidade

#### `app/ai/threads` - Sistema de processamento concorrente utilizando a Threadpool e filas do python

Como grande parte das operações da pipeline são I/O heavy e bloqueam a execução de uma thread (que espera que esse I/O termine) nós implementamos um sistema para execução concorrente de jobs da pipeline. Para isso utilizamos a Threadpool do python 
para alocar uma buffer de threads para executar jobs e utilizamos uma fila de jobs separados pelo tipo da operação (equivale à cada passo do pipeline). Isso permite que um parte do código submeta jobs como extração de links, busca por evidência... de forma async, e quando esses jobs precisarem ser processados, um outro trecho de código espera
pela execução deles por meio da fila concorrente que implementamos.

#### `app/api/` - API REST
Implementação dos endpoints da API usando FastAPI:

- **`endpoints/`** - Definições de rotas
  - Endpoints para receber mensagens
  - Endpoints de análise e verificação
  - Endpoints de consulta e relatórios

#### `app/core/` - Configuração
Configurações centrais da aplicação:
- Gerenciamento de variáveis de ambiente
- Configurações de logging
- Constantes e parâmetros do sistema

#### `app/models/` - Modelos de Dados
Schemas Pydantic para validação de dados:
- Modelos de request/response
- Estruturas de dados internas da pipeline (entradas e saídas de cada passo da pipeline)
- Validadores



### 📁 `scripts/`
Scripts de automação para facilitar o desenvolvimento e operação:

- `docker-start.sh` - Inicia o ambiente Docker
- `docker-stop.sh` - Para o ambiente Docker
- Outros scripts auxiliares



### 📁 `logs/`
Diretório para armazenamento de logs da aplicação:
- Logs são persistidos em arquivos
- Útil para debugging e monitoramento
- Ignorado pelo Git (configurado em `.gitignore`)



## Arquivos de Configuração

### `docker-compose.yml`
Define os serviços Docker e suas configurações:
- Configuração de containers
- Mapeamento de portas
- Volumes e redes

### `Dockerfile`
Instruções para construção da imagem Docker:
- Imagem base Python
- Instalação de dependências
- Configuração do ambiente

### `requirements.txt`
Lista de todas as dependências Python do projeto:
- FastAPI para a API REST
- OpenAI para processamento de linguagem
- Bibliotecas de scraping e análise
- Ferramentas de teste e desenvolvimento

### `.env` (não versionado)
Arquivo de variáveis de ambiente (criado a partir de `env.example`):
- Chaves de API
- Configurações específicas do ambiente
- **Nunca deve ser commitado no Git**



## Fluxo de Dados

```mermaid
graph LR
    A[Usuário] -->|Mensagem WhatsApp| B[API Endpoints]
    B --> C[Pipeline AI]
    C --> D[Claim Extraction]
    D --> E[Context Enrichment]
    E --> F[Fact Checking]
    F --> G[Judgment]
    G -->|Resultado| A
```

O sistema processa mensagens através de um pipeline que:
1. Recebe o conteúdo via API
2. Extrai claims principais
3. Busca contexto e evidências
4. Realiza fact-checking
5. Gera um julgamento final
6. Retorna o resultado ao usuário

## APIs utilizadas

### API de Fact-checking do google

Utilizamos a API de fact-checking do google para buscar fontes confiáveis sobre a afirmação. 
Documentação da API pode ser acessada neste [link](https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims)

E o mesmo recurso de fact-checking pode ser acessado de forma manual neste [website](https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=pt) da google

### Busca na web em domínios confiáveis

Utilizamos da API de busca customizada da google, restringindo os domínios retornados para uma lista de domínios confiáveis (ex: g1.globo.com e .gov.br)

## 📚 Documentação Relacionada

- [📋 Requisitos](./REQUISITOS.md) - Requisitos do sistema
- [⚙️ Configuração](./CONFIGURACAO.md) - Configure o ambiente
- [🛠️ Execução](./EXECUCAO.md) - Como executar o backend
