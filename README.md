# Fact-Checking Bot Backend

Backend do Chatbot no WhatsApp que recebe conteúdo de usuários, extrai claims centrais e verifica contra fontes de informação externas.

---

## Requisitos

- **Python 3.11+**
- **pip** (gerenciador de pacotes Python)
- Chaves de API:
  - **OPENAI_API_KEY** (obrigatório)
  - **GOOGLE_SEARCH_API_KEY** (obrigatório)
  - **GOOGLE_API_KEY** (obrigatório)
  - **APIFY_TOKEN** (obrigatório para scraping de links)

---

## Configuração inicial

### 1. Crie e ative o ambiente virtual

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# No Linux/Mac:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

Copie o arquivo de exemplo:

```bash
cp env.example .env
```

Edite o `.env` e configure as chaves de API obrigatórias.

Carregue as variáveis de ambiente no terminal:

**No Linux/Mac:**
```bash
set -a
source .env
set +a
```

**No Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:GOOGLE_SEARCH_API_KEY="..."
$env:GOOGLE_API_KEY="..."
$env:APIFY_TOKEN="apify_api_..."
```

---

## Executando o projeto

### Modo Desenvolvimento

Para desenvolvimento local com recarga automática de código:

```bash
make run
```

### Endpoints Disponíveis

Após inicialização, a API estará disponível em:

- **API Backend**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Endpoint Principal**: http://localhost:8000/text

---

## Testes de Integração

Os testes de integração verificam o funcionamento end-to-end da API com diferentes cenários e configurações.

### Testes Disponíveis

| Teste | Descrição |
|-------|-----------|
| `text_only_test` | Testa fact-checking de texto puro sem links |
| `text_and_links_test` | Testa processamento de texto com links (usa Apify) |
| `text_and_links_no_browser` | Testa processamento de texto com links (sem Apify) |
| `text_and_links_gemini` | Testa pipeline completo usando Gemini |
| `adjudication_with_search_fallback` | Testa fallback para busca web quando fontes são insuficientes |
| `prod_load_test` | Teste de carga com 6 requisições simultâneas |

### Como Executar

**1. Executar um teste específico:**

```bash
# Primeiro, inicie o servidor
make run

# Em outro terminal, execute o teste
make test TEST=text_only_test
```

Substitua `text_only_test` pelo nome de qualquer teste da tabela acima.

**2. Executar o teste de carga:**

```bash
# Terminal 1: Inicie o servidor em modo produção
make run-prod

# Terminal 2: Execute o teste de carga
make run-load-test
```

O teste de carga envia 6 requisições simultâneas com payloads grandes e verifica:
- Todos retornam HTTP 200
- Respostas contêm vereditos válidos
- Pipeline processa afirmações falsos corretamente

---

## Estrutura do projeto

```
├── app/
│   ├── ai/                      # ⭐ PIPELINE PRINCIPAL - Lógica de fact-checking
│   │   ├── context/             # Expansão de links (Apify) e enrichment
│   │   ├── factchecking/        # Busca de evidências e sources
│   │   ├── pipeline/            # Extração de claims e adjudicação
│   │   └── threads/             # ⭐ SISTEMA DE CONCORRÊNCIA
│   │       └── ...              # ThreadPool e concorrência baseada em filas
│   │
│   ├── models/                  # ⭐ SCHEMAS PYDANTIC
│   │   ├── api.py              # Request/Response da API
│   │   ├── factchecking.py     # Modelos do pipeline (Claims, Verdicts, etc)
│   │   └── ...                 # Outros schemas
│   │
│   ├── observability/          # ⭐ ANALYTICS E LOGGING
│   │   ├── analytics.py        # AnalyticsCollector para métricas
│   │   └── logger/             # Sistema de logging customizado
│   │
│   ├── clients/                # ⭐ CLIENTES EXTERNOS
│   │   └── analytics_service.py # Cliente para envio de analytics
│   │
│   ├── api/                    # Endpoints FastAPI
│   │   └── endpoints/          # Rotas (/text, /health, etc)
│   │
│   └── core/                   # Configuração e utilities
│
├── integration_tests/          # Testes end-to-end
```

### Componentes Principais

**⭐ `ai/` - Pipeline Principal**
Orquestra todo o fluxo de fact-checking: extração de claims do texto, busca de evidências em múltiplas fontes, e julgamento final usando LLMs com raciocínio fundamentado em citações.

**⭐ `models/` - Schemas Pydantic**
Define os contratos de dados entre todas as camadas do sistema: requests/responses da API, estruturas internas de claims e verdicts, e modelos de evidências e citações.

**⭐ `observability/` - Analytics e Logging**
Captura métricas detalhadas de cada execução do pipeline (tempos, tokens, fontes utilizadas) e prove sistema de logging estruturado para debugging e monitoramento.

**⭐ `clients/` - Clientes Externos**
Implementa comunicação com serviços externos, incluindo envio de analytics para a plataforma [tacertoissoai](https://tacertoissoai.com.br).

**⭐ `threads/` - Sistema de Concorrência**
Biblioteca customizada de ThreadPool (Buffer pré-alocado de Threads para execução de tarefas) e filas para processamento paralelo de múltiplas fontes de dados, garantindo processamento mais rápido e menor latência.

---

## Como funciona o Pipeline de Fact-Checking

O pipeline de fact-checking processa mensagens dos usuários em etapas sequenciais, cada uma enriquecendo as informações até produzir o veredito final com citações.

### Visão Geral do Fluxo

```
Mensagem do Usuário

1. Separação por fonte do dados (texto, imagem...)
    ↓
2. Expansão de Links
    ↓
3. Extração de Claims
    ↓
4. Busca de Evidências
    ↓
5. Adjudicação
    ↓
Resposta com Veredito e Citações
```


### Diagrama de arquitetura da Pipeline
![diagrama_pipeline](./images/diagrama_pipeline_2.png)

### Etapas Detalhadas

**1. Expansão de Links (Link Expansion)**

- **O que faz:** Identifica URLs no texto original e busca o conteúdo desses links
- **Como funciona:**
  - Extrai automaticamente todos os URLs da mensagem do usuário
  - Acessa cada URL e obtém o conteúdo completo da página (artigos, posts, etc)
  - Cria novos "data sources" do tipo `link_context` com o conteúdo expandido
- **Tecnologia:** Executa em paralelo usando ThreadPoolManager para processar múltiplos links simultaneamente
- **Resultado:** Lista de fontes de dados expandidas que serão analisadas junto com o texto original

**2. Extração de Claims (Claim Extraction)**

- **O que faz:** Identifica afirmações verificáveis no texto usando LLM (Large Language Model)
- **Como funciona:**
  - Processa cada fonte de dados (texto original + conteúdo dos links expandidos)
  - Usa um LLM para extrair claims factuais que podem ser verificados
  - Cada claim recebe um ID único e referência à fonte de onde foi extraído
- **Modelo:** Configurável via `claim_extraction_llm_config` (padrão: Gemini ou GPT-4)
- **Resultado:** Lista de `ClaimExtractionOutput` com todas as afirmações extraídas de todas as fontes

**3. Busca de Evidências (Evidence Gathering)**

- **O que faz:** Para cada claim extraído, busca evidências que suportem ou refutem a afirmação
- **Como funciona:**
  - Executa múltiplas buscas em paralelo para cada claim:
    - **Google Fact Check API:** Busca fact-checks já publicados sobre o claim
    - **Web Search:** Realiza buscas na web filtradas por domínios confiáveis
  - Agrega todas as citações encontradas para cada claim
  - Normaliza e deduplica as fontes
- **Concorrência:** Usa gatherers assíncronos para buscar em múltiplas fontes simultaneamente
- **Resultado:** `EvidenceRetrievalResult` mapeando cada claim_id para um `EnrichedClaim` com citações

**4. Adjudicação (Adjudication)**

- **O que faz:** Analisa todos os claims e evidências para produzir vereditos finais
- **Como funciona:**
  - Recebe o texto original, claims extraídos e todas as evidências coletadas
  - Um LLM forte (configurável, padrão: Gemini ou GPT-4) analisa o conjunto completo
  - Gera vereditos para cada claim: Verdadeiro, Falso, Enganoso ou Fontes Insuficientes
  - Produz um resumo geral explicando a análise
  - Inclui citações específicas que fundamentam cada veredito
- **Fallback:** Se as fontes forem insuficientes, executa `adjudication_with_search` que usa busca web integrada do OpenAI para encontrar mais evidências em tempo real
- **Resultado:** `FactCheckResult` com vereditos detalhados, citações e explicação para o usuário

### Características Técnicas

- **Execução Assíncrona:** Todas as etapas usam async/await para operações I/O eficientes
- **Processamento Paralelo:** ThreadPoolManager processa múltiplos claims e links simultaneamente
- **Isolamento de Pipeline:** Cada requisição recebe um `pipeline_id` único para rastreamento e limpeza de recursos
- **Observabilidade:** Analytics collector captura métricas de cada etapa
- **Resiliência:** Sistema de fallback garante resposta mesmo quando fontes primárias falham

---
Este projeto faz parte da iniciativa **Tá Certo Isso AI**.

