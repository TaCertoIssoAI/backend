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
Este projeto faz parte da iniciativa **Tá Certo Isso AI**.

