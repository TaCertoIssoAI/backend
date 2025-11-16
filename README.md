# Fake News Detector - Backend

WhatsApp chatbot que recebe conteúdo de usuários, extrai claims centrais e verifica contra fontes de informação externas.

---

## Requisitos

- **Docker** e **Docker Compose**
- **Git**
- Chaves de API:
  - **OPENAI_API_KEY** (obrigatório)
  - **APIFY_TOKEN** (obrigatório para scraping de links)

---

## Configuração inicial


### 1. Configure as variáveis de ambiente

Copie o arquivo de exemplo:

```bash
cp env.example .env
```

Edite o `.env` e configure as chaves de API obrigatórias:

```bash
# Obrigatório
OPENAI_API_KEY=sk-...
APIFY_TOKEN=apify_api_...

```

### 3. Dê permissão de execução aos scripts

```bash
chmod +x scripts/*.sh
```

---

## Executando o projeto

### Iniciar o backend

```bash
./scripts/docker-start.sh
```

Esse script:
- Verifica se o `.env` existe
- Constrói as imagens Docker
- Inicia os containers
- Mostra o status e logs iniciais

Após inicialização, a API estará disponível em:

- **API Backend**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health


### Parar o backend

```bash
./scripts/docker-stop.sh
```
---

## Estrutura do projeto

```
backend/
├── app/
│   ├── ai/                   # lógica de AI e fact-checking
│   │   ├── context/          # Apify e enrichment
│   │   ├── factchecking/     # evidence retrieval
│   │   └── pipeline/         # claim extraction e judgment
│   ├── api/                  # endpoints FastAPI
│   │   └── endpoints/
│   ├── core/                 # configuração
│   └── models/               # schemas Pydantic
├── scripts/                  # scripts de execução
└── logs/                     # logs persistidos
```

---
Este projeto faz parte da iniciativa **Tá Certo Isso AI**.

