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

# Opcional (para integração com WhatsApp)
EVOLUTION_API_URL=your_evolution_api_url_here
EVOLUTION_API_KEY=your_evolution_api_key_here
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

### Ver logs em tempo real

```bash
./scripts/docker-logs.sh
```

Para ver logs de um serviço específico:

```bash
./scripts/docker-logs.sh selenium  # logs do Selenium Grid
```

### Testar o backend

```bash
./scripts/docker-test.sh
```

Esse script:
- Verifica se os containers estão rodando
- Testa health check
- Testa endpoints principais
- Verifica status do Selenium Grid

### Parar o backend

```bash
./scripts/docker-stop.sh
```

Para remover volumes também:

```bash
docker-compose down -v
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

