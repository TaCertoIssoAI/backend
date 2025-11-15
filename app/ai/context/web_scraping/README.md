# 🕷️ Web Scraping Module

Módulo Python para fazer scraping de páginas web com fallback automático para Selenium.

## ✨ Features

- ⚡ **Rápido**: Usa `requests` por padrão
- 🔄 **Fallback inteligente**: Muda automaticamente para Selenium quando necessário
- 🐳 **Docker ready**: Suporta Selenium em containers
- 📦 **Plug and play**: Basta importar e usar

## 🚀 Como Usar no Seu Projeto

### 1. Copiar para seu projeto

```bash
# Na raiz do seu projeto
cp -r web_scraping app/ai/context/
```

### 2. Instalar dependências

```bash
# Adicione ao seu requirements.txt principal:
cat web_scraping/requirements.txt >> requirements.txt

# Ou instale diretamente:
pip install -r web_scraping/requirements.txt
```

### 3. Configurar Docker (opcional, apenas se precisar de Selenium)

```bash
# Copie docker-compose.yml para a raiz do seu projeto
cp docker-compose.yml ../

# Inicie o container Selenium
docker compose up -d
```

### 4. Usar no código

```python
from app.ai.context.web_scraping import get_page_content

# Simples assim!
text = get_page_content("https://example.com")
print(text)

# Forçar Selenium (para sites JavaScript pesados)
text = get_page_content("https://facebook.com/some-post", force_selenium=True)
```

## 📖 Exemplos

### Exemplo Básico

```python
from app.ai.context.web_scraping import get_page_content

try:
    content = get_page_content("https://example.com")
    print(f"Extraído {len(content)} caracteres")
    print(content[:200])  # Primeiros 200 chars
except RuntimeError as e:
    print(f"Erro ao fazer scraping: {e}")
```

### Exemplo com Tratamento de Erro

```python
from app.ai.context.web_scraping import get_page_content
import logging

logging.basicConfig(level=logging.INFO)

def scrape_safely(url: str) -> str | None:
    """Faz scraping com tratamento de erro"""
    try:
        return get_page_content(url)
    except RuntimeError as e:
        logging.error(f"Falha ao scraping {url}: {e}")
        return None

# Usar
content = scrape_safely("https://example.com")
if content:
    print("✅ Sucesso!")
```

### Exemplo com Múltiplas URLs

```python
from app.ai.context.web_scraping import get_page_content
from concurrent.futures import ThreadPoolExecutor

urls = [
    "https://example.com",
    "https://wikipedia.org",
    "https://httpbin.org/html"
]

def scrape_url(url):
    try:
        return get_page_content(url)
    except:
        return None

# Scraping paralelo
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(scrape_url, urls))

for url, content in zip(urls, results):
    if content:
        print(f"✅ {url}: {len(content)} chars")
```

## ⚙️ Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do seu projeto:

```bash
# Usar Selenium remoto (Docker)?
# Valores: auto | true | false
USE_SELENIUM_REMOTE=auto

# URL do Selenium Grid (se usar Docker)
SELENIUM_REMOTE_URL=http://localhost:4444/wd/hub
```

### Opções:

- **`USE_SELENIUM_REMOTE=auto`** (padrão): Detecta automaticamente
  - Tenta Docker primeiro
  - Se falhar, usa ChromeDriver local

- **`USE_SELENIUM_REMOTE=true`**: Força uso do Docker
  - Requer `docker compose up -d` rodando
  - Mais estável e isolado

- **`USE_SELENIUM_REMOTE=false`**: Força ChromeDriver local
  - Requer ChromeDriver instalado no sistema
  - Mais leve, sem Docker

## 🐳 Docker (Opcional)

### Se você precisa de Selenium:

1. **Copie `docker-compose.yml` para a raiz do projeto**:
   ```bash
   cp docker-compose.yml ../
   ```

2. **Inicie o container**:
   ```bash
   docker compose up -d
   ```

3. **Verifique**:
   ```bash
   docker compose ps
   curl http://localhost:4444/wd/hub/status
   ```

4. **Use normalmente**:
   ```python
   from app.ai.context.web_scraping import get_page_content
   
   # Vai usar o Selenium do Docker automaticamente!
   text = get_page_content("https://facebook.com/post")
   ```

### Quando NÃO precisa de Docker:

Se você só vai fazer scraping de sites simples (HTML), **não precisa do Docker**:

```python
# Sites HTML simples funcionam sem Selenium
text = get_page_content("https://example.com")  # ✅ Funciona
text = get_page_content("https://wikipedia.org")  # ✅ Funciona
```

## 🔧 Como Funciona

1. **Tenta `requests`** primeiro (rápido, leve)
2. Se falhar, **tenta Selenium** (sites JavaScript)
3. Selenium detecta automaticamente Docker ou local

## 📦 Estrutura

```
web_scraping/
├── __init__.py          # Expõe get_page_content()
├── scraper.py           # Lógica principal
├── requirements.txt     # Dependências
└── README.md           # Esta documentação
```

## 🚨 Troubleshooting

### Erro: "selenium não está instalado"

```bash
pip install selenium
```

### Erro: "erro do selenium"

**Opção 1**: Use Docker (recomendado)
```bash
docker compose up -d
```

**Opção 2**: Instale ChromeDriver local
```bash
# Ubuntu/Debian
sudo apt install chromium-chromedriver

# Mac
brew install chromedriver

# Ou baixe de: https://chromedriver.chromium.org/
```

### Sites não funcionam mesmo com Selenium

Alguns sites bloqueiam scraping. Soluções:
- Use proxies rotativos
- Configure cookies/sessões
- Considere APIs oficiais se disponíveis

## 📝 Notas

- **Limite de caracteres**: 128.000 por padrão (configurável no código)
- **Timeout**: 20s para requests, 30s para Selenium
- **User-Agent**: Aleatorizado automaticamente
- **Suporta**: HTML, JSON, text/plain

## 🔗 Integração Completa

### Estrutura no seu projeto:

```
seu-projeto/
├── app/
│   └── ai/
│       └── context/
│           └── web_scraping/    ← Pasta copiada
│               ├── __init__.py
│               ├── scraper.py
│               └── requirements.txt
├── docker-compose.yml           ← Copiado da pasta web_scraping
├── .env                         ← Suas configurações
├── requirements.txt             ← Inclui dependências do web_scraping
└── main.py                      ← Seu código que usa o módulo
```

### Exemplo de uso no seu projeto:

```python
# app/ai/context/main.py
from app.ai.context.web_scraping import get_page_content

def process_url(url: str):
    """Processa conteúdo de uma URL"""
    print(f"Buscando {url}...")
    
    content = get_page_content(url)
    
    # Agora você tem o texto limpo!
    print(f"Extraídos {len(content)} caracteres")
    
    # Faça o que quiser com o conteúdo
    # - Envie para LLM
    # - Salve no banco
    # - Processe com IA
    return content

if __name__ == "__main__":
    text = process_url("https://example.com")
    print(text[:500])
```

## ✅ Checklist de Integração

- [ ] Copiar pasta `web_scraping` para `app/ai/context/`
- [ ] Adicionar dependências ao `requirements.txt` principal
- [ ] Instalar: `pip install -r requirements.txt`
- [ ] (Opcional) Copiar `docker-compose.yml` para raiz
- [ ] (Opcional) Criar `.env` com configurações
- [ ] Importar e usar: `from app.ai.context.web_scraping import get_page_content`
- [ ] Testar com URL simples
- [ ] (Opcional) Iniciar Docker: `docker compose up -d`
- [ ] Testar com URL JavaScript

## 🎉 Pronto!

Agora você tem scraping inteligente no seu projeto!

```python
from app.ai.context.web_scraping import get_page_content

# É só isso! 🚀
text = get_page_content("https://qualquer-site.com")
```
