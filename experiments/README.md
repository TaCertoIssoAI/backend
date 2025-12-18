# Testes da API com Datasets

Esta pasta contém um script simples para testar a API de fact-checking contra datasets.

## Estrutura

```
experiments/
├── run_against_dataset.py        # Script principal de teste
├── requirements.txt               # Dependências Python
├── README.md                      # Este arquivo
└── meta_ai_2025_fake_news_g1/    # Outro dataset
    └── ...
```

## Configuração

1. Crie e ative o ambiente virtual:
```bash
cd experiments
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Certifique-se de que o servidor da API está rodando:
```bash
# Em outro terminal, a partir da raiz do projeto:
uvicorn app.main:app --reload --port 8000
```

## Uso

Execute o script com uma pasta de dataset:

```bash
# Execute contra outro dataset
python run_against_dataset.py meta_ai_2025_fake_news_g1
```

**Sem argumentos?** O script mostrará as pastas de dataset disponíveis:
```bash
python run_against_dataset.py
```

## Configuração

Edite as constantes em `run_against_dataset.py`:

```python
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/text"
MAX_CONCURRENT_REQUESTS = 5
```

## Como Funciona

1. **Lê o CSV** - Encontra o primeiro arquivo `.csv` na pasta do dataset
2. **Auto-detecta a coluna de alegação** - Procura por nomes de coluna comuns como "Título da checagem", "claim_text", "text", "claim"
3. **Chama a API concorrentemente** - Máximo de 5 requisições em paralelo
4. **Escreve as respostas** - Salva todas as respostas da API em um arquivo de texto

## Seleção de Dataset

Quando você executa o script, ele:
1. Procura na pasta do dataset especificado por arquivos `.csv`
2. Se múltiplos CSVs existirem, lista-os e usa o **primeiro em ordem alfabética**
3. Auto-detecta qual coluna contém o texto da alegação
4. Escreve os resultados em `api_responses_AAAAMMDD_HHMMSS.txt` na **mesma pasta do dataset**

**Dica:** Se você tiver múltiplos arquivos CSV e quiser que um específico seja usado:
- Nomeie-o para que venha primeiro em ordem alfabética (ex: `01_dataset.csv`)
- Mova ou delete os outros arquivos CSV temporariamente

## Formato do Arquivo de Saída

O script gera um arquivo de texto com todas as respostas da API:

```
================================================================================
RESULTADOS DOS TESTES DA API COM DATASET
================================================================================
Dataset: filtered_dataset.csv
API: http://localhost:8000/text
Data: 2025-12-17 15:30:45
Total de alegações: 145
================================================================================

================================================================================
ALEGAÇÃO 1
================================================================================

Texto: Sergio Moro tirou foto ao lado do Lula após seu partido...

Status: SUCESSO

Resposta:
--------------------------------------------------------------------------------
[Resposta completa da API com veredito, justificativa e citações]
--------------------------------------------------------------------------------

================================================================================
ALEGAÇÃO 2
================================================================================

...
```

## Exemplo de Saída no Console

```
================================================================================
TESTES DA API COM DATASET
================================================================================
Entrada:  /caminho/para/filtered_dataset.csv
Saída: /caminho/para/api_responses_20251217_153045.txt
API:    http://localhost:8000/text
Máximo de requisições concorrentes: 5
================================================================================

Lendo CSV...
✓ Usando coluna 'Título da checagem' para texto da alegação
Carregadas 145 alegações do CSV

Processando com máximo de 5 requisições concorrentes...

[1] Processando: Sergio Moro tirou foto ao lado do Lula após seu partido...
[2] Processando: A vacina X causa infertilidade...
[1] ✓ Sucesso
[3] Processando: O governo vai aumentar impostos...
[2] ✓ Sucesso
...

================================================================================
RESUMO
================================================================================
Total processado:  145
✓ Bem-sucedidos:   142 (97.9%)
✗ Falhas:          3 (2.1%)
================================================================================

Resultados salvos em: api_responses_20251217_153045.txt
```
