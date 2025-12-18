[⬅️ Voltar ao README Principal](../README.md)



# Configuração

Este documento detalha o processo de configuração do **Fake News Detector - Backend**.

## 1. Arquivo de Variáveis de Ambiente

### Copiar o arquivo de exemplo

O projeto inclui um arquivo `env.example` com todas as variáveis necessárias. Primeiro, copie-o para criar seu arquivo `.env`:

```bash
cp env.example .env
```

### Configurar as variáveis obrigatórias

Abra o arquivo `.env` em seu editor de texto preferido e configure as seguintes variáveis:

```bash
# Obrigatório - Chave de API da OpenAI
OPENAI_API_KEY=sk-...

# Obrigatório - Token do Apify para scraping
APIFY_TOKEN=apify_api_...
```

### Variáveis Disponíveis

O arquivo `.env` pode conter outras variáveis de configuração. Consulte o arquivo `env.example` para ver todas as opções disponíveis e suas descrições.

> [!TIP]
> Mantenha seu arquivo `.env` seguro e **nunca** o compartilhe publicamente ou faça commit dele no Git. Ele já está incluído no `.gitignore` por segurança.



## 2. Permissões dos Scripts

Para executar os scripts de automação do projeto, é necessário dar permissão de execução:

```bash
chmod +x scripts/*.sh
```

Isso permitirá que você execute scripts como:
- `./scripts/docker-start.sh` - Iniciar o backend
- `./scripts/docker-stop.sh` - Parar o backend



## Verificação da Configuração

Após concluir estas etapas, você deve ter:
- ✅ Arquivo `.env` criado com as chaves de API configuradas
- ✅ Permissões de execução nos scripts

Agora você está pronto para executar o projeto! Consulte a [documentação de execução](./EXECUCAO.md) para os próximos passos.



## 📚 Documentação Relacionada

- [📋 Requisitos](./REQUISITOS.md) - Verifique se você tem tudo instalado
- [🛠️ Execução](./EXECUCAO.md) - Próximo passo: inicie o backend
- [📁 Estrutura do Projeto](./ESTRUTURA.md) - Entenda a organização do código
