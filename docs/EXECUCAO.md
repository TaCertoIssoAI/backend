[⬅️ Voltar ao README Principal](../README.md)



# Execução

Este documento explica como iniciar, parar e acessar o **Fake News Detector - Backend**.

## Iniciar o Backend

Para iniciar o backend, execute o seguinte comando na raiz do projeto:

```bash
./scripts/docker-start.sh
```

### O que o script faz?

O script `docker-start.sh` automatiza o processo de inicialização:

1. ✅ Verifica se o arquivo `.env` existe
2. 🏗️ Constrói as imagens Docker (se necessário)
3. 🚀 Inicia os containers
4. 📊 Mostra o status dos containers
5. 📝 Exibe os logs iniciais

### Tempo de inicialização

A primeira execução pode demorar alguns minutos enquanto as imagens Docker são construídas. Execuções subsequentes serão muito mais rápidas.

## Acessar o Backend

Após a inicialização bem-sucedida, os seguintes endpoints estarão disponíveis:

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API Backend** | http://localhost:8000 | Endpoint principal da API |
| **API Docs (Swagger)** | http://localhost:8000/docs | Documentação interativa da API |
| **Health Check** | http://localhost:8000/health | Verificação de saúde do sistema |

> [!TIP]
> Acesse http://localhost:8000/docs para explorar interativamente todos os endpoints disponíveis da API usando a interface Swagger.



## Parar o Backend

Para parar o backend e todos os containers em execução:

```bash
./scripts/docker-stop.sh
```

Este comando irá:
- 🛑 Parar todos os containers
- 🧹 Limpar recursos temporários
- 📝 Exibir o status final



## Comando Direto (Alternativo)

Se preferir executar o servidor diretamente sem Docker (não recomendado para produção):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> [!CAUTION]
> Executar fora do Docker requer ter todas as dependências Python instaladas localmente e pode causar problemas de compatibilidade. Use Docker sempre que possível.



## Solução de Problemas

### O backend não inicia
- Verifique se o arquivo `.env` está configurado corretamente
- Confirme que as portas 8000 não estão sendo usadas por outros serviços
- Verifique os logs com `docker-compose logs`

### Erro de permissão ao executar scripts
- Execute `chmod +x scripts/*.sh` para dar permissão de execução



## 📚 Documentação Relacionada

- [📋 Requisitos](./REQUISITOS.md) - Verifique os requisitos do sistema
- [⚙️ Configuração](./CONFIGURACAO.md) - Configure o ambiente
- [📁 Estrutura do Projeto](./ESTRUTURA.md) - Entenda a organização do código
