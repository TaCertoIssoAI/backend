[⬅️ Voltar ao README Principal](../README.md)



# Requisitos

Este documento descreve todos os requisitos necessários para executar o **Fake News Detector - Backend**.

## Requisitos de Sistema

### Docker e Docker Compose
O projeto utiliza containers Docker para facilitar a configuração e execução do ambiente. Você precisará ter instalado:

- **Docker** (versão 20.10 ou superior recomendada)
- **Docker Compose** (versão 2.0 ou superior recomendada)

Para verificar se você tem o Docker instalado:
```bash
docker --version
docker-compose --version
```

### Git
Necessário para clonar o repositório e gerenciar versões do código.

```bash
git --version
```



## Chaves de API Necessárias

O sistema requer as seguintes chaves de API para funcionar corretamente:

### OPENAI_API_KEY (Obrigatório)
- **Descrição**: Chave de API da OpenAI para processamento de linguagem natural
- **Uso**: Utilizada para extração de claims, análise de conteúdo e geração de respostas
- **Como obter**: Acesse [platform.openai.com](https://platform.openai.com/) e crie uma conta para obter sua chave

### APIFY_TOKEN (Obrigatório)
- **Descrição**: Token de autenticação do Apify
- **Uso**: Necessário para realizar scraping de links e coleta de informações de fontes externas
- **Como obter**: Crie uma conta em [apify.com](https://apify.com/) e gere seu token de API

> [!IMPORTANT]
> Sem essas chaves de API configuradas, o sistema não funcionará corretamente. Certifique-se de obtê-las antes de prosseguir com a configuração.

## 📚 Documentação Relacionada

- [⚙️ Configuração](./CONFIGURACAO.md) - Próximo passo: configure suas variáveis de ambiente
- [🛠️ Execução](./EXECUCAO.md) - Como iniciar e parar o backend
- [📁 Estrutura do Projeto](./ESTRUTURA.md) - Entenda a organização do código
