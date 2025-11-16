#!/bin/bash
# Script para iniciar o backend no Docker

echo "🚀 Iniciando o Backend - Fake News Detector"
echo "==========================================="
echo ""

# Verificar se o arquivo .env existe
if [ ! -f .env ]; then
    echo "⚠️  Arquivo .env não encontrado!"
    echo "Criando .env a partir do env.example..."
    cp env.example .env
    echo "✅ Arquivo .env criado. Configure suas variáveis de ambiente!"
    echo ""
    echo "⚠️  IMPORTANTE: Configure o APIFY_TOKEN no arquivo .env"
    echo ""
fi

# Verificar se APIFY_TOKEN está configurado
if [ -f .env ]; then
    if grep -q "APIFY_TOKEN=your_apify_token_here" .env || ! grep -q "APIFY_TOKEN=" .env; then
        echo "⚠️  ATENÇÃO: APIFY_TOKEN não está configurado no .env"
        echo "Por favor, adicione seu token do Apify em .env"
        echo ""
    fi
fi

# Build das imagens
echo "📦 Construindo imagens Docker..."
docker-compose build

# Iniciar serviços
echo ""
echo "🐳 Iniciando containers..."
docker-compose up -d

# Aguardar inicialização
echo ""
echo "⏳ Aguardando inicialização dos serviços..."
sleep 10

# Verificar status
echo ""
echo "📊 Status dos containers:"
docker-compose ps

# Verificar health
echo ""
echo "🏥 Verificando health checks..."
sleep 5

# Mostrar logs do backend
echo ""
echo "📋 Últimos logs do backend:"
docker-compose logs --tail=20 backend

echo ""
echo "✅ Backend iniciado com sucesso!"
echo ""
echo "📍 URLs disponíveis:"
echo "   - API Backend: http://localhost:8000"
echo "   - API Docs (Swagger): http://localhost:8000/docs"
echo "   - Health Check: http://localhost:8000/health"
echo "   - Scraping Status: http://localhost:8000/api/scraping-status"
echo ""
echo "📝 Comandos úteis:"
echo "   - Ver logs: ./docker-logs.sh"
echo "   - Parar: ./docker-stop.sh"
echo "   - Restart: docker-compose restart backend"
echo ""
echo "🧪 Testar scraping:"
echo "   curl http://localhost:8000/api/scraping-status"
echo ""
