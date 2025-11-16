#!/bin/bash
# Script para testar o backend no Docker

echo "🧪 Testando o Backend - Fake News Detector"
echo "==========================================="
echo ""

# Verificar se os containers estão rodando
echo "1️⃣  Verificando containers..."
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Containers não estão rodando!"
    echo "Execute: ./docker-start.sh"
    exit 1
fi
echo "✅ Containers rodando"
echo ""

# Testar health check
echo "2️⃣  Testando health check..."
HEALTH=$(curl -s http://localhost:8000/health)
if [ $? -eq 0 ]; then
    echo "✅ Health check OK: $HEALTH"
else
    echo "❌ Health check falhou!"
    exit 1
fi
echo ""

# Testar endpoint raiz
echo "3️⃣  Testando endpoint raiz..."
ROOT=$(curl -s http://localhost:8000/)
if [ $? -eq 0 ]; then
    echo "✅ Endpoint raiz OK: $ROOT"
else
    echo "❌ Endpoint raiz falhou!"
    exit 1
fi
echo ""

# Testar web scraping (teste simples)
echo "4️⃣  Testando web scraping com Selenium..."
echo "   Isso pode levar alguns segundos..."

# Criar payload de teste
TEST_PAYLOAD='{"text":"Teste de web scraping","chatId":"test"}'

# Fazer requisição
RESPONSE=$(curl -s -X POST http://localhost:8000/api/text \
    -H "Content-Type: application/json" \
    -d "$TEST_PAYLOAD")

if [ $? -eq 0 ]; then
    echo "✅ Endpoint de texto respondeu"
    echo "   Resposta: ${RESPONSE:0:100}..."
else
    echo "❌ Endpoint de texto falhou!"
fi
echo ""

# Verificar Selenium
echo "5️⃣  Verificando Selenium Grid..."
SELENIUM_STATUS=$(curl -s http://localhost:4444/wd/hub/status)
if [ $? -eq 0 ]; then
    echo "✅ Selenium Grid OK"
else
    echo "❌ Selenium Grid não está respondendo!"
fi
echo ""

echo "======================================"
echo "✅ Testes básicos concluídos!"
echo ""
echo "📚 Para testar web scraping completo:"
echo "   python test_selenium.py"
echo ""
echo "📖 Documentação interativa da API:"
echo "   http://localhost:8000/docs"
echo ""

