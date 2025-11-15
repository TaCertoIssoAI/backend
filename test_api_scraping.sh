#!/bin/bash
# Script para testar a API de scraping completa

# Configuração
API_URL="${API_URL:-http://localhost:8000}"
echo "🧪 TESTE COMPLETO DA API DE SCRAPING"
echo "====================================="
echo "API URL: $API_URL"
echo ""

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para testar endpoint
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    echo -e "${YELLOW}► Testando: $name${NC}"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}✅ PASSOU${NC} (HTTP $http_code)"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        echo -e "${RED}❌ FALHOU${NC} (HTTP $http_code)"
        echo "$body"
    fi
    echo ""
}

# Teste 1: Health Check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Health Check" "GET" "/health" ""

# Teste 2: Root endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Root Endpoint" "GET" "/" ""

# Teste 3: Scraping Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Scraping Status" "GET" "/api/scraping-status" ""

# Teste 4: Scrape Test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Scrape Test (Quick)" "GET" "/api/scrape-test" ""

# Teste 5: Scrape URL simples
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Scrape URL (example.com)" "POST" "/api/scrape" '{
  "url": "https://example.com",
  "force_selenium": false
}'

# Teste 6: Scrape com Selenium
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Scrape com Selenium" "POST" "/api/scrape" '{
  "url": "https://example.com",
  "force_selenium": true,
  "max_chars": 500
}'

# Teste 7: Múltiplas URLs (se disponível)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_endpoint "Scrape Múltiplas URLs" "POST" "/api/scrape-multiple?force_selenium=false" '{
  "urls": ["https://example.com", "https://httpbin.org/html"]
}'

# Resumo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 TESTES CONCLUÍDOS!${NC}"
echo ""
echo "Para testar em produção (Render):"
echo "  export API_URL=https://seu-app.onrender.com"
echo "  ./test_api_scraping.sh"
echo ""
echo "Para ver documentação interativa:"
echo "  $API_URL/docs"

