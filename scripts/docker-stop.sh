#!/bin/bash
# Script para parar o backend no Docker

echo "🛑 Parando o Backend - Fake News Detector"
echo "=========================================="
echo ""

docker-compose down

echo ""
echo "✅ Containers parados com sucesso!"
echo ""
echo "💡 Para remover volumes também: docker-compose down -v"
echo "💡 Para remover imagens: docker-compose down --rmi all"
echo ""

