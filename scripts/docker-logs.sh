#!/bin/bash
# Script para ver logs do backend no Docker

echo "📋 Logs do Backend - Fake News Detector"
echo "========================================"
echo ""

# Se passou argumento, usa; senão mostra backend
SERVICE="${1:-backend}"

echo "Mostrando logs de: $SERVICE"
echo ""
echo "Pressione Ctrl+C para sair"
echo ""

docker-compose logs -f "$SERVICE"

