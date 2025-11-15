#!/usr/bin/env python3
"""
Script para testar o web scraping dentro do Docker
Testa tanto requests quanto Selenium
"""

import requests
import json
import sys
from time import sleep


def test_api_health():
    """Testa se a API está respondendo"""
    print("\n1️⃣  Testando Health Check da API...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API está saudável:", response.json())
            return True
        else:
            print(f"   ❌ API retornou status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Erro ao conectar na API: {e}")
        return False


def test_selenium_grid():
    """Testa se o Selenium Grid está acessível"""
    print("\n2️⃣  Testando Selenium Grid...")
    try:
        response = requests.get("http://localhost:4444/wd/hub/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            ready = status.get("value", {}).get("ready", False)
            if ready:
                print("   ✅ Selenium Grid está pronto")
                return True
            else:
                print("   ⚠️  Selenium Grid não está pronto ainda")
                return False
        else:
            print(f"   ❌ Selenium retornou status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Erro ao conectar no Selenium: {e}")
        return False


def test_simple_url_scraping():
    """Testa scraping de URL simples (que deve funcionar com requests)"""
    print("\n3️⃣  Testando scraping de URL simples (requests)...")
    
    # Example.com é um site simples e estável
    test_url = "http://example.com"
    
    try:
        # Aqui você deve adaptar para o seu endpoint real que faz scraping
        # Este é um exemplo genérico
        print(f"   Tentando fazer scraping de: {test_url}")
        
        # Se você tiver um endpoint específico para scraping, use-o aqui
        # Por enquanto, vamos testar diretamente o módulo
        from app.ai.context.web_scraping.scraper import get_page_content
        
        content = get_page_content(test_url, force_selenium=False)
        
        if content and len(content) > 0:
            print(f"   ✅ Scraping com requests funcionou!")
            print(f"   📄 Primeiros 200 chars: {content[:200]}...")
            return True
        else:
            print("   ❌ Nenhum conteúdo foi extraído")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro no scraping com requests: {e}")
        return False


def test_selenium_scraping():
    """Testa scraping com Selenium (JavaScript)"""
    print("\n4️⃣  Testando scraping com Selenium (JavaScript)...")
    
    # Vamos testar com um site que geralmente requer JavaScript
    test_url = "http://example.com"
    
    try:
        print(f"   Tentando fazer scraping de: {test_url}")
        print("   ⏳ Isso pode levar 10-20 segundos...")
        
        from app.ai.context.web_scraping.scraper import get_page_content
        
        # Força uso do Selenium
        content = get_page_content(test_url, force_selenium=True)
        
        if content and len(content) > 0:
            print(f"   ✅ Scraping com Selenium funcionou!")
            print(f"   📄 Primeiros 200 chars: {content[:200]}...")
            return True
        else:
            print("   ❌ Nenhum conteúdo foi extraído")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro no scraping com Selenium: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_api_endpoint():
    """Testa o endpoint /api/text"""
    print("\n5️⃣  Testando endpoint /api/text...")
    
    try:
        payload = {
            "text": "A Terra é plana segundo cientistas",
            "chatId": "test-docker"
        }
        
        print("   Enviando requisição...")
        response = requests.post(
            "http://localhost:8000/api/text",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Endpoint respondeu com sucesso!")
            print(f"   📊 Resposta: {json.dumps(result, indent=2, ensure_ascii=False)[:300]}...")
            return True
        else:
            print(f"   ❌ Endpoint retornou status {response.status_code}")
            print(f"   Resposta: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao chamar endpoint: {e}")
        return False


def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("🧪 TESTE COMPLETO DO BACKEND DOCKER")
    print("    Fake News Detector - Web Scraping")
    print("=" * 60)
    
    results = []
    
    # Teste 1: Health Check
    results.append(("Health Check API", test_api_health()))
    sleep(1)
    
    # Teste 2: Selenium Grid
    results.append(("Selenium Grid", test_selenium_grid()))
    sleep(1)
    
    # Teste 3: Scraping simples
    results.append(("Scraping com Requests", test_simple_url_scraping()))
    sleep(2)
    
    # Teste 4: Scraping com Selenium
    results.append(("Scraping com Selenium", test_selenium_scraping()))
    sleep(2)
    
    # Teste 5: Endpoint da API
    results.append(("Endpoint /api/text", test_text_api_endpoint()))
    
    # Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status:12} - {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "-" * 60)
    print(f"Total: {passed}/{total} testes passaram ({passed*100//total}%)")
    print("-" * 60)
    
    if passed == total:
        print("\n🎉 Todos os testes passaram! Backend está funcionando perfeitamente.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam. Verifique os logs acima.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Teste interrompido pelo usuário")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

