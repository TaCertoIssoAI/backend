#!/usr/bin/env python3
"""
Script de teste rápido para o módulo web_scraping
Simula como seria usado no projeto principal
"""

import sys
import os

# Adiciona o diretório raiz do projeto ao path para importar
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

from app.ai.context.web_scraping import get_page_content

def test_basic():
    """Teste básico"""
    print("🧪 Teste 1: Scraping básico")
    print("-" * 60)
    
    url = "https://example.com"
    print(f"URL: {url}")
    
    try:
        content = get_page_content(url)
        print(f"✅ Sucesso! {len(content)} caracteres")
        print(f"\nPrimeiros 150 caracteres:")
        print(content[:150])
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def test_multiple():
    """Teste com múltiplas URLs"""
    print("\n\n🧪 Teste 2: Múltiplas URLs")
    print("-" * 60)
    
    urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]
    
    success_count = 0
    for url in urls:
        print(f"\n{url}")
        try:
            content = get_page_content(url)
            print(f"  ✅ {len(content)} chars")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {e}")
    
    print(f"\n{success_count}/{len(urls)} URLs processadas com sucesso")
    return success_count == len(urls)


def main():
    print("\n" + "="*60)
    print("🕷️  TESTE DO MÓDULO WEB_SCRAPING")
    print("="*60 + "\n")
    
    results = []
    
    # Teste 1
    results.append(test_basic())
    
    # Teste 2
    results.append(test_multiple())
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ Todos os {total} testes passaram!")
        print("\n🎉 Módulo pronto para uso no projeto!")
        print("\nImporte usando:")
        print("  from app.ai.context.web_scraping import get_page_content")
        print("\nUse:")
        print("  content = get_page_content(url)")
        return 0
    else:
        print(f"❌ {total - passed}/{total} testes falharam")
        print("\nVerifique:")
        print("- Dependências instaladas? pip install -r web_scraping/requirements.txt")
        print("- Conexão com internet ok?")
        return 1


if __name__ == "__main__":
    sys.exit(main())
