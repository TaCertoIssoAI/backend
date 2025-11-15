"""
Web Scraping Module

Módulo para fazer scraping de páginas web com fallback automático para Selenium.
Uso simples:

    from app.ai.context.web_scraping import get_page_content
    
    text = get_page_content("https://example.com")
    print(text)
"""

from .scraper import get_page_content

__version__ = "1.0.0"
__all__ = ["get_page_content"]
