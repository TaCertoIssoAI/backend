# scraper.py - Módulo principal de web scraping
from __future__ import annotations

import logging
import os
from typing import Tuple

import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, RequestException

# Fake headers para evitar bloqueios
FAKE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Limite grande o suficiente para a maioria das páginas
MAX_CHAR_LIMIT = 128000

# URL do Selenium Grid (Docker)
SELENIUM_REMOTE_URL = os.getenv("SELENIUM_REMOTE_URL", "http://localhost:4444/wd/hub")


def _parse_html(html_text: str) -> str:
    """Parse HTML e extrai texto limpo"""
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script e style
    for script in soup(["script", "style"]):
        script.decompose()

    # Pega o texto e normaliza quebras de linha
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = (
        phrase.strip()
        for line in lines
        for phrase in line.split("  ")
    )
    parsed_content = "\n".join(chunk for chunk in chunks if chunk)
    return parsed_content


def _get_content_from_url(
    url: str,
    max_chars: int = MAX_CHAR_LIMIT,
    use_fake_headers: bool = True,
) -> Tuple[str, str]:
    """
    Busca conteúdo usando requests (rápido, sem JavaScript)
    Retorna: (status, conteudo_ou_erro)
    """
    headers: dict = {}

    if use_fake_headers:
        headers.update(FAKE_HEADERS)

    logging.info(f"fazendo requisição para {url} com headers: {headers}")

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except (HTTPError, RequestException) as e:
        status_code = (
            e.response.status_code
            if isinstance(e, HTTPError) and e.response is not None
            else "N/A"
        )
        logging.error(f"falha na requisição para {url} (status: {status_code}): {e}")
        return "error", f"erro HTTP {status_code}"

    mime_type = response.headers.get("Content-Type", "")

    # Só trata texto e JSON
    if mime_type.startswith(("text/", "application/json")):
        content = response.text

        if mime_type.startswith("text/html"):
            content = _parse_html(response.text)

        if len(content) > max_chars:
            logging.warning(
                f"conteúdo de {url} passou de {max_chars} caracteres, truncando"
            )
            content = content[:max_chars]

        return "success", content

    error_msg = f"tipo de conteúdo não suportado: {mime_type}"
    logging.error(error_msg)
    return "error", error_msg


def _get_content_with_selenium(url: str, max_chars: int = MAX_CHAR_LIMIT) -> Tuple[str, str]:
    """
    Busca conteúdo usando Selenium (para sites com JavaScript)
    Detecta automaticamente se usa Docker ou ChromeDriver local
    Retorna: (status, conteudo_ou_erro)
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.common.exceptions import WebDriverException
    except ImportError:
        return "error", "selenium não está instalado. instale com: pip install selenium"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = None
    use_remote = os.getenv("USE_SELENIUM_REMOTE", "auto").lower()
    
    try:
        if use_remote == "false":
            # Força uso local
            logging.info(f"usando chromedriver local para {url}")
            driver = webdriver.Chrome(options=chrome_options)
            
        elif use_remote == "true":
            # Força uso remoto (Docker)
            logging.info(f"usando selenium remote ({SELENIUM_REMOTE_URL}) para {url}")
            driver = webdriver.Remote(
                command_executor=SELENIUM_REMOTE_URL,
                options=chrome_options
            )
        else:
            # Auto: tenta remote primeiro, fallback para local
            logging.info(f"tentando selenium remote ({SELENIUM_REMOTE_URL}) para {url}")
            try:
                driver = webdriver.Remote(
                    command_executor=SELENIUM_REMOTE_URL,
                    options=chrome_options
                )
                logging.info("conectado ao selenium remote")
            except Exception as e:
                logging.warning(f"falha ao conectar ao selenium remote: {e}")
                logging.info("tentando chromedriver local como fallback")
                driver = webdriver.Chrome(options=chrome_options)
                logging.info("usando chromedriver local")
        
        driver.set_page_load_timeout(30)
        driver.get(url)
        driver.implicitly_wait(5)
        
        html_content = driver.page_source
        content = _parse_html(html_content)
        
        if len(content) > max_chars:
            logging.warning(
                f"conteúdo de {url} passou de {max_chars} caracteres, truncando"
            )
            content = content[:max_chars]
        
        return "success", content
        
    except WebDriverException as e:
        error_msg = f"erro do selenium: {str(e)}"
        logging.error(error_msg)
        return "error", error_msg
    except Exception as e:
        error_msg = f"erro inesperado com selenium: {str(e)}"
        logging.error(error_msg)
        return "error", error_msg
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def get_page_content(url: str, force_selenium: bool = False) -> str:
    """
    Função principal: busca e retorna o texto de uma página web.
    
    Tenta primeiro com requests (rápido). Se falhar, usa Selenium (JavaScript).
    
    Args:
        url: URL da página
        force_selenium: Se True, pula requests e usa Selenium direto
    
    Returns:
        str: Texto extraído da página
    
    Raises:
        RuntimeError: Se ambos os métodos falharem
    
    Example:
        >>> from app.ai.context.web_scraping import get_page_content
        >>> text = get_page_content("https://example.com")
        >>> print(text[:100])
    """
    if force_selenium:
        logging.info(f"forçando uso do selenium para {url}")
        status, content = _get_content_with_selenium(url)
        if status == "error":
            raise RuntimeError(content)
        return content
    
    # Primeira tentativa: requests (mais rápido)
    logging.info(f"tentando buscar {url} com requests")
    status, content = _get_content_from_url(url)

    if status == "success":
        logging.info(f"sucesso ao buscar {url} com requests")
        return content

    # Se falhou, tenta com Selenium
    logging.warning(f"falha com requests para {url}: {content}")
    logging.info(f"tentando fallback com selenium")
    
    status, content = _get_content_with_selenium(url)
    
    if status == "error":
        raise RuntimeError(f"falha tanto com requests quanto com selenium: {content}")

    logging.info(f"sucesso ao buscar {url} com selenium")
    return content
