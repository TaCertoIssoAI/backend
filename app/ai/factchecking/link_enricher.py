"""
Link Enrichment Module - Step 2.5 of Fact-Checking Pipeline

This module takes claims with URLs and enriches them by extracting content from those URLs.
Based on the existing webscraping module using newspaper3k library.

Follows LangChain best practices:
- Async-first design
- Structured outputs with Pydantic models
- Error handling and fallback mechanisms
- Detailed logging and processing notes
"""

import asyncio
import time
import logging
import re
import random
import requests
import os
from bs4 import BeautifulSoup

from newspaper import Article, Config
from readability import Document
import trafilatura
from goose3 import Goose

# Selenium imports - only if available
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    # Logger will be defined later, so we'll handle this warning in the functions

from app.models.factchecking import (
    ClaimExtractionOutput,
    ExtractedClaim,
    EnrichedLink,
    EnrichedClaim,
    LinkEnrichmentOutput
)

logger = logging.getLogger(__name__)


def _is_render_environment():
    """Detecta se está rodando no Render."""
    return (
        os.getenv('RENDER') == 'true' or 
        os.getenv('DYNO') is not None or  # Heroku
        os.path.exists('/.dockerenv') or   # Docker
        os.getenv('CONTAINER') == 'true'
    )


def _is_selenium_available():
    """Verifica se Selenium está disponível e funcionando."""
    if not SELENIUM_AVAILABLE:
        return False
    
    # Verifica se Chrome/Chromium está disponível
    try:
        import subprocess
        result = subprocess.run(['google-chrome', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        try:
            # Tenta chromium como fallback
            result = subprocess.run(['chromium-browser', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False


def _is_invalid_content(texto):
    """
    Verifica se o conteúdo extraído é inválido (ex: mensagens de JavaScript desabilitado)
    """
    if not texto:
        return True
    
    texto_lower = texto.lower().strip()
    
    # Lista expandida de padrões que indicam conteúdo inválido
    invalid_patterns = [
        # JavaScript/erro patterns
        "javascript is not available",
        "javascript is disabled",
        "please enable javascript",
        "switch to a supported browser",
        "we've detected that javascript is disabled",
        "enable javascript or switch to a supported browser",
        "something went wrong",
        "try again",
        "privacy related extensions",
        "disable them and try again",
        
        # X/Twitter specific patterns
        "help center",
        "terms of service",
        "privacy policy",
        "cookie policy",
        "ads info",
        "imprint",
        "© 2025 x corp",
        "© 2024 x corp",
        "© 2023 x corp",
        
        # Generic error patterns
        "access denied",
        "forbidden",
        "not found",
        "page not found",
        "server error",
        "temporarily unavailable",
        "maintenance mode",
        "under construction",
        "coming soon",
        "this page is not available",
        "content not available",
        "unable to load",
        "loading failed",
        "connection error",
        "timeout",
        "rate limited",
        "too many requests"
    ]
    
    # Verifica se o texto contém principalmente padrões inválidos
    invalid_count = sum(1 for pattern in invalid_patterns if pattern in texto_lower)
    total_words = len(texto_lower.split())
    
    # Se mais de 15% das palavras são padrões inválidos, considera inválido
    if total_words > 0 and (invalid_count / total_words) > 0.15:
        return True
    
    # Se o texto contém qualquer padrão inválido e é curto
    if len(texto_lower) < 300 and any(pattern in texto_lower for pattern in invalid_patterns):
        return True
    
    # Verifica se o texto é principalmente sobre JavaScript/erro
    js_related_words = ["javascript", "browser", "enable", "switch", "supported", "detected", "error", "failed", "unavailable"]
    js_word_count = sum(1 for word in js_related_words if word in texto_lower)
    
    if js_word_count >= 2 and len(texto_lower) < 400:
        return True
    
    # Verifica se o texto é muito curto e contém palavras de erro
    error_words = ["error", "failed", "unavailable", "denied", "forbidden", "not found", "disabled"]
    if len(texto_lower) < 100 and any(word in texto_lower for word in error_words):
        return True
    
    # Verifica se o texto é principalmente links de navegação/footer
    footer_words = ["help", "terms", "privacy", "policy", "cookie", "ads", "imprint", "corp", "©"]
    footer_count = sum(1 for word in footer_words if word in texto_lower)
    
    if footer_count >= 3 and len(texto_lower) < 200:
        return True
    
    return False


def _extrair_com_trafilatura(url):
    """Método 1: Trafilatura (muito robusto)"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            metadata = trafilatura.extract_metadata(downloaded)
            
            return {
                "titulo": metadata.title if metadata else None,
                "autores": [metadata.author] if metadata and metadata.author else None,
                "data_publicacao": metadata.date if metadata else None,
                "texto_completo": content
            }
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed for {url}: {e}")
    return None


def _extrair_com_newspaper3k(url):
    """Método 2: Newspaper3k (especializado em notícias)"""
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        config.request_timeout = 10
        
        artigo = Article(url, language='pt', config=config)
        artigo.download()
        artigo.parse()
        
        return {
            "titulo": artigo.title,
            "autores": artigo.authors,
            "data_publicacao": artigo.publish_date,
            "texto_completo": artigo.text
        }
    except Exception as e:
        logger.debug(f"Newspaper3k extraction failed for {url}: {e}")
    return None


def _extrair_com_readability(url):
    """Método 3: Readability-lxml (focado em conteúdo principal)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            doc = Document(response.text)
            soup = BeautifulSoup(doc.summary(), 'html.parser')
            
            # Extrair título da página original
            title_soup = BeautifulSoup(response.text, 'html.parser')
            title = title_soup.find('title')
            title_text = title.get_text().strip() if title else None
            
            return {
                "titulo": title_text,
                "autores": None,
                "data_publicacao": None,
                "texto_completo": soup.get_text()
            }
    except Exception as e:
        logger.debug(f"Readability extraction failed for {url}: {e}")
    return None


def _extrair_com_goose3(url):
    """Método 4: Goose3 (especializado em notícias)"""
    try:
        g = Goose()
        article = g.extract(url=url)
        
        return {
            "titulo": article.title,
            "autores": [article.authors] if article.authors else None,
            "data_publicacao": article.publish_date,
            "texto_completo": article.cleaned_text
        }
    except Exception as e:
        logger.debug(f"Goose3 extraction failed for {url}: {e}")
    return None


def _extrair_com_requests_session(url):
    """Método 5: Requests com sessão (para sites que requerem cookies)"""
    try:
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        session.headers.update(headers)
        
        # Primeira requisição para estabelecer sessão
        response = session.get(url, timeout=15, allow_redirects=True)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts e estilos
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Tenta encontrar título
            title = soup.find('title')
            title_text = title.get_text().strip() if title else None
            
            # Para X/Twitter, tenta seletores específicos
            if 'x.com' in url or 'twitter.com' in url:
                content_selectors = [
                    '[data-testid="tweetText"]',  # Texto do tweet
                    '[data-testid="tweet"]',      # Container do tweet
                    'article[data-testid="tweet"]',  # Artigo do tweet
                    '[role="article"]',           # Artigo genérico
                    '.tweet-text',                # Classe do texto do tweet
                    '[data-testid="card.wrapper"]'  # Card wrapper
                ]
            else:
                content_selectors = [
                    'article', 'main', '.content', '.post-content', 
                    '.article-content', '.entry-content', '[role="main"]'
                ]
            
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text()
                    if content and len(content.strip()) > 10:
                        break
            
            if not content:
                content = soup.get_text()
            
            # Limpa o texto
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                "titulo": title_text,
                "autores": None,
                "data_publicacao": None,
                "texto_completo": content
            }
    except Exception as e:
        logger.debug(f"Requests session extraction failed for {url}: {e}")
    return None


def _extrair_com_beautifulsoup(url):
    """Método 6: BeautifulSoup (último recurso)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts e estilos
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Tenta encontrar título
            title = soup.find('title')
            title_text = title.get_text().strip() if title else None
            
            # Tenta encontrar conteúdo principal
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.article-content', '.entry-content', '[role="main"]'
            ]
            
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text()
                    break
            
            if not content:
                content = soup.get_text()
            
            # Limpa o texto
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                "titulo": title_text,
                "autores": None,
                "data_publicacao": None,
                "texto_completo": content
            }
    except Exception as e:
        logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")
    return None


def _extrair_com_selenium(url):
    """Método 7: Selenium (para sites com JavaScript)"""
    if not SELENIUM_AVAILABLE:
        logger.debug("Selenium não disponível - pulando método selenium")
        return None
        
    if not _is_selenium_available():
        logger.debug("Selenium não disponível - pulando método selenium")
        return None
        
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Configurações específicas para ambiente containerizado
        if _is_render_environment():
            chrome_options.add_argument("--remote-debugging-port=9222")
            chrome_options.add_argument("--disable-background-timer-throttling")
            chrome_options.add_argument("--disable-backgrounding-occluded-windows")
            chrome_options.add_argument("--disable-renderer-backgrounding")
        
        # Tenta usar ChromeDriver do sistema primeiro, depois webdriver-manager
        try:
            service = Service('/usr/local/bin/chromedriver')
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            driver.get(url)
            time.sleep(3)  # Aguarda carregar
            
            # Tenta extrair título
            title = driver.title
            
            # Tenta extrair conteúdo principal
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.article-content', '.entry-content', '[role="main"]'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    content = elements[0].text
                    break
            
            if not content:
                content = driver.find_element(By.TAG_NAME, "body").text
            
            return {
                "titulo": title,
                "autores": None,
                "data_publicacao": None,
                "texto_completo": content
            }
            
        finally:
            driver.quit()
    except Exception as e:
        logger.debug(f"Selenium extraction failed for {url}: {e}")
    return None


def _extrair_com_selenium_avancado(url):
    """Método 8: Selenium Avançado (especialmente para X/Twitter)"""
    if not SELENIUM_AVAILABLE:
        logger.debug("Selenium não disponível - pulando método selenium avançado")
        return None
        
    if not _is_selenium_available():
        logger.debug("Selenium não disponível - pulando método selenium avançado")
        return None
        
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        
        # Configurações específicas para ambiente containerizado
        if _is_render_environment():
            chrome_options.add_argument("--remote-debugging-port=9222")
            chrome_options.add_argument("--disable-background-timer-throttling")
            chrome_options.add_argument("--disable-backgrounding-occluded-windows")
            chrome_options.add_argument("--disable-renderer-backgrounding")
        else:
            chrome_options.add_argument("--disable-javascript")  # Desabilita JS para evitar detecção
        
        # Tenta usar ChromeDriver do sistema primeiro, depois webdriver-manager
        try:
            service = Service('/usr/local/bin/chromedriver')
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            # Executa script para ocultar que é um bot
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Adiciona delay aleatório
            time.sleep(random.uniform(2, 4))
            
            driver.get(url)
            
            # Aguarda mais tempo para carregar
            time.sleep(5)
            
            # Tenta extrair título
            title = driver.title
            
            # Para X/Twitter, tenta seletores específicos
            if 'x.com' in url or 'twitter.com' in url:
                content_selectors = [
                    '[data-testid="tweetText"]',  # Texto do tweet
                    '[data-testid="tweet"]',      # Container do tweet
                    'article[data-testid="tweet"]',  # Artigo do tweet
                    '[role="article"]',           # Artigo genérico
                    '.tweet-text',                # Classe do texto do tweet
                    '[data-testid="card.wrapper"]'  # Card wrapper
                ]
            else:
                content_selectors = [
                    'article', 'main', '.content', '.post-content', 
                    '.article-content', '.entry-content', '[role="main"]'
                ]
            
            content = ""
            for selector in content_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        content = elements[0].text
                        if content and len(content.strip()) > 10:
                            break
                except:
                    continue
            
            if not content:
                try:
                    content = driver.find_element(By.TAG_NAME, "body").text
                except:
                    content = ""
            
            return {
                "titulo": title,
                "autores": None,
                "data_publicacao": None,
                "texto_completo": content
            }
            
        finally:
            driver.quit()
    except Exception as e:
        logger.debug(f"Selenium avançado extraction failed for {url}: {e}")
    return None


def extrair_noticia_principal_de_link(url):
    """
    Tenta extrair o conteúdo de uma notícia usando múltiplas bibliotecas em sequência.
    Usa uma abordagem de fallback otimizada: métodos rápidos primeiro, pesados depois.
    
    Args:
        url (str): O link da notícia.
        
    Returns:
        dict: Um dicionário com o título, autores, data e o texto limpo da notícia.
        None: Se todas as tentativas falharem.
    """
    
    # PIPELINE OTIMIZADO: Métodos rápidos primeiro, pesados depois
    # Fase 1: Métodos rápidos e baratos (HTTP requests)
    metodos_rapidos = [
        ("trafilatura", _extrair_com_trafilatura),
        ("newspaper3k", _extrair_com_newspaper3k),
        ("readability", _extrair_com_readability),
        ("requests_session", _extrair_com_requests_session),
        ("beautifulsoup", _extrair_com_beautifulsoup)
    ]
    
    # Fase 2: Métodos pesados (apenas se Selenium estiver disponível)
    metodos_pesados = []
    if _is_selenium_available():
        metodos_pesados = [
            ("selenium", _extrair_com_selenium),
            ("selenium_avancado", _extrair_com_selenium_avancado)
        ]
        logger.debug(f"🚀 Selenium disponível - métodos pesados habilitados")
    else:
        logger.debug(f"⚠️ Selenium não disponível - usando apenas métodos rápidos")
    
    # Para X/Twitter, adiciona goose3 na fase rápida
    if 'x.com' in url or 'twitter.com' in url:
        metodos_rapidos.insert(1, ("goose3", _extrair_com_goose3))
    
    logger.debug(f"🚀 FASE 1: Tentando métodos rápidos para {url}")
    
    # Tenta métodos rápidos primeiro
    for nome_metodo, funcao_metodo in metodos_rapidos:
        try:
            logger.debug(f"Tentando extrair com {nome_metodo}...")
            resultado = funcao_metodo(url)
            
            if resultado and resultado.get('texto_completo'):
                texto = resultado['texto_completo'].strip()
                
                # Verifica se o conteúdo é válido
                if len(texto) > 50 and not _is_invalid_content(texto):
                    logger.debug(f"✅ Sucesso com {nome_metodo}!")
                    resultado['metodo_usado'] = nome_metodo
                    return resultado
                else:
                    logger.debug(f"❌ {nome_metodo} extraiu conteúdo inválido (JS disabled ou muito curto)")
            else:
                logger.debug(f"❌ {nome_metodo} não conseguiu extrair conteúdo suficiente")
                
        except Exception as e:
            logger.debug(f"❌ Erro com {nome_metodo}: {e}")
            continue
    
    # Se métodos rápidos falharam e Selenium está disponível, tenta métodos pesados
    if metodos_pesados:
        logger.debug(f"🔄 FASE 2: Tentando métodos pesados (Selenium) para {url}")
        
        for nome_metodo, funcao_metodo in metodos_pesados:
            try:
                logger.debug(f"Tentando extrair com {nome_metodo}...")
                resultado = funcao_metodo(url)
                
                if resultado and resultado.get('texto_completo'):
                    texto = resultado['texto_completo'].strip()
                    
                    # Verifica se o conteúdo é válido
                    if len(texto) > 50 and not _is_invalid_content(texto):
                        logger.debug(f"✅ Sucesso com {nome_metodo}!")
                        resultado['metodo_usado'] = nome_metodo
                        return resultado
                    else:
                        logger.debug(f"❌ {nome_metodo} extraiu conteúdo inválido (JS disabled ou muito curto)")
                else:
                    logger.debug(f"❌ {nome_metodo} não conseguiu extrair conteúdo suficiente")
                    
            except Exception as e:
                logger.debug(f"❌ Erro com {nome_metodo}: {e}")
                continue
    
    logger.debug(f"❌ Todos os métodos falharam para {url}")
    return None


class LinkEnricher:
    """
    Enriches claims by extracting content from their associated URLs.
    
    Uses newspaper3k for web scraping with async execution.
    """
    
    def __init__(self, content_limit: int = 5000):
        """Initialize the link enricher with web scraping capabilities."""
        
        # Configure newspaper3k for web scraping
        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.newspaper_config.request_timeout = 15  # Increased timeout
        
        # Content limit for extracted text
        self.content_limit = content_limit

    async def enrich_links(self, claims_result: ClaimExtractionOutput) -> LinkEnrichmentOutput:
        """
        Main method to enrich all claims with link content.
        
        Args:
            claims_result: Result from claim extraction step
            
        Returns:
            LinkEnrichmentOutput with enriched claims
        """
        start_time = time.time()
        
        enriched_claims = []
        total_links = 0
        successful_extractions = 0
        
        for claim in claims_result.claims:
            if claim.links:
                total_links += len(claim.links)
                enriched_claim = await self._enrich_single_claim(claim)
                
                # Count successful extractions
                for enriched_link in enriched_claim.enriched_links:
                    if enriched_link.extraction_status == "success":
                        successful_extractions += 1
                        
                enriched_claims.append(enriched_claim)
            else:
                # No links to enrich, convert to EnrichedClaim as-is
                enriched_claim = EnrichedClaim(
                    text=claim.text,
                    original_links=[],
                    enriched_links=[],
                    llm_comment=claim.llm_comment,
                    entities=claim.entities
                )
                enriched_claims.append(enriched_claim)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        processing_notes = (
            f"Processados {total_links} links. "
            f"{successful_extractions} extrações bem-sucedidas. "
            f"{total_links - successful_extractions} falhas."
        )
        
        return LinkEnrichmentOutput(
            original_claims=claims_result.claims,
            enriched_claims=enriched_claims,
            total_links_processed=total_links,
            successful_extractions=successful_extractions,
            processing_time_ms=processing_time,
            processing_notes=processing_notes
        )

    async def _enrich_single_claim(self, claim: ExtractedClaim) -> EnrichedClaim:
        """Enrich a single claim by extracting content from its links."""
        
        enriched_links = []
        
        # Process each link in the claim
        for url in claim.links:
            enriched_link = await self._extract_link_content(url)
            enriched_links.append(enriched_link)
        
        return EnrichedClaim(
            text=claim.text,
            original_links=claim.links,
            enriched_links=enriched_links,
            llm_comment=claim.llm_comment,
            entities=claim.entities
        )

    async def _extract_link_content(self, url: str) -> EnrichedLink:
        """
        Extract content from a single URL using newspaper3k.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            EnrichedLink with extracted content and summary
        """
        enriched_link = EnrichedLink(
            url=url,
            extraction_status="pending"
        )
        
        try:
            # Use asyncio to run newspaper3k extraction in thread pool
            # (newspaper3k is synchronous, so we need to wrap it)
            loop = asyncio.get_event_loop()
            extraction_result = await loop.run_in_executor(
                None, 
                self._extract_with_newspaper, 
                url
            )
            
            if extraction_result:
                enriched_link.title = extraction_result.get("titulo", "")
                
                # Apply content limit from the existing webscraping logic
                full_content = extraction_result.get("texto_completo", "")
                enriched_link.content = full_content[:self.content_limit] if full_content else ""
                
                # Create a simple summary based on title and first paragraph
                enriched_link.summary = self._create_simple_summary(
                    enriched_link.title, 
                    enriched_link.content
                )
                
                enriched_link.extraction_status = "success"
                method_used = extraction_result.get("metodo_usado", "unknown")
                enriched_link.extraction_notes = f"Conteúdo extraído com {method_used}. Tamanho: {len(full_content)} chars, limitado a {len(enriched_link.content)} chars"
            else:
                enriched_link.extraction_status = "failed"
                enriched_link.extraction_notes = "Falha na extração de conteúdo"
                
        except Exception as e:
            logger.error(f"Link enrichment failed for {url}: {e}")
            enriched_link.extraction_status = "failed"
            enriched_link.extraction_notes = f"Erro durante extração: {str(e)[:100]}"
        
        return enriched_link

    def _extract_with_newspaper(self, url: str) -> dict:
        """
        Enhanced extraction using the optimized pipeline with multiple fallback methods.
        Uses the new pipeline that tries multiple extraction methods in order of efficiency.
        """
        try:
            # Use the optimized pipeline with multiple extraction methods
            extraction_result = extrair_noticia_principal_de_link(url)
            
            if extraction_result:
                # Convert to the expected format
                info_noticia = {
                    "titulo": extraction_result.get("titulo", ""),
                    "autores": extraction_result.get("autores", []),
                    "data_publicacao": extraction_result.get("data_publicacao"),
                    "texto_completo": extraction_result.get("texto_completo", ""),
                    "metodo_usado": extraction_result.get("metodo_usado", "unknown")
                }
                
                logger.debug(f"Successful extraction for {url} using {info_noticia['metodo_usado']}")
                return info_noticia
            else:
                logger.warning(f"All extraction methods failed for {url}")
                return None
            
        except Exception as e:
            logger.error(f"Pipeline extraction failed for {url}: {e}")
            return None

    def _create_simple_summary(self, title: str, content: str) -> str:
        """
        Create a simple summary without LLM, using basic text processing.
        """
        if not content:
            return "Conteúdo não disponível"
        
        # Use title and first paragraph as summary
        summary_parts = []
        
        if title:
            summary_parts.append(f"Título: {title}")
        
        # Get first paragraph (up to first double newline or first 200 chars)
        first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content
        first_paragraph = first_paragraph[:200] + "..." if len(first_paragraph) > 200 else first_paragraph
        
        if first_paragraph.strip():
            summary_parts.append(f"Resumo: {first_paragraph.strip()}")
        
        return " | ".join(summary_parts) if summary_parts else "Conteúdo extraído mas sem informações resumíveis"


# Factory function for creating enricher
def create_link_enricher(content_limit: int = 5000) -> LinkEnricher:
    """
    Factory function to create a LinkEnricher instance.
    """
    return LinkEnricher(content_limit=content_limit)


# Async helper function for direct usage
async def enrich_claim_links(claims_result: ClaimExtractionOutput) -> LinkEnrichmentOutput:
    """
    Convenience function to enrich links from claim extraction result.
    
    Args:
        claims_result: ClaimExtractionOutput from step 2
        
    Returns:
        LinkEnrichmentOutput with enriched claims
    """
    enricher = create_link_enricher()
    return await enricher.enrich_links(claims_result)
