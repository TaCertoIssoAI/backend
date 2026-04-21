"""
tests for domain-specific news scrapers and platform routing.
"""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from app.agentic_ai.context.web.apify_utils import detectPlatform, PlatformType, scrapeGenericUrl
from app.agentic_ai.context.web.news_scrapers import (
    scrape_g1_article,
    scrape_estadao_article,
    scrape_folha_article,
    scrape_aosfatos_article,
    _build_result,
)


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

G1_HTML = """
<html><head><title>G1 Test</title></head>
<body>
<h1>Vacina contra gripe é segura, dizem especialistas</h1>
<div class="content-text">
  <p>Este é o primeiro parágrafo do artigo com conteúdo suficiente para passar o filtro de tamanho mínimo.</p>
  <p>Segundo parágrafo com informações adicionais sobre a vacina e seus efeitos colaterais mínimos.</p>
  <div class="content-intertitle">Veja também</div>
</div>
</body></html>
"""

ESTADAO_HTML = """
<html><head><title>Estadao Test</title></head>
<body>
<h1>Economia brasileira cresce acima do esperado no trimestre</h1>
<div class="news-body">
  <p>O crescimento econômico superou as expectativas dos analistas no último trimestre do ano.</p>
  <p class="headlines">Manchete de ruído que deve ser ignorada pelo scraper</p>
  <p>Os dados do IBGE mostram avanço de 3,1% no PIB, impulsionado pelo setor de serviços.</p>
</div>
</body></html>
"""

FOLHA_HTML = """
<html><head><meta charset="utf-8"><title>Folha Test</title></head>
<body>
<h1>Eleições municipais registram recorde de candidaturas femininas</h1>
<div class="c-news__body">
  <p>O número de mulheres candidatas em 2026 bateu o recorde histórico, com aumento de 15% em relação ao pleito anterior.</p>
  <p class="some-noise-class">Texto com classe deve ser ignorado pelo scraper</p>
  <p>A participação feminina cresceu em todas as regiões do país, especialmente no Nordeste e Centro-Oeste.</p>
</div>
</body></html>
"""

AOSFATOS_HTML = """
<html><head><title>Aos Fatos Test</title></head>
<body>
<h1>É falso que vacinas causam autismo em crianças</h1>
<div class="prose">
  <p>A afirmação de que vacinas causam autismo é falsa e já foi desmentida por diversos estudos científicos.</p>
  <p class="text-sm text-gray-500">Texto com classes Tailwind deve ser ignorado</p>
  <p>O estudo original de 1998 que fazia essa associação foi retratado pela revista Lancet por fraude.</p>
</div>
</body></html>
"""

EMPTY_BODY_HTML = """
<html><head><title>Empty</title></head>
<body><h1>Title Only</h1><div class="content-text"></div></body></html>
"""


# ---------------------------------------------------------------------------
# 1. detectPlatform routing tests
# ---------------------------------------------------------------------------

class TestDetectPlatform:
    """verify URL routing to correct platform types"""

    def test_g1(self):
        assert detectPlatform("https://g1.globo.com/sp/noticia.ghtml") == PlatformType.G1

    def test_g1_with_path(self):
        assert detectPlatform("https://g1.globo.com/economia/noticia/2026/02/article.ghtml") == PlatformType.G1

    def test_estadao(self):
        assert detectPlatform("https://www.estadao.com.br/politica/artigo") == PlatformType.ESTADAO

    def test_estadao_bare(self):
        assert detectPlatform("https://estadao.com.br/economia/artigo") == PlatformType.ESTADAO

    def test_folha_www1(self):
        assert detectPlatform("https://www1.folha.uol.com.br/cotidiano/artigo.shtml") == PlatformType.FOLHA

    def test_folha_www(self):
        assert detectPlatform("https://www.folha.uol.com.br/cotidiano/artigo.shtml") == PlatformType.FOLHA

    def test_aosfatos(self):
        assert detectPlatform("https://www.aosfatos.org/noticias/check") == PlatformType.AOSFATOS

    def test_generic_unchanged(self):
        assert detectPlatform("https://bbc.com/news/article") == PlatformType.GENERIC

    def test_facebook_still_works(self):
        assert detectPlatform("https://www.facebook.com/post/123") == PlatformType.FACEBOOK

    def test_instagram_still_works(self):
        assert detectPlatform("https://instagram.com/p/abc123") == PlatformType.INSTAGRAM

    def test_twitter_still_works(self):
        assert detectPlatform("https://x.com/user/status/123") == PlatformType.TWITTER

    def test_tiktok_still_works(self):
        assert detectPlatform("https://www.tiktok.com/@user/video/123") == PlatformType.TIKTOK


# ---------------------------------------------------------------------------
# 2. scraper extraction tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestG1Scraper:

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_successful_extraction(self, mock_traf, mock_session):
        mock_resp = MagicMock()
        mock_resp.text = G1_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_traf.extract.return_value = "trafilatura fallback text"

        result = scrape_g1_article("https://g1.globo.com/test/article.ghtml")

        assert result["success"] is True
        assert "Vacina contra gripe" in result["metadata"]["title"]
        assert result["metadata"]["extraction_tool"] == "g1_scraper"
        assert "primeiro parágrafo" in result["content"]
        assert result["error"] is None

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_stops_at_nav_marker(self, mock_traf, mock_session):
        mock_resp = MagicMock()
        mock_resp.text = G1_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_traf.extract.return_value = ""

        result = scrape_g1_article("https://g1.globo.com/test/article.ghtml")
        # "Veja também" intertitle should stop extraction
        assert "Veja também" not in result["content"]

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_falls_back_to_trafilatura(self, mock_traf, mock_session):
        mock_resp = MagicMock()
        mock_resp.text = EMPTY_BODY_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_traf.extract.return_value = "A" * 60  # long enough to pass threshold

        result = scrape_g1_article("https://g1.globo.com/test/article.ghtml")
        assert result["success"] is True
        assert result["content"] == "A" * 60

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    def test_http_error(self, mock_session):
        mock_session.get.side_effect = Exception("connection refused")

        result = scrape_g1_article("https://g1.globo.com/test/article.ghtml")
        assert result["success"] is False
        assert "connection refused" in result["error"]


class TestEstadaoScraper:

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_successful_extraction(self, mock_traf, mock_session):
        mock_resp = MagicMock()
        mock_resp.text = ESTADAO_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_traf.extract.return_value = ""

        result = scrape_estadao_article("https://www.estadao.com.br/test")

        assert result["success"] is True
        assert "Economia brasileira" in result["metadata"]["title"]
        assert result["metadata"]["extraction_tool"] == "estadao_scraper"
        assert "crescimento econômico" in result["content"]
        # noise class paragraph should be excluded
        assert "Manchete de ruído" not in result["content"]
        assert result["error"] is None

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    def test_http_error(self, mock_session):
        mock_session.get.side_effect = Exception("timeout")

        result = scrape_estadao_article("https://www.estadao.com.br/test")
        assert result["success"] is False
        assert "timeout" in result["error"]


class TestFolhaScraper:

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_successful_extraction(self, mock_traf, mock_session):
        mock_resp = MagicMock()
        mock_resp.content = FOLHA_HTML.encode("utf-8")
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_traf.extract.return_value = ""

        result = scrape_folha_article("https://www1.folha.uol.com.br/cotidiano/test.shtml")

        assert result["success"] is True
        assert "Eleições municipais" in result["metadata"]["title"]
        assert result["metadata"]["extraction_tool"] == "folha_scraper"
        assert "recorde histórico" in result["content"]
        # noise class paragraph should be excluded
        assert "deve ser ignorado" not in result["content"]

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_url_normalization(self, mock_traf, mock_session):
        mock_resp = MagicMock()
        mock_resp.content = FOLHA_HTML.encode("utf-8")
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_traf.extract.return_value = ""

        # www. should be normalized to www1.
        scrape_folha_article("https://www.folha.uol.com.br/test.shtml")
        call_url = mock_session.get.call_args[0][0]
        assert "www1.folha.uol.com.br" in call_url

    @patch("app.agentic_ai.context.web.news_scrapers._SESSION")
    def test_http_error(self, mock_session):
        mock_session.get.side_effect = Exception("ssl error")

        result = scrape_folha_article("https://www1.folha.uol.com.br/test")
        assert result["success"] is False


class TestAosFatosScraper:

    @patch("app.agentic_ai.context.web.news_scrapers._fetch_aosfatos")
    @patch("app.agentic_ai.context.web.news_scrapers.trafilatura")
    def test_successful_extraction(self, mock_traf, mock_fetch):
        mock_fetch.return_value = (AOSFATOS_HTML, 200)
        mock_traf.extract.return_value = ""

        result = scrape_aosfatos_article("https://www.aosfatos.org/noticias/test")

        assert result["success"] is True
        assert "falso que vacinas" in result["metadata"]["title"]
        assert result["metadata"]["extraction_tool"] == "aosfatos_scraper"
        assert "desmentida por diversos estudos" in result["content"]
        # noise class paragraph should be excluded
        assert "deve ser ignorado" not in result["content"]

    @patch("app.agentic_ai.context.web.news_scrapers._fetch_aosfatos")
    def test_http_error(self, mock_fetch):
        mock_fetch.side_effect = Exception("UNEXPECTED_EOF")

        result = scrape_aosfatos_article("https://www.aosfatos.org/test")
        assert result["success"] is False
        assert "UNEXPECTED_EOF" in result["error"]


# ---------------------------------------------------------------------------
# 3. _build_result helper
# ---------------------------------------------------------------------------

class TestBuildResult:

    def test_prefers_body_text(self):
        body = "body content that is long enough to pass the minimum threshold check"
        result = _build_result(body, "trafilatura text", "Title", "test")
        assert result["success"] is True
        assert result["content"] == body

    def test_falls_back_to_trafilatura(self):
        result = _build_result("", "trafilatura fallback that is long enough to pass the threshold", "Title", "test")
        assert result["success"] is True
        assert "trafilatura fallback" in result["content"]

    def test_fails_when_both_empty(self):
        result = _build_result("", "", "Title", "test")
        assert result["success"] is False

    def test_fails_when_content_too_short(self):
        result = _build_result("short", "", "Title", "test")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# 4. scrapeGenericUrl integration (routing + fallback)
# ---------------------------------------------------------------------------

class TestScrapeGenericUrlRouting:

    @pytest.mark.asyncio
    @patch("app.agentic_ai.context.web.apify_utils.scrape_g1_article")
    async def test_routes_g1(self, mock_scraper):
        mock_scraper.return_value = {
            "success": True, "content": "g1 content", "metadata": {"extraction_tool": "g1_scraper"}, "error": None
        }

        result = await scrapeGenericUrl("https://g1.globo.com/test/article.ghtml")

        mock_scraper.assert_called_once_with("https://g1.globo.com/test/article.ghtml")
        assert result["success"] is True
        assert result["metadata"]["extraction_tool"] == "g1_scraper"

    @pytest.mark.asyncio
    @patch("app.agentic_ai.context.web.apify_utils.scrape_estadao_article")
    async def test_routes_estadao(self, mock_scraper):
        mock_scraper.return_value = {
            "success": True, "content": "estadao content", "metadata": {"extraction_tool": "estadao_scraper"}, "error": None
        }

        result = await scrapeGenericUrl("https://www.estadao.com.br/politica/test")

        mock_scraper.assert_called_once()
        assert result["metadata"]["extraction_tool"] == "estadao_scraper"

    @pytest.mark.asyncio
    @patch("app.agentic_ai.context.web.apify_utils.scrape_folha_article")
    async def test_routes_folha(self, mock_scraper):
        mock_scraper.return_value = {
            "success": True, "content": "folha content", "metadata": {"extraction_tool": "folha_scraper"}, "error": None
        }

        result = await scrapeGenericUrl("https://www1.folha.uol.com.br/test.shtml")

        mock_scraper.assert_called_once()
        assert result["metadata"]["extraction_tool"] == "folha_scraper"

    @pytest.mark.asyncio
    @patch("app.agentic_ai.context.web.apify_utils.scrape_aosfatos_article")
    async def test_routes_aosfatos(self, mock_scraper):
        mock_scraper.return_value = {
            "success": True, "content": "aosfatos content", "metadata": {"extraction_tool": "aosfatos_scraper"}, "error": None
        }

        result = await scrapeGenericUrl("https://www.aosfatos.org/noticias/test")

        mock_scraper.assert_called_once()
        assert result["metadata"]["extraction_tool"] == "aosfatos_scraper"

    @pytest.mark.asyncio
    @patch("app.agentic_ai.context.web.apify_utils.scrapeGenericSimple")
    @patch("app.agentic_ai.context.web.apify_utils.scrape_g1_article")
    async def test_fallback_on_scraper_failure(self, mock_g1, mock_generic):
        mock_g1.return_value = {
            "success": False, "content": "", "metadata": {}, "error": "extraction failed"
        }
        mock_generic.return_value = {
            "success": True, "content": "generic content", "metadata": {"scraping_method": "simple_http"}, "error": None
        }

        result = await scrapeGenericUrl("https://g1.globo.com/test/article.ghtml")

        mock_g1.assert_called_once()
        mock_generic.assert_called_once()
        assert result["success"] is True
        assert result["metadata"]["scraping_method"] == "simple_http"
