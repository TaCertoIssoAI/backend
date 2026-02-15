import pytest

# Configure pytest to automatically handle async tests
pytest_plugins = ('pytest_asyncio',)

from app.ai.pipeline.link_context_expander import (
    extract_links,
    expand_link_contexts,
)
from app.models import DataSource
from app.config import get_default_pipeline_config
from app.ai.threads.thread_utils import ThreadPoolManager


_cfg = get_default_pipeline_config()


@pytest.fixture(scope="session", autouse=True)
def initialize_thread_pool():
    """initialize ThreadPoolManager once for all tests"""
    manager = ThreadPoolManager.get_instance()
    manager.initialize()
    yield
    # cleanup after all tests
    manager.shutdown()

# ===== UNIT TESTS FOR extract_links =====

def test_extract_single_https_url():
    """should extract a single https URL from text"""
    text = "Check out this article at https://example.com for more info."
    result = extract_links(text)
    assert result == ["https://example.com"]


def test_extract_single_http_url():
    """should extract a single http URL from text"""
    text = "Visit http://test.org to learn more."
    result = extract_links(text)
    assert result == ["http://test.org"]


def test_extract_multiple_urls():
    """should extract multiple URLs from text"""
    text = "Check https://example.com and http://test.org for details."
    result = extract_links(text)
    assert result == ["https://example.com", "http://test.org"]


def test_extract_urls_with_paths():
    """should extract URLs with paths and query parameters"""
    text = "See https://example.com/article/123?ref=social and http://test.org/page"
    result = extract_links(text)
    assert result == ["https://example.com/article/123?ref=social", "http://test.org/page"]


def test_remove_duplicate_urls():
    """should remove duplicate URLs while preserving order"""
    text = "Visit https://example.com and also https://example.com again."
    result = extract_links(text)
    assert result == ["https://example.com"]
    assert len(result) == 1


def test_empty_text():
    """should return empty list for empty text"""
    result = extract_links("")
    assert result == []


def test_text_without_urls():
    """should return empty list when no URLs are present"""
    text = "This is just plain text with no links at all."
    result = extract_links(text)
    assert result == []


def test_url_with_special_characters():
    """should handle URLs with hyphens, underscores and other valid chars"""
    text = "Check https://my-site.example.com/path_to/resource-123"
    result = extract_links(text)
    assert result == ["https://my-site.example.com/path_to/resource-123"]


def test_multiple_urls_preserves_order():
    """should preserve the order of URLs as they appear in text"""
    text = "First https://first.com then https://second.com and https://third.com"
    result = extract_links(text)
    assert result == ["https://first.com", "https://second.com", "https://third.com"]


def test_url_at_end_of_sentence():
    """should extract URL that ends with punctuation"""
    text = "Visit our website at https://example.com."
    result = extract_links(text)
    # the dot should not be part of the URL
    assert result == ["https://example.com"]


def test_url_in_parentheses():
    """should extract URL surrounded by parentheses"""
    text = "See the docs (https://docs.example.com) for details."
    result = extract_links(text)
    assert result == ["https://docs.example.com"]


def test_multiple_protocols_mixed():
    """should handle mix of http and https URLs"""
    text = "http://old.example.com and https://secure.example.com"
    result = extract_links(text)
    assert result == ["http://old.example.com", "https://secure.example.com"]


def test_url_with_port():
    """should extract URLs with port numbers"""
    text = "Connect to https://localhost:8080/api for testing."
    result = extract_links(text)
    assert result == ["https://localhost:8080/api"]


def test_url_with_fragment():
    """should extract URLs with fragments/anchors"""
    text = "Jump to https://example.com/page#section-2 directly."
    result = extract_links(text)
    assert result == ["https://example.com/page#section-2"]


def test_multiline_text_with_urls():
    """should extract URLs from multiline text"""
    text = """First line with https://example.com
    Second line with http://test.org
    Third line with https://another.com"""
    result = extract_links(text)
    assert result == ["https://example.com", "http://test.org", "https://another.com"]


def test_urls_without_protocol_not_extracted():
    """should not extract URLs without http/https protocol"""
    text = "Visit www.example.com or example.com for info."
    result = extract_links(text)
    assert result == []


def test_real_world_whatsapp_message():
    """should handle typical WhatsApp message with URLs"""
    text = "Olha essa notícia importante: https://g1.globo.com/economia/noticia.html compartilha aí!"
    result = extract_links(text)
    assert result == ["https://g1.globo.com/economia/noticia.html"]


# ===== INTEGRATION TESTS FOR WEB SCRAPING =====
# expand_link_context tests moved to app/agentic_ai/tests/nodes/test_link_expander.py

@pytest.mark.asyncio
async def test_expand_link_contexts_with_multiple_real_urls():
    """should extract and expand multiple real URLs from DataSource"""
    # create a DataSource with text containing multiple URLs
    text = """
    Veja essas notícias importantes:

    1. Arsenal encontrado em SP: https://g1.globo.com/sp/sao-paulo/noticia/2025/11/16/policia-encontra-arsenal-de-guerra-na-zona-sul-de-sp.ghtml

    2. Cúpula dos Povos em Belém: https://www.cnnbrasil.com.br/nacional/em-belem-cupula-dos-povos-cobra-participacao-popular-nas-acoes-climaticas/

    3. Arte moderna na BBC: https://www.bbc.com/culture/article/20251112-why-this-1768-painting-could-be-the-real-birth-of-modern-art
    """

    data_source = DataSource(
        id="msg-test-001",
        source_type="original_text",
        original_text=text,
        locale="pt-BR"
    )

    # expand all links
    expanded_sources = await expand_link_contexts(data_source,_cfg)

    # validate results
    assert len(expanded_sources) == 3, f"Expected 3 expanded sources, got {len(expanded_sources)}"

    # validate each expanded source
    for i, source in enumerate(expanded_sources, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPANDED SOURCE {i}")
        print(f"{'=' * 80}")

        assert source.source_type == "link_context"
        assert source.metadata["parent_source_id"] == "msg-test-001"
        assert "url" in source.metadata
        assert source.metadata["success"] is True, f"Source {i} scraping failed"
        assert source.original_text != "", f"Source {i} content is empty"
        assert source.metadata["content_length"] > 0

        print(f"ID: {source.id}")
        print(f"URL: {source.metadata['url']}")
        print(f"Success: {source.metadata['success']}")
        print(f"Content length: {source.metadata['content_length']} chars")
        print(f"Content preview (first 150 chars):\n{source.original_text[:150]}...")
        print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_expand_link_contexts_no_links():
    """should return empty list when DataSource has no links"""
    data_source = DataSource(
        id="msg-no-links",
        source_type="original_text",
        original_text="This is just plain text with no URLs at all."
    )

    expanded_sources = await expand_link_contexts(data_source,_cfg)

    assert expanded_sources == []
    assert len(expanded_sources) == 0


@pytest.mark.asyncio
async def test_expand_link_contexts_validates_source_type():
    """should raise ValueError if DataSource is not original_text type"""
    data_source = DataSource(
        id="link-001",
        source_type="link_context",  # wrong type!
        original_text="Some text with https://example.com"
    )

    with pytest.raises(ValueError) as exc_info:
        await expand_link_contexts(data_source,_cfg)

    assert "original_text" in str(exc_info.value)
    assert "link_context" in str(exc_info.value)


@pytest.mark.asyncio
async def test_expand_link_context_preserves_locale_and_timestamp():
    """should preserve locale and timestamp from original DataSource"""
    text = "Check this: https://g1.globo.com/sp/sao-paulo/noticia/2025/11/16/policia-encontra-arsenal-de-guerra-na-zona-sul-de-sp.ghtml"

    data_source = DataSource(
        id="msg-locale-test",
        source_type="original_text",
        original_text=text,
        locale="en-US",
        timestamp="2025-11-16T10:30:00Z"
    )

    expanded_sources = await expand_link_contexts(data_source,_cfg)

    assert len(expanded_sources) == 1
    expanded = expanded_sources[0]

    assert expanded.locale == "en-US"
    assert expanded.timestamp == "2025-11-16T10:30:00Z"


@pytest.mark.asyncio
async def test_expand_link_contexts_single_url():
    """should handle DataSource with single URL"""
    text = "Veja esta notícia: https://g1.globo.com/sp/sao-paulo/noticia/2025/11/16/policia-encontra-arsenal-de-guerra-na-zona-sul-de-sp.ghtml"

    data_source = DataSource(
        id="msg-single",
        source_type="original_text",
        original_text=text
    )

    expanded_sources = await expand_link_contexts(data_source,_cfg)

    assert len(expanded_sources) == 1
    assert expanded_sources[0].metadata["success"] is True
    assert expanded_sources[0].original_text != ""


@pytest.mark.asyncio
async def test_expand_link_contexts_timeout_with_nonexistent_site():
    """should handle timeout gracefully when scraping takes too long or site doesn't exist"""
    from app.models import PipelineConfig, LLMConfig, TimeoutConfig
    from langchain_openai import ChatOpenAI

    # create a config with very short timeouts
    short_timeout_config = PipelineConfig(
        claim_extraction_llm_config=LLMConfig(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, timeout=30.0)
        ),
        adjudication_llm_config=LLMConfig(
            llm=ChatOpenAI(model="o3-mini", timeout=60.0)
        ),
        timeout_config=TimeoutConfig(
            link_content_expander_timeout_per_link=2.0,  # very short timeout
            link_content_expander_timeout_total=5.0,     # very short total timeout
            claim_extractor_timeout_per_source=10.0,
            claim_extractor_timeout_total=20.0,
            evidence_retrieval_timeout_per_claim=20.0,
            evidence_retrieval_timeout_total=40.0,
            adjudication_timeout=20.0
        ),
        max_links_to_expand=5,
        max_claims_to_extract=10,
        max_evidence_sources_per_claim=5
    )

    # create a DataSource with non-existent sites
    text = """
    Check these sites:
    http://this-site-definitely-does-not-exist-12345.com
    http://another-fake-site-that-will-timeout-67890.net
    http://third-nonexistent-domain-99999.org
    """

    data_source = DataSource(
        id="msg-timeout-test",
        source_type="original_text",
        original_text=text
    )

    # expand links with short timeout config
    expanded_sources = await expand_link_contexts(data_source, short_timeout_config)

    # should return empty list or partial results due to timeouts
    # the function should handle timeouts gracefully and not crash
    assert isinstance(expanded_sources, list)

    # if any sources were expanded before timeout, they should have proper structure
    for source in expanded_sources:
        assert source.source_type == "link_context"
        assert "url" in source.metadata
        assert "parent_source_id" in source.metadata
        assert source.metadata["parent_source_id"] == "msg-timeout-test"

    # extract timeout values for printing
    timeout_cfg: TimeoutConfig = short_timeout_config.timeout_config
    per_link_timeout = timeout_cfg.link_content_expander_timeout_per_link
    total_timeout = timeout_cfg.link_content_expander_timeout_total

    print(f"\n{'=' * 80}")
    print(f"TEST: Timeout Handling with Non-Existent Sites")
    print(f"{'=' * 80}")
    print(f"Total expanded sources: {len(expanded_sources)}")
    print(f"Config timeout per link: {per_link_timeout}s")
    print(f"Config timeout total: {total_timeout}s")
    print(f"Result: Timeout handled gracefully without crashing")
    print(f"{'=' * 80}\n")


@pytest.mark.asyncio
async def test_parallel_expansion_with_social_media_and_news():
    """
    simple test: scrape 4 links in parallel and measure time.
    """
    import time
    from app.ai.threads.thread_utils import ThreadPoolManager, OperationType
    from app.ai.pipeline.link_context_expander import expand_link_context_sync

    # 4 links to test
    urls = [
        "https://www.facebook.com/share/p/1GXv2qwKbE/",
        "https://www.instagram.com/p/DRKnubCjgtM/?utm_source=ig_web_copy_link&igsh=NTc4MTIwNjQ2YQ==",
        "https://www.tiktok.com/@roteiro.em.dia/video/7572058473853226260?is_from_webapp=1&sender_device=pc",
        "https://noticias.uol.com.br/colunas/jamil-chade/2025/11/17/conselho-da-onu-aprova-plano-de-trump-para-gaza-com-tropas-internacionais.htm"
    ]

    print(f"\n{'=' * 80}")
    print(f"PARALLEL SCRAPING TEST - 4 LINKS")
    print(f"{'=' * 80}")
    for i, url in enumerate(urls, 1):
        platform = "Unknown"
        if "facebook.com" in url:
            platform = "Facebook"
        elif "instagram.com" in url:
            platform = "Instagram"
        elif "tiktok.com" in url:
            platform = "TikTok"
        elif "uol.com.br" in url:
            platform = "UOL"
        print(f"{i}. {platform}: {url[:60]}...")
    print(f"{'=' * 80}\n")

    # get thread pool manager
    manager = ThreadPoolManager.get_instance()
    if not manager._initialized:
        manager.initialize()

    print(f"ThreadPool: {manager.max_workers} workers")
    print(f"Starting parallel scraping...\n")

    # submit all 4 jobs at once
    start_time = time.time()

    futures = []
    for url in urls:
        future = manager.submit(
            OperationType.LINK_CONTEXT_EXPANDING,
            expand_link_context_sync,
            url,
            timeout_per_link=90.0  # 90s timeout per link (increased for TikTok)
        )
        futures.append(future)

    print(f"✓ Submitted {len(futures)} jobs to ThreadPool")
    print(f"⏳ Waiting for all to complete...\n")

    # wait for all results
    from app.ai.threads.thread_utils import wait_all
    try:
        results = wait_all(futures, timeout=120.0)
    except TimeoutError:
        print("❌ Timeout waiting for results")
        results = []

    end_time = time.time()
    elapsed = end_time - start_time

    # print results
    print(f"\n{'=' * 80}")
    print(f"RESULTS")
    print(f"{'=' * 80}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Results received: {len(results)}/{len(urls)}")
    print(f"{'=' * 80}\n")

    success_count = 0
    fail_count = 0

    for i, result in enumerate(results, 1):
        platform = "Unknown"
        url = urls[i-1]
        if "facebook.com" in url:
            platform = "Facebook"
        elif "instagram.com" in url:
            platform = "Instagram"
        elif "tiktok.com" in url:
            platform = "TikTok"
        elif "uol.com.br" in url:
            platform = "UOL"

        print(f"--- LINK {i}: {platform} ---")
        print(f"   URL: {url}")

        if result is None:
            print(f"❌ FAILED (returned None - check logs above for timeout/error)")
            fail_count += 1
        elif result.success:
            print(f"✅ SUCCESS")
            print(f"   Content: {result.content_length} chars")
            if result.content:
                print(f"   Preview: {result.content[:100]}...")
            if result.metadata:
                print(f"   Metadata: {result.metadata}")
            success_count += 1
        else:
            print(f"❌ FAILED")
            print(f"   Error: {result.error}")
            print(f"   Content length: {result.content_length}")
            if result.metadata:
                print(f"   Metadata: {result.metadata}")
            fail_count += 1
        print()

    print(f"{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Success: {success_count}/{len(urls)}")
    print(f"Failed: {fail_count}/{len(urls)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg per link: {elapsed/len(urls):.2f}s")
    print(f"{'=' * 80}\n")

    # simple assertion: at least 1 should succeed
    assert success_count > 0, "At least one link should scrape successfully"
