import pytest

from app.agentic_ai.nodes.format_input import extract_links


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
