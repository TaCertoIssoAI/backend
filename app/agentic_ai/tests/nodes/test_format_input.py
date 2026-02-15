"""tests for the format_input node."""

import pytest

from app.models.commondata import DataSource
from app.agentic_ai.nodes.format_input import _format_data_sources, format_input_node


# --- _format_data_sources ---

def test_empty_list_returns_placeholder():
    result = _format_data_sources([])
    assert result == "(nenhum conteudo fornecido)"


def test_single_data_source_returns_to_llm_string():
    ds = DataSource(id="ds-1", source_type="original_text", original_text="some claim")
    result = _format_data_sources([ds])
    assert result == ds.to_llm_string()
    # should NOT contain numbered section headers
    assert "=== Fonte" not in result


def test_multiple_data_sources_numbered():
    ds1 = DataSource(id="ds-1", source_type="original_text", original_text="claim A")
    ds2 = DataSource(id="ds-2", source_type="link_context", original_text="article text")
    result = _format_data_sources([ds1, ds2])

    assert "=== Fonte 1 ===" in result
    assert "=== Fonte 2 ===" in result
    assert ds1.to_llm_string() in result
    assert ds2.to_llm_string() in result


def test_data_source_with_metadata():
    ds = DataSource(
        id="ds-meta",
        source_type="link_context",
        original_text="article content",
        metadata={"title": "Test Article", "url": "https://example.com"},
    )
    result = _format_data_sources([ds])
    assert "Test Article" in result
    assert "https://example.com" in result


def test_different_source_types():
    ds_text = DataSource(id="t-1", source_type="original_text", original_text="text")
    ds_link = DataSource(id="l-1", source_type="link_context", original_text="link")
    ds_image = DataSource(id="i-1", source_type="image", original_text="ocr text")

    for ds in [ds_text, ds_link, ds_image]:
        result = _format_data_sources([ds])
        assert ds.source_type in result


# --- format_input_node (async) ---

@pytest.mark.asyncio
async def test_format_input_node_reads_state():
    ds = DataSource(id="ds-1", source_type="original_text", original_text="test claim")
    state = {"data_sources": [ds]}
    result = await format_input_node(state)

    assert "formatted_data_sources" in result
    assert result["formatted_data_sources"] == ds.to_llm_string()


@pytest.mark.asyncio
async def test_format_input_node_empty_state():
    state = {"data_sources": []}
    result = await format_input_node(state)
    assert result["formatted_data_sources"] == "(nenhum conteudo fornecido)"


@pytest.mark.asyncio
async def test_format_input_node_missing_key():
    state = {}
    result = await format_input_node(state)
    assert result["formatted_data_sources"] == "(nenhum conteudo fornecido)"
