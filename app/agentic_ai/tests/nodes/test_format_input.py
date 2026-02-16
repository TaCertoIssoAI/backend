"""tests for the format_input node."""

from unittest.mock import patch, AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from app.models.commondata import DataSource
from app.agentic_ai.nodes.format_input import (
    _format_data_sources,
    _is_links_only,
    format_input_node,
)


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

    assert "=== Entrada A ===" in result
    assert "=== Entrada B ===" in result
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


def test_format_data_sources_link_context_priority_header():
    ds_text = DataSource(id="t-1", source_type="original_text", original_text="text")
    ds_link = DataSource(id="l-1", source_type="link_context", original_text="link content")
    result = _format_data_sources([ds_text, ds_link])
    assert "PRIORIDADE ALTA" in result
    # original_text source should NOT have priority header
    assert result.index("PRIORIDADE ALTA") > result.index("=== Entrada A ===")


# --- _is_links_only ---


def test_is_links_only_true():
    assert _is_links_only("https://a.com https://b.com", ["https://a.com", "https://b.com"])


def test_is_links_only_true_with_whitespace():
    assert _is_links_only("  https://a.com  ", ["https://a.com"])


def test_is_links_only_false():
    assert not _is_links_only("check this https://a.com", ["https://a.com"])


# --- format_input_node (async) ---

@pytest.mark.asyncio
async def test_format_input_node_reads_state():
    ds = DataSource(id="ds-1", source_type="original_text", original_text="test claim")
    state = {"data_sources": [ds]}
    result = await format_input_node(state)

    assert "formatted_data_sources" in result
    assert result["formatted_data_sources"] == ds.to_llm_string()

    # should include a HumanMessage with the formatted content
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert "test claim" in result["messages"][0].content


@pytest.mark.asyncio
async def test_format_input_node_empty_state():
    state = {"data_sources": []}
    result = await format_input_node(state)
    assert result["formatted_data_sources"] == "(nenhum conteudo fornecido)"
    assert isinstance(result["messages"][0], HumanMessage)


@pytest.mark.asyncio
async def test_format_input_node_missing_key():
    state = {}
    result = await format_input_node(state)
    assert result["formatted_data_sources"] == "(nenhum conteudo fornecido)"
    assert isinstance(result["messages"][0], HumanMessage)


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.format_input.fire_link_expansion", return_value=1)
@patch("app.agentic_ai.nodes.format_input.extract_links", return_value=["https://a.com"])
async def test_format_input_fires_link_expansion(mock_extract, mock_fire):
    ds = DataSource(
        id="ds-1", source_type="original_text",
        original_text="some claim https://a.com",
    )
    state = {"data_sources": [ds]}
    result = await format_input_node(state)

    mock_fire.assert_called_once()
    assert result.get("pending_async_count") == 1
    assert "run_id" in result


@pytest.mark.asyncio
async def test_format_input_no_links_no_pending():
    ds = DataSource(id="ds-1", source_type="original_text", original_text="just text")
    state = {"data_sources": [ds]}
    result = await format_input_node(state)

    assert "pending_async_count" not in result
    assert "formatted_data_sources" in result


@pytest.mark.asyncio
@patch(
    "app.agentic_ai.nodes.format_input.expand_all_links",
    new_callable=AsyncMock,
    return_value=[
        DataSource(
            id="link-1", source_type="link_context",
            original_text="expanded content",
            metadata={"url": "https://a.com"},
        )
    ],
)
@patch("app.agentic_ai.nodes.format_input.extract_links", return_value=["https://a.com"])
async def test_format_input_links_only_blocks(mock_extract, mock_expand):
    ds = DataSource(
        id="ds-1", source_type="original_text",
        original_text="https://a.com",
    )
    state = {"data_sources": [ds]}
    result = await format_input_node(state)

    mock_expand.assert_called_once()
    # links-only should NOT set pending_async_count
    assert "pending_async_count" not in result
    # should have data_sources with expanded content
    assert len(result.get("data_sources", [])) == 1
    assert result["data_sources"][0].source_type == "link_context"


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.format_input.fire_link_expansion", return_value=1)
@patch("app.agentic_ai.nodes.format_input.extract_links", return_value=["https://a.com"])
async def test_format_input_text_plus_links_fires(mock_extract, mock_fire):
    ds = DataSource(
        id="ds-1", source_type="original_text",
        original_text="claim text https://a.com",
    )
    state = {"data_sources": [ds]}
    result = await format_input_node(state)

    mock_fire.assert_called_once()
    assert result.get("pending_async_count") == 1
