"""tests for the async link expander module."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.agentic_ai.nodes.link_expander import (
    _scrape_single_url,
    expand_all_links,
    fire_link_expansion,
    await_link_expansion,
    _pending_link_tasks,
)
from app.agentic_ai.config import MAX_LINKS_TO_EXPAND
from app.ai.context.web.models import WebContentResult
from app.models.commondata import DataSource


def _make_web_result(success=True, content="article text", url="https://example.com"):
    return WebContentResult(
        success=success,
        url=url,
        content=content,
        content_length=len(content) if content else 0,
    )


# --- _scrape_single_url ---


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_scrape_single_url_success(mock_expand):
    mock_expand.return_value = _make_web_result()
    ds = await _scrape_single_url("https://example.com", "parent-1", "pt-BR", None)

    assert ds is not None
    assert isinstance(ds, DataSource)
    assert ds.source_type == "link_context"
    assert ds.original_text == "article text"
    assert ds.metadata["url"] == "https://example.com"
    assert ds.metadata["parent_source_id"] == "parent-1"


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_scrape_single_url_timeout(mock_expand):
    async def slow_expand(url):
        await asyncio.sleep(10)
        return _make_web_result()

    mock_expand.side_effect = slow_expand
    ds = await _scrape_single_url("https://example.com", "p-1", "pt-BR", None, timeout=0.01)
    assert ds is None


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_scrape_single_url_exception(mock_expand):
    mock_expand.side_effect = RuntimeError("network error")
    ds = await _scrape_single_url("https://example.com", "p-1", "pt-BR", None)
    assert ds is None


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_scrape_single_url_no_content(mock_expand):
    mock_expand.return_value = _make_web_result(success=True, content="")
    ds = await _scrape_single_url("https://example.com", "p-1", "pt-BR", None)
    assert ds is None


# --- expand_all_links ---


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_expand_all_links_concurrent(mock_expand):
    mock_expand.return_value = _make_web_result()

    urls = ["https://a.com", "https://b.com", "https://c.com"]
    results = await expand_all_links(urls, "parent-1", "pt-BR", None)

    assert len(results) == 3
    assert all(isinstance(r, DataSource) for r in results)
    assert mock_expand.call_count == 3


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_expand_all_links_partial_failure(mock_expand):
    """1 of 2 links fails, the successful one is still returned."""

    async def side_effect(url):
        if "fail" in url:
            raise RuntimeError("scrape failed")
        return _make_web_result(url=url)

    mock_expand.side_effect = side_effect

    results = await expand_all_links(
        ["https://good.com", "https://fail.com"], "p-1", "pt-BR", None
    )
    assert len(results) == 1
    assert results[0].metadata["url"] == "https://good.com"


# --- fire_link_expansion + await_link_expansion ---


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_fire_and_await_roundtrip(mock_expand):
    mock_expand.return_value = _make_web_result()

    run_id = "test-run-1"
    count = fire_link_expansion(run_id, ["https://a.com"], "p-1", "pt-BR", None)
    assert count == 1
    assert run_id in _pending_link_tasks

    results = await await_link_expansion(run_id)
    assert len(results) == 1
    assert isinstance(results[0], DataSource)
    # task should be removed from registry
    assert run_id not in _pending_link_tasks


@pytest.mark.asyncio
async def test_await_missing_run_id():
    results = await await_link_expansion("nonexistent-run-id")
    assert results == []


@pytest.mark.asyncio
async def test_await_task_exception():
    """if the task raises, await_link_expansion returns [] instead of propagating."""
    run_id = "test-exception-run"

    async def failing():
        raise RuntimeError("boom")

    _pending_link_tasks[run_id] = asyncio.create_task(failing())
    # let the task start
    await asyncio.sleep(0)

    results = await await_link_expansion(run_id)
    assert results == []
    assert run_id not in _pending_link_tasks


def test_fire_with_no_urls():
    count = fire_link_expansion("empty-run", [], "p-1", "pt-BR", None)
    assert count == 0
    assert "empty-run" not in _pending_link_tasks


@pytest.mark.asyncio
@patch("app.agentic_ai.nodes.link_expander.expand_link_context")
async def test_max_links_respected(mock_expand):
    mock_expand.return_value = _make_web_result()

    urls = [f"https://site{i}.com" for i in range(20)]
    results = await expand_all_links(urls, "p-1", "pt-BR", None)

    assert len(results) == MAX_LINKS_TO_EXPAND
    assert mock_expand.call_count == MAX_LINKS_TO_EXPAND
