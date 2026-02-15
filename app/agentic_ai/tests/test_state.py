"""tests for state schema and reducers."""

from app.agentic_ai.state import _merge_search_results
from app.models.agenticai import GoogleSearchContext, SourceReliability


def _make_entry(id="1"):
    return GoogleSearchContext(
        id=id,
        url="https://test.com",
        parent_id=None,
        reliability=SourceReliability.NEUTRO,
        title="test",
        snippet="snippet",
        domain="test.com",
    )


def test_merge_search_results_empty():
    result = _merge_search_results({}, {})
    assert result == {}


def test_merge_search_results_new_key():
    existing = {"geral": [_make_entry("1")]}
    new = {"g1": [_make_entry("2")]}
    result = _merge_search_results(existing, new)
    assert len(result["geral"]) == 1
    assert len(result["g1"]) == 1


def test_merge_search_results_extends_existing():
    existing = {"geral": [_make_entry("1")]}
    new = {"geral": [_make_entry("2")]}
    result = _merge_search_results(existing, new)
    assert len(result["geral"]) == 2


def test_merge_does_not_mutate_existing():
    e1 = _make_entry("1")
    existing = {"geral": [e1]}
    new = {"geral": [_make_entry("2")]}
    _merge_search_results(existing, new)
    assert len(existing["geral"]) == 1
