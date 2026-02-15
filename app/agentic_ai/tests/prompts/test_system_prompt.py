"""tests for system_prompt builder."""

from app.agentic_ai.prompts.system_prompt import build_system_prompt


def test_prompt_contains_iteration_info():
    prompt = build_system_prompt(
        formatted_data_sources="test content",
        iteration_count=2,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
        max_iterations=5,
    )
    assert "2/5" in prompt


def test_prompt_contains_data_sources():
    prompt = build_system_prompt(
        formatted_data_sources="Vacina X causa autismo",
        iteration_count=0,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "Vacina X causa autismo" in prompt


def test_prompt_contains_tool_descriptions():
    prompt = build_system_prompt(
        formatted_data_sources="test",
        iteration_count=0,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "search_fact_check_api" in prompt
    assert "search_web" in prompt
    assert "scrape_pages" in prompt


def test_prompt_contains_stop_criteria():
    prompt = build_system_prompt(
        formatted_data_sources="test",
        iteration_count=0,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "Muito confiável" in prompt
    assert "SUFICIENTES" in prompt
