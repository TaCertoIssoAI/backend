"""tests for system_prompt builder."""

from app.agentic_ai.prompts.system_prompt import build_system_prompt


def test_prompt_contains_iteration_info():
    prompt = build_system_prompt(
        iteration_count=2,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
        max_iterations=5,
    )
    assert "2/5" in prompt


def test_prompt_does_not_contain_data_sources():
    """data sources are now sent as HumanMessages, not in the system prompt."""
    prompt = build_system_prompt(
        iteration_count=0,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "usuário enviará" in prompt


def test_prompt_contains_tool_descriptions():
    prompt = build_system_prompt(
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
        iteration_count=0,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "Muito confiável" in prompt
    assert "SUFICIENTES" in prompt


def test_prompt_contains_adaptive_strategy():
    prompt = build_system_prompt(
        iteration_count=0,
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "_summary" in prompt
    assert "POUCOS resultados" in prompt
