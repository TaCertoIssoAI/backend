"""tests for retry_system_prompt builder."""

from app.agentic_ai.prompts.retry_system_prompt import build_retry_system_prompt


def test_retry_prompt_contains_retry_context():
    prompt = build_retry_system_prompt(
        iteration_count=1,
        retry_context="## Queries ja utilizadas\n1. old query",
        max_iterations=3,
    )
    assert "Queries ja utilizadas" in prompt
    assert "old query" in prompt


def test_retry_prompt_does_not_contain_data_sources():
    """data sources are sent as HumanMessages, not in the system prompt."""
    prompt = build_retry_system_prompt(
        iteration_count=0,
        retry_context="retry info",
        max_iterations=3,
    )
    assert "usuario enviara" in prompt


def test_retry_prompt_omits_collected_sources():
    """retry prompt should not have a 'Fontes ja coletadas' section."""
    prompt = build_retry_system_prompt(
        iteration_count=0,
        retry_context="context",
        max_iterations=3,
    )
    assert "Fontes ja coletadas" not in prompt
    assert "SEGUNDA TENTATIVA" in prompt
