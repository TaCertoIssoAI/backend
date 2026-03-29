"""tests for adjudication prompt instructions."""

from app.ai.pipeline.prompts import ADJUDICATION_SYSTEM_PROMPT


def test_prompt_requires_extremely_concise_overall_summary_with_ai_signal():
    assert "EXTREMAMENTE conciso" in ADJUDICATION_SYSTEM_PROMPT
    assert "A PRIMEIRA frase do sumário deve destacar explicitamente se há indícios de conteúdo gerado por IA" in ADJUDICATION_SYSTEM_PROMPT


def test_prompt_requires_more_concise_claim_justification():
    assert "justification: sua explicação um pouco mais concisa" in ADJUDICATION_SYSTEM_PROMPT
