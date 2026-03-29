"""tests for deep-fake prompt building in adjudication_prompt.py."""

from app.agentic_ai.prompts.adjudication_prompt import (
    build_adjudication_prompt,
    _format_deep_fake_results,
    DEEP_FAKE_SYSTEM_BLOCK,
)


_SAMPLE_DEEP_FAKE_DATA = {
    "results": [
        {
            "label": "fake",
            "score": 0.8392,
            "model_used": "frame_sampler(InternVideo2)",
            "media_type": "video",
            "processing_time_ms": 1500,
        },
        {
            "label": "real",
            "score": 0.1608,
            "model_used": "frame_sampler(InternVideo2)",
            "media_type": "video",
            "processing_time_ms": 1500,
        },
        {
            "label": "fake",
            "score": 0.0052,
            "model_used": "VoiceGen (Dual-RawNet2)",
            "media_type": "audio",
            "processing_time_ms": 800,
        },
    ],
}


# ---- _format_deep_fake_results ----

def test_format_deep_fake_results_valid():
    result = _format_deep_fake_results(_SAMPLE_DEEP_FAKE_DATA)
    assert "video" in result
    assert "fake" in result
    assert "0.8392" in result
    assert "frame_sampler(InternVideo2)" in result
    lines = result.strip().split("\n")
    assert len(lines) == 3


def test_format_deep_fake_results_none():
    assert _format_deep_fake_results(None) == ""


def test_format_deep_fake_results_empty():
    assert _format_deep_fake_results({"results": []}) == ""


def test_format_deep_fake_results_missing_results_key():
    assert _format_deep_fake_results({}) == ""


# ---- build_adjudication_prompt without deep-fake ----

def test_prompt_without_deep_fake_no_deep_fake_content():
    system, user = build_adjudication_prompt(
        formatted_data_sources="some text",
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "Deep Fake" not in system
    assert "Deep Fake" not in user


# ---- build_adjudication_prompt with deep-fake ----

def test_prompt_with_deep_fake_system_contains_instructions():
    system, _ = build_adjudication_prompt(
        formatted_data_sources="some text",
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
        deep_fake_verification_result=_SAMPLE_DEEP_FAKE_DATA,
    )
    assert "Deep Fake" in system
    assert "Fora de Contexto" in system
    assert "automatizada" in system


def test_prompt_with_deep_fake_user_contains_results():
    _, user = build_adjudication_prompt(
        formatted_data_sources="some text",
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
        deep_fake_verification_result=_SAMPLE_DEEP_FAKE_DATA,
    )
    assert "Resultados de Deteccao de Deep Fake" in user
    assert "0.8392" in user
    assert "video" in user


# ---- both has_audio and deep-fake ----

def test_prompt_with_audio_and_deep_fake_both_blocks():
    system, user = build_adjudication_prompt(
        formatted_data_sources="some text",
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
        has_audio=True,
        deep_fake_verification_result=_SAMPLE_DEEP_FAKE_DATA,
    )
    assert "audio_script" in system  # audio block
    assert "Deep Fake" in system  # deep-fake block
    assert "Resultados de Deteccao de Deep Fake" in user


def test_prompt_requires_extremely_concise_summary_and_ai_signal():
    system, _ = build_adjudication_prompt(
        formatted_data_sources="some text",
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "EXTREMAMENTE conciso" in system
    assert "a PRIMEIRA frase do resumo deve destacar explicitamente se ha indicios de conteudo gerado por IA" in system


def test_prompt_requires_more_concise_justification():
    system, _ = build_adjudication_prompt(
        formatted_data_sources="some text",
        fact_check_results=[],
        search_results={},
        scraped_pages=[],
    )
    assert "justificativa, deixando-a um pouco mais concisa" in system
