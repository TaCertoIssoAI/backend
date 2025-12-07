# -*- coding: utf-8 -*-
"""
tests for the no claims fallback pipeline step.

these tests make REAL calls to the LLM (Gemini API) to validate:
- the structure of outputs
- the LangChain chain works correctly
- the prompt produces valid results
- fallback logic works properly
- pipeline steps integration

IMPORTANT: Set GOOGLE_API_KEY in your environment before running.

run with:
    pytest app/ai/pipeline/tests/no_claims_fallback_test.py -v -s

the -s flag shows stdout so you can see the LLM responses for debugging.
"""

import pytest
from typing import List

from app.models import DataSource, PipelineConfig
from app.ai.pipeline.no_claims_fallback import (
    NoClaimsFallbackOutput,
    should_use_fallback,
    get_combined_text_from_sources,
)
from app.ai.pipeline.steps import DefaultPipelineSteps
from app.ai.tests.fixtures.mock_pipelinesteps import WithoutBrowsingPipelineSteps
from app.config.gemini_models import get_gemini_default_pipeline_config


# ===== HELPER FUNCTIONS =====

def print_fallback_result(
    result: NoClaimsFallbackOutput,
    test_name: str,
    input_text: str | None = None
):
    """print fallback result for debugging, including input for verification."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)

    # print input for verification
    if input_text:
        print(f"\nINPUT TEXT:")
        print(f"  {input_text}")

    # print output
    print(f"\nOUTPUT:")
    print(f"  Explanation:")
    print(f"    {result.explanation}")
    print()
    print(f"  Original Text (stored):")
    print(f"    {result.original_text[:100]}..." if len(result.original_text) > 100 else f"    {result.original_text}")
    print()


def validate_fallback_output(result: NoClaimsFallbackOutput):
    """validate that a fallback output has the correct structure."""
    # type check
    assert isinstance(result, NoClaimsFallbackOutput), "result should be NoClaimsFallbackOutput"

    # required fields
    assert result.explanation is not None and result.explanation != "", "explanation should not be empty"
    assert result.original_text is not None, "original_text should not be None"

    # type checks
    assert isinstance(result.explanation, str), "explanation should be a string"
    assert isinstance(result.original_text, str), "original_text should be a string"

    # content validation
    assert len(result.explanation) > 10, "explanation should be meaningful (>10 chars)"


# ===== TESTS FOR HELPER FUNCTIONS =====

def test_should_use_fallback_zero_claims():
    """test that should_use_fallback returns true when no claims."""
    assert should_use_fallback(0) == True, "should use fallback when 0 claims"


def test_should_use_fallback_with_claims():
    """test that should_use_fallback returns false when claims exist."""
    assert should_use_fallback(1) == False, "should not use fallback when 1 claim"
    assert should_use_fallback(5) == False, "should not use fallback when 5 claims"


def test_get_combined_text_single_source():
    """test combining text from a single data source."""
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text="Olá, bom dia!"
        )
    ]

    result = get_combined_text_from_sources(sources)
    assert result == "Olá, bom dia!"


def test_get_combined_text_multiple_sources():
    """test combining text from multiple data sources."""
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text="Primeira mensagem."
        ),
        DataSource(
            id="msg-002",
            source_type="link_context",
            original_text="Segunda mensagem."
        )
    ]

    result = get_combined_text_from_sources(sources)
    assert result == "Primeira mensagem.\n\nSegunda mensagem."


def test_get_combined_text_empty_sources():
    """test combining text from empty sources list."""
    sources = []
    result = get_combined_text_from_sources(sources)
    assert result == ""


def test_get_combined_text_sources_without_text():
    """test combining when sources have empty original_text."""
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text=""  # empty string instead of None
        )
    ]

    result = get_combined_text_from_sources(sources)
    assert result == ""


# ===== PIPELINE STEPS INTEGRATION TESTS =====

@pytest.mark.asyncio
async def test_default_pipeline_steps_fallback():
    """test fallback through DefaultPipelineSteps with gemini config."""
    # create data sources
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text="Olá! Como vai?"
        )
    ]

    # get gemini config
    config = get_gemini_default_pipeline_config()

    # create pipeline steps
    steps = DefaultPipelineSteps()

    # execute fallback through pipeline steps
    result = await steps.handle_no_claims_fallback(sources, config)

    # print for debugging
    print_fallback_result(
        result,
        "DefaultPipelineSteps Fallback (Gemini)",
        input_text=sources[0].original_text
    )

    # validate structure
    validate_fallback_output(result)

    # validate content
    assert sources[0].original_text in result.original_text, "should contain source text"


@pytest.mark.asyncio
async def test_without_browsing_pipeline_steps_fallback():
    """test fallback through WithoutBrowsingPipelineSteps with gemini config."""
    # create data sources
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text="Bom dia! Tudo bem?"
        )
    ]

    # get gemini config
    config = get_gemini_default_pipeline_config()

    # create pipeline steps
    steps = WithoutBrowsingPipelineSteps()

    # execute fallback through pipeline steps
    result = await steps.handle_no_claims_fallback(sources, config)

    # print for debugging
    print_fallback_result(
        result,
        "WithoutBrowsingPipelineSteps Fallback (Gemini)",
        input_text=sources[0].original_text
    )

    # validate structure
    validate_fallback_output(result)

    # validate content
    assert sources[0].original_text in result.original_text, "should contain source text"


@pytest.mark.asyncio
async def test_pipeline_steps_multiple_sources():
    """test fallback with multiple data sources through pipeline steps."""
    # create multiple data sources
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text="Olá!"
        ),
        DataSource(
            id="link-001",
            source_type="link_context",
            original_text="Como posso ajudar?"
        ),
        DataSource(
            id="img-001",
            source_type="image",  # valid source_type
            original_text="Obrigado!"
        )
    ]

    # get gemini config
    config = get_gemini_default_pipeline_config()

    # create pipeline steps
    steps = DefaultPipelineSteps()

    # execute fallback through pipeline steps
    result = await steps.handle_no_claims_fallback(sources, config)

    # print for debugging
    combined = get_combined_text_from_sources(sources)
    print_fallback_result(
        result,
        "Pipeline Steps - Multiple Sources (Gemini)",
        input_text=combined
    )

    # validate structure
    validate_fallback_output(result)

    # validate all sources are combined
    assert "Olá!" in result.original_text, "should contain first source"
    assert "Como posso ajudar?" in result.original_text, "should contain second source"
    assert "Obrigado!" in result.original_text, "should contain third source"


@pytest.mark.asyncio
async def test_pipeline_steps_empty_sources():
    """test fallback with empty sources through pipeline steps."""
    # create empty sources
    sources = []

    # get gemini config
    config = get_gemini_default_pipeline_config()

    # create pipeline steps
    steps = DefaultPipelineSteps()

    # execute fallback through pipeline steps
    result = await steps.handle_no_claims_fallback(sources, config)

    # print for debugging
    print_fallback_result(
        result,
        "Pipeline Steps - Empty Sources (Gemini)",
        input_text="(no sources)"
    )

    # validate structure
    assert isinstance(result, NoClaimsFallbackOutput), "should return valid output"
    assert result.original_text == "", "should have empty original text"


@pytest.mark.asyncio
async def test_pipeline_config_fallback_llm_is_used():
    """test that the fallback LLM from config is actually used."""
    # create data source
    sources = [
        DataSource(
            id="msg-001",
            source_type="original_text",
            original_text="Oi, tudo bem?"
        )
    ]

    # get gemini config (uses gemini-2.5-flash for fallback)
    config = get_gemini_default_pipeline_config()

    # verify config has fallback LLM
    assert config.fallback_llm_config is not None, "config should have fallback_llm_config"
    assert config.fallback_llm_config.llm is not None, "fallback_llm_config should have llm"

    # create pipeline steps
    steps = DefaultPipelineSteps()

    # execute fallback
    result = await steps.handle_no_claims_fallback(sources, config)

    # print for debugging
    print("\n" + "=" * 80)
    print("TEST: Pipeline Config Fallback LLM Usage")
    print("=" * 80)
    print(f"\nConfig fallback LLM model: {config.fallback_llm_config.llm.model}")
    print(f"\nGenerated explanation:")
    print(f"  {result.explanation}")
    print()

    # validate structure
    validate_fallback_output(result)

    # validate content
    assert result.original_text == sources[0].original_text, "should store original text"
