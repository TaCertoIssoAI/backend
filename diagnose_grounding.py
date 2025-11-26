"""
diagnose why google search grounding is not being used.
"""

import os
import json
from pprint import pprint
from app.llms.gemini_langchain import create_gemini_model
from app.config.gemini_models import get_gemini_default_pipeline_config


def diagnose_model_creation():
    """check if model is created correctly with tools."""
    print("\n" + "=" * 80)
    print("1. Model Creation Diagnostics")
    print("=" * 80)

    model = create_gemini_model(
        model="gemini-2.5-flash",
        enable_search=True,
        temperature=0.0
    )

    print(f"\n✓ Model created: {model}")
    print(f"✓ Model type: {type(model)}")
    print(f"✓ Model class name: {model.__class__.__name__}")

    # Check if it's our custom subclass
    if hasattr(model, '_tools_list'):
        print(f"✓ Has _tools_list attribute: {model._tools_list is not None}")
        if model._tools_list:
            print(f"✓ Number of tools: {len(model._tools_list)}")
            print(f"✓ Tools: {model._tools_list}")
    else:
        print("✗ No _tools_list attribute found")

    # Check parent class attributes
    if hasattr(model, 'model'):
        print(f"\n✓ Model name: {model.model}")

    if hasattr(model, 'google_api_key'):
        print(f"✓ API key configured: {bool(model.google_api_key)}")

    return model


def diagnose_direct_invocation(model):
    """test direct model invocation and inspect response."""
    print("\n" + "=" * 80)
    print("2. Direct Model Invocation Diagnostics")
    print("=" * 80)

    prompt = "What is the current weather in Tokyo? Include temperature."

    print(f"\n📤 Prompt: {prompt}")
    print("\n⏳ Invoking model...\n")

    response = model.invoke(prompt)

    print("=" * 80)
    print("Response Object Inspection")
    print("=" * 80)

    print(f"\n✓ Response type: {type(response)}")
    print(f"✓ Response class: {response.__class__.__name__}")

    # Check all attributes
    print(f"\n✓ Response attributes:")
    for attr in dir(response):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Check content
    if hasattr(response, 'content'):
        print(f"\n✓ Content (first 200 chars): {response.content[:200]}...")

    # Check response_metadata
    print("\n" + "=" * 80)
    print("response_metadata inspection:")
    print("=" * 80)
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        print(f"\n✓ response_metadata exists: {metadata is not None}")
        if metadata:
            print(f"\n✓ Keys in response_metadata:")
            for key in metadata.keys():
                print(f"  - {key}")

            print(f"\n✓ Full response_metadata:")
            pprint(metadata, indent=2, width=100)
    else:
        print("\n✗ No response_metadata attribute")

    # Check additional_kwargs
    print("\n" + "=" * 80)
    print("additional_kwargs inspection:")
    print("=" * 80)
    if hasattr(response, 'additional_kwargs'):
        kwargs = response.additional_kwargs
        print(f"\n✓ additional_kwargs exists: {kwargs is not None}")
        if kwargs:
            print(f"\n✓ Keys in additional_kwargs:")
            for key in kwargs.keys():
                print(f"  - {key}")

            print(f"\n✓ Full additional_kwargs:")
            pprint(kwargs, indent=2, width=100)
    else:
        print("\n✗ No additional_kwargs attribute")

    return response


def diagnose_config():
    """check pipeline config."""
    print("\n" + "=" * 80)
    print("3. Pipeline Config Diagnostics")
    print("=" * 80)

    config = get_gemini_default_pipeline_config()

    print(f"\n✓ Config loaded: {config}")
    print(f"\n✓ Adjudication LLM type: {type(config.adjudication_llm_config.llm)}")
    print(f"✓ Adjudication LLM class: {config.adjudication_llm_config.llm.__class__.__name__}")

    llm = config.adjudication_llm_config.llm

    if hasattr(llm, '_tools_list'):
        print(f"\n✓ Has _tools_list: {llm._tools_list is not None}")
        if llm._tools_list:
            print(f"✓ Number of tools: {len(llm._tools_list)}")
            print(f"✓ Tools: {llm._tools_list}")
    else:
        print("\n✗ No _tools_list attribute")


def diagnose_with_structured_output(model):
    """test with structured output disabled."""
    print("\n" + "=" * 80)
    print("4. Structured Output Override Test")
    print("=" * 80)

    from pydantic import BaseModel, Field

    class TestOutput(BaseModel):
        answer: str = Field(..., description="The answer")

    print("\n✓ Calling with_structured_output...")
    result = model.with_structured_output(TestOutput)

    print(f"✓ Result type: {type(result)}")
    print(f"✓ Result class: {result.__class__.__name__}")

    # Check if it's the model itself (when tools are present)
    if result is model:
        print("\n✅ Structured output was DISABLED (returns model itself)")
        print("   This means tools are present and our override is working!")
    else:
        print("\n⚠️  Structured output was ENABLED")
        print("   This means tools might not be configured correctly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GOOGLE SEARCH GROUNDING DIAGNOSTICS")
    print("=" * 80)

    # Run diagnostics
    model = diagnose_model_creation()
    response = diagnose_direct_invocation(model)
    diagnose_config()
    diagnose_with_structured_output(model)

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print("\nLook for:")
    print("  - _tools_list should be present and non-empty")
    print("  - response_metadata or additional_kwargs should contain grounding data")
    print("  - with_structured_output should return the model itself (not a chain)")
    print("=" * 80)
