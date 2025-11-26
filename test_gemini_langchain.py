"""
quick test to verify langchain-google-genai supports both google search and structured output.
"""

from pydantic import BaseModel, Field
from app.llms.gemini_langchain import create_gemini_model


class SimpleVerdict(BaseModel):
    """simple verdict schema for testing."""
    claim: str = Field(..., description="the claim being verified")
    verdict: str = Field(..., description="true, false, or uncertain")
    reasoning: str = Field(..., description="brief reasoning for the verdict")


def test_structured_output_without_search():
    """test structured output without google search."""
    print("\n" + "=" * 80)
    print("TEST 1: Structured Output WITHOUT Google Search")
    print("=" * 80)

    try:
        model = create_gemini_model(
            model="gemini-2.5-flash",
            enable_search=False,
            temperature=0.0
        )

        structured_model = model.with_structured_output(SimpleVerdict)

        result = structured_model.invoke("The Eiffel Tower is in Paris")

        print(f"\n✅ Success!")
        print(f"Claim: {result.claim}")
        print(f"Verdict: {result.verdict}")
        print(f"Reasoning: {result.reasoning}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_structured_output_with_search():
    """test structured output WITH google search - this is the critical test."""
    print("\n" + "=" * 80)
    print("TEST 2: Structured Output WITH Google Search")
    print("=" * 80)

    try:
        model = create_gemini_model(
            model="gemini-2.5-flash",
            enable_search=True,
            temperature=0.0
        )

        structured_model = model.with_structured_output(SimpleVerdict)

        result = structured_model.invoke("What was the weather in Paris yesterday?")

        print(f"\n✅ Success!")
        print(f"Claim: {result.claim}")
        print(f"Verdict: {result.verdict}")
        print(f"Reasoning: {result.reasoning}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nThis error means LangChain's implementation has the same limitation:")
        print(f"Google Search tools and structured output cannot be used together.")
        import traceback
        traceback.print_exc()


def test_search_without_structured_output():
    """test google search without structured output."""
    print("\n" + "=" * 80)
    print("TEST 3: Google Search WITHOUT Structured Output")
    print("=" * 80)

    try:
        model = create_gemini_model(
            model="gemini-2.5-flash",
            enable_search=True,
            temperature=0.0
        )

        result = model.invoke("What was the weather in Paris yesterday?")

        print(f"\n✅ Success!")
        print(f"Response: {result.content[:500]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing LangChain Google GenAI - Search + Structured Output")
    print("=" * 80)

    test_structured_output_without_search()
    test_search_without_structured_output()
    test_structured_output_with_search()

    print("\n" + "=" * 80)
    print("Tests Complete")
    print("=" * 80)
