"""
Quick verification script to check that all imports work correctly.
Run this before running the full examples to ensure the module is properly set up.
"""

def verify_imports():
    """Verify that all imports work correctly."""
    print("Verifying imports...\n")

    # Test 1: Import from pipeline module
    print("✓ Testing: from app.ai.pipeline import ...")
    try:
        from app.ai.pipeline import (
            build_claim_extraction_chain,
        )
        print("  ✓ All functions imported successfully\n")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}\n")
        return False

    # Test 2: Import data models
    print("✓ Testing: from app.models import ...")
    try:
        from app.models import ExtractedClaim
        from app.models.commondata import CommonPipelineData
        print("  ✓ Data models imported successfully\n")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}\n")
        return False

    # Test 3: Create sample data models
    print("✓ Testing: Creating sample data models...")
    try:
        common_data = CommonPipelineData(
            message_id="test-msg-001",
            message_text="Test message",
            locale="en-US"
        )
        print(f"  ✓ Created CommonPipelineData: {common_data.message_id}\n")

        claim = ExtractedClaim(
            id="test-claim-001",
            text="Sample claim for testing",
            links=[],
            llm_comment="Test comment",
            entities=["test"]
        )
        print(f"  ✓ Created ExtractedClaim: {claim.id}\n")
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}\n")
        return False

    # Test 4: Build chain (without invoking)
    print("✓ Testing: Building LCEL chain...")
    try:
        chain = build_claim_extraction_chain(
            model_name="gpt-4o-mini",
            temperature=0.0
        )
        print(f"  ✓ Chain built successfully: {type(chain).__name__}\n")
    except Exception as e:
        print(f"  ✗ Chain building failed: {e}\n")
        return False

    # Test 5: Format context
    print("✓ Testing: Formatting expanded context...")
    try:
        formatted = format_expanded_context({
            "https://example.com": "Test content"
        })
        print(f"  ✓ Context formatted successfully ({len(formatted)} chars)\n")
    except Exception as e:
        print(f"  ✗ Formatting failed: {e}\n")
        return False

    print("=" * 60)
    print("✓ All imports and basic functionality verified!")
    print("=" * 60)
    print("\nYou can now run the full examples:")
    print("  python -m app.ai.pipeline.example_claim_extraction")
    print("\nMake sure to set OPENAI_API_KEY in your environment first:")
    print("  export OPENAI_API_KEY='sk-...'")

    return True


if __name__ == "__main__":
    success = verify_imports()
    exit(0 if success else 1)
