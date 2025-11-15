# Claim Extractor Tests

Integration tests for the claim extraction pipeline step that make **real calls to the LLM**.

## Prerequisites

1. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

2. **Install dependencies**:
   ```bash
   pip install pytest
   # Also ensure langchain and other dependencies are installed
   ```

## Running Tests

### Run all tests with output:
```bash
pytest app/ai/pipeline/tests/claim_extractor_test.py -v -s
```

### Run a specific test:
```bash
pytest app/ai/pipeline/tests/claim_extractor_test.py::test_basic_claim_extraction -v -s
```

### Run without LLM output (quieter):
```bash
pytest app/ai/pipeline/tests/claim_extractor_test.py -v
```

## Flags Explained

- `-v` : Verbose output (shows test names)
- `-s` : Show stdout (you'll see the LLM responses for debugging)
- `-k <pattern>` : Run tests matching pattern (e.g., `-k portuguese`)

## What These Tests Do

✅ **Make real LLM calls** - Not mocked, actual OpenAI API requests
✅ **Validate structure** - Check that outputs have correct types and fields
✅ **Print results** - Show LLM responses in stdout for debugging
✅ **Don't validate content** - We don't check if LLM answers are "correct" (that's subjective)

## Test Coverage

1. **test_basic_claim_extraction** - Single claim with context
2. **test_multiple_claims_extraction** - Multiple claims in one message
3. **test_portuguese_message_extraction** - Language preservation
4. **test_no_context_extraction** - Claim extraction without expanded context
5. **test_empty_message** - Edge case: empty input
6. **test_opinion_vs_claim** - LLM should distinguish opinions from facts
7. **test_validate_claims_function** - Test the validation helper
8. **test_chain_building** - Ensure chain builds without errors
9. **test_return_type_is_list** - Verify we return List, not wrapper

## Expected Behavior

Each test will:
1. Create test data (user message + expanded context)
2. Call the claim extractor
3. Print the LLM's response to stdout
4. Validate the structure (types, required fields)
5. Assert structural requirements (not content accuracy)

## Cost Warning ⚠️

These tests make **real API calls** to OpenAI, which costs money. Running all tests might make ~10 API calls using the `gpt-4o-mini` model (cheaper option).

Estimated cost per full test run: **< $0.01 USD**

## Debugging Failed Tests

If a test fails:
1. Check the stdout output to see what the LLM returned
2. Verify `OPENAI_API_KEY` is set correctly
3. Check your OpenAI API quota/billing
4. Look at the assertion error to see which validation failed

## Example Output

```
TEST: Basic Claim Extraction
================================================================================

✓ Extracted 1 claim(s):

  Claim 1:
    ID: test-msg-001-claim-uuid-abc123
    Text: Vaccine X causes infertility in women
    Entities: ['Vaccine X', 'infertility', 'women']
    Links: ['https://example.com/vaccine-article']
    LLM Comment: This is a medical claim that can be verified...

```
