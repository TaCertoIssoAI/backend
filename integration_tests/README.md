# Integration Tests

Integration tests for the fact-checking API endpoints.

## Prerequisites

1. **Server must be running:**
   ```bash
   cd /Users/caue.lira/Desktop/facul/TaCertoIssoAI
   source venv/bin/activate
   uvicorn app.main:app --reload --port 8000
   ```

2. **Environment variables must be set:**
   - `OPENAI_API_KEY` - required for LLM claim extraction
   - `APIFY_TOKEN` - optional, needed for link scraping tests

## Running Tests

### Text-Only Integration Test

Tests the `/api/text` endpoint with pure text content (no links).

**Run directly:**
```bash
python integration_tests/text_only_test.py
```

**Run with pytest:**
```bash
pytest integration_tests/text_only_test.py -v -s
```

**What it tests:**
- ✅ Server health check
- ✅ Simple text claim
- ✅ Multiple claims in text
- ✅ Short text
- ✅ Long text with multiple claims
- ✅ Opinion-based text

### Text + Links Integration Test

Tests the `/api/text` endpoint with text containing links.

**Run directly:**
```bash
python integration_tests/text_and_links_test.py
```

**Run with pytest:**
```bash
pytest integration_tests/text_and_links_test.py -v -s
```

**What it tests:**
- ✅ Text with single link
- ✅ Text with multiple links
- ✅ Link-only content (minimal surrounding text)
- ✅ Link at end of sentence (with punctuation)
- ✅ Social media link (Facebook)
- ✅ Broken/invalid link handling

**⚠️ Important:**
These tests **require `APIFY_TOKEN`** to be set in `.env` for link scraping to work.

## Expected Output

When tests pass, you should see:

```
================================================================================
TEXT-ONLY INTEGRATION TESTS
================================================================================
Target: http://localhost:8000/api/text
================================================================================

✅ Server is up and running
   Response: {'status': 'healthy'}

================================================================================
TEST 1: Simple Text Claim
================================================================================

📤 Request payload:
{
  "content": [
    {
      "textContent": "Neymar está dando tudo de si...",
      "type": "text"
    }
  ]
}

📥 Response status: 200

📄 Response body:
{
  "message_id": "abc-123",
  "verdict": "2 claim(s) extracted",
  "rationale": "Análise das alegações extraídas:\n\n1. Neymar está ajudando...",
  ...
}

✅ Test passed: Response schema is correct

...

================================================================================
TEST SUMMARY
================================================================================
✅ Passed: 5
❌ Failed: 0
📊 Total: 5
================================================================================

🎉 All tests passed!
```

## Troubleshooting

### Server Not Running
```
❌ Could not connect to server at http://localhost:8000
   Make sure the server is running: uvicorn app.main:app --reload --port 8000
```
**Fix:** Start the server first

### Missing API Key
```
Error processing request: No API key provided
```
**Fix:** Set `OPENAI_API_KEY` in your `.env` file

### Test Timeout
```
requests.exceptions.Timeout
```
**Fix:** Increase timeout in test file or check LLM service availability

## Test Files

- `text_only_test.py` - Tests with pure text content (no links)
- `text_and_links_test.py` - Tests with text containing URLs

