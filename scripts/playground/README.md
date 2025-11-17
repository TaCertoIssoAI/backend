# playground - interactive API testing tools

interactive CLI tools for testing and exploring various APIs used in the fact-checking pipeline.

## setup

make sure you have the required environment variables set:

```bash
export GOOGLE_API_KEY='your-google-api-key-here'
```

## available tools

### 1. google fact-check API tester

interactive CLI for testing the Google Fact-Check Tools API.

**run:**
```bash
python scripts/playground/google_factcheck_cli.py
```

**features:**
- query custom claims interactively
- test with predefined example claims
- batch query multiple claims at once
- compare results across languages (English vs Portuguese)
- view detailed citation information
- test API connection
- view configuration

**menu options:**

1. **query custom claim** - enter any claim text to search
2. **query example claim** - choose from predefined test claims
3. **query multiline claim** - enter longer, multi-paragraph claims
4. **batch query multiple claims** - test multiple claims in one session
5. **view last query results** - review previous query in detail
6. **compare language results** - test same claim in English and Portuguese
7. **test API connection** - verify API is accessible
8. **show configuration** - display current API settings

**example usage:**

```bash
$ python scripts/playground/google_factcheck_cli.py

================================================================================
Google Fact-Check API Interactive CLI
================================================================================

✓ Google Fact-Check API gatherer initialized

================================================================================
Google Fact-Check API Tester
================================================================================

  1. Query custom claim
  2. Query example claim
  3. Query multiline claim
  4. Batch query multiple claims
  ────────────────────────────────────────
  5. View last query results
  6. Compare language results
  ────────────────────────────────────────
  7. Test API connection
  8. Show configuration
  0. Exit

Select option: 1

================================================================================
Query Custom Claim
================================================================================

Enter claim to fact-check: Vaccines cause autism

[API calls and results display here...]
```

**output features:**
- colored terminal output for better readability
- formatted citation lists with ratings
- detailed API response information
- JSON export of results
- progress indicators for batch operations

## common utilities (common.py)

reusable CLI components used by all playground tools:

### text formatting
- `print_header()` - bold headers with separators
- `print_section()` - section dividers
- `print_success()` - green success messages (✓)
- `print_error()` - red error messages (✗)
- `print_warning()` - yellow warnings (⚠)
- `print_info()` - cyan info messages (ℹ)
- `print_json()` - formatted JSON output
- `print_dict_table()` - key-value tables

### user input
- `prompt_input()` - text input with optional default
- `prompt_yes_no()` - yes/no confirmation
- `prompt_choice()` - select from list
- `prompt_multiline()` - multi-line text input

### interactive menus
- `Menu` class - build interactive CLI menus

### utilities
- `with_spinner()` - loading spinner for async operations
- `handle_api_error()` - formatted error display
- `print_api_response()` - formatted API response display
- `print_citation_list()` - formatted citation lists

## adding new API testers

to create a new interactive API tester:

1. import common utilities:
```python
from scripts.playground.common import (
    Menu, print_header, print_section,
    prompt_input, print_citation_list
)
```

2. create menu and handlers:
```python
def handle_query():
    claim = prompt_input("Enter claim")
    # ... call your API
    print_citation_list(results)

menu = Menu("My API Tester")
menu.add_option("Query claim", handle_query)
menu.run()
```

3. make executable:
```bash
chmod +x scripts/playground/my_api_cli.py
```

## examples

### test google API with custom claim:
```bash
$ python scripts/playground/google_factcheck_cli.py
> Select option: 1
> Enter claim: Climate change is a hoax
```

### batch test multiple claims:
```bash
$ python scripts/playground/google_factcheck_cli.py
> Select option: 4
> 1. Vaccines are safe
> 2. Earth is flat
> 3. DONE
```

### compare language coverage:
```bash
$ python scripts/playground/google_factcheck_cli.py
> Select option: 6
[Shows results for same claim in English vs Portuguese]
```

## tips

- use ctrl+c to cancel any operation
- use option 0 to exit the menu
- most prompts have sensible defaults (shown in brackets)
- use "view last results" to re-examine previous queries
- API calls are logged with detailed output for debugging

## troubleshooting

**"GOOGLE_API_KEY environment variable not set"**
- run: `export GOOGLE_API_KEY='your-key'`
- add to ~/.bashrc or ~/.zshrc for persistence

**"API connection failed"**
- check your API key is valid
- verify network connectivity
- check API quota limits

**"No citations found"**
- this is normal for many claims (API has limited coverage)
- try well-known claims like "vaccines cause autism"
- English claims have better coverage than Portuguese

## future tools

planned additions:
- web search API tester
- brazilian fact-checker API tester
- news API tester
- evidence retrieval pipeline tester
- full pipeline interactive debugger
