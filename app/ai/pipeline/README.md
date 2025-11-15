# Claim Extraction Pipeline Step

This module implements the **Claim Extraction** step of the fact-checking pipeline using LangChain best practices.

## Overview

The claim extractor takes a user message (optionally enriched with context from linked articles) and extracts fact-checkable claims using an LLM with structured outputs.

## Architecture

```
Input (ExpandedUserInput)
    ↓
Format Context
    ↓
LCEL Chain: Prompt | Model | Structured Output
    ↓
Post-Process (UUID generation, validation)
    ↓
Output (List[ExtractedClaim])
```

## LangChain Best Practices Applied

### ✅ 1. LCEL Composition
The chain uses declarative LCEL syntax for clean, composable pipelines:

```python
chain = prompt | model.with_structured_output(schema)
```

This automatically provides:
- `.invoke()` for sync execution
- `.ainvoke()` for async execution
- `.stream()` and `.batch()` capabilities
- Retry and timeout handling

### ✅ 2. Structured Outputs
Uses Pydantic models with `with_structured_output()` for type-safe LLM responses:

```python
class ClaimExtractionOutput(BaseModel):
    claims: List[ExtractedClaim]

structured_model = model.with_structured_output(ClaimExtractionOutput)
```

This ensures:
- Validated JSON output
- Type safety throughout the pipeline
- Clear schema for the LLM

### ✅ 3. Stateless Design
All functions accept explicit parameters - no global state:

```python
def extract_claims(
    expanded_context_by_source: dict[str, str],
    user_message: str,
    common_data: CommonPipelineData,
    ...
) -> List[ExtractedClaim]:
```

### ✅ 4. Type Annotations
Every function, parameter, and return value is typed:

```python
def format_expanded_context(
    expanded_context_by_source: dict[str, str]
) -> str:
```

### ✅ 5. Async Support
Both sync and async versions provided for all operations:

```python
claims = extract_claims(...)           # Sync
claims = await extract_claims_async(...)  # Async
```

### ✅ 6. ChatPromptTemplate
Consistent message handling using LangChain's prompt templates:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_PROMPT)
])
```

### ✅ 7. Separation of Concerns
Clean module structure:
- `prompts.py` - Prompt templates
- `claim_extractor.py` - Chain logic and extraction
- `example_claim_extraction.py` - Usage examples

### ✅ 8. Configuration Over Hardcoding
Model parameters are configurable:

```python
def build_claim_extraction_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: Optional[float] = None
) -> Runnable:
```

## Usage

### Basic Usage

```python
from app.ai.pipeline.claim_extractor import extract_and_validate_claims
from app.models.commondata import CommonPipelineData

# Prepare input
expanded_context = {
    "https://example.com": "Article text here..."
}
user_message = "I heard vaccine X causes infertility"
common_data = CommonPipelineData(
    message_id="msg-123",
    message_text=user_message
)

# Extract claims
claims = extract_and_validate_claims(
    expanded_context_by_source=expanded_context,
    user_message=user_message,
    common_data=common_data
)

# Use the extracted claims
for claim in claims:
    print(f"Claim: {claim.text}")
    print(f"Entities: {claim.entities}")
```

### Async Usage

```python
import asyncio

async def process_message():
    claims = await extract_claims_async(
        expanded_context_by_source=expanded_context,
        user_message=user_message,
        common_data=common_data
    )
    return claims

claims = asyncio.run(process_message())
```

### Custom Model Configuration

```python
# Use a different model
claims = extract_claims(
    ...,
    model_name="gpt-4o",      # More powerful model
    temperature=0.1,          # Slightly more creative
    timeout=60.0              # Longer timeout
)
```

## Input Format

The claim extractor expects input in this format:

```
=== Context from https://example.com/article1 ===
Title: Article Title
Content: Article content here...

=== Context from https://example.com/article2 ===
Title: Another Article
Content: More content...

====Original User Message Below====
User's original message text here
```

This is automatically formatted by the `format_expanded_context()` function.

## Output Format

Returns a list of `ExtractedClaim` objects:

```python
ExtractedClaim(
    id="msg-123-claim-uuid-...",
    text="Vaccine X causes infertility in women",
    links=["https://example.com/article1"],
    llm_comment="This is a medical claim that requires scientific verification",
    entities=["Vaccine X", "infertility", "women"]
)
```

## Data Flow

1. **Input**: User message + expanded context from links (from previous pipeline step)
2. **Format**: Context is formatted into a structured prompt
3. **Extract**: LLM extracts claims with structured output
4. **Post-process**:
   - Generate unique IDs (prefixed with message_id)
   - Validate claims (remove empty, duplicates, too short)
5. **Output**: List of validated `ExtractedClaim` objects

## Error Handling

The module handles common errors:

- **Timeout**: Set `timeout` parameter for long-running requests
- **Invalid API key**: Raises clear error if `OPENAI_API_KEY` not set
- **Empty claims**: `validate_claims()` filters out empty results
- **Duplicate claims**: Validation removes duplicates

## Testing

Run the example file to test:

```bash
python -m app.ai.pipeline.example_claim_extraction
```

Make sure `OPENAI_API_KEY` is set in your environment.

## Configuration

Model configuration can be set via:

1. **Function parameters** (preferred for runtime config):
   ```python
   extract_claims(..., model_name="gpt-4o", timeout=30.0)
   ```

2. **Environment variables** (for API keys):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Performance Considerations

- **Model choice**: `gpt-4o-mini` is faster and cheaper, good for most cases
- **Temperature**: `0.0` for deterministic extraction (recommended)
- **Timeout**: Default 30s, increase for complex messages
- **Async**: Use `extract_claims_async()` for concurrent processing

## Integration with Pipeline

This step fits into the larger pipeline:

```
1. User Input
2. Context Expansion ← enriches links
3. **Claim Extraction** ← YOU ARE HERE
4. Evidence Gathering ← uses extracted claims
5. Final Adjudication
```

Input from previous step: `ExpandedUserInput`
Output to next step: `List[ExtractedClaim]`

## Next Steps

After claim extraction, the pipeline proceeds to:
- **Evidence Gathering**: Search for sources supporting/refuting each claim
- **Adjudication**: Final verdict based on evidence

## References

- [LangChain LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [Structured Outputs Guide](https://python.langchain.com/docs/how_to/structured_output/)
- [ChatPromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/quick_start/)
