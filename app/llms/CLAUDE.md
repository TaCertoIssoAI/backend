# LLM Implementations

Custom LangChain model wrappers for LLM providers.

## gemini.py — GeminiChatModel

A `BaseChatModel` implementation wrapping the Google Gemini native API (`google.genai`). Exists because the official `langchain-google-genai` package may not support all Gemini-specific features (e.g., `thinking_config`).

### Features
- Extended thinking via `thinking_level` parameter (`"low"`, `"medium"`, `"high"`)
- Structured output via `with_structured_output(schema)` using Gemini's native `response_schema` + `response_mime_type="application/json"`
- Retry with exponential backoff (3 attempts) for `ServerError`, `ConnectError`, `RemoteProtocolError`
- Both sync (`_generate`) and async (`_agenerate`) implementations
- Configurable: `model`, `temperature`, `max_output_tokens`, `top_p`, `top_k`

### Usage

```python
from app.llms.gemini import GeminiChatModel

llm = GeminiChatModel(model="gemini-2.5-flash-lite", thinking_level="low")
response = await llm.ainvoke([HumanMessage(content="...")])

# structured output
structured = llm.with_structured_output(MyPydanticModel)
result = structured.invoke([...])  # returns MyPydanticModel instance
```

### Note
The main pipeline uses `ChatGoogleGenerativeAI` from `langchain-google-genai` for the context agent (needs tool binding), and this custom wrapper is available when native Gemini features are needed.
