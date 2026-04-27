# LLM Helpers

Thin factories for LLM provider clients used in the fact-checking pipeline.

## vertex.py — `make_vertex_chat`

Factory that builds `ChatGoogleGenerativeAI` (from `langchain-google-genai>=4.2.2`) configured for **Vertex AI** routing. Centralizes `project`, `location`, and `vertexai=True` so call sites stay compact.

### Auth

Resolves automatically via `google-auth` discovery order:
1. `GOOGLE_APPLICATION_CREDENTIALS` env var → SA JSON path
2. ADC default (`~/.config/gcloud/application_default_credentials.json`)
3. GCE/GKE/Cloud Run metadata server

`.env` typically sets `GOOGLE_APPLICATION_CREDENTIALS` to the SA JSON.

### Required env vars

- `VERTEX_PROJECT_ID` — GCP project id
- `VERTEX_LOCATION` — Vertex region (e.g. `us-central1`)
- `GOOGLE_APPLICATION_CREDENTIALS` — path to SA JSON (or use ADC)

### Usage

```python
from app.llms.vertex import make_vertex_chat

# basic
llm = make_vertex_chat("gemini-2.5-flash-lite", temperature=0)
response = await llm.ainvoke([HumanMessage(content="...")])

# with extended thinking (token budget, not level)
llm = make_vertex_chat("gemini-3-pro-preview", temperature=0, thinking_budget=8192)

# structured output (pydantic schema)
structured = llm.with_structured_output(MyPydanticModel)
result = await structured.ainvoke([...])  # returns MyPydanticModel instance

# tool binding (used by the context/search agent)
model_with_tools = llm.bind_tools(tools)
```

### Why `ChatGoogleGenerativeAI` and not `ChatVertexAI`

`ChatVertexAI` from `langchain-google-vertexai` is deprecated as of LangChain 3.2.0. LangChain has consolidated Vertex AI support into `langchain-google-genai>=4.0.0`'s `ChatGoogleGenerativeAI` via the `vertexai=True` flag. Single class for both Developer API and Vertex modes.
