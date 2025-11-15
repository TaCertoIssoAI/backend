# LangChain (Python) — Practical Guide

> A condensed, deployment-first overview of the LangChain Python stack. Focuses on structure, core concepts, useful primitives, best practices, coding standards, and common pitfalls. Inspired by the “Claude.md” sketch you shared.

---

## 2) Deployment‑First

- **Expose runnables as REST** with LangServe + FastAPI. Start with a single `/invoke` or `/stream` endpoint per chain, then layer auth/rate limits. citeturn0search7
- **Trace everything** to LangSmith from day one for debugging and offline evaluation. Store datasets and evaluators early. citeturn0search8turn0search18
- **Environment isolation:** pin model providers and versions, keep model config in code (temperature, max tokens), and set timeouts/retries. (See callbacks and retry helpers.) citeturn0search0

---

## 3) Core Principles

- **Runnable interface + LCEL composition.** Build chains declaratively and get streaming, retries, parallelism, and batch for free. 
- **Retrieval as a first‑class boundary.** Wrap vector stores as **retrievers**; swap implementations without touching prompts. 
- **Structured outputs.** Bind schemas so models return typed objects; prefer `with_structured_output(...)` or Pydantic parsers. 
- **Streaming UX.** Stream tokens and tool events via callbacks for responsive UIs. citeturn0search4turn0search22
- **Agents when needed.** For tool‑using workflows or multi‑step control, use LangGraph‑based agents. 

---

## 4) Tool Usage (Primitives & Concepts)

### 4.1 Models and LCEL
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("user", "{question}")
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | model | StrOutputParser()

print(chain.invoke({"question": "What is LCEL?"}))
```

LCEL composes `Runnable` stages with `|`, and the same chain supports `.invoke`, `.stream`, `.batch`, and `.ainvoke`. citeturn0search1

### 4.2 Retrieval
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

emb = OpenAIEmbeddings()
vs = FAISS.from_texts(["alpha", "beta about LCEL", "gamma docs"], emb)
retriever = vs.as_retriever()  # standard retriever interface

rag = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Use context: {context}\nQuestion: {question}")
    | model
    | StrOutputParser()
)
```

Use the retriever interface instead of calling the vector store directly for portability. citeturn0search2

### 4.3 Tools for Agents
```python
from langchain_core.tools import tool

@tool
def search_docs(q: str) -> str:
    "Search internal docs and return a short snippet."
    return "stub result"
```

Create tools with `@tool` to package a function and schema for agent/tool‑calling models. citeturn0search20

### 4.4 Structured Output
```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    verdict: bool = Field(..., description="true if answer is found in context")
    gist: str

typed_chain = (prompt | model).with_structured_output(Answer)
result = typed_chain.invoke({"question": "Is LCEL declarative?"})
```

Use `with_structured_output` or `PydanticOutputParser` to get validated objects. citeturn0search23turn0search17

### 4.5 Streaming and Callbacks
```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

streaming_model = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
list(streaming_model.stream([("user", "stream please")]))  # yields chunks
```

Streaming is enabled via model support and callback handlers. citeturn0search4turn0search15

---

## 5) Best Practices

- **Keep chains stateless;** pass state explicitly or attach to graph state, not globals. Use retrievers not raw store calls. citeturn0search2
- **Prefer LCEL over ad‑hoc Python.** You get retry, streaming, parallel, and tracing hooks consistently. citeturn0search1
- **Validate outputs.** Bind JSON schema or Pydantic models; fail fast on parse errors. citeturn0search23
- **Stream user‑facing flows.** Surface intermediate events for better UX and lower perceived latency. citeturn0search4
- **Evaluate continuously.** Use LangSmith datasets and evaluators to prevent regressions. citeturn0search24
- **Deploy early with LangServe.** Treat chains as APIs; version them and pin configs. citeturn0search7

---

## 6) Coding Standards

- **Type everything** (tool inputs, outputs, chain configs). Prefer `TypedDict` or `pydantic` models for IO. citeturn0search23
- **Consistent message handling.** Always construct prompts with `ChatPromptTemplate` and message tuples to avoid format drift. citeturn0search10
- **Callbacks policy.** Register standard handlers for tracing, streaming, and error logging across all runnables. citeturn0search0
- **Separation of concerns.** Keep prompt templates, models, and parsers in distinct modules; prefer dependency injection in tests.
- **Async first.** Use `.ainvoke` and `.astream` where IO bound; avoid mixing sync and async in the same call path.

**References:** Concepts overview; Callbacks; LCEL. citeturn0search10turn0search0turn0search1

---

## 7) Common Pitfalls and Anti‑Patterns

- **Using deprecated agent APIs.** Modern agents should be built on **LangGraph**; migrate away from older `AgentExecutor` stacks. citeturn0search3
- **Incorrect interrupt or state updates** in agent workflows. Model the state machine explicitly with graph nodes and typed state. citeturn0search9
- **Assuming free‑form text outputs.** Without schema binding, outputs drift and downstream code breaks. Use structured outputs. citeturn0search23
- **Tight coupling to a single vector store.** Wrap behind the retriever interface to swap FAISS/Milvus/etc. without code churn. citeturn0search19
- **No streaming or tracing.** Harder debugging and poor UX; enable callbacks and LangSmith from the start. citeturn0search4turn0search8

---

