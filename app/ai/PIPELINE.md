# Fact-Checking Pipeline

## Overview

The fact-checking pipeline is a multi-stage, streaming system that receives user content (text, links, images, video transcripts), extracts verifiable claims, gathers evidence from multiple sources, and returns structured verdicts with citations.

The pipeline is designed for high parallelism using a **fire-and-forget streaming** model: stages do not wait for each other to fully complete before kicking off downstream work. As soon as a claim is extracted, evidence retrieval begins for that claim — without waiting for all other claims to be extracted.

---

## Pipeline Stages

```
Input: List[DataSource]
       │
       ▼
[Stage 1] Link Context Expansion
   – Scrape URLs found in text
   – Produce new DataSources with scraped content
       │
       ▼
[Stage 2] Claim Extraction  (parallel per DataSource)
   – LLM extracts fact-checkable claims from each source
   – Produces ExtractedClaim objects
       │
       ▼
[Stage 3] Evidence Retrieval  (parallel per claim, per gatherer)
   – Google Fact-Check API
   – Web Search (Google Custom Search)
   – Produces EnrichedClaim (claim + citations)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Stage 4A] Adjudication           [Stage 4B] Adjudication with Search
   (evidence-based, primary)         (OpenAI real-time search, fallback)
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
               Output: FactCheckResult
```

If no claims are extracted from any source, a **no-claims fallback** is triggered, returning a user-friendly explanation.

---

## Stage 1 — Link Context Expansion

**File**: `pipeline/link_context_expander.py`

### What it does

Finds URLs in the original message text, scrapes their content, and adds that content as additional `DataSource` objects so downstream stages can also fact-check claims from the linked pages.

### Input

`DataSource` objects with `source_type = "original_text"`.

### Process

1. Extracts URLs using a regex pattern (`https?://...`), stripping trailing punctuation.
2. Submits one async scraping job per URL via `ThreadPoolManager`.
3. Enforces per-link and total timeouts; skips URLs that time out.

### Output

`List[DataSource]` with `source_type = "link_context"`. Each source includes:
- Scraped page text
- URL and domain metadata
- Social media platform info (author, likes, shares) when available
- Success/error status flag

### Configuration

| Key | Description |
|-----|-------------|
| `max_links_to_expand` | Max number of links processed |
| `link_content_expander_timeout_per_link` | Per-URL timeout |
| `link_content_expander_timeout_total` | Timeout for all link expansion |

---

## Stage 2 — Claim Extraction

**File**: `pipeline/claim_extractor.py`

### What it does

Uses an LLM to identify and normalize fact-checkable claims from a data source. Only concrete, verifiable assertions are extracted — not opinions, greetings, or vague statements.

### Input

`ClaimExtractionInput`:
```python
data_source: DataSource  # text content + source_type + metadata
```

### Process

1. Selects a prompt based on `source_type`:
   - `original_text` → general claim extraction
   - `link_context` → web article extraction
   - `image` → OCR/visual description extraction
   - `video_transcript` → transcript extraction
2. Builds a LangChain LCEL chain: `prompt | model.with_structured_output(...)`.
3. Normalizes claims — e.g., "X is being shared as Y" → extracts "Y happened".
4. Assigns each claim a UUID and tracks its originating source.
5. Deduplicates and removes empty claims.

**Extraction directives**: prefer fewer, richer claims over many vague ones. Every claim must be self-contained (include WHO, WHAT, WHEN, WHERE) and relate to something independently verifiable.

### Output

`ClaimExtractionOutput`:
```python
data_source: DataSource
claims: List[ExtractedClaim]

# ExtractedClaim fields:
id: str           # UUID
text: str         # Normalized claim text
source: ClaimSource  # Reference to originating DataSource
entities: List[str]  # Named entities extracted
llm_comment: str  # LLM reasoning on why claim is fact-checkable
```

---

## Stage 3 — Evidence Retrieval

**File**: `pipeline/evidence_retrieval.py`

### What it does

Queries multiple external sources in parallel to gather citations and evidence for each extracted claim.

### Input

`EvidenceRetrievalInput`:
```python
claims: List[ExtractedClaim]
```

### Process

For each claim, all configured evidence gatherers run concurrently with a per-gatherer timeout. Errors or timeouts on one gatherer do not block the others.

**Gatherer 1 — Google Fact-Check API** (`context/google_factcheck_gatherer.py`)
- Queries `https://factchecktools.googleapis.com/v1alpha1/claims:search`.
- Returns structured fact-checks from organizations such as PolitiFact, FactCheck.org, Agência Lupa, etc.
- Translates English ratings to the verdict vocabulary used by the pipeline:
  - "True / Correct / Mostly True" → `Verdadeiro`
  - "False / Incorrect / Pants on Fire" → `Falso`
  - "Misleading / Missing Context" → `Fora de Contexto`
  - "Unverifiable / Satire" → `Fontes insuficientes para verificar`

**Gatherer 2 — Web Search** (`context/web_search_gatherer.py`)
- Queries Google Custom Search API.
- Supports language filtering (`pt-BR`) and trusted domain constraints.
- Returns title, URL, snippet, and domain.

After all gatherers complete, citations are optionally:
- Deduplicated by URL.
- Filtered by quality (minimum snippet length, required fields).

### Output

`EvidenceRetrievalResult`:
```python
claim_evidence_map: Dict[str, EnrichedClaim]
# maps claim_id → EnrichedClaim

# EnrichedClaim adds to ExtractedClaim:
citations: List[Citation]

# Citation fields:
url: str
title: str
publisher: str
citation_text: str       # Relevant excerpt
source: str              # Which gatherer produced this
rating: Optional[str]    # Fact-check rating label
rating_comment: str      # Additional context
date: Optional[str]      # Publication/review date
```

---

## Stage 4A — Adjudication (Primary)

**File**: `pipeline/judgement.py`

### What it does

Uses an LLM to reason over each claim and its gathered citations, then produces a structured verdict with justification.

### Input

`AdjudicationInput`:
```python
sources_with_claims: List[DataSourceWithClaims]
# each contains: data_source + list of EnrichedClaims (claim + citations)
additional_context: Optional[str]
```

### Process

1. Formats all claims and their citations into a structured text block.
2. Builds a LangChain LCEL chain with `structured_output` binding.
3. Calls the LLM with the current date, the formatted evidence, and any additional context.
4. Maps LLM output back to original claim IDs (by ID, with positional fallback).

**Adjudication directives** (from `pipeline/prompts.py`):
- Assess individual claims first, then evaluate overall context.
- Look for temporal/spatial descontextualization (true fact, wrong time or place).
- Prioritize specialized fact-checking organizations.
- Favor more recent sources when evidence conflicts.
- For atemporal facts (math, definitions), internal knowledge may supplement.
- Cite sources using `[1]`, `[2]`, `[3]` inline notation in the justification.

### Verdict categories

| Verdict | Portuguese | Meaning |
|---------|-----------|---------|
| True | Verdadeiro | Confirmed by reliable evidence |
| False | Falso | Contradicted by reliable sources |
| Out of Context | Fora de Contexto | Factually true but misleadingly framed |
| Insufficient Sources | Fontes insuficientes para verificar | Cannot confirm or refute |

### Output

`FactCheckResult`:
```python
results: List[DataSourceResult]
overall_summary: Optional[str]      # 3–4 line summary across all verdicts
sources_with_claims: List[...]      # Full lineage for traceability

# DataSourceResult fields:
data_source_id: str
source_type: str
claim_verdicts: List[ClaimVerdict]

# ClaimVerdict fields:
claim_id: str
claim_text: str
verdict: str              # One of the 4 categories above
justification: str        # Reasoning with inline [N] citations
citations_used: List[int] # Indices of citations used
```

---

## Stage 4B — Adjudication with Search (Fallback)

**File**: `pipeline/adjudication_with_search.py`

### What it does

An alternative adjudication path that uses OpenAI's real-time web search API. It is submitted as a fire-and-forget job in parallel with the primary adjudication, and used as the final result only if primary adjudication fails or returns empty results.

### Input

`List[DataSourceWithExtractedClaims]` — claims **without** pre-gathered evidence.

### Process

1. Formats claims grouped by data source into a prompt.
2. Calls the OpenAI Responses API with `tools=[{"type": "web_search"}]` and structured output.
3. OpenAI performs its own web searches during reasoning.
4. Handles UTF-8 encoding issues and repairs malformed JSON in responses.

### Trade-offs vs. primary adjudication

| | Primary (4A) | With Search (4B) |
|--|--|--|
| Search control | Full (we choose queries/sources) | Delegated to OpenAI |
| Latency | Depends on evidence retrieval | Single API call |
| Reliability | Depends on our gatherers | Depends on OpenAI |
| Cost | Multiple API calls | One call |

### When it is used

The function `_chose_fact_checking_result()` in `main_pipeline.py` selects between the two paths: it uses 4A by default, and falls back to 4B when 4A returns no verdicts.

---

## No-Claims Fallback

**File**: `pipeline/no_claims_fallback.py`

### When it is triggered

After claim extraction, if no valid claims were found across all data sources.

### What it does

Calls a lightweight LLM with a friendly prompt that explains why the content does not contain fact-checkable claims. Common reasons include:
- Personal opinions without verifiable assertions.
- Greetings or casual conversation.
- Questions without implicit claims.
- Vague statements lacking specific details.

### Output

`NoClaimsFallbackOutput`:
```python
explanation: str    # User-friendly message explaining the situation
original_text: str  # Echo of the input
```

The pipeline wraps this in a `FactCheckResult` with `results=[]` and sets `overall_summary` to the explanation.

---

## Orchestration and Parallelism

**Files**: `main_pipeline.py`, `async_code.py`, `threads/thread_utils.py`

### Entry point

```python
run_fact_check_pipeline(
    data_sources: List[DataSource],
    config: PipelineConfig,
    steps: PipelineSteps,
    analytics: ...,
    message_id: str,
) -> FactCheckResult
```

### Fire-and-forget streaming

The core async function `fire_and_forget_streaming_pipeline()` uses a `ThreadPoolManager` (singleton, priority queue) to execute work without blocking:

1. Claim extraction jobs are submitted immediately for all `original_text` sources.
2. Link expansion runs in parallel.
3. As each claim extraction completes, evidence retrieval jobs are submitted for those claims without waiting for other extractions.
4. As link expansion completes, claim extraction is triggered for each expanded source.
5. Adjudication with search (4B) is submitted as a background fire-and-forget task.
6. The main loop waits only for all evidence retrieval to finish, then proceeds to adjudication (4A).

### Job priorities

| Operation | Priority |
|-----------|----------|
| Claim extraction | 10 (highest) |
| Adjudication with search | 8 |
| Link expansion (pipeline) | 6 |
| Link context expanding | 5 |
| Evidence retrieval | 3 (lowest) |

### Error handling and resilience

| Failure | Behavior |
|---------|----------|
| Evidence gatherer timeout | Empty citations for that gatherer; continue |
| Link expansion timeout | Skip that link; continue with others |
| Primary adjudication fails | Fall back to adjudication with search (4B) |
| No claims extracted | Use no-claims fallback |
| Malformed JSON from LLM | Attempt JSON repair before raising |

---

## Extensibility

### PipelineSteps protocol

`steps.py` defines a `PipelineSteps` protocol with one method per pipeline stage:

```python
expand_links_from_sources(...)
extract_claims_from_all_sources(...)
get_evidence_gatherers(...)
gather_evidence(...)
handle_no_claims_fallback(...)
adjudicate_claims(...)
adjudicate_claims_with_search(...)
```

A `DefaultPipelineSteps` implementation is provided and used by default. Swapping it out allows custom implementations for testing or alternative integrations.

### EvidenceGatherer protocol

```python
class EvidenceGatherer(Protocol):
    async def gather(claim: ExtractedClaim) -> List[Citation]
    def gather_sync(claim: ExtractedClaim) -> List[Citation]
    source_name: str
```

Any class implementing this interface can be added as a new evidence source without touching the pipeline logic.

---

## Configuration Reference

`PipelineConfig` controls all pipeline behavior:

| Key | Controls |
|-----|---------|
| `claim_extraction_llm_config` | LLM used for claim extraction |
| `adjudication_llm_config` | LLM used for adjudication |
| `fallback_llm_config` | LLM used for no-claims fallback |
| `max_links_to_expand` | How many URLs to scrape |
| `link_content_expander_timeout_per_link` | Per-URL scraping timeout |
| `link_content_expander_timeout_total` | Total link expansion timeout |
| `enable_adjudication_with_search` | Toggle fallback adjudication path |
