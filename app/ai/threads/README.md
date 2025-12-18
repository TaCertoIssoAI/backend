
# async parallelization plan for fact-checking pipeline

## executive summary

this document outlines the strategy for parallelizing the IO-heavy fact-checking pipeline to achieve ~4x latency reduction (from 420s to 105s) through async concurrency and centralized resource management.

**status**: planning complete, implementation deferred

**key improvements**:
- parallel link expansion (10 links: 60s → 15s)
- parallel claim extraction (10 sources: 300s → 75s)
- parallel evidence gathering (15 claims × 3 gatherers: 60s → 15s)
- centralized async resource pool with semaphore limits
- real-time progress tracking and observability



## current state analysis

### sequential execution flow

```
link expansion (sequential)
├─ source 1: extract links → expand link 1 (6s) → expand link 2 (6s) → ... = 60s
└─ total: ~60s for 10 links

claim extraction (sequential)
├─ source 1: extract claims (30s)
├─ source 2: extract claims (30s)
└─ total: ~300s for 10 sources

evidence gathering (sequential)
├─ claim 1: google api (2s) + web search (2s) = 4s
├─ claim 2: google api (2s) + web search (2s) = 4s
└─ total: ~60s for 15 claims

TOTAL LATENCY: ~420s
```

### bottlenecks

1. **link expansion**: fetching 10 URLs takes 60s sequentially (6s each)
2. **claim extraction**: 10 LLM calls take 300s sequentially (30s each)
3. **evidence gathering**: 15 claims × 3 gatherers = 45 sequential API calls (60s)

all three stages are IO-bound and can be parallelized.



## proposed architecture

### parallel execution flow

```
link expansion (parallel with concurrency=5)
├─ batch 1: [link 1, link 2, link 3, link 4, link 5] → max(6s) = 6s
├─ batch 2: [link 6, link 7, link 8, link 9, link 10] → max(6s) = 6s
└─ total: ~15s (4x faster)

claim extraction (parallel with concurrency=4)
├─ batch 1: [source 1, source 2, source 3, source 4] → max(30s) = 30s
├─ batch 2: [source 5, source 6, source 7, source 8] → max(30s) = 30s
├─ batch 3: [source 9, source 10] → max(30s) = 30s
└─ total: ~90s (3.3x faster, limited by LLM quota)

evidence gathering (parallel with nested concurrency)
├─ per claim: [google, web, news] in parallel → max(2s) = 2s
├─ all claims: process 5 claims at once → 15 claims / 5 = 3 batches
└─ total: ~15s (4x faster)

TOTAL LATENCY: ~105s (4x improvement)
```

### resource management

**centralized async pool manager** (singleton pattern):
- manages httpx.AsyncClient lifecycle
- enforces semaphore limits per operation type
- provides observability hooks for monitoring

**semaphore limits**:
- link expansion: 5 concurrent fetches
- claim extraction: 4 concurrent LLM calls (rate limit protection)
- evidence gathering: 5 concurrent claims, 3 gatherers per claim



## implementation plan

### phase 1: core async utilities

**new files**:
- `app/ai/threads/pool_manager.py` - centralized async resource manager
- `app/ai/threads/parallel_helpers.py` - reusable parallel execution utilities
- `app/ai/threads/progress.py` - progress tracking for long operations

**code structure**:

```python
# pool_manager.py
class AsyncPoolManager:
    """singleton async resource pool with semaphore limits."""

    _instance: Optional["AsyncPoolManager"] = None

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.link_expansion_semaphore = asyncio.Semaphore(5)
        self.claim_extraction_semaphore = asyncio.Semaphore(4)
        self.evidence_gathering_semaphore = asyncio.Semaphore(5)

    @classmethod
    def get_instance(cls) -> "AsyncPoolManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """initialize async resources."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """cleanup async resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None


# parallel_helpers.py
async def map_async_with_concurrency[T, R](
    items: List[T],
    async_fn: Callable[[T], Awaitable[R]],
    semaphore: asyncio.Semaphore,
    description: str = "processing"
) -> List[R]:
    """
    map async function over items with semaphore-based concurrency control.

    args:
        items: list of items to process
        async_fn: async function to apply to each item
        semaphore: semaphore for concurrency limit
        description: description for progress tracking

    returns:
        list of results in same order as input items
    """
    async def _bounded_task(item: T) -> R:
        async with semaphore:
            return await async_fn(item)

    tasks = [_bounded_task(item) for item in items]

    # use asyncio.gather to run all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # handle exceptions: log and convert to None or re-raise
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"task {i} failed: {result}")
            processed_results.append(None)  # or re-raise
        else:
            processed_results.append(result)

    return processed_results


# progress.py
class ProgressTracker:
    """track progress of parallel operations with real-time updates."""

    def __init__(self, total: int, description: str):
        self.total = total
        self.completed = 0
        self.description = description
        self.lock = asyncio.Lock()

    async def increment(self, count: int = 1):
        async with self.lock:
            self.completed += count
            self._print_progress()

    def _print_progress(self):
        pct = (self.completed / self.total) * 100 if self.total > 0 else 0
        print(f"[{self.description}] {self.completed}/{self.total} ({pct:.1f}%)")
```

### phase 2: parallelize link expansion

**modify**: `app/ai/pipeline/link_context_expander.py`

**changes**:

```python
async def expand_link_contexts(
    data_source: DataSource,
    config: PipelineConfig
) -> List[DataSource]:
    """expand links from original text into link_context data sources (parallel)."""

    # extract URLs
    urls = extract_urls_from_text(data_source.original_text)
    if not urls:
        return []

    # get pool manager
    pool = AsyncPoolManager.get_instance()
    await pool.initialize()

    # parallel fetch with semaphore
    async def _fetch_one_url(url: str) -> Optional[DataSource]:
        async with pool.link_expansion_semaphore:
            return await fetch_and_create_link_context(url, data_source, config)

    # use asyncio.gather for parallel execution
    results = await asyncio.gather(
        *[_fetch_one_url(url) for url in urls],
        return_exceptions=True
    )

    # filter out None and exceptions
    expanded_sources = [r for r in results if isinstance(r, DataSource)]

    return expanded_sources
```

**expected improvement**: 60s → 15s (4x faster for 10 links with concurrency=5)

### phase 3: parallelize claim extraction

**modify**: `app/ai/pipeline/steps.py` (DefaultPipelineSteps)

**changes**:

```python
async def extract_claims_from_all_sources(
    self,
    data_sources: List[DataSource],
    llm_config: LLMConfig
) -> List[ClaimExtractionOutput]:
    """extract claims from all sources in parallel."""

    pool = AsyncPoolManager.get_instance()

    async def _extract_one_source(source: DataSource) -> ClaimExtractionOutput:
        extraction_input = ClaimExtractionInput(data_source=source)

        # semaphore protects LLM rate limits
        async with pool.claim_extraction_semaphore:
            return await self._extract_claims(extraction_input, llm_config)

    # parallel execution
    claim_outputs = await map_async_with_concurrency(
        items=data_sources,
        async_fn=_extract_one_source,
        semaphore=pool.claim_extraction_semaphore,
        description="claim extraction"
    )

    return claim_outputs
```

**expected improvement**: 300s → 75s (4x faster with concurrency=4)

### phase 4: parallelize evidence gathering

**modify**: `app/ai/pipeline/evidence_retrieval.py`

**changes**:

```python
async def gather_evidence_async(
    retrieval_input: EvidenceRetrievalInput,
    gatherers: List[EvidenceGatherer]
) -> EvidenceRetrievalResult:
    """gather evidence for all claims in parallel."""

    pool = AsyncPoolManager.get_instance()

    async def _gather_for_one_claim(claim: ExtractedClaim) -> EnrichedClaim:
        # for each claim, run all gatherers in parallel
        async def _run_one_gatherer(gatherer: EvidenceGatherer) -> List[Citation]:
            return await gatherer.gather(claim)

        # nested parallelism: all gatherers for this claim run concurrently
        all_citations_nested = await asyncio.gather(
            *[_run_one_gatherer(g) for g in gatherers],
            return_exceptions=True
        )

        # flatten citations
        all_citations = []
        for cits in all_citations_nested:
            if isinstance(cits, list):
                all_citations.extend(cits)

        return EnrichedClaim(
            id=claim.id,
            text=claim.text,
            source=claim.source,
            entities=claim.entities,
            llm_comment=claim.llm_comment,
            citations=all_citations
        )

    # parallel across claims with semaphore
    enriched_claims = await map_async_with_concurrency(
        items=retrieval_input.claims,
        async_fn=_gather_for_one_claim,
        semaphore=pool.evidence_gathering_semaphore,
        description="evidence gathering"
    )

    # build result map
    claim_evidence_map = {
        claim.id: claim for claim in enriched_claims
    }

    return EvidenceRetrievalResult(claim_evidence_map=claim_evidence_map)
```

**expected improvement**: 60s → 15s (4x faster with concurrency=5 claims, 3 gatherers each)

### phase 5: integration and cleanup

**modify**: `app/ai/main_pipeline.py`

**changes**:

```python
async def run_fact_check_pipeline(
    data_sources: List[DataSource],
    config: PipelineConfig,
    steps: PipelineSteps,
) -> List[ClaimExtractionOutput]:
    """run the fact-checking pipeline with async parallelization."""

    # initialize async pool
    pool = AsyncPoolManager.get_instance()
    await pool.initialize()

    try:
        # existing pipeline logic (now with parallel steps)
        expanded_link_sources = await steps.expand_data_sources_with_links(
            data_sources, config
        )

        all_data_sources = list(data_sources) + expanded_link_sources

        claim_outputs = await steps.extract_claims_from_all_sources(
            data_sources=all_data_sources,
            llm_config=config.claim_extraction_llm_config
        )

        return claim_outputs

    finally:
        # cleanup resources
        await pool.cleanup()
```

**add lifecycle management in FastAPI app**:

```python
# app/main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    pool = AsyncPoolManager.get_instance()
    await pool.initialize()
    yield
    # shutdown
    await pool.cleanup()

app = FastAPI(lifespan=lifespan)
```



## testing strategy

### unit tests

1. **pool_manager_test.py**:
   - singleton behavior
   - semaphore limits enforced
   - resource initialization and cleanup

2. **parallel_helpers_test.py**:
   - map_async_with_concurrency with mock async functions
   - exception handling (partial failures)
   - semaphore respects concurrency limits

3. **progress_test.py**:
   - thread-safe increment operations
   - accurate progress calculations

### integration tests

1. **parallel_link_expansion_test.py**:
   - expand 10 URLs in parallel
   - measure latency improvement (should be ~4x faster)
   - verify all results are correct

2. **parallel_claim_extraction_test.py**:
   - extract claims from 10 sources in parallel
   - verify LLM rate limits are respected
   - check result quality matches sequential version

3. **parallel_evidence_gathering_test.py**:
   - gather evidence for 15 claims with 3 gatherers
   - verify nested parallelism (claims + gatherers)
   - measure latency improvement

### load tests

1. **stress test with 100 claims**:
   - verify semaphores prevent resource exhaustion
   - check memory usage stays bounded
   - ensure no deadlocks or race conditions

2. **real-world scenario**:
   - run full pipeline with 5 original sources (each with 2 links)
   - measure end-to-end latency
   - target: <120s (vs current ~420s)



## migration strategy

### rollout phases

**week 1-2**: implement core utilities (phase 1)
- create pool_manager.py, parallel_helpers.py, progress.py
- write unit tests
- review and merge

**week 3**: parallelize link expansion (phase 2)
- modify link_context_expander.py
- add integration tests
- measure latency improvements
- feature flag: `PARALLEL_LINK_EXPANSION=true`

**week 4**: parallelize claim extraction (phase 3)
- modify steps.py
- add integration tests
- feature flag: `PARALLEL_CLAIM_EXTRACTION=true`

**week 5**: parallelize evidence gathering (phase 4)
- modify evidence_retrieval.py
- add integration tests
- feature flag: `PARALLEL_EVIDENCE_GATHERING=true`

**week 6**: full integration and cleanup (phase 5)
- modify main_pipeline.py
- add FastAPI lifecycle management
- end-to-end load tests
- remove feature flags, make parallel execution default

### feature flags

use environment variables for gradual rollout:

```python
# config.py
PARALLEL_LINK_EXPANSION = os.getenv("PARALLEL_LINK_EXPANSION", "false").lower() == "true"
PARALLEL_CLAIM_EXTRACTION = os.getenv("PARALLEL_CLAIM_EXTRACTION", "false").lower() == "true"
PARALLEL_EVIDENCE_GATHERING = os.getenv("PARALLEL_EVIDENCE_GATHERING", "false").lower() == "true"
```

fallback to sequential execution if flags are disabled.



## performance expectations

### latency improvements

| stage | current (sequential) | proposed (parallel) | speedup |
|-------|---------------------|---------------------|---------|
| link expansion (10 links) | 60s | 15s | 4x |
| claim extraction (10 sources) | 300s | 75s | 4x |
| evidence gathering (15 claims) | 60s | 15s | 4x |
| **total pipeline** | **420s** | **105s** | **4x** |

### resource usage

- **memory**: slight increase due to concurrent tasks (~10-20% more)
- **CPU**: minimal (IO-bound operations)
- **network**: concurrent connections limited by semaphores (safe)
- **LLM quota**: protected by claim_extraction_semaphore (max 4 concurrent)

### observability

- progress tracking: real-time completion percentages
- error rates: logged per operation type
- latency distribution: P50, P95, P99 per stage
- semaphore saturation: track how often limits are hit



## risks and mitigations

### risk 1: rate limit violations

**impact**: LLM provider or API throttling errors

**mitigation**:
- use semaphores to enforce strict concurrency limits
- implement exponential backoff in parallel_helpers.py
- monitor rate limit headers and adjust semaphores dynamically

### risk 2: resource exhaustion

**impact**: too many concurrent connections, memory spikes

**mitigation**:
- centralized pool manager with bounded resources
- semaphore limits tuned conservatively (start with 4-5 concurrent)
- integration tests with stress scenarios (100+ claims)

### risk 3: partial failures

**impact**: some tasks fail, others succeed - inconsistent results

**mitigation**:
- asyncio.gather with return_exceptions=True
- log all exceptions with context (claim ID, source ID)
- return partial results with clear indication of failures
- retry logic for transient errors (network timeouts)

### risk 4: debugging complexity

**impact**: harder to trace execution flow with concurrent tasks

**mitigation**:
- add request_id to all log messages
- progress tracker shows which tasks are running
- structured logging with correlation IDs
- keep sequential execution as fallback (feature flags)



## future optimizations

### stream processing (advanced)

for very large batches (100+ claims), implement streaming:

```python
async def stream_process_claims(
    data_sources: AsyncIterator[DataSource],
    llm_config: LLMConfig
) -> AsyncIterator[ClaimExtractionOutput]:
    """process claims as they arrive, don't wait for all sources."""

    async for source in data_sources:
        result = await extract_claims(source, llm_config)
        yield result
```

**benefits**:
- start evidence gathering while still extracting claims
- reduce peak memory usage
- lower time-to-first-result

### adaptive concurrency

dynamically adjust semaphore limits based on:
- current system load (CPU, memory)
- rate limit headers from providers
- historical latency data

### caching layer

cache expanded link contexts and LLM responses:
- redis for short-term cache (1 hour)
- database for long-term cache (similar claims)
- cache key: hash of claim text + source type



## conclusion

this parallelization plan provides:
- **4x latency reduction** (420s → 105s)
- **bounded resource usage** with semaphore limits
- **incremental rollout** with feature flags
- **strong observability** and error handling
- **maintainable code** with centralized async utilities

**next steps**:
1. review and approve this plan
2. create app/ai/threads/ directory
3. implement phase 1 (core utilities)
4. proceed with phases 2-5 over 6 weeks

**status**: ready for implementation when prioritized.
