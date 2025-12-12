"""
thread-based job queue system for fact-checking pipeline.

provides priority-based job scheduling using ThreadPoolExecutor and PriorityQueue,
with async bridge for awaitable completion and structured observability.
"""

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# default max workers optimized for Apify free tier (8GB RAM)
# with complex actors needing 1-2GB, safer to limit to 4 concurrent jobs
# can be overridden via THREAD_POOL_MAX_WORKERS env var
DEFAULT_MAX_WORKERS = int(os.getenv("THREAD_POOL_MAX_WORKERS", "4"))


class OperationType(Enum):
    """
    operation types for fact-checking pipeline with priority weights.

    higher weight = higher priority in job queue.
    """
    CLAIMS_EXTRACTION = 10               # highest priority - critical path
    ADJUDICATION_WITH_SEARCH = 8         # high priority - final adjudication with real-time search
    LINK_EXPANSION_PIPELINE = 6          # high priority - full link expansion + DataSource creation
    LINK_CONTEXT_EXPANDING = 5           # medium priority - individual URL scraping
    LINK_EVIDENCE_RETRIEVER = 3          # lowest priority - evidence retrieval

    @property
    def weight(self) -> int:
        """get priority weight for this operation type."""
        return self.value


@dataclass(order=True)
class Job:
    """
    represents a job in the priority queue.

    jobs are ordered by priority (higher weight first), then by creation time (FIFO).
    """
    # priority field for heap ordering (negated for max-heap behavior)
    priority: int = field(init=False, compare=True)

    # actual job data (not used in comparison)
    id: str = field(compare=False)
    operation_type: OperationType = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    future: Future = field(default_factory=Future, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    pipeline_id: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        """calculate priority after initialization."""
        # negate weight for max-heap (higher weight = lower priority value = processed first)
        # add small timestamp component to break ties with FIFO
        self.priority = -self.operation_type.weight - (self.created_at / 1e10)


class ThreadPoolManager:
    """
    singleton thread pool manager with priority-based job scheduling.

    manages a fixed-size thread pool and dispatches jobs from a priority queue.
    threads are pre-allocated and named for observability.
    """

    _instance: Optional["ThreadPoolManager"] = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = None):
        """
        initialize thread pool manager.

        args:
            max_workers: number of worker threads to pre-allocate
                        (default: 10, optimized for Apify 8GB RAM free tier)
        """
        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()

        # job tracking
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.running_jobs_lock = threading.Lock()

        # completion queues for consumer pattern (per operation type)
        self.completion_queues: Dict[OperationType, queue.Queue] = {
            op_type: queue.Queue() for op_type in OperationType
        }
        # global completion queue for all operations
        self.global_completion_queue: queue.Queue = queue.Queue()

        # dispatcher control
        self.dispatcher_thread: Optional[threading.Thread] = None
        self.dispatcher_running = False
        self.shutdown_event = threading.Event()

        self._initialized = False

    @classmethod
    def get_instance(cls, max_workers: int = None) -> "ThreadPoolManager":
        """
        get or create singleton instance (thread-safe).

        args:
            max_workers: number of worker threads (only used on first call)
                        (default: 10, optimized for Apify 8GB RAM free tier)

        returns:
            ThreadPoolManager singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers=max_workers)
        return cls._instance

    def initialize(self):
        """
        initialize thread pool and start dispatcher.

        creates ThreadPoolExecutor with named threads and starts background
        dispatcher thread. safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # create thread pool with named threads
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="FactCheck-Worker"
            )

            # start dispatcher thread
            self.dispatcher_running = True
            self.shutdown_event.clear()
            self.dispatcher_thread = threading.Thread(
                target=self._dispatch_loop,
                name="FactCheck-Dispatcher",
                daemon=True
            )
            self.dispatcher_thread.start()

            self._initialized = True

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        shutdown thread pool and dispatcher.

        args:
            wait: if True, wait for running jobs to complete
            timeout: maximum time to wait in seconds (None = wait forever)
        """
        if not self._initialized:
            return

        # stop dispatcher
        self.dispatcher_running = False
        self.shutdown_event.set()

        # wait for dispatcher to finish
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.dispatcher_thread.join(timeout=5.0)

        # shutdown executor
        if self.executor:
            self.executor.shutdown(wait=wait, cancel_futures=not wait)
            self.executor = None

        self._initialized = False

    def submit(
        self,
        operation_type: OperationType,
        func: Callable,
        *args,
        pipeline_id: Optional[str] = None,
        **kwargs
    ) -> Future:
        """
        submit a job to the priority queue.

        args:
            operation_type: type of operation (determines priority)
            func: function to execute
            *args: positional arguments for func
            pipeline_id: optional pipeline ID for request isolation
            **kwargs: keyword arguments for func

        returns:
            Future that will contain the result when job completes

        example:
            >>> manager = ThreadPoolManager.get_instance()
            >>> manager.initialize()
            >>> future = manager.submit(
            ...     OperationType.CLAIMS_EXTRACTION,
            ...     extract_claims,
            ...     text="some text",
            ...     pipeline_id="request-123"
            ... )
            >>> result = future.result()  # blocks until complete
        """
        if not self._initialized:
            raise RuntimeError("thread pool manager not initialized. call initialize() first")

        # create job
        job = Job(
            id=str(uuid.uuid4()),
            operation_type=operation_type,
            func=func,
            args=args,
            kwargs=kwargs,
            pipeline_id=pipeline_id
        )

        # add to priority queue
        self.job_queue.put(job)

        return job.future

    async def submit_async(
        self,
        operation_type: OperationType,
        func: Callable,
        *args,
        pipeline_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        submit a job and await completion (async bridge).

        args:
            operation_type: type of operation (determines priority)
            func: function to execute (can be sync or async)
            *args: positional arguments for func
            pipeline_id: optional pipeline ID for request isolation
            **kwargs: keyword arguments for func

        returns:
            result from func

        example:
            >>> manager = ThreadPoolManager.get_instance()
            >>> manager.initialize()
            >>> result = await manager.submit_async(
            ...     OperationType.CLAIMS_EXTRACTION,
            ...     extract_claims,
            ...     text="some text",
            ...     pipeline_id="request-123"
            ... )
        """
        future = self.submit(operation_type, func, *args, pipeline_id=pipeline_id, **kwargs)

        # bridge sync Future to async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, future.result)

    def _dispatch_loop(self):
        """
        background dispatcher loop.

        continuously pulls jobs from priority queue and submits them to thread pool.
        runs in dedicated dispatcher thread.
        """
        while self.dispatcher_running or not self.job_queue.empty():
            try:
                # get next job from priority queue (1 second timeout)
                job = self.job_queue.get(timeout=1.0)

                # track running job
                with self.running_jobs_lock:
                    self.running_jobs[job.id] = job

                # submit to thread pool
                self.executor.submit(self._execute_job, job)

                # logger.debug can be slow, use print for dispatcher debugging
                # print(f"[DISPATCHER] Dispatched {job.operation_type.name} job {job.id}")

            except queue.Empty:
                # no jobs available, continue loop
                continue
            except Exception as e:
                logger.error(f"dispatcher error: {e}", exc_info=True)

    def _execute_job(self, job: Job):
        """
        execute a job and set result/exception on future.

        args:
            job: job to execute
        """
        start_time = time.time()

        try:
            # execute function
            result = job.func(*job.args, **job.kwargs)

            # set result on future
            job.future.set_result(result)

            # put result in completion queues for consumer pattern
            self.completion_queues[job.operation_type].put((job.id, result))
            self.global_completion_queue.put((job.operation_type, job.id, result))

        except Exception as e:
            # set exception on future
            job.future.set_exception(e)

            # put exception in completion queues for consumer pattern
            self.completion_queues[job.operation_type].put((job.id, e))
            self.global_completion_queue.put((job.operation_type, job.id, e))

            # only print on error for debugging
            elapsed = time.time() - start_time
            print(f"[THREAD ERROR] Job {job.id} ({job.operation_type.name}) FAILED in {elapsed:.2f}s: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # move from running to completed
            with self.running_jobs_lock:
                if job.id in self.running_jobs:
                    del self.running_jobs[job.id]
                self.completed_jobs[job.id] = job

    def get_status(self) -> dict:
        """
        get current status of thread pool and job queue.

        returns:
            dict with queue_size, running_jobs, completed_jobs, max_workers
        """
        with self.running_jobs_lock:
            return {
                "queue_size": self.job_queue.qsize(),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": len(self.completed_jobs),
                "max_workers": self.max_workers,
                "initialized": self._initialized,
            }

    def wait_next_completed(
        self,
        operation_type: OperationType,
        timeout: Optional[float] = None,
        raise_on_error: bool = True,
        pipeline_id: Optional[str] = None,
    ) -> Any:
        """
        wait for and return result of next completed job of given operation type.

        useful for processing results as they arrive instead of waiting for all.

        args:
            operation_type: operation type to wait for
            timeout: max time to wait in seconds (None = wait forever)
            raise_on_error: if True, raise exception if job failed; if False, return exception object
            pipeline_id: optional pipeline ID for request isolation. if provided, only jobs
                        with matching pipeline_id will be returned. jobs with non-matching
                        pipeline_id will be put back in the queue for other waiters.

        returns:
            tuple of (job_id, result) where result is the job's return value or exception

        raises:
            TimeoutError: if timeout exceeded with no completion (via queue.Empty)
            Exception: if job failed and raise_on_error is True

        example:
            >>> # submit 10 claim extraction jobs for pipeline "req-123"
            >>> for claim in claims:
            ...     manager.submit(
            ...         OperationType.CLAIMS_EXTRACTION,
            ...         extract,
            ...         claim,
            ...         pipeline_id="req-123"
            ...     )
            >>>
            >>> # process results as they complete (streaming pattern)
            >>> for _ in range(10):
            ...     job_id, result = manager.wait_next_completed(
            ...         OperationType.CLAIMS_EXTRACTION,
            ...         pipeline_id="req-123"
            ...     )
            ...     print(f"job {job_id} completed: {result}")
        """
        # track non-matching jobs to put back
        non_matching_jobs = []

        try:
            while True:
                try:
                    job_id, result = self.completion_queues[operation_type].get(
                        block=True,
                        timeout=timeout
                    )

                    # if pipeline_id filtering is enabled, check if job matches
                    if pipeline_id is not None:
                        # look up the job to check its pipeline_id
                        with self.running_jobs_lock:
                            job = self.completed_jobs.get(job_id)

                        # if job doesn't match our pipeline_id, save it for later
                        if job is None or job.pipeline_id != pipeline_id:
                            non_matching_jobs.append((job_id, result))
                            print(f"trying to match job of type: {operation_type}")
                            continue  # try next job

                    # job matches (or no filtering) - return it
                    # if result is an exception, handle based on raise_on_error
                    if isinstance(result, Exception):
                        if raise_on_error:
                            raise result
                        return job_id, result

                    return job_id, result

                except queue.Empty:
                    raise TimeoutError(
                        f"no completed jobs of type {operation_type.name} "
                        f"{'for pipeline ' + pipeline_id if pipeline_id else ''} "
                        f"within {timeout}s"
                    )
        finally:
            # put non-matching jobs back into queue for other waiters
            for job_id, result in non_matching_jobs:
                self.completion_queues[operation_type].put((job_id, result))

    def clear_completed_jobs(
        self,
        pipeline_id: str,
        operation_type: Optional[OperationType] = None,
    ) -> int:
        """
        remove all completed jobs for a specific pipeline_id.

        this is a non-blocking cleanup operation that drains the completion queue
        for the specified pipeline without waiting for or processing results.

        args:
            pipeline_id: pipeline ID to filter by (required)
            operation_type: optional type of jobs to clear. if None, clears all operation types.

        returns:
            number of jobs cleared

        example:
            >>> # clear all completed jobs for request-123 (all operation types)
            >>> manager = ThreadPoolManager.get_instance()
            >>> cleared = manager.clear_completed_jobs(pipeline_id="request-123")
            >>> print(f"cleared {cleared} jobs")

            >>> # clear only claims extraction jobs for request-123
            >>> cleared = manager.clear_completed_jobs(
            ...     pipeline_id="request-123",
            ...     operation_type=OperationType.CLAIMS_EXTRACTION
            ... )
            >>> print(f"cleared {cleared} jobs")
        """
        cleared_count = 0

        # if operation_type is None, clear from all operation types
        if operation_type is None:
            operation_types = list(OperationType)
        else:
            operation_types = [operation_type]

        # drain jobs from each operation type
        for op_type in operation_types:
            while True:
                try:
                    # use very short timeout for non-blocking behavior
                    _job_id, _result = self.wait_next_completed(
                        operation_type=op_type,
                        timeout=0.001,  # 1ms timeout = effectively non-blocking
                        raise_on_error=False,  # don't raise on job errors
                        pipeline_id=pipeline_id
                    )
                    cleared_count += 1
                except TimeoutError:
                    # no more jobs available for this operation type
                    break

        return cleared_count

    def clear_completed_jobs_async(
        self,
        pipeline_id: str,
        operation_type: Optional[OperationType] = None,
    ) -> Future:
        """
        non-blocking background cleanup of completed jobs for a specific pipeline_id.

        this submits the cleanup as a background task and returns immediately,
        allowing the server to continue responding while cleanup happens in the background.

        args:
            pipeline_id: pipeline ID to filter by (required)
            operation_type: optional type of jobs to clear. if None, clears all operation types.

        returns:
            Future that will contain the number of jobs cleared when complete

        example:
            >>> # submit cleanup in background and continue
            >>> manager = ThreadPoolManager.get_instance()
            >>> future = manager.clear_completed_jobs_async(pipeline_id="request-123")
            >>> # server continues, cleanup happens in background
            >>> # optionally check result later
            >>> if future.done():
            ...     cleared = future.result()
            ...     print(f"cleared {cleared} jobs")
        """
        # use threading to avoid blocking the calling thread
        import threading

        def cleanup_task():
            """background cleanup task"""
            return self.clear_completed_jobs(
                pipeline_id=pipeline_id,
                operation_type=operation_type
            )

        # create future to track completion
        future = Future()

        def run_cleanup():
            """wrapper that sets result on future"""
            try:
                result = cleanup_task()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        # run in background thread
        cleanup_thread = threading.Thread(
            target=run_cleanup,
            daemon=True,  # daemon thread won't block shutdown
            name=f"cleanup-{pipeline_id}"
        )
        cleanup_thread.start()

        return future

    def wait_next_completed_any(
        self,
        timeout: Optional[float] = None,
        raise_on_error: bool = True,
    ) -> tuple:
        """
        wait for and return result of next completed job of any operation type.

        useful for processing results from mixed operation types as they arrive.

        args:
            timeout: max time to wait in seconds (None = wait forever)
            raise_on_error: if True, raise exception if job failed; if False, return exception object

        returns:
            tuple of (operation_type, job_id, result)

        raises:
            TimeoutError: if timeout exceeded with no completion (via queue.Empty)
            Exception: if job failed and raise_on_error is True

        example:
            >>> # submit mixed jobs
            >>> manager.submit(OperationType.CLAIMS_EXTRACTION, extract, claim1)
            >>> manager.submit(OperationType.LINK_CONTEXT_EXPANDING, expand, link1)
            >>> manager.submit(OperationType.CLAIMS_EXTRACTION, extract, claim2)
            >>>
            >>> # process ANY result as it completes
            >>> for _ in range(3):
            ...     op_type, job_id, result = manager.wait_next_completed_any()
            ...     print(f"{op_type.name} job {job_id}: {result}")
        """
        try:
            operation_type, job_id, result = self.global_completion_queue.get(
                block=True,
                timeout=timeout
            )

            # if result is an exception, handle based on raise_on_error
            if isinstance(result, Exception):
                if raise_on_error:
                    raise result
                return operation_type, job_id, result

            return operation_type, job_id, result

        except queue.Empty:
            raise TimeoutError(
                f"no completed jobs of any type within {timeout}s"
            )


# helper functions


def map_threaded(
    items: List[T],
    func: Callable[[T], R],
    operation_type: OperationType,
    manager: Optional[ThreadPoolManager] = None,
) -> List[R]:
    """
    map function over items using thread pool (blocking).

    args:
        items: list of items to process
        func: function to apply to each item
        operation_type: operation type (determines priority)
        manager: thread pool manager (uses singleton if None)

    returns:
        list of results in same order as input items

    example:
        >>> def process_link(url: str) -> str:
        ...     return fetch_url(url)
        >>>
        >>> results = map_threaded(
        ...     items=urls,
        ...     func=process_link,
        ...     operation_type=OperationType.LINK_CONTEXT_EXPANDING
        ... )
    """
    if manager is None:
        manager = ThreadPoolManager.get_instance()

    if not manager._initialized:
        raise RuntimeError("thread pool manager not initialized")

    # submit all jobs
    futures = [
        manager.submit(operation_type, func, item)
        for item in items
    ]

    # wait for all to complete and collect results
    results = [future.result() for future in futures]

    return results


async def map_threaded_async(
    items: List[T],
    func: Callable[[T], R],
    operation_type: OperationType,
    manager: Optional[ThreadPoolManager] = None,
) -> List[R]:
    """
    map function over items using thread pool (async).

    args:
        items: list of items to process
        func: function to apply to each item
        operation_type: operation type (determines priority)
        manager: thread pool manager (uses singleton if None)

    returns:
        list of results in same order as input items

    example:
        >>> results = await map_threaded_async(
        ...     items=urls,
        ...     func=fetch_url,
        ...     operation_type=OperationType.LINK_CONTEXT_EXPANDING
        ... )
    """
    if manager is None:
        manager = ThreadPoolManager.get_instance()

    if not manager._initialized:
        raise RuntimeError("thread pool manager not initialized")

    # submit all jobs and await completion
    results = await asyncio.gather(*[
        manager.submit_async(operation_type, func, item)
        for item in items
    ])

    return list(results)


def wait_all(futures: List[Future], timeout: Optional[float] = None) -> List[Any]:
    """
    wait for all futures to complete and return results.

    args:
        futures: list of futures to wait for
        timeout: maximum time to wait in seconds (None = wait forever)

    returns:
        list of results in same order as futures

    raises:
        TimeoutError: if timeout is exceeded
    """
    results = []
    start_time = time.time()

    for future in futures:
        if timeout is not None:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(f"wait_all timed out after {timeout}s")
            result = future.result(timeout=remaining)
        else:
            result = future.result()

        results.append(result)

    return results


class ThreadPoolContext:
    """
    context manager for thread pool lifecycle.

    ensures proper initialization and cleanup of ThreadPoolManager.
    """

    def __init__(self, max_workers: int = 25):
        """
        initialize context manager.

        args:
            max_workers: number of worker threads
        """
        self.max_workers = max_workers
        self.manager: Optional[ThreadPoolManager] = None

    def __enter__(self) -> ThreadPoolManager:
        """initialize pool on context entry."""
        self.manager = ThreadPoolManager.get_instance(max_workers=self.max_workers)
        self.manager.initialize()
        return self.manager

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """cleanup pool on context exit."""
        if self.manager:
            self.manager.shutdown(wait=True)
        return False


async def __aenter__(self) -> ThreadPoolManager:
    """initialize pool on async context entry."""
    self.manager = ThreadPoolManager.get_instance(max_workers=self.max_workers)
    self.manager.initialize()
    return self.manager


async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
    """cleanup pool on async context exit."""
    if self.manager:
        self.manager.shutdown(wait=True)
    return False


# convenience function
def with_thread_pool(
    func: Callable[[ThreadPoolManager], R],
    max_workers: int = 25
) -> R:
    """
    execute function with automatic pool lifecycle management.

    args:
        func: function that takes ThreadPoolManager as argument
        max_workers: number of worker threads

    returns:
        result from func

    example:
        >>> def process_data(manager: ThreadPoolManager):
        ...     return map_threaded(
        ...         items=data,
        ...         func=process_item,
        ...         operation_type=OperationType.CLAIMS_EXTRACTION,
        ...         manager=manager
        ...     )
        >>>
        >>> results = with_thread_pool(process_data)
    """
    with ThreadPoolContext(max_workers=max_workers) as manager:
        return func(manager)
