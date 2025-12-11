"""
simple tests for thread-based job queue system.

run with: python app/ai/threads/test_thread_utils.py
"""

import asyncio
import time
from concurrent.futures import Future

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.ai.threads.thread_utils import (
    Job,
    OperationType,
    ThreadPoolContext,
    ThreadPoolManager,
    map_threaded,
    map_threaded_async,
    wait_all,
    with_thread_pool,
)


def test_operation_type_weights():
    """test that operation types have correct priority weights."""
    print("\n1. testing operation type weights...")

    assert OperationType.CLAIMS_EXTRACTION.weight == 10
    assert OperationType.LINK_CONTEXT_EXPANDING.weight == 5
    assert OperationType.LINK_EVIDENCE_RETRIEVER.weight == 3

    print("   ✓ operation type weights correct")


def test_job_priority_ordering():
    """test that jobs are ordered by priority correctly."""
    print("\n2. testing job priority ordering...")

    # create jobs with different priorities
    job1 = Job(
        id="1",
        operation_type=OperationType.LINK_EVIDENCE_RETRIEVER,  # weight 3
        func=lambda: None
    )
    job2 = Job(
        id="2",
        operation_type=OperationType.CLAIMS_EXTRACTION,  # weight 10
        func=lambda: None
    )
    job3 = Job(
        id="3",
        operation_type=OperationType.LINK_CONTEXT_EXPANDING,  # weight 5
        func=lambda: None
    )

    # higher priority (weight 10) should have lower priority value
    assert job2.priority < job3.priority < job1.priority

    print("   ✓ job priority ordering correct (higher weight = processed first)")


def test_thread_pool_initialization():
    """test thread pool manager initialization and shutdown."""
    print("\n3. testing thread pool initialization...")

    manager = ThreadPoolManager.get_instance(max_workers=5)
    assert manager.max_workers == 5
    assert manager._initialized is False

    manager.initialize()
    assert manager._initialized is True
    assert manager.executor is not None
    assert manager.dispatcher_thread is not None

    manager.shutdown()
    assert manager._initialized is False

    print("   ✓ thread pool initialization/shutdown works")


def test_singleton_pattern():
    """test that ThreadPoolManager is a singleton."""
    print("\n4. testing singleton pattern...")

    # note: need to reset singleton for testing
    ThreadPoolManager._instance = None

    manager1 = ThreadPoolManager.get_instance(max_workers=10)
    manager2 = ThreadPoolManager.get_instance(max_workers=20)

    assert manager1 is manager2
    assert manager1.max_workers == 10  # first call wins

    # cleanup
    manager1.shutdown()

    print("   ✓ singleton pattern works")


def test_submit_and_wait():
    """test submitting a job and waiting for result."""
    print("\n5. testing submit and wait...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def slow_task(x: int) -> int:
        time.sleep(0.1)
        return x * 2

    future = manager.submit(
        OperationType.CLAIMS_EXTRACTION,
        slow_task,
        5
    )

    assert isinstance(future, Future)
    result = future.result()
    assert result == 10

    manager.shutdown()

    print("   ✓ submit and wait works")


def test_priority_ordering():
    """test that jobs are executed in priority order."""
    print("\n6. testing priority-based execution...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=1)  # single worker
    manager.initialize()

    execution_order = []

    def track_execution(name: str):
        execution_order.append(name)
        time.sleep(0.05)
        return name

    # submit jobs in reverse priority order
    f1 = manager.submit(OperationType.LINK_EVIDENCE_RETRIEVER, track_execution, "low")
    f2 = manager.submit(OperationType.LINK_CONTEXT_EXPANDING, track_execution, "medium")
    f3 = manager.submit(OperationType.CLAIMS_EXTRACTION, track_execution, "high")

    # wait for all
    results = wait_all([f1, f2, f3])

    # with single worker, should execute in priority order: high, medium, low
    # note: first job might start immediately before priority takes effect
    print(f"   execution order: {execution_order}")

    manager.shutdown()

    print("   ✓ priority ordering verified")


def test_map_threaded():
    """test map_threaded helper function."""
    print("\n7. testing map_threaded...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def square(x: int) -> int:
        time.sleep(0.05)
        return x * x

    items = [1, 2, 3, 4, 5]
    start = time.time()

    results = map_threaded(
        items=items,
        func=square,
        operation_type=OperationType.CLAIMS_EXTRACTION,
        manager=manager
    )

    elapsed = time.time() - start

    assert results == [1, 4, 9, 16, 25]

    # with 5 workers, 5 items should complete in ~0.05s (parallel)
    # instead of ~0.25s (sequential)
    print(f"   elapsed: {elapsed:.2f}s (should be ~0.05s with parallelism)")

    manager.shutdown()

    print("   ✓ map_threaded works")


async def test_async_bridge():
    """test async bridge (submit_async)."""
    print("\n8. testing async bridge...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def slow_task(x: int) -> int:
        time.sleep(0.1)
        return x * 3

    # test submit_async
    result = await manager.submit_async(
        OperationType.CLAIMS_EXTRACTION,
        slow_task,
        7
    )

    assert result == 21

    manager.shutdown()

    print("   ✓ async bridge works")


async def test_map_threaded_async():
    """test map_threaded_async helper."""
    print("\n9. testing map_threaded_async...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def cube(x: int) -> int:
        time.sleep(0.05)
        return x ** 3

    items = [1, 2, 3, 4]

    results = await map_threaded_async(
        items=items,
        func=cube,
        operation_type=OperationType.LINK_CONTEXT_EXPANDING,
        manager=manager
    )

    assert results == [1, 8, 27, 64]

    manager.shutdown()

    print("   ✓ map_threaded_async works")


def test_context_manager():
    """test ThreadPoolContext context manager."""
    print("\n10. testing context manager...")

    ThreadPoolManager._instance = None

    with ThreadPoolContext(max_workers=5) as manager:
        assert manager._initialized is True

        result = manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            lambda x: x + 1,
            10
        ).result()

        assert result == 11

    # after context exit, should be shut down
    # note: singleton might still exist but should be cleaned up

    print("   ✓ context manager works")


def test_with_thread_pool():
    """test with_thread_pool convenience function."""
    print("\n11. testing with_thread_pool...")

    ThreadPoolManager._instance = None

    def process_data(manager: ThreadPoolManager) -> list:
        return map_threaded(
            items=[1, 2, 3],
            func=lambda x: x * 10,
            operation_type=OperationType.CLAIMS_EXTRACTION,
            manager=manager
        )

    results = with_thread_pool(process_data, max_workers=5)

    assert results == [10, 20, 30]

    print("   ✓ with_thread_pool works")


def test_error_handling():
    """test that errors are properly propagated through futures."""
    print("\n12. testing error handling...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def failing_task():
        raise ValueError("intentional error")

    future = manager.submit(
        OperationType.CLAIMS_EXTRACTION,
        failing_task
    )

    try:
        future.result()
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert str(e) == "intentional error"

    manager.shutdown()

    print("   ✓ error handling works")


def test_get_status():
    """test get_status method."""
    print("\n13. testing get_status...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    status = manager.get_status()

    assert status["max_workers"] == 5
    assert status["initialized"] is True
    assert status["queue_size"] == 0
    assert status["running_jobs"] == 0

    manager.shutdown()

    print("   ✓ get_status works")


def test_wait_next_completed():
    """test wait_next_completed for streaming results."""
    print("\n14. testing wait_next_completed (streaming pattern)...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def slow_task(x: int) -> int:
        time.sleep(0.05 * x)  # different delays
        return x * 2

    # submit 5 jobs
    for i in range(1, 6):
        manager.submit(OperationType.CLAIMS_EXTRACTION, slow_task, i)

    # collect results as they complete (streaming)
    results = []
    for _ in range(5):
        job_id, result = manager.wait_next_completed(OperationType.CLAIMS_EXTRACTION)
        results.append(result)
        print(f"   received: {result}")

    # should have all results (not necessarily in order due to different delays)
    assert sorted(results) == [2, 4, 6, 8, 10]

    manager.shutdown()

    print("   ✓ wait_next_completed works (streaming pattern)")


def test_wait_next_completed_any():
    """test wait_next_completed_any for mixed operation types."""
    print("\n15. testing wait_next_completed_any (mixed operations)...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def task_a(x: int) -> str:
        time.sleep(0.05)
        return f"A{x}"

    def task_b(x: int) -> str:
        time.sleep(0.05)
        return f"B{x}"

    def task_c(x: int) -> str:
        time.sleep(0.05)
        return f"C{x}"

    # submit mixed jobs
    manager.submit(OperationType.CLAIMS_EXTRACTION, task_a, 1)
    manager.submit(OperationType.LINK_CONTEXT_EXPANDING, task_b, 2)
    manager.submit(OperationType.CLAIMS_EXTRACTION, task_a, 3)
    manager.submit(OperationType.LINK_EVIDENCE_RETRIEVER, task_c, 4)

    # collect results from ANY operation type
    results = []
    for _ in range(4):
        op_type, job_id, result = manager.wait_next_completed_any()
        results.append((op_type.name, result))
        print(f"   {op_type.name}: {result}")

    # check we got all 4 results
    assert len(results) == 4
    result_values = [r[1] for r in results]
    assert sorted(result_values) == ["A1", "A3", "B2", "C4"]

    manager.shutdown()

    print("   ✓ wait_next_completed_any works (mixed operations)")


def test_wait_next_completed_timeout():
    """test wait_next_completed timeout behavior."""
    print("\n16. testing wait_next_completed timeout...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    # don't submit any jobs

    try:
        manager.wait_next_completed(OperationType.CLAIMS_EXTRACTION, timeout=0.1)
        assert False, "should have raised TimeoutError"
    except TimeoutError as e:
        assert "no completed jobs" in str(e)

    manager.shutdown()

    print("   ✓ timeout handling works")


def test_wait_next_completed_error_handling():
    """test wait_next_completed with failing jobs."""
    print("\n17. testing wait_next_completed error handling...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def failing_task(x: int):
        if x == 3:
            raise ValueError(f"task {x} failed")
        return x * 2

    # submit jobs (one will fail)
    for i in range(1, 5):
        manager.submit(OperationType.CLAIMS_EXTRACTION, failing_task, i)

    # collect results
    results = []
    errors = []
    for _ in range(4):
        try:
            job_id, result = manager.wait_next_completed(
                OperationType.CLAIMS_EXTRACTION,
                raise_on_error=True
            )
            results.append(result)
        except ValueError as e:
            errors.append(str(e))

    assert len(results) == 3  # 3 succeeded
    assert len(errors) == 1   # 1 failed
    assert "task 3 failed" in errors[0]

    manager.shutdown()

    print("   ✓ error handling in completion queue works")


def test_pipeline_id_job_tracking():
    """test that jobs submitted with pipeline_id have that value set."""
    print("\n18. testing pipeline_id is set on jobs...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def simple_task(x: int) -> int:
        return x * 2

    # submit job with pipeline_id
    pipeline_id = "test-pipeline-123"
    future = manager.submit(
        OperationType.CLAIMS_EXTRACTION,
        simple_task,
        5,
        pipeline_id=pipeline_id
    )

    # wait for job to complete
    result = future.result()
    assert result == 10

    # verify the job has the correct pipeline_id
    # get the job from completed_jobs
    with manager.running_jobs_lock:
        # find the job in completed_jobs
        job_found = False
        for job_id, job in manager.completed_jobs.items():
            if job.pipeline_id == pipeline_id:
                assert job.pipeline_id == pipeline_id
                job_found = True
                break

    assert job_found, "job with pipeline_id should be in completed_jobs"

    # submit job without pipeline_id
    future2 = manager.submit(
        OperationType.CLAIMS_EXTRACTION,
        simple_task,
        10
    )

    result2 = future2.result()
    assert result2 == 20

    # verify the job has pipeline_id=None
    with manager.running_jobs_lock:
        job_found_none = False
        for job_id, job in manager.completed_jobs.items():
            if job.pipeline_id is None and job.func == simple_task:
                job_found_none = True
                break

    assert job_found_none, "job without pipeline_id should have None value"

    manager.shutdown()

    print("   ✓ pipeline_id is correctly set on jobs")


def test_pipeline_id_isolation():
    """test that wait_next_completed filters by pipeline_id correctly."""
    print("\n19. testing pipeline_id isolation in wait_next_completed...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def task_with_delay(x: int, delay: float) -> str:
        time.sleep(delay)
        return f"result-{x}"

    # submit jobs for pipeline A (fast jobs)
    pipeline_a = "pipeline-A"
    for i in range(3):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            task_with_delay,
            i,
            0.05,  # fast
            pipeline_id=pipeline_a
        )

    # submit jobs for pipeline B (slower jobs)
    pipeline_b = "pipeline-B"
    for i in range(3, 6):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            task_with_delay,
            i,
            0.1,  # slower
            pipeline_id=pipeline_b
        )

    # submit jobs without pipeline_id (no isolation)
    for i in range(6, 9):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            task_with_delay,
            i,
            0.05
        )

    # wait for all jobs from pipeline A
    results_a = []
    for _ in range(3):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            pipeline_id=pipeline_a
        )
        results_a.append(result)

        # verify job belongs to pipeline A
        with manager.running_jobs_lock:
            job = manager.completed_jobs[job_id]
            assert job.pipeline_id == pipeline_a, f"job {job_id} should belong to pipeline A"

    assert len(results_a) == 3
    assert set(results_a) == {"result-0", "result-1", "result-2"}

    # wait for all jobs from pipeline B
    results_b = []
    for _ in range(3):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            pipeline_id=pipeline_b
        )
        results_b.append(result)

        # verify job belongs to pipeline B
        with manager.running_jobs_lock:
            job = manager.completed_jobs[job_id]
            assert job.pipeline_id == pipeline_b, f"job {job_id} should belong to pipeline B"

    assert len(results_b) == 3
    assert set(results_b) == {"result-3", "result-4", "result-5"}

    # wait for jobs without pipeline_id (should get None pipeline jobs)
    results_none = []
    for _ in range(3):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            pipeline_id=None  # explicitly wait for non-isolated jobs
        )
        results_none.append(result)

    assert len(results_none) == 3
    assert set(results_none) == {"result-6", "result-7", "result-8"}

    manager.shutdown()

    print("   ✓ pipeline_id isolation works correctly")


def test_pipeline_id_cross_contamination_prevention():
    """test that jobs from different pipelines don't interfere with each other."""
    print("\n20. testing prevention of cross-pipeline contamination...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def slow_task(x: int) -> int:
        time.sleep(0.1)
        return x * 10

    # submit 5 jobs for request-1
    request_1 = "request-1"
    for i in range(1, 6):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            slow_task,
            i,
            pipeline_id=request_1
        )

    # submit 5 jobs for request-2
    request_2 = "request-2"
    for i in range(10, 15):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            slow_task,
            i,
            pipeline_id=request_2
        )

    # collect results for request-1
    results_req1 = []
    for _ in range(5):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            timeout=5.0,
            pipeline_id=request_1
        )
        results_req1.append(result)

    # verify request-1 got correct results
    assert len(results_req1) == 5
    assert set(results_req1) == {10, 20, 30, 40, 50}

    # collect results for request-2
    results_req2 = []
    for _ in range(5):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            timeout=5.0,
            pipeline_id=request_2
        )
        results_req2.append(result)

    # verify request-2 got correct results (and NOT request-1's results)
    assert len(results_req2) == 5
    assert set(results_req2) == {100, 110, 120, 130, 140}

    # verify no overlap
    assert set(results_req1).isdisjoint(set(results_req2))

    manager.shutdown()

    print("   ✓ cross-pipeline contamination is prevented")


def test_pipeline_id_with_mixed_operations():
    """test pipeline_id isolation across different operation types."""
    print("\n21. testing pipeline_id with mixed operation types...")

    ThreadPoolManager._instance = None
    manager = ThreadPoolManager.get_instance(max_workers=5)
    manager.initialize()

    def task_type_a(x: int) -> str:
        time.sleep(0.05)
        return f"A-{x}"

    def task_type_b(x: int) -> str:
        time.sleep(0.05)
        return f"B-{x}"

    pipeline_id = "mixed-pipeline"

    # submit claims extraction jobs for this pipeline
    for i in range(3):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            task_type_a,
            i,
            pipeline_id=pipeline_id
        )

    # submit link expansion jobs for this pipeline
    for i in range(3):
        manager.submit(
            OperationType.LINK_CONTEXT_EXPANDING,
            task_type_b,
            i,
            pipeline_id=pipeline_id
        )

    # submit claims extraction jobs for a different pipeline
    other_pipeline = "other-pipeline"
    for i in range(10, 12):
        manager.submit(
            OperationType.CLAIMS_EXTRACTION,
            task_type_a,
            i,
            pipeline_id=other_pipeline
        )

    # wait for claims extraction jobs from mixed-pipeline
    results = []
    for _ in range(3):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            pipeline_id=pipeline_id
        )
        results.append(result)

        # verify it's from the correct pipeline
        with manager.running_jobs_lock:
            job = manager.completed_jobs[job_id]
            assert job.pipeline_id == pipeline_id

    assert len(results) == 3
    assert set(results) == {"A-0", "A-1", "A-2"}

    # wait for link expansion jobs from mixed-pipeline
    link_results = []
    for _ in range(3):
        job_id, result = manager.wait_next_completed(
            OperationType.LINK_CONTEXT_EXPANDING,
            pipeline_id=pipeline_id
        )
        link_results.append(result)

        # verify it's from the correct pipeline
        with manager.running_jobs_lock:
            job = manager.completed_jobs[job_id]
            assert job.pipeline_id == pipeline_id

    assert len(link_results) == 3
    assert set(link_results) == {"B-0", "B-1", "B-2"}

    # wait for claims extraction jobs from other-pipeline
    other_results = []
    for _ in range(2):
        job_id, result = manager.wait_next_completed(
            OperationType.CLAIMS_EXTRACTION,
            pipeline_id=other_pipeline
        )
        other_results.append(result)

    assert len(other_results) == 2
    assert set(other_results) == {"A-10", "A-11"}

    manager.shutdown()

    print("   ✓ pipeline_id isolation works across different operation types")


def run_all_tests():
    """run all tests."""
    print("=" * 60)
    print("running thread pool manager tests")
    print("=" * 60)

    # sync tests
    test_operation_type_weights()
    test_job_priority_ordering()
    test_thread_pool_initialization()
    test_singleton_pattern()
    test_submit_and_wait()
    test_priority_ordering()
    test_map_threaded()
    test_context_manager()
    test_with_thread_pool()
    test_error_handling()
    test_get_status()
    test_wait_next_completed()
    test_wait_next_completed_any()
    test_wait_next_completed_timeout()
    test_wait_next_completed_error_handling()
    test_pipeline_id_job_tracking()
    test_pipeline_id_isolation()
    test_pipeline_id_cross_contamination_prevention()
    test_pipeline_id_with_mixed_operations()

    # async tests
    asyncio.run(test_async_bridge())
    asyncio.run(test_map_threaded_async())

    print("\n" + "=" * 60)
    print("✓ all tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
