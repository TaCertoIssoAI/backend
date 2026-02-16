"""tests for check_edges router."""

from app.agentic_ai.nodes.check_edges import check_edges
from app.agentic_ai.config import MAX_ITERATIONS, MAX_RETRY_ITERATIONS


def _make_state(iteration_count=0, pending_async_count=0, retry_count=0):
    return {
        "iteration_count": iteration_count,
        "pending_async_count": pending_async_count,
        "retry_count": retry_count,
        "messages": [],
        "fact_check_results": [],
        "search_results": {},
        "scraped_pages": [],
        "formatted_data_sources": "",
    }


def test_routes_to_end_when_no_pending_async():
    state = _make_state(iteration_count=1, pending_async_count=0)
    assert check_edges(state) == "end"


def test_routes_to_wait_when_async_pending():
    state = _make_state(iteration_count=1, pending_async_count=2)
    assert check_edges(state) == "wait_for_async"


def test_routes_to_end_when_max_iterations_reached_no_pending():
    state = _make_state(iteration_count=MAX_ITERATIONS, pending_async_count=0)
    assert check_edges(state) == "end"


def test_pending_async_takes_priority_over_max_iterations():
    """even at max iterations, pending async results must be collected first."""
    state = _make_state(iteration_count=MAX_ITERATIONS, pending_async_count=3)
    assert check_edges(state) == "wait_for_async"


def test_routes_to_end_on_zero_iteration_no_pending():
    state = _make_state(iteration_count=0, pending_async_count=0)
    assert check_edges(state) == "end"


def test_routes_to_end_at_retry_iteration_limit():
    """in retry mode, use MAX_RETRY_ITERATIONS instead of MAX_ITERATIONS."""
    state = _make_state(
        iteration_count=MAX_RETRY_ITERATIONS,
        pending_async_count=0,
        retry_count=1,
    )
    assert check_edges(state) == "end"


def test_retry_mode_allows_fewer_iterations():
    """retry mode uses smaller budget — iteration below MAX_RETRY_ITERATIONS still ends."""
    state = _make_state(
        iteration_count=MAX_RETRY_ITERATIONS - 1,
        pending_async_count=0,
        retry_count=1,
    )
    # below limit, should still route to "end" (context gathering complete)
    assert check_edges(state) == "end"
