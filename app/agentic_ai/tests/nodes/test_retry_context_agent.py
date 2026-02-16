"""tests for retry_context_agent node."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from app.agentic_ai.nodes.retry_context_agent import make_retry_context_agent_node


def _make_mock_model():
    """create a mock LLM that returns an AIMessage with no tool calls."""
    model = AsyncMock()
    model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Retry search complete.")
    )
    return model


@pytest.mark.asyncio
async def test_retry_agent_uses_retry_prompt():
    """verify the retry prompt contains 'SEGUNDA TENTATIVA'."""
    model = _make_mock_model()
    node = make_retry_context_agent_node(model)

    state = {
        "messages": [],
        "formatted_data_sources": "some data sources",
        "iteration_count": 0,
        "retry_context": "previous queries and justifications",
    }

    await node(state)

    # check the system message passed to the model
    call_args = model.ainvoke.call_args[0][0]
    system_msg = call_args[0]
    assert "SEGUNDA TENTATIVA" in system_msg.content
    assert "previous queries and justifications" in system_msg.content


@pytest.mark.asyncio
async def test_retry_agent_increments_iteration():
    model = _make_mock_model()
    node = make_retry_context_agent_node(model)

    state = {
        "messages": [],
        "formatted_data_sources": "",
        "iteration_count": 0,
        "retry_context": "",
    }

    result = await node(state)
    assert result["iteration_count"] == 1

    # simulate second call
    state["iteration_count"] = 1
    result = await node(state)
    assert result["iteration_count"] == 2
