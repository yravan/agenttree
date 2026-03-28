"""Unit tests for the decider (branch/execute decision)."""

from __future__ import annotations

import json

import pytest

from agenttree.config import Settings
from agenttree.nodes.decider import decide, Decision


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
        self._idx = 0

    async def call(self, messages, *, node_id="", role="", model=None,
                   temperature=None, max_tokens=None, tools=None, response_format=None):
        self.calls.append({"messages": messages, "role": role, "node_id": node_id, "tools": tools})
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            return resp
        return {"content": "{}", "tool_calls": [], "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_simple_task_executes():
    """When the LLM returns an execute decision, decide() returns action='execute'."""
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps({
                "action": "execute",
                "reasoning": "Simple task, can be done in one tab.",
                "subtasks": [],
            }),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    result = await decide(
        task="Search Google for 'python asyncio tutorial'",
        depth=0,
        settings=settings,
        llm=llm,
        node_id="node-1",
    )

    assert isinstance(result, Decision)
    assert result.action == "execute"
    assert result.subtasks == []
    assert len(llm.calls) == 1
    assert llm.calls[0]["role"] == "planner"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complex_task_branches():
    """When the LLM returns a branch decision with subtasks, decide() returns them."""
    subtasks = [
        "Search site A for product prices",
        "Search site B for product prices",
        "Search site C for product prices",
    ]
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps({
                "action": "branch",
                "reasoning": "Complex task needing multiple sites.",
                "subtasks": subtasks,
            }),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    result = await decide(
        task="Compare prices across three retailers",
        depth=0,
        settings=settings,
        llm=llm,
        node_id="node-2",
    )

    assert result.action == "branch"
    assert result.subtasks == subtasks
    assert len(result.subtasks) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decision_includes_reasoning():
    """The reasoning field should be populated from the LLM response."""
    reasoning_text = "This is a straightforward single-page lookup."
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps({
                "action": "execute",
                "reasoning": reasoning_text,
                "subtasks": [],
            }),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    result = await decide(
        task="Look up the weather",
        depth=0,
        settings=settings,
        llm=llm,
        node_id="node-3",
    )

    assert result.reasoning == reasoning_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_max_depth_forces_execute():
    """At max depth, decide() returns execute without making an LLM call."""
    llm = MockLLMClient(responses=[])
    settings = Settings()
    max_depth = settings.tree.max_depth  # default is 5

    result = await decide(
        task="This should be forced to execute",
        depth=max_depth,  # at max depth
        settings=settings,
        llm=llm,
        node_id="node-4",
    )

    assert result.action == "execute"
    assert "Maximum depth" in result.reasoning
    assert result.subtasks == []
    # No LLM call should have been made
    assert len(llm.calls) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subtasks_limited_by_max_breadth():
    """If the LLM returns more subtasks than max_breadth, only max_breadth are kept."""
    twenty_subtasks = [f"Subtask {i}" for i in range(20)]
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps({
                "action": "branch",
                "reasoning": "Many subtasks needed.",
                "subtasks": twenty_subtasks,
            }),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()
    max_breadth = settings.tree.max_breadth  # default is 10

    result = await decide(
        task="Do a huge multi-site comparison",
        depth=0,
        settings=settings,
        llm=llm,
        node_id="node-5",
    )

    assert result.action == "branch"
    assert len(result.subtasks) == max_breadth
    # Should keep the first max_breadth subtasks
    assert result.subtasks == twenty_subtasks[:max_breadth]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decision_handles_malformed_response():
    """A non-JSON response from the LLM should default to execute."""
    llm = MockLLMClient(responses=[
        {
            "content": "I'm not valid JSON at all!",
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    result = await decide(
        task="Some task",
        depth=0,
        settings=settings,
        llm=llm,
        node_id="node-6",
    )

    assert result.action == "execute"
    assert result.subtasks == []
    assert "Failed to parse" in result.reasoning
