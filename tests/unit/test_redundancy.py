"""Unit tests for adaptive redundancy (execute-once, evaluate, escalate, vote)."""

from __future__ import annotations

import json

import pytest

from agenttree.config import Settings
from agenttree.engine.tree import TreeNode
from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.redundancy import execute_with_redundancy


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


def _done_response(result_text: str = "found data") -> dict:
    """Create a standard LLM response that calls the 'done' tool."""
    return {
        "content": "Completing task.",
        "tool_calls": [
            {
                "name": "done",
                "arguments": {"result": result_text, "success": True},
            }
        ],
        "input_tokens": 10,
        "output_tokens": 10,
        "cost": 0.001,
        "model": "test",
    }


def _eval_response(score: float) -> dict:
    """Create a confidence evaluation response."""
    return {
        "content": json.dumps({"score": score, "reasoning": "evaluation"}),
        "tool_calls": [],
        "input_tokens": 10,
        "output_tokens": 10,
        "cost": 0.001,
        "model": "test",
    }


def _vote_response(best_index: int = 0) -> dict:
    """Create a voter response picking a result."""
    return {
        "content": json.dumps({
            "best_index": best_index,
            "confidence": 0.85,
            "reasoning": "Best result",
        }),
        "tool_calls": [],
        "input_tokens": 10,
        "output_tokens": 10,
        "cost": 0.001,
        "model": "test",
    }


def _make_settings(adaptive: bool = True, action_delay: float = 0.0) -> Settings:
    """Create Settings with zero delay for fast tests."""
    return Settings(
        redundancy={"adaptive": adaptive, "redundancy_n": 3},
        timing={"max_steps_per_leaf": 5, "action_delay_seconds": action_delay},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_high_confidence_skips_redundancy():
    """A high confidence score (>= threshold) should accept the first result without extra runs."""
    llm = MockLLMClient(responses=[
        # Run 1: execute_leaf calls LLM once, returns via done tool
        _done_response("high quality data"),
        # Confidence evaluation
        _eval_response(0.92),
    ])
    settings = _make_settings(adaptive=True)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_pool = TabPool(bridge, max_tabs=5)
    node = TreeNode(task="Simple lookup", id="r1")

    result = await execute_with_redundancy(node, tab_pool, bridge, llm, settings)

    assert result.success is True
    assert result.data == "high quality data"
    assert result.steps_taken == 1
    # Only 1 execute_leaf call (1 LLM call) + 1 evaluator call = 2 total
    executor_calls = [c for c in llm.calls if c["role"] == "executor"]
    evaluator_calls = [c for c in llm.calls if c["role"] == "evaluator"]
    assert len(executor_calls) == 1
    assert len(evaluator_calls) == 1
    # Node should have exactly 1 redundancy result
    assert len(node.redundancy_results) == 1

    await bridge.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_low_confidence_runs_full_n():
    """A low confidence score (< medium threshold) should run redundancy_n times and vote."""
    llm = MockLLMClient(responses=[
        # Run 1
        _done_response("result from run 1"),
        # Confidence evaluation (low score)
        _eval_response(0.3),
        # Run 2
        _done_response("result from run 2"),
        # Run 3
        _done_response("result from run 3"),
        # Voter picks best
        _vote_response(best_index=1),
    ])
    settings = _make_settings(adaptive=True)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_pool = TabPool(bridge, max_tabs=5)
    node = TreeNode(task="Difficult extraction", id="r2")

    result = await execute_with_redundancy(node, tab_pool, bridge, llm, settings)

    assert result.data == "result from run 2"  # voter picked index 1
    # Should have 3 redundancy results recorded on the node
    assert len(node.redundancy_results) == 3
    # Vote outcome should be set
    assert node.vote_outcome is not None
    assert node.vote_outcome["best_index"] == 1

    await bridge.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adaptive_disabled_runs_once():
    """With adaptive=False, only one execution is performed regardless of confidence."""
    llm = MockLLMClient(responses=[
        _done_response("single run result"),
    ])
    settings = _make_settings(adaptive=False)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_pool = TabPool(bridge, max_tabs=5)
    node = TreeNode(task="One-shot task", id="r3")

    result = await execute_with_redundancy(node, tab_pool, bridge, llm, settings)

    assert result.success is True
    assert result.data == "single run result"
    assert len(node.redundancy_results) == 1
    # No evaluator call should have been made
    evaluator_calls = [c for c in llm.calls if c["role"] == "evaluator"]
    assert len(evaluator_calls) == 0

    await bridge.disconnect()
