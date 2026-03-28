"""E2E test: single node that executes directly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import (
    CostsConfig,
    RedundancyConfig,
    Settings,
    TimingConfig,
    TreeConfig,
)
from agenttree.engine.executor import execute_node
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store


_DEFAULT_RESP = {
    "content": "{}",
    "tool_calls": [],
    "input_tokens": 10,
    "output_tokens": 10,
    "cost": 0.001,
    "model": "test",
}


class RoleAwareMockLLM:
    """Mock LLM that returns different responses based on the role parameter."""

    def __init__(self, planner_response, executor_responses, aggregator_response, evaluator_response=None):
        self.planner = planner_response
        self.executor_responses = list(executor_responses)
        self.executor_idx = 0
        self.aggregator = aggregator_response
        self.evaluator = evaluator_response or {
            "content": json.dumps({"score": 0.9, "reasoning": "good"}),
            "tool_calls": [],
            "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
        }
        self.calls = []

    async def call(self, messages, *, node_id="", role="", **kwargs):
        self.calls.append({"role": role, "node_id": node_id})
        if role == "planner":
            return self.planner
        elif role == "executor":
            if self.executor_idx < len(self.executor_responses):
                r = self.executor_responses[self.executor_idx]
                self.executor_idx += 1
                return r
            return {
                "content": "",
                "tool_calls": [{"name": "done", "arguments": {"result": "mock result", "success": True}}],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        elif role == "aggregator":
            return self.aggregator
        elif role == "evaluator":
            return self.evaluator
        elif role == "voter":
            return {
                "content": json.dumps({"best_index": 0, "confidence": 0.9, "reasoning": "best"}),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        return dict(_DEFAULT_RESP)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_simple_execute(tmp_path: Path):
    """Single node that decides to execute directly, completes in one step."""
    store = Store(run_dir=tmp_path / "e2e_simple")
    await store.init()
    try:
        settings = Settings(
            tree=TreeConfig(max_depth=2, max_breadth=3, max_total_nodes=50),
            costs=CostsConfig(budget_usd=5.0),
            timing=TimingConfig(
                action_delay_seconds=0.0,
                timeout_per_node_seconds=30,
                max_steps_per_leaf=3,
            ),
            redundancy=RedundancyConfig(adaptive=False),
        )
        bridge = MockBrowserBridge()
        await bridge.connect()
        tab_pool = TabPool(bridge=bridge, max_tabs=5)
        cost_tracker = CostTracker(budget_usd=5.0)

        llm = RoleAwareMockLLM(
            planner_response={
                "content": json.dumps({
                    "action": "execute",
                    "reasoning": "simple direct task",
                    "subtasks": [],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
            executor_responses=[
                {
                    "content": "doing the task",
                    "tool_calls": [{"name": "done", "arguments": {"result": "e2e simple result", "success": True}}],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                },
            ],
            aggregator_response=dict(_DEFAULT_RESP),
        )

        root = TreeNode(task="get website title", depth=0)
        await store.save_node(root)

        result = await execute_node(root, store, settings, tab_pool, bridge, llm, cost_tracker)

        assert result.status == NodeStatus.COMPLETE
        assert result.result == "e2e simple result"
        assert result.decision == "execute"

        # Verify event log was written
        assert store.events_path.exists()
        lines = store.events_path.read_text().strip().split("\n")
        assert len(lines) >= 2  # at least status_change and node_complete

        await bridge.disconnect()
    finally:
        await store.close()
