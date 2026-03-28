"""Integration tests for crash recovery."""

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
from agenttree.engine.recovery import recover_and_resume
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
        self._idx = 0

    async def call(self, messages, *, node_id="", role="", model=None,
                   temperature=None, max_tokens=None, tools=None, response_format=None):
        self.calls.append({"messages": messages, "role": role, "node_id": node_id})
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            return resp
        return {"content": "{}", "tool_calls": [], "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test"}


def _make_settings() -> Settings:
    return Settings(
        tree=TreeConfig(max_depth=2, max_breadth=3, max_total_nodes=50),
        costs=CostsConfig(budget_usd=5.0),
        timing=TimingConfig(
            action_delay_seconds=0.0,
            timeout_per_node_seconds=30,
            max_steps_per_leaf=3,
        ),
        redundancy=RedundancyConfig(adaptive=False),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_pending_nodes(tmp_path: Path):
    """Save tree with incomplete (PENDING) nodes, verify recovery finds them."""
    store = Store(run_dir=tmp_path / "recovery1")
    await store.init()
    try:
        root = TreeNode(task="root task", depth=0, status=NodeStatus.COMPLETE)
        child1 = TreeNode(task="child 1", parent_id=root.id, depth=1, status=NodeStatus.COMPLETE)
        child2 = TreeNode(task="child 2", parent_id=root.id, depth=1, status=NodeStatus.PENDING)
        root.children_ids = [child1.id, child2.id]
        root.result = "partial"

        await store.save_node(root)
        await store.save_node(child1)
        await store.save_node(child2)

        incomplete = await store.load_incomplete_nodes()
        assert len(incomplete) == 1
        assert incomplete[0].id == child2.id
        assert incomplete[0].status == NodeStatus.PENDING
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_executing_node(tmp_path: Path):
    """EXECUTING node gets reset to PENDING during recovery."""
    store = Store(run_dir=tmp_path / "recovery2")
    await store.init()
    try:
        settings = _make_settings()
        bridge = MockBrowserBridge()
        await bridge.connect()
        tab_pool = TabPool(bridge=bridge, max_tabs=3)
        cost_tracker = CostTracker(budget_usd=5.0)

        # The mock LLM needs to provide responses for the recovery:
        # 1. decide (planner) -> execute
        # 2. executor step 1 -> done
        llm = MockLLMClient(responses=[
            # planner: decide to execute directly
            {
                "content": json.dumps({"action": "execute", "reasoning": "simple task"}),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
            # executor: call done
            {
                "content": "Task complete",
                "tool_calls": [{"name": "done", "arguments": {"result": "recovered result", "success": True}}],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
            # evaluator (not called because adaptive=False, but just in case)
            {
                "content": json.dumps({"score": 0.9, "reasoning": "good"}),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
        ])

        root = TreeNode(task="executing task", depth=0, status=NodeStatus.EXECUTING)
        await store.save_node(root)

        result = await recover_and_resume(store, settings, tab_pool, bridge, llm, cost_tracker)

        assert result is not None
        assert result.status == NodeStatus.COMPLETE
        await bridge.disconnect()
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_aggregating(tmp_path: Path):
    """AGGREGATING node re-runs aggregation from saved child results."""
    store = Store(run_dir=tmp_path / "recovery3")
    await store.init()
    try:
        settings = _make_settings()
        bridge = MockBrowserBridge()
        await bridge.connect()
        tab_pool = TabPool(bridge=bridge, max_tabs=3)
        cost_tracker = CostTracker(budget_usd=5.0)

        # The mock LLM provides the aggregation response
        llm = MockLLMClient(responses=[
            # aggregator response
            {
                "content": json.dumps({
                    "final_result": "aggregated data",
                    "confidence": 0.85,
                    "completeness": 0.9,
                    "gaps": [],
                    "needs_more_work": False,
                    "additional_subtasks": [],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
        ])

        # Create a parent node in AGGREGATING status with completed children
        parent = TreeNode(task="aggregate task", depth=0, status=NodeStatus.AGGREGATING)
        child1 = TreeNode(
            task="sub A", parent_id=parent.id, depth=1,
            status=NodeStatus.COMPLETE, result="result A", confidence=0.8,
        )
        child2 = TreeNode(
            task="sub B", parent_id=parent.id, depth=1,
            status=NodeStatus.COMPLETE, result="result B", confidence=0.9,
        )
        parent.children_ids = [child1.id, child2.id]

        await store.save_node(parent)
        await store.save_node(child1)
        await store.save_node(child2)

        result = await recover_and_resume(store, settings, tab_pool, bridge, llm, cost_tracker)

        assert result is not None
        assert result.status == NodeStatus.COMPLETE
        assert result.result == "aggregated data"
        assert result.confidence == 0.85
        await bridge.disconnect()
    finally:
        await store.close()
