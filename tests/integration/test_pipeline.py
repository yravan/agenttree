"""Integration tests for the full execute_node pipeline with mocked LLM."""

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


# ---------------------------------------------------------------------------
# Shared mock response factory
# ---------------------------------------------------------------------------
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
            # Default: call done
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
async def test_simple_flat_decomposition(tmp_path: Path):
    """Root branches to 3 leaves, all execute and aggregate."""
    store = Store(run_dir=tmp_path / "pipeline1")
    await store.init()
    try:
        settings = _make_settings()
        bridge = MockBrowserBridge()
        await bridge.connect()
        tab_pool = TabPool(bridge=bridge, max_tabs=5)
        cost_tracker = CostTracker(budget_usd=5.0)

        llm = RoleAwareMockLLM(
            planner_response={
                "content": json.dumps({
                    "action": "branch",
                    "reasoning": "decompose into subtasks",
                    "subtasks": ["subtask A", "subtask B", "subtask C"],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
            executor_responses=[
                # Each child will call the executor; return done immediately
                {
                    "content": "executing",
                    "tool_calls": [{"name": "done", "arguments": {"result": "result A", "success": True}}],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                },
                {
                    "content": "executing",
                    "tool_calls": [{"name": "done", "arguments": {"result": "result B", "success": True}}],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                },
                {
                    "content": "executing",
                    "tool_calls": [{"name": "done", "arguments": {"result": "result C", "success": True}}],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                },
            ],
            aggregator_response={
                "content": json.dumps({
                    "final_result": "combined A+B+C",
                    "confidence": 0.9,
                    "completeness": 0.95,
                    "gaps": [],
                    "needs_more_work": False,
                    "additional_subtasks": [],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
        )

        root = TreeNode(task="main task", depth=0)
        await store.save_node(root)

        result = await execute_node(root, store, settings, tab_pool, bridge, llm, cost_tracker)

        assert result.status == NodeStatus.COMPLETE
        assert result.result == "combined A+B+C"
        assert result.confidence == 0.9

        # Verify children were created
        all_nodes = await store.load_all_nodes()
        # root + 3 children = 4
        assert len(all_nodes) >= 4

        # Verify all children are complete
        children = [n for n in all_nodes if n.parent_id == root.id]
        assert len(children) == 3
        for child in children:
            assert child.status == NodeStatus.COMPLETE

        await bridge.disconnect()
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_leaf_only_execution(tmp_path: Path):
    """Root executes directly (no branching)."""
    store = Store(run_dir=tmp_path / "pipeline2")
    await store.init()
    try:
        settings = _make_settings()
        bridge = MockBrowserBridge()
        await bridge.connect()
        tab_pool = TabPool(bridge=bridge, max_tabs=5)
        cost_tracker = CostTracker(budget_usd=5.0)

        llm = RoleAwareMockLLM(
            planner_response={
                "content": json.dumps({
                    "action": "execute",
                    "reasoning": "simple enough to execute directly",
                    "subtasks": [],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            },
            executor_responses=[
                # Step 1: navigate
                {
                    "content": "navigating to page",
                    "tool_calls": [{"name": "navigate", "arguments": {"url": "https://example.com"}}],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                },
                # Step 2: done
                {
                    "content": "extracted data",
                    "tool_calls": [{"name": "done", "arguments": {"result": "leaf result data", "success": True}}],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                },
            ],
            aggregator_response=dict(_DEFAULT_RESP),  # not used
        )

        root = TreeNode(task="simple task", depth=0)
        await store.save_node(root)

        result = await execute_node(root, store, settings, tab_pool, bridge, llm, cost_tracker)

        assert result.status == NodeStatus.COMPLETE
        assert result.result == "leaf result data"
        assert result.decision == "execute"

        # No children should be created
        all_nodes = await store.load_all_nodes()
        assert len(all_nodes) == 1

        await bridge.disconnect()
    finally:
        await store.close()
