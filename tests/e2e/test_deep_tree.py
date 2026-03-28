"""E2E test: root branches, children execute."""

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
    """Mock LLM that branches at depth 0, executes at depth >= 1."""

    def __init__(self):
        self.calls = []
        self._executor_idx = 0

    async def call(self, messages, *, node_id="", role="", **kwargs):
        self.calls.append({"role": role, "node_id": node_id})
        if role == "planner":
            # Inspect the prompt to determine depth
            prompt_text = " ".join(m.get("content", "") for m in messages)
            # The decision prompt includes depth info; check if depth is 0
            # At depth 0 we branch, at depth >= 1 we force execute (max_depth=2 means
            # depth >= 2 is forced execute by the decider; depth 1 we also return execute)
            if "depth 0" in prompt_text.lower() or "depth: 0" in prompt_text.lower():
                return {
                    "content": json.dumps({
                        "action": "branch",
                        "reasoning": "split into subtasks",
                        "subtasks": ["sub A", "sub B"],
                    }),
                    "tool_calls": [],
                    "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
                }
            # At deeper depths, execute
            return {
                "content": json.dumps({
                    "action": "execute",
                    "reasoning": "leaf node",
                    "subtasks": [],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        elif role == "executor":
            self._executor_idx += 1
            return {
                "content": f"doing task {self._executor_idx}",
                "tool_calls": [{"name": "done", "arguments": {"result": f"result_{self._executor_idx}", "success": True}}],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        elif role == "aggregator":
            return {
                "content": json.dumps({
                    "final_result": "aggregated from children",
                    "confidence": 0.88,
                    "completeness": 0.9,
                    "gaps": [],
                    "needs_more_work": False,
                    "additional_subtasks": [],
                }),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        elif role == "evaluator":
            return {
                "content": json.dumps({"score": 0.9, "reasoning": "good"}),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        elif role == "voter":
            return {
                "content": json.dumps({"best_index": 0, "confidence": 0.9, "reasoning": "best"}),
                "tool_calls": [],
                "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test",
            }
        return dict(_DEFAULT_RESP)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_depth_2_tree(tmp_path: Path):
    """Root branches to 2 children, children execute, results aggregate."""
    store = Store(run_dir=tmp_path / "e2e_deep")
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

        llm = RoleAwareMockLLM()

        root = TreeNode(task="complex task requiring depth 0 decomposition", depth=0)
        await store.save_node(root)

        result = await execute_node(root, store, settings, tab_pool, bridge, llm, cost_tracker)

        assert result.status == NodeStatus.COMPLETE
        assert result.result == "aggregated from children"
        assert result.confidence == 0.88

        # Verify tree structure
        all_nodes = await store.load_all_nodes()
        assert len(all_nodes) >= 3  # root + 2 children

        root_loaded = await store.load_node(root.id)
        assert root_loaded is not None
        assert len(root_loaded.children_ids) == 2
        assert root_loaded.decision == "branch"

        # All children should be complete
        for child_id in root_loaded.children_ids:
            child = await store.load_node(child_id)
            assert child is not None
            assert child.status == NodeStatus.COMPLETE
            assert child.decision == "execute"

        await bridge.disconnect()
    finally:
        await store.close()
