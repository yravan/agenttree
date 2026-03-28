"""Stress test — deep recursive tree."""

import asyncio
import re
import pytest
from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings
from agenttree.engine.executor import execute_node
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store


class DepthAwareMockLLM:
    def __init__(self, force_max=3):
        self.force_max = force_max
        self.calls = []

    async def call(self, messages, *, node_id="", role="", **kwargs):
        self.calls.append({"role": role, "node_id": node_id})
        if role == "planner":
            prompt_text = str(messages)
            depth = 0
            m = re.search(r"Current depth: (\d+)", prompt_text)
            if m:
                depth = int(m.group(1))
            if depth < self.force_max:
                return {"content": '{"action": "branch", "reasoning": "go deeper", "subtasks": ["a", "b"]}',
                        "tool_calls": [], "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
            return {"content": '{"action": "execute", "reasoning": "leaf"}',
                    "tool_calls": [], "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
        elif role == "executor":
            return {"content": "", "tool_calls": [{"name": "done", "arguments": {"result": "leaf", "success": True}}],
                    "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
        elif role == "aggregator":
            return {"content": '{"final_result": "agg", "confidence": 0.85, "completeness": 0.9, "gaps": [], "needs_more_work": false, "additional_subtasks": []}',
                    "tool_calls": [], "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
        return {"content": "{}", "tool_calls": [], "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test"}


@pytest.mark.slow
@pytest.mark.asyncio
async def test_depth_3_binary_tree(tmp_path):
    settings = Settings()
    settings.tree.max_depth = 3
    settings.tree.max_breadth = 2
    settings.tree.max_total_nodes = 100
    settings.redundancy.adaptive = False
    settings.timing.max_steps_per_leaf = 2
    settings.timing.action_delay_seconds = 0.0
    settings.timing.timeout_per_node_seconds = 15

    cost_tracker = CostTracker(budget_usd=5.0)
    llm = DepthAwareMockLLM(force_max=3)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_pool = TabPool(bridge, max_tabs=5)
    store = Store(tmp_path)
    await store.init()

    root = TreeNode(task="Deep task", depth=0)
    await store.save_node(root)

    result = await asyncio.wait_for(
        execute_node(root, store, settings, tab_pool, bridge, llm, cost_tracker),
        timeout=30,
    )

    assert result.status == NodeStatus.COMPLETE
    all_nodes = await store.load_all_nodes()
    assert len(all_nodes) >= 7  # 1 + 2 + 4
    leaves = [n for n in all_nodes if not n.children_ids]
    for leaf in leaves:
        assert leaf.status in (NodeStatus.COMPLETE, NodeStatus.FAILED)

    await store.close()
    await bridge.disconnect()
