"""E2E test — crash and resume from SQLite checkpoint."""

import pytest
from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings
from agenttree.engine.recovery import recover_and_resume
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store


class ResumeMockLLM:
    def __init__(self):
        self.calls = []

    async def call(self, messages, *, node_id="", role="", **kwargs):
        self.calls.append({"role": role, "node_id": node_id})
        if role == "planner":
            return {"content": '{"action": "execute", "reasoning": "simple"}', "tool_calls": [],
                    "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
        elif role == "executor":
            return {"content": "", "tool_calls": [{"name": "done", "arguments": {"result": "resumed", "success": True}}],
                    "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
        elif role == "aggregator":
            return {"content": '{"final_result": "agg", "confidence": 0.9, "completeness": 0.9, "gaps": [], "needs_more_work": false, "additional_subtasks": []}',
                    "tool_calls": [], "input_tokens": 100, "output_tokens": 50, "cost": 0.001, "model": "test"}
        return {"content": "{}", "tool_calls": [], "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_crash_and_resume(tmp_path):
    settings = Settings()
    settings.tree.max_depth = 2
    settings.redundancy.adaptive = False
    settings.timing.max_steps_per_leaf = 2
    settings.timing.action_delay_seconds = 0.0
    settings.timing.timeout_per_node_seconds = 10

    store = Store(tmp_path)
    await store.init()

    root = TreeNode(task="Root task", depth=0)
    root.status = NodeStatus.WAITING_CHILDREN
    root.decision = "branch"
    root.children_ids = ["child1", "child2"]
    await store.save_node(root)

    child1 = TreeNode(task="Child 1 done", depth=1)
    child1.id = "child1"
    child1.parent_id = root.id
    child1.status = NodeStatus.COMPLETE
    child1.result = "child1 result"
    child1.confidence = 0.9
    await store.save_node(child1)

    child2 = TreeNode(task="Child 2 crashed", depth=1)
    child2.id = "child2"
    child2.parent_id = root.id
    child2.status = NodeStatus.EXECUTING
    await store.save_node(child2)
    await store.close()

    store2 = Store(tmp_path)
    await store2.init()
    cost_tracker = CostTracker(budget_usd=5.0)
    llm = ResumeMockLLM()
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_pool = TabPool(bridge, max_tabs=3)

    result = await recover_and_resume(store2, settings, tab_pool, bridge, llm, cost_tracker)

    assert result is not None
    assert result.status == NodeStatus.COMPLETE
    child1_calls = [c for c in llm.calls if c["node_id"] == "child1"]
    assert len(child1_calls) == 0
    child2_calls = [c for c in llm.calls if c["node_id"] == "child2"]
    assert len(child2_calls) > 0

    await store2.close()
    await bridge.disconnect()
