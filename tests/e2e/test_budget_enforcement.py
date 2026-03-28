"""E2E test — budget enforcement stops execution."""

import asyncio
import pytest

from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings
from agenttree.engine.executor import execute_node
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store


class ExpensiveMockLLM:
    """Mock LLM that updates a cost tracker with $0.10 per call."""

    def __init__(self, cost_tracker):
        self.calls = []
        self._cost_tracker = cost_tracker

    async def call(self, messages, *, node_id="", role="", **kwargs):
        self.calls.append({"role": role, "node_id": node_id})
        await self._cost_tracker.add(
            cost=0.10, input_tokens=100, output_tokens=50,
            model="test", node_id=node_id, role=role,
        )
        if role == "planner":
            return {
                "content": '{"action": "branch", "reasoning": "need subtasks", "subtasks": ["sub1", "sub2", "sub3"]}',
                "tool_calls": [], "input_tokens": 500, "output_tokens": 200, "cost": 0.10, "model": "test",
            }
        return {
            "content": "",
            "tool_calls": [{"name": "done", "arguments": {"result": "done", "success": True}}],
            "input_tokens": 100, "output_tokens": 50, "cost": 0.10, "model": "test",
        }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_budget_stops_execution(tmp_path):
    settings = Settings()
    settings.costs.budget_usd = 0.25
    settings.tree.max_depth = 3
    settings.tree.max_breadth = 3
    settings.redundancy.adaptive = False
    settings.timing.max_steps_per_leaf = 2
    settings.timing.action_delay_seconds = 0.0
    settings.timing.timeout_per_node_seconds = 10

    cost_tracker = CostTracker(budget_usd=0.25, warn_at_usd=0.20)
    llm = ExpensiveMockLLM(cost_tracker)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_pool = TabPool(bridge, max_tabs=3)
    store = Store(tmp_path)
    await store.init()

    root = TreeNode(task="Expensive task", depth=0)
    await store.save_node(root)

    result = await asyncio.wait_for(
        execute_node(root, store, settings, tab_pool, bridge, llm, cost_tracker),
        timeout=15,
    )

    assert cost_tracker.total >= 0.25
    all_nodes = await store.load_all_nodes()
    failed = [n for n in all_nodes if n.status == NodeStatus.FAILED]
    assert len(failed) > 0
    budget_failures = [n for n in failed if n.error and "budget" in n.error.lower()]
    assert len(budget_failures) > 0

    await store.close()
    await bridge.disconnect()
