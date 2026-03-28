"""Integration tests for the Rich dashboard rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenttree.dashboard import Dashboard
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dashboard_renders_tree(tmp_path: Path):
    """Create nodes, build tree widget, check labels contain expected info."""
    store = Store(run_dir=tmp_path / "dash1")
    await store.init()
    try:
        cost_tracker = CostTracker(budget_usd=5.0)
        dashboard = Dashboard(store=store, cost_tracker=cost_tracker)

        # Build a small tree
        root = TreeNode(task="root task", depth=0, status=NodeStatus.COMPLETE, confidence=0.95)
        child1 = TreeNode(task="child task 1", parent_id=root.id, depth=1, status=NodeStatus.COMPLETE)
        child2 = TreeNode(task="child task 2", parent_id=root.id, depth=1, status=NodeStatus.FAILED)
        root.children_ids = [child1.id, child2.id]

        nodes = [root, child1, child2]

        tree_widget = dashboard._build_tree_widget(nodes)

        # Render the tree to a string to check its contents
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        console.print(tree_widget)
        rendered = buf.getvalue()

        # Check that node IDs and tasks appear in the rendered output
        assert root.id in rendered
        assert "root task" in rendered
        assert child1.id in rendered
        assert child2.id in rendered
        assert "AgentTree" in rendered
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dashboard_renders_stats(tmp_path: Path):
    """Check stats table has expected metrics."""
    store = Store(run_dir=tmp_path / "dash2")
    await store.init()
    try:
        cost_tracker = CostTracker(budget_usd=5.0)
        # Add some mock cost data
        await cost_tracker.add(
            cost=0.05, input_tokens=500, output_tokens=200,
            model="test-model", node_id="n1", role="planner",
        )

        dashboard = Dashboard(store=store, cost_tracker=cost_tracker)

        # Build nodes for stats
        nodes = [
            TreeNode(task="t1", depth=0, status=NodeStatus.COMPLETE),
            TreeNode(task="t2", depth=1, status=NodeStatus.COMPLETE),
            TreeNode(task="t3", depth=1, status=NodeStatus.FAILED),
            TreeNode(task="t4", depth=1, status=NodeStatus.EXECUTING),
            TreeNode(task="t5", depth=1, status=NodeStatus.PENDING),
        ]

        stats_table = dashboard._build_stats_table(nodes)

        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        console.print(stats_table)
        rendered = buf.getvalue()

        # Check that key metrics are present
        assert "Total nodes" in rendered
        assert "5" in rendered  # total count
        assert "Complete" in rendered
        assert "Failed" in rendered
        assert "Executing" in rendered
        assert "Pending" in rendered
        assert "LLM calls" in rendered
        assert "Total cost" in rendered
        assert "Budget left" in rendered
        assert "Run Statistics" in rendered
    finally:
        await store.close()
