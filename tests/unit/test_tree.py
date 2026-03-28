"""Unit tests for TreeNode and NodeStatus."""

from __future__ import annotations

import pytest

from agenttree.engine.tree import NodeStatus, TreeNode


pytestmark = pytest.mark.unit


class TestNodeStatus:
    """Verify that all nine statuses are defined."""

    def test_node_status_values(self):
        expected = {
            "PENDING",
            "DECIDING",
            "BRANCHING",
            "WAITING_CHILDREN",
            "EXECUTING",
            "VOTING",
            "AGGREGATING",
            "COMPLETE",
            "FAILED",
        }
        actual = {s.value for s in NodeStatus}
        assert actual == expected, f"Missing or extra statuses: {actual ^ expected}"


class TestTreeNode:

    def test_create_node_defaults(self):
        node = TreeNode(task="find flights")
        assert node.task == "find flights"
        assert node.status == NodeStatus.PENDING
        assert node.depth == 0
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.result is None
        assert node.confidence == 0.0
        assert node.completeness == 0.0
        assert node.tokens_used == 0
        assert node.cost_usd == 0.0
        assert node.error is None
        assert node.browser_history == []
        assert node.tab_id is None
        assert node.id  # non-empty

    def test_create_child_node(self):
        parent = TreeNode(task="parent task")
        child = TreeNode(
            task="child task",
            parent_id=parent.id,
            depth=parent.depth + 1,
        )
        parent.children_ids.append(child.id)

        assert child.parent_id == parent.id
        assert child.depth == 1
        assert child.id in parent.children_ids
        assert child.id != parent.id

    def test_node_serialization_roundtrip(self):
        node = TreeNode(
            task="search hotels",
            depth=2,
            status=NodeStatus.EXECUTING,
            decision="execute",
            decision_reasoning="simple task",
            confidence=0.85,
            completeness=0.9,
            tokens_used=1500,
            cost_usd=0.003,
            result={"hotels": ["Hotel A", "Hotel B"]},
            children_ids=["abc", "def"],
            browser_history=[{"action": "navigate", "url": "https://example.com"}],
        )
        data = node.to_dict()
        restored = TreeNode.from_dict(data)

        assert restored.id == node.id
        assert restored.task == node.task
        assert restored.depth == node.depth
        assert restored.status == node.status
        assert restored.decision == node.decision
        assert restored.decision_reasoning == node.decision_reasoning
        assert restored.confidence == node.confidence
        assert restored.completeness == node.completeness
        assert restored.tokens_used == node.tokens_used
        assert restored.cost_usd == node.cost_usd
        assert restored.result == node.result
        assert restored.children_ids == node.children_ids
        assert restored.browser_history == node.browser_history

    def test_node_id_uniqueness(self):
        ids = {TreeNode(task="t").id for _ in range(1000)}
        assert len(ids) == 1000, "Node IDs must be unique"

    def test_node_cost_accumulation(self):
        node = TreeNode(task="expensive task")
        assert node.cost_usd == 0.0

        node.cost_usd += 0.001
        node.cost_usd += 0.002
        node.cost_usd += 0.0005

        assert abs(node.cost_usd - 0.0035) < 1e-9
        assert node.tokens_used == 0  # tokens tracked separately

        node.tokens_used += 500
        node.tokens_used += 300
        assert node.tokens_used == 800
