"""Integration tests for Store (SQLite persistence + event log)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agenttree.engine.tree import NodeStatus, TreeNode
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
async def test_save_and_load_node(tmp_path: Path):
    """Create, save, load by ID, assert match."""
    store = Store(run_dir=tmp_path / "run1")
    await store.init()
    try:
        node = TreeNode(task="find info", depth=0, status=NodeStatus.PENDING)
        await store.save_node(node)

        loaded = await store.load_node(node.id)
        assert loaded is not None
        assert loaded.id == node.id
        assert loaded.task == "find info"
        assert loaded.depth == 0
        assert loaded.status == NodeStatus.PENDING
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_overwrites_existing(tmp_path: Path):
    """Save, modify status, save again, load and verify updated value."""
    store = Store(run_dir=tmp_path / "run2")
    await store.init()
    try:
        node = TreeNode(task="task A", depth=0, status=NodeStatus.PENDING)
        await store.save_node(node)

        node.status = NodeStatus.EXECUTING
        node.result = "partial data"
        await store.save_node(node)

        loaded = await store.load_node(node.id)
        assert loaded is not None
        assert loaded.status == NodeStatus.EXECUTING
        assert loaded.result == "partial data"
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_nonexistent_node(tmp_path: Path):
    """Loading a node that does not exist returns None."""
    store = Store(run_dir=tmp_path / "run3")
    await store.init()
    try:
        loaded = await store.load_node("nonexistent_id_abc")
        assert loaded is None
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_all_nodes(tmp_path: Path):
    """Save 5 nodes, load all, assert 5 returned."""
    store = Store(run_dir=tmp_path / "run4")
    await store.init()
    try:
        for i in range(5):
            node = TreeNode(task=f"task_{i}", depth=0, status=NodeStatus.PENDING)
            await store.save_node(node)

        all_nodes = await store.load_all_nodes()
        assert len(all_nodes) == 5
        tasks = {n.task for n in all_nodes}
        for i in range(5):
            assert f"task_{i}" in tasks
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_incomplete_nodes(tmp_path: Path):
    """Mix of COMPLETE, FAILED, PENDING nodes -- only incomplete returned."""
    store = Store(run_dir=tmp_path / "run5")
    await store.init()
    try:
        statuses = [
            NodeStatus.COMPLETE,
            NodeStatus.FAILED,
            NodeStatus.PENDING,
            NodeStatus.EXECUTING,
            NodeStatus.DECIDING,
        ]
        nodes = []
        for i, status in enumerate(statuses):
            node = TreeNode(task=f"task_{i}", depth=0, status=status)
            nodes.append(node)
            await store.save_node(node)

        incomplete = await store.load_incomplete_nodes()
        incomplete_ids = {n.id for n in incomplete}

        # COMPLETE and FAILED should NOT be in incomplete
        assert nodes[0].id not in incomplete_ids  # COMPLETE
        assert nodes[1].id not in incomplete_ids  # FAILED
        # PENDING, EXECUTING, DECIDING should be in incomplete
        assert nodes[2].id in incomplete_ids  # PENDING
        assert nodes[3].id in incomplete_ids  # EXECUTING
        assert nodes[4].id in incomplete_ids  # DECIDING
        assert len(incomplete) == 3
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_log_append(tmp_path: Path):
    """Log 5 events, read jsonl, assert 5 lines."""
    store = Store(run_dir=tmp_path / "run6")
    await store.init()
    try:
        for i in range(5):
            await store.log_event({"type": "test_event", "index": i})

        lines = store.events_path.read_text().strip().split("\n")
        assert len(lines) == 5

        for i, line in enumerate(lines):
            event = json.loads(line)
            assert event["type"] == "test_event"
            assert event["index"] == i
            assert "ts" in event
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_call_logging(tmp_path: Path):
    """Log LLM calls, verify they are stored in the database."""
    store = Store(run_dir=tmp_path / "run7")
    await store.init()
    try:
        await store.log_llm_call(
            node_id="node_1",
            role="planner",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
        )
        await store.log_llm_call(
            node_id="node_2",
            role="executor",
            model="test-model",
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.010,
        )

        assert store._db is not None
        cursor = await store._db.execute("SELECT COUNT(*) FROM llm_calls")
        row = await cursor.fetchone()
        assert row[0] == 2

        cursor = await store._db.execute(
            "SELECT node_id, role, model, input_tokens, output_tokens, cost_usd FROM llm_calls ORDER BY id"
        )
        rows = await cursor.fetchall()
        assert rows[0] == ("node_1", "planner", "test-model", 100, 50, 0.005)
        assert rows[1] == ("node_2", "executor", "test-model", 200, 100, 0.010)
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_writes(tmp_path: Path):
    """10 async tasks saving nodes simultaneously should not corrupt data."""
    store = Store(run_dir=tmp_path / "run8")
    await store.init()
    try:
        async def save_task(idx: int):
            node = TreeNode(task=f"concurrent_task_{idx}", depth=0, status=NodeStatus.PENDING)
            await store.save_node(node)
            return node.id

        node_ids = await asyncio.gather(*[save_task(i) for i in range(10)])

        all_nodes = await store.load_all_nodes()
        assert len(all_nodes) == 10

        loaded_ids = {n.id for n in all_nodes}
        for nid in node_ids:
            assert nid in loaded_ids
    finally:
        await store.close()
