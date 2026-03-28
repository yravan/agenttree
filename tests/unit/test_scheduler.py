"""Unit tests for Scheduler (breadth-first priority queue)."""

from __future__ import annotations

import pytest

from agenttree.engine.scheduler import Scheduler
from agenttree.engine.tree import TreeNode


pytestmark = pytest.mark.unit


class TestScheduler:

    @pytest.mark.asyncio
    async def test_breadth_first_ordering(self):
        """Shallower nodes should be dequeued before deeper ones."""
        scheduler = Scheduler()

        deep = TreeNode(task="deep task", depth=3)
        mid = TreeNode(task="mid task", depth=1)
        shallow = TreeNode(task="shallow task", depth=0)

        # Submit in arbitrary order
        await scheduler.submit(deep)
        await scheduler.submit(shallow)
        await scheduler.submit(mid)

        first = await scheduler.next()
        second = await scheduler.next()
        third = await scheduler.next()

        assert first.depth == 0, "Shallowest should come first"
        assert second.depth == 1
        assert third.depth == 3

    @pytest.mark.asyncio
    async def test_same_depth_fifo(self):
        """Nodes at the same depth should come out in FIFO (created_at) order."""
        scheduler = Scheduler()

        import time
        # Ensure distinguishable created_at values
        a = TreeNode(task="A", depth=1)
        a.created_at = 1000.0

        b = TreeNode(task="B", depth=1)
        b.created_at = 1001.0

        c = TreeNode(task="C", depth=1)
        c.created_at = 1002.0

        await scheduler.submit(c)
        await scheduler.submit(a)
        await scheduler.submit(b)

        first = await scheduler.next()
        second = await scheduler.next()
        third = await scheduler.next()

        assert first.task == "A"
        assert second.task == "B"
        assert third.task == "C"

    @pytest.mark.asyncio
    async def test_empty_queue(self):
        """is_empty and pending should reflect queue state correctly."""
        scheduler = Scheduler()
        assert scheduler.is_empty is True
        assert scheduler.pending == 0
        assert scheduler.submitted == 0
        assert scheduler.completed == 0

        node = TreeNode(task="test", depth=0)
        await scheduler.submit(node)

        assert scheduler.is_empty is False
        assert scheduler.pending == 1
        assert scheduler.submitted == 1

        _ = await scheduler.next()
        scheduler.mark_complete()

        assert scheduler.is_empty is True
        assert scheduler.pending == 0
        assert scheduler.completed == 1
