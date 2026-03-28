"""Priority queue for leaf node scheduling (breadth-first by default)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedNode:
    priority: tuple[int, float]  # (depth, creation_time) — lower is higher priority
    node: Any = field(compare=False)


class Scheduler:
    """Breadth-first priority scheduler for leaf nodes."""

    def __init__(self):
        self._queue: asyncio.PriorityQueue[PrioritizedNode] = asyncio.PriorityQueue()
        self._submitted: int = 0
        self._completed: int = 0

    async def submit(self, node: Any) -> None:
        """Add a node to the scheduling queue."""
        priority = (node.depth, node.created_at)
        await self._queue.put(PrioritizedNode(priority=priority, node=node))
        self._submitted += 1

    async def next(self) -> Any:
        """Get the next highest-priority node."""
        item = await self._queue.get()
        return item.node

    def mark_complete(self) -> None:
        self._completed += 1
        self._queue.task_done()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def submitted(self) -> int:
        return self._submitted

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()
