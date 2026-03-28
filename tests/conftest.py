"""Shared fixtures for AgentTree tests."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import (
    CostsConfig,
    LLMConfig,
    RedundancyConfig,
    Settings,
    TimingConfig,
    TreeConfig,
)
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Fake LLM client that returns canned responses.

    Configure via:
    - ``response_sequence``: list of responses returned in order (cycles).
    - ``response_map``: dict mapping substrings of the user message to a response.
    - ``failure_at``: set of call indices (0-based) that should raise RuntimeError.
    """

    def __init__(
        self,
        response_sequence: list[str] | None = None,
        response_map: dict[str, str] | None = None,
        failure_at: set[int] | None = None,
    ):
        self.response_sequence = response_sequence or ['{"action": "execute", "reasoning": "test"}']
        self.response_map = response_map or {}
        self.failure_at = failure_at or set()
        self.calls: list[dict[str, Any]] = []
        self._call_index = 0

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Simulate an LLM chat call."""
        idx = self._call_index
        self._call_index += 1
        self.calls.append({"messages": messages, "kwargs": kwargs, "index": idx})

        if idx in self.failure_at:
            raise RuntimeError(f"Injected failure at call {idx}")

        # Check response_map first (match against user message content)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                for key, response in self.response_map.items():
                    if key in content:
                        return response
                break

        # Fall back to sequence (cycling)
        return self.response_sequence[idx % len(self.response_sequence)]

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings() -> Settings:
    """Settings with test-friendly defaults."""
    return Settings(
        tree=TreeConfig(max_depth=3, max_breadth=5),
        costs=CostsConfig(budget_usd=5.0),
        timing=TimingConfig(
            action_delay_seconds=0.0,
            timeout_per_node_seconds=10,
            max_steps_per_leaf=5,
        ),
        redundancy=RedundancyConfig(adaptive=False),
        llm=LLMConfig(),
    )


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A MockLLMClient with a single default response."""
    return MockLLMClient()


@pytest.fixture
def mock_bridge() -> MockBrowserBridge:
    """MockBrowserBridge instance ready for use."""
    return MockBrowserBridge()


@pytest.fixture
def mock_tab_pool(mock_bridge: MockBrowserBridge) -> TabPool:
    """TabPool backed by MockBrowserBridge with max_tabs=3."""
    return TabPool(bridge=mock_bridge, max_tabs=3)


@pytest.fixture
def temp_store(tmp_path: Path):
    """Yields a Store backed by a temporary directory. Cleans up on teardown."""
    store = Store(run_dir=tmp_path / "test_run")

    async def _init_and_yield():
        await store.init()
        return store

    # We cannot yield from an async generator in a sync fixture,
    # so we yield the store and let each test call init().
    yield store

    # Cleanup
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def cost_tracker() -> CostTracker:
    """CostTracker with a $5.00 budget."""
    return CostTracker(budget_usd=5.0, warn_at_usd=3.75)


@pytest.fixture
def sample_nodes():
    """Factory fixture for creating pre-built tree structures.

    Returns a callable that produces a root node with children.
    Usage::

        root, children = sample_nodes(breadth=3, depth=2)
    """

    def _factory(
        breadth: int = 3,
        depth: int = 1,
        root_task: str = "root task",
    ) -> tuple[TreeNode, list[TreeNode]]:
        root = TreeNode(task=root_task, depth=0, status=NodeStatus.PENDING)
        all_children: list[TreeNode] = []

        def _build(parent: TreeNode, current_depth: int) -> None:
            if current_depth > depth:
                return
            for i in range(breadth):
                child = TreeNode(
                    task=f"subtask-d{current_depth}-{i}",
                    parent_id=parent.id,
                    depth=current_depth,
                    status=NodeStatus.PENDING,
                )
                parent.children_ids.append(child.id)
                all_children.append(child)
                _build(child, current_depth + 1)

        _build(root, 1)
        return root, all_children

    return _factory
