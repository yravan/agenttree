"""Unit tests for TabPool."""

from __future__ import annotations

import asyncio

import pytest

from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool


pytestmark = pytest.mark.unit


class TestTabPool:

    @pytest.mark.asyncio
    async def test_acquire_returns_tab_id(self, mock_tab_pool: TabPool):
        """Acquiring a tab should return a non-empty string tab ID."""
        tab_id = await mock_tab_pool.acquire()
        assert isinstance(tab_id, str)
        assert len(tab_id) > 0
        assert mock_tab_pool.active_count == 1
        assert mock_tab_pool.total_count == 1

    @pytest.mark.asyncio
    async def test_release_returns_tab_to_pool(self, mock_tab_pool: TabPool):
        """Releasing a tab should make it available for reuse."""
        tab_id = await mock_tab_pool.acquire()
        assert mock_tab_pool.active_count == 1

        await mock_tab_pool.release(tab_id)
        assert mock_tab_pool.active_count == 0

        # Re-acquire should reuse the same tab
        reused_tab_id = await mock_tab_pool.acquire()
        assert reused_tab_id == tab_id
        assert mock_tab_pool.total_count == 1  # no new tab created

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_empty(self):
        """When all tabs are in use and semaphore is exhausted, acquire should block."""
        bridge = MockBrowserBridge()
        pool = TabPool(bridge=bridge, max_tabs=1)

        # Acquire the only available slot
        tab1 = await pool.acquire()
        assert pool.active_count == 1

        acquired = False

        async def _try_acquire():
            nonlocal acquired
            _ = await pool.acquire()
            acquired = True

        # Start a task that tries to acquire; it should block
        task = asyncio.create_task(_try_acquire())
        await asyncio.sleep(0.05)
        assert acquired is False, "acquire() should block when pool is exhausted"

        # Release the tab to unblock the waiting task
        await pool.release(tab1)
        await asyncio.sleep(0.05)
        assert acquired is True

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_acquire_release(self):
        """Multiple coroutines acquiring and releasing tabs concurrently."""
        bridge = MockBrowserBridge()
        pool = TabPool(bridge=bridge, max_tabs=3)
        results: list[str] = []

        async def _worker(worker_id: int) -> None:
            tab_id = await pool.acquire()
            results.append(f"acquired-{worker_id}")
            await asyncio.sleep(0.01)  # simulate work
            await pool.release(tab_id)
            results.append(f"released-{worker_id}")

        await asyncio.gather(*[_worker(i) for i in range(6)])

        # All 6 workers should have acquired and released
        assert len([r for r in results if r.startswith("acquired")]) == 6
        assert len([r for r in results if r.startswith("released")]) == 6
        assert pool.active_count == 0
