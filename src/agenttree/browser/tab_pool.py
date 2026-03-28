"""Tab pool — asyncio.Semaphore + Queue for Chrome tab lifecycle."""

from __future__ import annotations

import asyncio

import structlog

from agenttree.browser.bridge import BrowserBridge

logger = structlog.get_logger()


class TabPool:
    """Manages a pool of Chrome tabs with bounded concurrency."""

    def __init__(self, bridge: BrowserBridge, max_tabs: int = 15):
        self.bridge = bridge
        self.max_tabs = max_tabs
        self._semaphore = asyncio.Semaphore(max_tabs)
        self._available: asyncio.Queue[str | int] = asyncio.Queue()
        self._all_tabs: set[str | int] = set()
        self._in_use: set[str | int] = set()

    async def acquire(self) -> str | int:
        """Acquire a tab from the pool, creating one if needed."""
        await self._semaphore.acquire()

        # Try to reuse an existing tab
        try:
            tab_id = self._available.get_nowait()
            self._in_use.add(tab_id)
            logger.debug("tab_reused", tab_id=tab_id, in_use=len(self._in_use))
            return tab_id
        except asyncio.QueueEmpty:
            pass

        # Create a new tab
        tab_id = await self.bridge.create_tab()
        self._all_tabs.add(tab_id)
        self._in_use.add(tab_id)
        logger.debug("tab_created", tab_id=tab_id, total=len(self._all_tabs), in_use=len(self._in_use))
        return tab_id

    async def release(self, tab_id: str | int) -> None:
        """Release a tab back to the pool.

        Navigates the tab to about:blank before returning it so the next
        consumer starts with a clean slate.
        """
        try:
            await self.bridge.navigate(tab_id, "about:blank")
        except Exception:
            logger.debug("tab_blank_navigate_failed", tab_id=tab_id)
        self._in_use.discard(tab_id)
        await self._available.put(tab_id)
        self._semaphore.release()
        logger.debug("tab_released", tab_id=tab_id, in_use=len(self._in_use))

    async def close_all(self) -> None:
        """Close all tabs and clean up."""
        for tab_id in list(self._all_tabs):
            try:
                await self.bridge.close_tab(tab_id)
            except Exception:
                pass
        self._all_tabs.clear()
        self._in_use.clear()
        # Drain the queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def active_count(self) -> int:
        return len(self._in_use)

    @property
    def total_count(self) -> int:
        return len(self._all_tabs)
