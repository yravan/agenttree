"""Stress test — concurrent tab acquisition."""

import asyncio
import time
import pytest
from agenttree.browser.bridge import MockBrowserBridge
from agenttree.browser.tab_pool import TabPool


@pytest.mark.slow
@pytest.mark.asyncio
async def test_many_concurrent_tabs():
    bridge = MockBrowserBridge()
    await bridge.connect()
    pool = TabPool(bridge, max_tabs=5)

    max_concurrent = 0
    current = 0
    lock = asyncio.Lock()

    async def worker(i):
        nonlocal max_concurrent, current
        tab_id = await pool.acquire()
        async with lock:
            current += 1
            max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        await pool.release(tab_id)

    await asyncio.gather(*[worker(i) for i in range(20)])
    assert max_concurrent <= 5
    assert pool.active_count == 0
    await pool.close_all()
    await bridge.disconnect()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_rapid_acquire_release():
    bridge = MockBrowserBridge()
    await bridge.connect()
    pool = TabPool(bridge, max_tabs=3)
    for _ in range(100):
        tab_id = await pool.acquire()
        await pool.release(tab_id)
    assert pool.active_count == 0
    await pool.close_all()
    await bridge.disconnect()
