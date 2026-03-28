"""Integration tests for MockBrowserBridge."""

from __future__ import annotations

import pytest

from agenttree.browser.bridge import MockBrowserBridge


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
async def test_mock_bridge_connect():
    """Connect MockBrowserBridge, assert connected."""
    bridge = MockBrowserBridge()
    assert not bridge.is_connected

    await bridge.connect()
    assert bridge.is_connected

    await bridge.disconnect()
    assert not bridge.is_connected


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mock_bridge_create_tab():
    """Create tab, get ID back."""
    bridge = MockBrowserBridge()
    await bridge.connect()
    try:
        tab_id = await bridge.create_tab("https://example.com")
        assert tab_id is not None
        assert isinstance(tab_id, str)
        assert tab_id.startswith("mock_tab_")
        assert tab_id in bridge._mock_tabs
        assert bridge._mock_tabs[tab_id]["url"] == "https://example.com"
    finally:
        await bridge.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mock_bridge_get_dom():
    """Get DOM from a mock tab."""
    bridge = MockBrowserBridge()
    await bridge.connect()
    try:
        tab_id = await bridge.create_tab("https://example.com")
        dom = await bridge.get_dom(tab_id)

        assert isinstance(dom, dict)
        assert "text" in dom
        assert "url" in dom
        assert "title" in dom
        assert "captcha_detected" in dom
        assert dom["url"] == "https://example.com"
        assert dom["title"] == "Mock Page"
        assert dom["captcha_detected"] is False
        assert "Interactive Elements" in dom["text"]
    finally:
        await bridge.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mock_bridge_execute_actions():
    """Execute navigate, click, type, scroll actions on mock bridge."""
    bridge = MockBrowserBridge()
    await bridge.connect()
    try:
        tab_id = await bridge.create_tab("about:blank")

        # Navigate
        result = await bridge.execute_action(tab_id, "navigate", {"url": "https://test.com"})
        assert result["success"] is True
        assert result["data"]["navigated"] == "https://test.com"
        assert bridge._mock_tabs[tab_id]["url"] == "https://test.com"

        # Click
        result = await bridge.execute_action(tab_id, "click", {"ref": "e1"})
        assert result["success"] is True
        assert result["data"]["clicked"] == "e1"

        # Type
        result = await bridge.execute_action(tab_id, "type_text", {"ref": "e1", "text": "hello world"})
        assert result["success"] is True
        assert result["data"]["typed"] == "hello world"

        # Scroll
        result = await bridge.execute_action(tab_id, "scroll", {"direction": "down"})
        assert result["success"] is True
        assert result["data"]["scrolled"] == "down"
    finally:
        await bridge.disconnect()
