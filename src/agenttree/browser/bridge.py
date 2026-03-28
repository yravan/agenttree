"""WebSocket server that the Nanobrowser Chrome extension connects to."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import structlog
import websockets
from websockets.asyncio.server import serve, ServerConnection

logger = structlog.get_logger()


class BrowserBridge:
    """WebSocket server that accepts connections from the Nanobrowser extension.

    The extension (ws-bridge.ts) connects outbound to us. We send commands
    and receive responses/events over this connection.
    """

    def __init__(self, port: int = 9223):
        self.port = port
        self._ws: ServerConnection | None = None
        self._server: Any = None
        self._pending: dict[str, asyncio.Future] = {}
        self._listener_task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._captcha_callbacks: list = []

    async def connect(self) -> None:
        """Start WebSocket server and wait for extension to connect."""
        self._server = await serve(
            self._handle_connection,
            "localhost",
            self.port,
        )
        logger.info("browser_bridge_server_started", port=self.port)
        # Wait for extension to connect (with timeout)
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=30)
            logger.info("browser_bridge_extension_connected")
        except asyncio.TimeoutError:
            logger.warning(
                "browser_bridge_no_extension",
                msg="No extension connected within 30s. Continuing without browser.",
            )

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle incoming WebSocket connection from the extension."""
        self._ws = ws
        self._connected.set()
        logger.info("extension_connected", remote=str(ws.remote_address))

        try:
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")

                # Handle keepalive pings
                if msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue

                # Handle CAPTCHA detection events
                if msg_type == "captcha_detected":
                    logger.warning(
                        "captcha_detected_ws",
                        tab_id=data.get("tab_id"),
                        patterns=data.get("data", {}).get("patterns"),
                    )
                    continue

                # Handle correlated responses
                msg_id = data.get("id")
                if msg_id and msg_id in self._pending:
                    future = self._pending[msg_id]
                    if not future.done():
                        future.set_result(data)
        except websockets.ConnectionClosed:
            logger.warning("extension_disconnected")
        finally:
            self._ws = None
            self._connected.clear()

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._connected.clear()

    async def _send(self, message: dict) -> dict:
        """Send a command and wait for the correlated response."""
        if not self._ws or not self._connected.is_set():
            raise ConnectionError("No browser extension connected")

        msg_id = uuid.uuid4().hex[:8]
        message["id"] = msg_id

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[msg_id] = future

        try:
            await self._ws.send(json.dumps(message))
            response = await asyncio.wait_for(future, timeout=60)
            return response
        finally:
            self._pending.pop(msg_id, None)

    async def create_tab(self, url: str = "about:blank") -> str:
        """Create a new Chrome tab and return its tab_id."""
        response = await self._send({
            "type": "create_tab",
            "params": {"url": url},
        })
        return str(response.get("data", {}).get("tab_id", ""))

    async def close_tab(self, tab_id: str) -> None:
        """Close a Chrome tab."""
        await self._send({
            "type": "close_tab",
            "tab_id": tab_id,
        })

    async def get_dom(self, tab_id: str) -> dict[str, Any]:
        """Get serialized DOM state from a tab."""
        response = await self._send({
            "type": "get_dom",
            "tab_id": tab_id,
        })
        data = response.get("data", {})
        return {
            "text": data.get("dom", ""),
            "url": data.get("url", ""),
            "title": data.get("title", ""),
            "captcha_detected": data.get("captcha_detected", False),
        }

    async def execute_action(
        self, tab_id: str, action: str, params: dict | None = None
    ) -> dict[str, Any]:
        """Execute a browser action in a tab."""
        response = await self._send({
            "type": action,
            "tab_id": tab_id,
            "params": params or {},
        })
        if response.get("type") == "error":
            return {"success": False, "error": response.get("data", {}).get("message", "Unknown error")}
        return {"success": True, "data": response.get("data", {})}

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()


class MockBrowserBridge(BrowserBridge):
    """Mock bridge for testing without Chrome extension."""

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client
        self._mock_tabs: dict[str, dict] = {}
        self._tab_counter = 0

    async def connect(self) -> None:
        self._connected.set()
        logger.info("mock_browser_bridge_connected")

    async def disconnect(self) -> None:
        self._connected.clear()

    async def create_tab(self, url: str = "about:blank") -> str:
        self._tab_counter += 1
        tab_id = f"mock_tab_{self._tab_counter}"
        self._mock_tabs[tab_id] = {"url": url, "dom": ""}
        return tab_id

    async def close_tab(self, tab_id: str) -> None:
        self._mock_tabs.pop(tab_id, None)

    async def get_dom(self, tab_id: str) -> dict[str, Any]:
        tab = self._mock_tabs.get(tab_id, {})
        return {
            "text": (
                f"Page: {tab.get('url', 'about:blank')}\n"
                f"Title: Mock Page\n\n"
                f"[Interactive Elements]\n"
                f'[e1] <input type="text" placeholder="Search..." />\n'
                f"[e2] <button>Search</button>\n\n"
                f"[Page Content]\n"
                f"This is a mock page for testing."
            ),
            "url": tab.get("url", "about:blank"),
            "title": "Mock Page",
            "captcha_detected": False,
        }

    async def execute_action(
        self, tab_id: str, action: str, params: dict | None = None
    ) -> dict[str, Any]:
        params = params or {}
        if action == "navigate":
            if tab_id in self._mock_tabs:
                self._mock_tabs[tab_id]["url"] = params.get("url", "about:blank")
            return {"success": True, "data": {"navigated": params.get("url")}}
        elif action == "click":
            return {"success": True, "data": {"clicked": params.get("ref")}}
        elif action == "type_text":
            return {"success": True, "data": {"typed": params.get("text")}}
        elif action == "scroll":
            return {"success": True, "data": {"scrolled": params.get("direction")}}
        elif action == "wait":
            await asyncio.sleep(min(params.get("seconds", 1), 2))
            return {"success": True, "data": {"waited": params.get("seconds", 1)}}
        elif action == "extract":
            return {"success": True, "data": {"extracted": f"Mock extraction for: {params.get('goal', '')}"}}
        return {"success": True, "data": {}}
