"""HTTP/MCP bridge to mcp-chrome for browser automation."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class BridgeError(Exception):
    """Raised when an MCP tool call fails."""


class BrowserBridge:
    """Talks to mcp-chrome over its HTTP/MCP endpoint using JSON-RPC.

    Every browser action is an MCP tool call sent as a POST to
    ``http://127.0.0.1:12306/mcp``.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:12306/mcp"):
        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._id_counter = itertools.count(1)
        self._available_tools: list[str] = []

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        return next(self._id_counter)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Send a JSON-RPC ``tools/call`` request to mcp-chrome."""
        if self._client is None:
            raise BridgeError("BrowserBridge is not connected")

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = await self._client.post(self._base_url, json=payload)
        result = response.json()
        if "error" in result:
            raise BridgeError(result["error"].get("message", "Unknown error"))
        return result.get("result", {})

    async def _list_tools(self) -> list[str]:
        """Call ``tools/list`` to discover available tool names."""
        if self._client is None:
            raise BridgeError("BrowserBridge is not connected")

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        response = await self._client.post(self._base_url, json=payload)
        result = response.json()
        if "error" in result:
            raise BridgeError(result["error"].get("message", "Unknown error"))
        tools = result.get("result", {}).get("tools", [])
        return [t["name"] for t in tools]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open an ``httpx.AsyncClient`` and verify mcp-chrome is reachable."""
        self._client = httpx.AsyncClient(timeout=60.0)
        try:
            self._available_tools = await self._list_tools()
            self._connected = True
            logger.info(
                "browser_bridge_connected",
                url=self._base_url,
                tools=self._available_tools,
            )
        except Exception as exc:
            await self._client.aclose()
            self._client = None
            logger.error("browser_bridge_connect_failed", error=str(exc))
            raise BridgeError(f"Cannot reach mcp-chrome at {self._base_url}: {exc}") from exc

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._available_tools = []
        logger.info("browser_bridge_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    async def create_tab(self, url: str = "about:blank") -> str:
        """Create a new Chrome tab and return its tab ID."""
        result = await self.call_tool("create_tab", {"url": url})
        return str(result.get("tab_id", result.get("tabId", "")))

    async def close_tab(self, tab_id: str) -> None:
        """Close a Chrome tab."""
        await self.call_tool("close_tab", {"tab_id": tab_id})

    async def list_tabs(self) -> list[dict]:
        """Return a list of open tabs."""
        result = await self.call_tool("list_tabs", {})
        return result.get("tabs", [])

    async def switch_tab(self, tab_id: str) -> None:
        """Activate / switch to a specific tab."""
        await self.call_tool("switch_tab", {"tab_id": tab_id})

    # ------------------------------------------------------------------
    # Navigation & interaction
    # ------------------------------------------------------------------

    async def navigate(self, tab_id: str, url: str) -> dict:
        """Navigate a tab to *url*."""
        return await self.call_tool("navigate", {"tab_id": tab_id, "url": url})

    async def click(self, tab_id: str, selector: str) -> dict:
        """Click an element identified by *selector*."""
        return await self.call_tool("click", {"tab_id": tab_id, "selector": selector})

    async def type_text(
        self,
        tab_id: str,
        selector: str,
        text: str,
        clear_first: bool = True,
    ) -> dict:
        """Type *text* into an element identified by *selector*."""
        return await self.call_tool(
            "type_text",
            {
                "tab_id": tab_id,
                "selector": selector,
                "text": text,
                "clear_first": clear_first,
            },
        )

    async def scroll(
        self,
        tab_id: str,
        direction: str = "down",
        amount: int = 500,
    ) -> dict:
        """Scroll the page in *tab_id*."""
        return await self.call_tool(
            "scroll",
            {"tab_id": tab_id, "direction": direction, "amount": amount},
        )

    # ------------------------------------------------------------------
    # Page content
    # ------------------------------------------------------------------

    async def get_page_content(self, tab_id: str) -> dict:
        """Return simplified page content (text, url, title)."""
        return await self.call_tool("get_page_content", {"tab_id": tab_id})

    async def get_page_html(self, tab_id: str) -> str:
        """Return raw HTML of the page."""
        result = await self.call_tool("get_page_html", {"tab_id": tab_id})
        return result.get("html", "")

    async def screenshot(self, tab_id: str) -> str:
        """Take a screenshot and return a base-64 encoded image."""
        result = await self.call_tool("screenshot", {"tab_id": tab_id})
        return result.get("data", "")

    async def execute_script(self, tab_id: str, script: str) -> Any:
        """Inject and execute JavaScript in *tab_id*."""
        result = await self.call_tool(
            "execute_script",
            {"tab_id": tab_id, "script": script},
        )
        return result.get("result", result)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def get_dom_for_llm(self, tab_id: str, max_tokens: int = 16000) -> dict[str, Any]:
        """Inject DOM-serialization JS and return a compact DOM snapshot.

        Falls back to ``get_page_content`` when script execution fails.
        """
        from agenttree.browser.dom_serializer import DOM_SERIALIZATION_JS

        js = DOM_SERIALIZATION_JS.replace("__MAX_TOKENS__", str(max_tokens))
        try:
            raw = await self.execute_script(tab_id, js)
            # The script is expected to return {text, url, title}
            if isinstance(raw, dict):
                return {
                    "text": raw.get("text", ""),
                    "url": raw.get("url", ""),
                    "title": raw.get("title", ""),
                    "captcha_detected": False,
                }
            # Stringified JSON from the extension
            if isinstance(raw, str):
                import json
                try:
                    parsed = json.loads(raw)
                    return {
                        "text": parsed.get("text", raw),
                        "url": parsed.get("url", ""),
                        "title": parsed.get("title", ""),
                        "captcha_detected": False,
                    }
                except json.JSONDecodeError:
                    return {
                        "text": raw,
                        "url": "",
                        "title": "",
                        "captcha_detected": False,
                    }
        except Exception:
            logger.warning(
                "dom_serialization_failed_falling_back",
                tab_id=tab_id,
            )

        # Fallback: simpler representation from get_page_content
        content = await self.get_page_content(tab_id)
        return {
            "text": content.get("text", content.get("content", "")),
            "url": content.get("url", ""),
            "title": content.get("title", ""),
            "captcha_detected": False,
        }

    async def check_captcha(self, tab_id: str) -> bool:
        """Inject CAPTCHA-detection JS and return *True* if a CAPTCHA is found."""
        from agenttree.browser.captcha_js import CAPTCHA_DETECTION_JS

        try:
            raw = await self.execute_script(tab_id, CAPTCHA_DETECTION_JS)
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, dict):
                return bool(raw.get("detected", False))
            return bool(raw)
        except Exception:
            logger.warning("captcha_detection_failed", tab_id=tab_id)
            return False

    async def click_by_ref(self, tab_id: str, ref: str) -> dict:
        """Translate a DOM-serializer ref ID (e.g. ``e1``) to a CSS selector
        ``[data-agent-ref="e1"]`` and click it."""
        selector = f'[data-agent-ref="{ref}"]'
        return await self.click(tab_id, selector)

    # ------------------------------------------------------------------
    # Legacy interface expected by executor / worker
    # ------------------------------------------------------------------

    async def get_dom(self, tab_id: str) -> dict[str, Any]:
        """Return serialised DOM state (calls ``get_dom_for_llm``)."""
        return await self.get_dom_for_llm(tab_id)

    async def execute_action(
        self, tab_id: str, action: str, params: dict | None = None,
    ) -> dict[str, Any]:
        """Dispatch a named browser action and return a result dict.

        This preserves the interface used by ``worker.py`` so that both
        ``BrowserBridge`` and ``MockBrowserBridge`` remain drop-in
        replacements for one another.
        """
        params = params or {}
        try:
            if action == "navigate":
                await self.navigate(tab_id, params.get("url", "about:blank"))
            elif action == "click":
                ref = params.get("ref", params.get("selector", ""))
                if ref and not ref.startswith("["):
                    await self.click_by_ref(tab_id, ref)
                else:
                    await self.click(tab_id, ref)
            elif action == "type_text":
                ref = params.get("ref", params.get("selector", ""))
                selector = f'[data-agent-ref="{ref}"]' if ref and not ref.startswith("[") else ref
                await self.type_text(
                    tab_id,
                    selector,
                    params.get("text", ""),
                    clear_first=params.get("clear_first", True),
                )
            elif action == "scroll":
                await self.scroll(
                    tab_id,
                    direction=params.get("direction", "down"),
                    amount=params.get("amount", 500),
                )
            elif action == "select":
                # Fallback to execute_script for select actions
                ref = params.get("ref", "")
                value = params.get("value", "")
                selector = f'[data-agent-ref="{ref}"]' if ref else ""
                js = (
                    f'document.querySelector(\'{selector}\').value = "{value}";'
                    f'document.querySelector(\'{selector}\').dispatchEvent(new Event("change"));'
                )
                await self.execute_script(tab_id, js)
            elif action == "wait":
                await asyncio.sleep(min(params.get("seconds", 1), 5))
            elif action == "extract":
                dom = await self.get_dom_for_llm(tab_id)
                return {"success": True, "data": {"extracted": dom.get("text", "")}}
            elif action == "screenshot":
                data = await self.screenshot(tab_id)
                return {"success": True, "data": {"screenshot": data}}
            else:
                # Try as a raw MCP tool call
                result = await self.call_tool(action, {"tab_id": tab_id, **params})
                return {"success": True, "data": result}
            return {"success": True, "data": {}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


# ======================================================================
# Mock bridge for testing
# ======================================================================


class MockBrowserBridge(BrowserBridge):
    """In-memory mock that records every call for assertion in tests.

    No HTTP requests are made.  The public interface is identical to
    ``BrowserBridge`` so the two are interchangeable in executor code and
    in test fixtures.
    """

    def __init__(self, llm_client: Any = None, **kwargs: Any):
        # Do NOT call super().__init__() -- we manage our own state to
        # avoid any httpx dependency.
        self._base_url = "mock://mcp-chrome"
        self._client = None
        self._connected = False
        self._id_counter = itertools.count(1)
        self._available_tools: list[str] = []

        self._llm = llm_client
        self._mock_tabs: dict[str, dict[str, Any]] = {}
        self._tab_counter = 0
        self.all_commands: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, method: str, **kwargs: Any) -> None:
        self.all_commands.append({"method": method, **kwargs})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True
        self._available_tools = [
            "create_tab", "close_tab", "list_tabs", "switch_tab",
            "navigate", "click", "type_text", "scroll",
            "get_page_content", "get_page_html", "screenshot",
            "execute_script",
        ]
        self._record("connect")
        logger.info("mock_browser_bridge_connected")

    async def disconnect(self) -> None:
        self._connected = False
        self._available_tools = []
        self._record("disconnect")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # call_tool (no-op for mock)
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Record but never send an HTTP request."""
        self._record("call_tool", name=name, arguments=arguments)
        return {}

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    async def create_tab(self, url: str = "about:blank") -> str:
        self._tab_counter += 1
        tab_id = f"mock_tab_{self._tab_counter}"
        self._mock_tabs[tab_id] = {"url": url, "title": "Mock Page", "dom": ""}
        self._record("create_tab", tab_id=tab_id, url=url)
        return tab_id

    async def close_tab(self, tab_id: str) -> None:
        self._mock_tabs.pop(tab_id, None)
        self._record("close_tab", tab_id=tab_id)

    async def list_tabs(self) -> list[dict]:
        tabs = [
            {"tab_id": tid, "url": info["url"], "title": info.get("title", "")}
            for tid, info in self._mock_tabs.items()
        ]
        self._record("list_tabs")
        return tabs

    async def switch_tab(self, tab_id: str) -> None:
        self._record("switch_tab", tab_id=tab_id)

    # ------------------------------------------------------------------
    # Navigation & interaction
    # ------------------------------------------------------------------

    async def navigate(self, tab_id: str, url: str) -> dict:
        if tab_id in self._mock_tabs:
            self._mock_tabs[tab_id]["url"] = url
        self._record("navigate", tab_id=tab_id, url=url)
        return {"navigated": url}

    async def click(self, tab_id: str, selector: str) -> dict:
        self._record("click", tab_id=tab_id, selector=selector)
        return {"clicked": selector}

    async def type_text(
        self,
        tab_id: str,
        selector: str,
        text: str,
        clear_first: bool = True,
    ) -> dict:
        self._record(
            "type_text",
            tab_id=tab_id,
            selector=selector,
            text=text,
            clear_first=clear_first,
        )
        return {"typed": text}

    async def scroll(
        self,
        tab_id: str,
        direction: str = "down",
        amount: int = 500,
    ) -> dict:
        self._record("scroll", tab_id=tab_id, direction=direction, amount=amount)
        return {"scrolled": direction}

    # ------------------------------------------------------------------
    # Page content
    # ------------------------------------------------------------------

    async def get_page_content(self, tab_id: str) -> dict:
        tab = self._mock_tabs.get(tab_id, {})
        self._record("get_page_content", tab_id=tab_id)
        return {
            "text": f"Mock page content for {tab.get('url', 'about:blank')}",
            "url": tab.get("url", "about:blank"),
            "title": tab.get("title", "Mock Page"),
        }

    async def get_page_html(self, tab_id: str) -> str:
        tab = self._mock_tabs.get(tab_id, {})
        self._record("get_page_html", tab_id=tab_id)
        url = tab.get("url", "about:blank")
        return f"<html><head><title>Mock Page</title></head><body><p>{url}</p></body></html>"

    async def screenshot(self, tab_id: str) -> str:
        self._record("screenshot", tab_id=tab_id)
        return "data:image/png;base64,MOCK_SCREENSHOT_DATA"

    async def execute_script(self, tab_id: str, script: str) -> Any:
        self._record("execute_script", tab_id=tab_id, script=script[:200])
        return {"result": "mock_script_result"}

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def get_dom_for_llm(self, tab_id: str, max_tokens: int = 16000) -> dict[str, Any]:
        tab = self._mock_tabs.get(tab_id, {})
        self._record("get_dom_for_llm", tab_id=tab_id, max_tokens=max_tokens)
        url = tab.get("url", "about:blank")
        return {
            "text": (
                f"Page: {url}\n"
                f"Title: Mock Page\n\n"
                f"[Interactive Elements]\n"
                f'[e1] <input type="text" placeholder="Search..." />\n'
                f"[e2] <button>Search</button>\n\n"
                f"[Page Content]\n"
                f"This is a mock page for testing."
            ),
            "url": url,
            "title": "Mock Page",
            "captcha_detected": False,
        }

    async def check_captcha(self, tab_id: str) -> bool:
        self._record("check_captcha", tab_id=tab_id)
        return False

    async def click_by_ref(self, tab_id: str, ref: str) -> dict:
        selector = f'[data-agent-ref="{ref}"]'
        self._record("click_by_ref", tab_id=tab_id, ref=ref, selector=selector)
        return {"clicked": ref}

    # ------------------------------------------------------------------
    # Legacy interface (executor / worker compatibility)
    # ------------------------------------------------------------------

    async def get_dom(self, tab_id: str) -> dict[str, Any]:
        """Return serialised DOM state (delegates to ``get_dom_for_llm``)."""
        return await self.get_dom_for_llm(tab_id)

    async def execute_action(
        self, tab_id: str, action: str, params: dict | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        self._record("execute_action", tab_id=tab_id, action=action, params=params)

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
            return {
                "success": True,
                "data": {"extracted": f"Mock extraction for: {params.get('goal', '')}"},
            }
        return {"success": True, "data": {}}

    # ------------------------------------------------------------------
    # Test assertion helpers
    # ------------------------------------------------------------------

    def assert_navigated_to(self, url: str) -> None:
        """Assert that ``navigate`` was called with *url* at some point."""
        for cmd in self.all_commands:
            if cmd["method"] == "navigate" and cmd.get("url") == url:
                return
            if (
                cmd["method"] == "execute_action"
                and cmd.get("action") == "navigate"
                and cmd.get("params", {}).get("url") == url
            ):
                return
        raise AssertionError(f"Never navigated to {url!r}. Commands: {self.all_commands}")

    def assert_clicked(self, ref: str) -> None:
        """Assert that a click action targeted *ref*."""
        for cmd in self.all_commands:
            if cmd["method"] == "click_by_ref" and cmd.get("ref") == ref:
                return
            if cmd["method"] == "click" and cmd.get("selector", "").endswith(f'"{ref}"]'):
                return
            if (
                cmd["method"] == "execute_action"
                and cmd.get("action") == "click"
                and cmd.get("params", {}).get("ref") == ref
            ):
                return
        raise AssertionError(f"Never clicked ref {ref!r}. Commands: {self.all_commands}")

    def get_commands_for_tab(self, tab_id: str) -> list[dict[str, Any]]:
        """Return every recorded command that mentions *tab_id*."""
        return [cmd for cmd in self.all_commands if cmd.get("tab_id") == tab_id]
