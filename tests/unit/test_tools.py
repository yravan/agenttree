"""Unit tests for BROWSER_TOOLS_SCHEMA."""

from __future__ import annotations

import pytest

from agenttree.browser.tools import BROWSER_TOOLS_SCHEMA


pytestmark = pytest.mark.unit

# Pre-index tools by name for convenience
_TOOLS_BY_NAME = {t["function"]["name"]: t for t in BROWSER_TOOLS_SCHEMA}


class TestBrowserToolsSchema:

    def test_all_tools_have_required_fields(self):
        """Every tool must have type, function.name, function.description, and function.parameters."""
        for tool in BROWSER_TOOLS_SCHEMA:
            assert "type" in tool, f"Tool missing 'type': {tool}"
            assert tool["type"] == "function"
            func = tool.get("function", {})
            assert "name" in func, f"Tool missing 'function.name': {tool}"
            assert "description" in func, f"Tool missing 'function.description': {func['name']}"
            assert "parameters" in func, f"Tool missing 'function.parameters': {func['name']}"
            params = func["parameters"]
            assert params.get("type") == "object", (
                f"Tool '{func['name']}' parameters.type must be 'object'"
            )
            assert "properties" in params, (
                f"Tool '{func['name']}' missing parameters.properties"
            )

    def test_navigate_tool_schema(self):
        """Navigate tool should require a 'url' string parameter."""
        tool = _TOOLS_BY_NAME["navigate"]
        params = tool["function"]["parameters"]
        assert "url" in params["properties"]
        assert params["properties"]["url"]["type"] == "string"
        assert "url" in params.get("required", [])

    def test_click_tool_schema(self):
        """Click tool should require a 'ref' string parameter."""
        tool = _TOOLS_BY_NAME["click"]
        params = tool["function"]["parameters"]
        assert "ref" in params["properties"]
        assert params["properties"]["ref"]["type"] == "string"
        assert "ref" in params.get("required", [])

    def test_type_text_tool_schema(self):
        """type_text tool should require 'ref' and 'text', with optional 'clear_first'."""
        tool = _TOOLS_BY_NAME["type_text"]
        params = tool["function"]["parameters"]
        props = params["properties"]

        assert "ref" in props
        assert "text" in props
        assert "clear_first" in props
        assert props["clear_first"]["type"] == "boolean"
        required = params.get("required", [])
        assert "ref" in required
        assert "text" in required

    def test_done_tool_schema(self):
        """Done tool should require 'result' and have optional 'success' boolean."""
        tool = _TOOLS_BY_NAME["done"]
        params = tool["function"]["parameters"]
        props = params["properties"]

        assert "result" in props
        assert props["result"]["type"] == "string"
        assert "success" in props
        assert props["success"]["type"] == "boolean"
        assert "result" in params.get("required", [])

    def test_all_tool_names_are_unique(self):
        """No two tools should share the same name."""
        names = [t["function"]["name"] for t in BROWSER_TOOLS_SCHEMA]
        assert len(names) == len(set(names)), f"Duplicate tool names found: {names}"
