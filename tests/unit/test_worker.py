"""Unit tests for the worker (ReAct browser automation loop)."""

from __future__ import annotations

import json

import pytest

from agenttree.config import Settings
from agenttree.engine.tree import TreeNode
from agenttree.browser.bridge import MockBrowserBridge
from agenttree.nodes.worker import execute_leaf, ExecutionResult


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
        self._idx = 0

    async def call(self, messages, *, node_id="", role="", model=None,
                   temperature=None, max_tokens=None, tools=None, response_format=None):
        self.calls.append({"messages": messages, "role": role, "node_id": node_id, "tools": tools})
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            return resp
        return {"content": "{}", "tool_calls": [], "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test"}


def _make_settings(max_steps: int = 5, action_delay: float = 0.0) -> Settings:
    """Create Settings with a small max_steps and zero delay for fast tests."""
    return Settings(
        timing={"max_steps_per_leaf": max_steps, "action_delay_seconds": action_delay},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_react_loop_done_on_tool_call():
    """When the LLM calls the 'done' tool, the loop exits and returns the result."""
    llm = MockLLMClient(responses=[
        {
            "content": "I found the data, calling done.",
            "tool_calls": [
                {
                    "name": "done",
                    "arguments": {"result": "Found: price is $29.99", "success": True},
                }
            ],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = _make_settings(max_steps=5)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_id = await bridge.create_tab("https://example.com")
    node = TreeNode(task="Find the price of product X", id="w1")

    result = await execute_leaf(node, tab_id, bridge, llm, settings)

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.data == "Found: price is $29.99"
    assert result.steps_taken == 1
    # Browser history should have one entry
    assert len(node.browser_history) == 1
    assert node.browser_history[0]["action"] == "done"

    await bridge.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_react_loop_max_steps():
    """When the LLM never calls done, the loop exits after max_steps."""
    # Return navigate actions but never call done -- use default fallback
    navigate_response = {
        "content": "Navigating to page.",
        "tool_calls": [
            {
                "name": "navigate",
                "arguments": {"url": "https://example.com/page"},
            }
        ],
        "input_tokens": 10,
        "output_tokens": 10,
        "cost": 0.001,
        "model": "test",
    }
    max_steps = 3
    llm = MockLLMClient(responses=[navigate_response] * max_steps)
    settings = _make_settings(max_steps=max_steps)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_id = await bridge.create_tab("https://example.com")
    node = TreeNode(task="Endless navigation task", id="w2")

    result = await execute_leaf(node, tab_id, bridge, llm, settings)

    assert result.success is False
    assert result.steps_taken == max_steps
    assert isinstance(result.data, dict)
    assert result.data.get("partial") is True
    assert len(node.browser_history) == max_steps

    await bridge.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_react_loop_text_content_result():
    """If the LLM returns text without tool calls after step 1, it is treated as the result."""
    long_text = "A" * 60  # > 50 chars to trigger the text-as-result path
    llm = MockLLMClient(responses=[
        # Step 1: no tool call, short text -- loop continues
        {
            "content": "Thinking...",
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        },
        # Step 2: no tool call, long text -- treated as result
        {
            "content": long_text,
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        },
    ])
    settings = _make_settings(max_steps=5)
    bridge = MockBrowserBridge()
    await bridge.connect()
    tab_id = await bridge.create_tab("https://example.com")
    node = TreeNode(task="Extract some data", id="w3")

    result = await execute_leaf(node, tab_id, bridge, llm, settings)

    assert result.success is True
    assert result.data == long_text
    assert result.steps_taken == 2
    assert len(node.browser_history) == 2

    await bridge.disconnect()
