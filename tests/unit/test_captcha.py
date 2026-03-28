"""Unit tests for CAPTCHA detection and resolution."""

from __future__ import annotations

import asyncio

import pytest

from agenttree.browser.captcha import _captcha_events, resolve_captcha


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_captcha_sets_event():
    """resolve_captcha() should set the event and remove it from the registry."""
    tab_id = "test_tab_42"
    event = asyncio.Event()
    _captcha_events[tab_id] = event

    assert not event.is_set()
    assert tab_id in _captcha_events

    resolve_captcha(tab_id)

    assert event.is_set()
    assert tab_id not in _captcha_events
