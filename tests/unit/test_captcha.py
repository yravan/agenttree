"""Unit tests for CAPTCHA detection (poll-based, mcp-chrome)."""

from __future__ import annotations

import pytest

from agenttree.browser.captcha import CaptchaHandler, CAPTCHA_DETECTION_JS, resolve_captcha


class FakeBridge:
    """Minimal mock bridge for CAPTCHA testing."""

    def __init__(self, script_result=None):
        self._script_result = script_result

    async def execute_script(self, tab_id, script):
        return self._script_result


@pytest.mark.unit
def test_captcha_detection_js_is_nonempty():
    assert len(CAPTCHA_DETECTION_JS) > 100


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_captcha_returns_none_when_clean():
    bridge = FakeBridge(script_result=None)
    handler = CaptchaHandler(bridge)
    result = await handler.check_captcha("tab1")
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_captcha_returns_dict_when_detected():
    bridge = FakeBridge(script_result={"detected": True, "patterns": ["selector:.g-recaptcha"]})
    handler = CaptchaHandler(bridge)
    result = await handler.check_captcha("tab1")
    assert result is not None
    assert result["detected"] is True
    assert "selector:.g-recaptcha" in result["patterns"]


@pytest.mark.unit
def test_resolve_captcha_does_not_raise():
    resolve_captcha("tab_99")
