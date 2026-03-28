"""CAPTCHA detection + poll-based resolution via BrowserBridge."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console

logger = structlog.get_logger()
console = Console()

# JavaScript snippet injected into pages to detect common CAPTCHAs and
# anti-bot challenges.  Returns {detected: true, patterns: [...]} or null.
CAPTCHA_DETECTION_JS = r"""
(function() {
    var patterns = [];

    // --- CSS selector probes ---
    var selectors = [
        '.g-recaptcha',
        '.h-captcha',
        '[class*="cf-turnstile"]',
        'iframe[src*="recaptcha"]',
        'iframe[src*="hcaptcha"]',
        'iframe[src*="challenges.cloudflare"]'
    ];
    for (var i = 0; i < selectors.length; i++) {
        if (document.querySelector(selectors[i])) {
            patterns.push('selector:' + selectors[i]);
        }
    }

    // --- Visible-text heuristics ---
    var bodyText = (document.body && document.body.innerText || '').toLowerCase();
    var textPatterns = [
        'checking your browser',
        'access denied',
        'please verify you are a human',
        'ray id',
        'one more step'
    ];
    for (var j = 0; j < textPatterns.length; j++) {
        if (bodyText.indexOf(textPatterns[j]) !== -1) {
            patterns.push('text:' + textPatterns[j]);
        }
    }

    // --- Script-based detection (DataDome, PerimeterX) ---
    var scripts = document.querySelectorAll('script[src]');
    for (var k = 0; k < scripts.length; k++) {
        var src = scripts[k].src || '';
        if (/\bdd\.js\b/.test(src)) {
            patterns.push('script:datadome(dd.js)');
        }
        if (/\/px\b/.test(src) || /perimeterx/i.test(src)) {
            patterns.push('script:perimeterx');
        }
    }

    if (patterns.length > 0) {
        return {detected: true, patterns: patterns};
    }
    return null;
})();
"""


class CaptchaHandler:
    """Poll-based CAPTCHA detector that works through the BrowserBridge."""

    def __init__(self, bridge: Any):
        """
        Parameters
        ----------
        bridge : BrowserBridge
            The MCP-chrome browser bridge instance used to execute scripts.
        """
        self.bridge = bridge

    async def check_captcha(self, tab_id: str | int) -> dict | None:
        """Inject the detection JS and return a dict if a CAPTCHA is found.

        Returns
        -------
        dict  – ``{"detected": True, "patterns": [...]}`` when a CAPTCHA is
                present.
        None  – when no CAPTCHA is detected.
        """
        result = await self.bridge.execute_script(tab_id, CAPTCHA_DETECTION_JS)
        if isinstance(result, dict) and result.get("detected"):
            return result
        return None

    async def wait_if_captcha(self, tab_id: str | int) -> None:
        """If a CAPTCHA is detected, alert the user and poll until it clears.

        Polls every 3 seconds.  The user is expected to solve the challenge
        manually in the browser window.
        """
        captcha = await self.check_captcha(tab_id)
        if captcha is None:
            return

        patterns = captcha.get("patterns", [])
        console.print(
            f"\n[bold red]CAPTCHA DETECTED[/bold red] in tab [bold]{tab_id}[/bold]"
        )
        console.print(
            f"[yellow]Patterns matched: {', '.join(patterns)}[/yellow]"
        )
        console.print(
            "[yellow]Please solve the CAPTCHA in Chrome. "
            "Polling every 3 s until resolved...[/yellow]"
        )

        while True:
            await asyncio.sleep(3)
            captcha = await self.check_captcha(tab_id)
            if captcha is None:
                logger.info("captcha_resolved", tab_id=tab_id)
                console.print(
                    f"[green]CAPTCHA resolved for tab {tab_id}.[/green]"
                )
                return


def resolve_captcha(tab_id: str | int) -> None:
    """Programmatically resolve a CAPTCHA wait (for testing).

    In the poll-based model this is a no-op convenience hook -- tests should
    manipulate the page so that the detection JS returns null on the next poll.
    """
    logger.info("resolve_captcha_called", tab_id=tab_id)
