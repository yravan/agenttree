"""CAPTCHA detection + pause/resume handler."""

from __future__ import annotations

import asyncio

import structlog
from rich.console import Console

logger = structlog.get_logger()

# Global registry of CAPTCHA events per tab
_captcha_events: dict[str, asyncio.Event] = {}

console = Console()


async def wait_for_captcha_resolution(tab_id: str) -> None:
    """Alert the user about a CAPTCHA and wait for them to solve it."""
    event = asyncio.Event()
    _captcha_events[tab_id] = event

    console.print(
        f"\n[bold red]CAPTCHA DETECTED[/bold red] in tab [bold]{tab_id}[/bold]"
    )
    console.print(
        "[yellow]Please solve the CAPTCHA in Chrome, then press Enter here...[/yellow]"
    )

    # Wait for user input in a non-blocking way
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input)

    event.set()
    _captcha_events.pop(tab_id, None)
    logger.info("captcha_resolved", tab_id=tab_id)


def resolve_captcha(tab_id: str) -> None:
    """Programmatically resolve a CAPTCHA wait (for testing)."""
    event = _captcha_events.get(tab_id)
    if event:
        event.set()
        _captcha_events.pop(tab_id, None)
