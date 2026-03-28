"""
Detect silent failures and soft blocks in scraped page content.
"""

import structlog

logger = structlog.get_logger()

DENIED_PATTERNS = [
    "access denied",
    "403 forbidden",
    "request blocked",
    "unusual traffic",
    "automated requests",
    "enable javascript",
    "please enable cookies",
    "this site can't be reached",
]


async def detect_silent_failure(page_content: str, task: str) -> str | None:
    """Detect soft blocks / silent failures from page content.

    Returns a short failure reason string, or None if the page looks normal.
    """
    content_lower = page_content.lower()

    # Empty page when results expected
    if len(page_content.strip()) < 100 and "search" in task.lower():
        return "suspiciously_empty"

    # Access denied patterns
    for pattern in DENIED_PATTERNS:
        if pattern in content_lower:
            logger.warning("silent_failure_detected", pattern=pattern)
            return f"blocked:{pattern}"

    return None
