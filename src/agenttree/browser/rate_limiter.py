"""
Per-domain rate limiting and concurrency control for browser automation.
"""

import asyncio
from collections import defaultdict
from urllib.parse import urlparse

from aiolimiter import AsyncLimiter


class DomainRateLimiter:
    """Enforce per-domain request rate limits."""

    def __init__(self):
        self._limiters: dict[str, AsyncLimiter] = {}
        self._defaults: dict[str, AsyncLimiter] = {
            "craigslist.org": AsyncLimiter(5, 60),
            "zillow.com": AsyncLimiter(10, 60),
            "facebook.com": AsyncLimiter(8, 60),
            "apartments.com": AsyncLimiter(10, 60),
        }
        self._default_limiter = AsyncLimiter(20, 60)

    @staticmethod
    def _get_domain(url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        # Strip www.
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    async def acquire(self, url: str):
        domain = self._get_domain(url)
        limiter = self._limiters.get(domain)
        if limiter is None:
            limiter = self._defaults.get(domain, self._default_limiter)
            self._limiters[domain] = limiter
        await limiter.acquire()


class DomainConcurrency:
    """Limit the number of concurrent requests to a single domain."""

    def __init__(self, max_per_domain: int = 3):
        self._max = max_per_domain
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    @staticmethod
    def _get_domain(url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _get_semaphore(self, domain: str) -> asyncio.Semaphore:
        if domain not in self._semaphores:
            self._semaphores[domain] = asyncio.Semaphore(self._max)
        return self._semaphores[domain]

    async def acquire(self, url: str):
        domain = self._get_domain(url)
        await self._get_semaphore(domain).acquire()

    def release(self, url: str):
        domain = self._get_domain(url)
        self._get_semaphore(domain).release()
