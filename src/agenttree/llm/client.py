"""AsyncOpenAI wrapper for OpenRouter with concurrency limiting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import structlog
from openai import AsyncOpenAI

from agenttree.config import Settings
from agenttree.llm.cost import CostTracker

logger = structlog.get_logger()


class LLMClient:
    def __init__(self, settings: Settings, cost_tracker: CostTracker):
        self.settings = settings
        self.cost_tracker = cost_tracker
        self._client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        self._semaphore = asyncio.Semaphore(settings.resources.max_concurrent_llm_calls)
        self._cache: dict[str, dict] = {}

    def _cache_key(self, node_id: str, role: str, messages: list[dict]) -> str:
        content = json.dumps({"node_id": node_id, "role": role, "messages": messages}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        node_id: str = "",
        role: str = "default",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """Make an LLM call with caching, concurrency limiting, and cost tracking."""
        resolved_model = model or self.settings.llm.model_for_role(role)
        resolved_temp = temperature if temperature is not None else self.settings.llm.temperature
        resolved_max_tokens = max_tokens or self.settings.llm.max_tokens

        # Check idempotency cache
        cache_key = self._cache_key(node_id, role, messages)
        if cache_key in self._cache:
            logger.debug("llm_cache_hit", node_id=node_id, role=role)
            return self._cache[cache_key]

        # Budget check
        if self.cost_tracker.total >= self.settings.costs.budget_usd:
            raise BudgetExceededError(
                f"Budget exhausted: ${self.cost_tracker.total:.2f} >= ${self.settings.costs.budget_usd:.2f}"
            )

        async with self._semaphore:
            start = time.time()
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "messages": messages,
                "temperature": resolved_temp,
                "max_tokens": resolved_max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format

            response = await self._client.chat.completions.create(**kwargs)
            elapsed = time.time() - start

            choice = response.choices[0]
            message = choice.message

            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            # Cost from response headers (OpenRouter) or estimation
            cost = 0.0
            if hasattr(response, "_raw_response"):
                raw = response._raw_response
                if hasattr(raw, "headers"):
                    cost_header = raw.headers.get("x-openrouter-cost")
                    if cost_header:
                        cost = float(cost_header)
            if cost == 0.0:
                cost = self.cost_tracker.estimate_cost(
                    resolved_model, input_tokens, output_tokens
                )

            await self.cost_tracker.add(
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=resolved_model,
                node_id=node_id,
                role=role,
            )

            result = {
                "content": message.content or "",
                "tool_calls": [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "elapsed": elapsed,
                "model": resolved_model,
            }

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in message.tool_calls
                ]

            # Cache
            self._cache[cache_key] = result
            logger.info(
                "llm_call",
                node_id=node_id,
                role=role,
                model=resolved_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=f"${cost:.4f}",
                elapsed=f"{elapsed:.1f}s",
            )
            return result


class BudgetExceededError(Exception):
    pass
