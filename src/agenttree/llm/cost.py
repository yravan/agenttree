"""Cost tracking with tokencost + OpenRouter header extraction."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# Try tokencost for estimation; fall back to simple heuristic
try:
    from tokencost import calculate_prompt_cost, calculate_completion_cost

    HAS_TOKENCOST = True
except ImportError:
    HAS_TOKENCOST = False


@dataclass
class LLMCallRecord:
    node_id: str
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    def __init__(self, budget_usd: float = 10.0, warn_at_usd: float = 7.5):
        self.budget_usd = budget_usd
        self.warn_at_usd = warn_at_usd
        self._lock = asyncio.Lock()
        self.total: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.records: list[LLMCallRecord] = []
        self._warned = False

    async def add(
        self,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
        node_id: str = "",
        role: str = "",
    ) -> None:
        async with self._lock:
            self.total += cost
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.records.append(
                LLMCallRecord(
                    node_id=node_id,
                    role=role,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            )
            if not self._warned and self.total >= self.warn_at_usd:
                self._warned = True
                logger.warning(
                    "cost_warning",
                    total=f"${self.total:.2f}",
                    budget=f"${self.budget_usd:.2f}",
                )

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if HAS_TOKENCOST:
            try:
                # Strip common prefixes for tokencost compatibility
                clean_model = model
                for prefix in ("openrouter/", "openai/", "anthropic/", "google/", "meta-llama/"):
                    if clean_model.startswith(prefix):
                        clean_model = clean_model[len(prefix):]
                        break
                prompt_cost = calculate_prompt_cost(
                    " " * input_tokens, clean_model
                )
                completion_cost = calculate_completion_cost(
                    " " * output_tokens, clean_model
                )
                return float(prompt_cost + completion_cost)
            except Exception:
                pass
        # Fallback: rough estimate at ~$1/1M tokens
        return (input_tokens + output_tokens) * 1e-6

    @property
    def budget_remaining(self) -> float:
        return max(0, self.budget_usd - self.total)

    @property
    def call_count(self) -> int:
        return len(self.records)
