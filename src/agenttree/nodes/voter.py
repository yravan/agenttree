"""Majority voting / LLM pick-best for redundancy results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog

from agenttree.config import Settings
from agenttree.llm.client import LLMClient
from agenttree.prompts.voter import format_voter_prompt

logger = structlog.get_logger()


@dataclass
class VoteResult:
    best_result: Any
    best_index: int
    confidence: float
    reasoning: str
    method: str  # "majority" or "llm"


async def vote(
    task: str,
    results: list[Any],
    llm: LLMClient,
    settings: Settings,
    node_id: str = "",
) -> VoteResult:
    """Pick the best result from N candidates."""
    if len(results) == 1:
        return VoteResult(
            best_result=results[0],
            best_index=0,
            confidence=1.0,
            reasoning="Only one result",
            method="single",
        )

    # Try DSPy majority first for string-like results
    try:
        str_results = [json.dumps(r) if not isinstance(r, str) else r for r in results]
        from dspy.predict.aggregation import majority
        majority_result = majority(str_results)
        if majority_result:
            idx = str_results.index(majority_result) if majority_result in str_results else 0
            return VoteResult(
                best_result=results[idx],
                best_index=idx,
                confidence=0.9,
                reasoning="DSPy majority vote",
                method="majority",
            )
    except Exception:
        pass

    # Fallback: LLM pick-best
    str_results = [json.dumps(r, default=str) if not isinstance(r, str) else r for r in results]
    messages = format_voter_prompt(task, str_results)

    voter_model = settings.redundancy.voter_model
    response = await llm.call(
        messages,
        node_id=node_id,
        role="voter",
        model=voter_model,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response["content"])
        best_index = int(parsed.get("best_index", 0))
        best_index = max(0, min(best_index, len(results) - 1))
        return VoteResult(
            best_result=results[best_index],
            best_index=best_index,
            confidence=float(parsed.get("confidence", 0.7)),
            reasoning=parsed.get("reasoning", "LLM selection"),
            method="llm",
        )
    except (json.JSONDecodeError, ValueError):
        # Just pick the first
        return VoteResult(
            best_result=results[0],
            best_index=0,
            confidence=0.5,
            reasoning="Fallback to first result",
            method="fallback",
        )
