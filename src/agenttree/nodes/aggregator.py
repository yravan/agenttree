"""Combine child results via LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog

from agenttree.config import Settings
from agenttree.llm.client import LLMClient
from agenttree.prompts.aggregator import format_aggregation_prompt

logger = structlog.get_logger()


@dataclass
class AggregationResult:
    final_result: Any
    confidence: float
    completeness: float
    gaps: list[str]
    needs_more_work: bool
    additional_subtasks: list[str]


async def aggregate(
    task: str,
    child_results: list[dict],
    llm: LLMClient,
    settings: Settings,
    node_id: str = "",
) -> AggregationResult:
    """Aggregate child results into a combined output."""
    messages = format_aggregation_prompt(task, child_results)

    response = await llm.call(
        messages,
        node_id=node_id,
        role="aggregator",
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response["content"])
    except json.JSONDecodeError:
        logger.warning("aggregation_parse_error", content=response["content"][:200])
        # Fallback: combine raw results
        combined = [cr.get("result") for cr in child_results if cr.get("result")]
        return AggregationResult(
            final_result=combined,
            confidence=0.5,
            completeness=0.5,
            gaps=["Failed to parse aggregation response"],
            needs_more_work=False,
            additional_subtasks=[],
        )

    return AggregationResult(
        final_result=parsed.get("final_result", parsed),
        confidence=float(parsed.get("confidence", 0.5)),
        completeness=float(parsed.get("completeness", 0.5)),
        gaps=parsed.get("gaps", []),
        needs_more_work=parsed.get("needs_more_work", False),
        additional_subtasks=parsed.get("additional_subtasks", []),
    )
