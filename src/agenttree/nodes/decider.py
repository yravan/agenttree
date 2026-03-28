"""LLM "branch or execute?" decision for each node."""

from __future__ import annotations

import json
from dataclasses import dataclass

import structlog

from agenttree.config import Settings
from agenttree.llm.client import LLMClient
from agenttree.prompts.decision import format_decision_prompt

logger = structlog.get_logger()


@dataclass
class Decision:
    action: str  # "branch" or "execute"
    reasoning: str
    subtasks: list[str]


async def decide(
    task: str,
    depth: int,
    settings: Settings,
    llm: LLMClient,
    node_id: str = "",
    parent_context: str = "",
) -> Decision:
    """Ask the LLM whether to branch or execute this task."""
    # Force execute at max depth
    if depth >= settings.tree.max_depth:
        return Decision(
            action="execute",
            reasoning=f"Maximum depth ({settings.tree.max_depth}) reached — forced execute.",
            subtasks=[],
        )

    messages = format_decision_prompt(
        task=task,
        depth=depth,
        max_depth=settings.tree.max_depth,
        max_breadth=settings.tree.max_breadth,
        max_steps=settings.timing.max_steps_per_leaf,
        parent_context=parent_context,
    )

    response = await llm.call(
        messages,
        node_id=node_id,
        role="planner",
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response["content"])
    except json.JSONDecodeError:
        logger.warning("decision_parse_error", content=response["content"][:200])
        # Default to execute if we can't parse
        return Decision(action="execute", reasoning="Failed to parse LLM response", subtasks=[])

    action = parsed.get("action", "execute")
    if action not in ("branch", "execute"):
        action = "execute"

    subtasks = parsed.get("subtasks", [])
    if action == "branch" and not subtasks:
        action = "execute"

    # Limit breadth
    subtasks = subtasks[: settings.tree.max_breadth]

    return Decision(
        action=action,
        reasoning=parsed.get("reasoning", ""),
        subtasks=subtasks,
    )
