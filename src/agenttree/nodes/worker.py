"""ReAct browser automation loop for leaf nodes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import structlog

from agenttree.browser.bridge import BrowserBridge
from agenttree.browser.tools import BROWSER_TOOLS_SCHEMA
from agenttree.config import Settings
from agenttree.engine.tree import TreeNode
from agenttree.llm.client import LLMClient
from agenttree.prompts.executor import format_executor_prompt

logger = structlog.get_logger()


@dataclass
class ExecutionResult:
    data: Any
    success: bool
    steps_taken: int


async def execute_leaf(
    node: TreeNode,
    tab_id: str,
    bridge: BrowserBridge,
    llm: LLMClient,
    settings: Settings,
) -> ExecutionResult:
    """Run the ReAct browser automation loop for a leaf node."""
    max_steps = settings.timing.max_steps_per_leaf
    memory = ""

    for step in range(1, max_steps + 1):
        # Get DOM state
        dom = await bridge.get_dom(tab_id)

        # Check for CAPTCHA
        if dom.get("captcha_detected"):
            logger.warning("captcha_detected", node_id=node.id, tab_id=tab_id)
            from agenttree.browser.captcha import wait_for_captcha_resolution
            await wait_for_captcha_resolution(tab_id)
            dom = await bridge.get_dom(tab_id)

        # Compress old history into memory
        recent_history = _format_recent_history(node.browser_history[-5:])
        if len(node.browser_history) > 5:
            memory = _compress_history(node.browser_history[:-5], memory)

        dom_text = dom.get("text", "(no DOM)")

        messages = format_executor_prompt(
            task=node.task,
            step=step,
            max_steps=max_steps,
            memory=memory,
            recent_history=recent_history,
            dom=dom_text,
        )

        response = await llm.call(
            messages,
            node_id=node.id,
            role="executor",
            tools=BROWSER_TOOLS_SCHEMA,
        )

        # Handle tool calls
        if response["tool_calls"]:
            tool_call = response["tool_calls"][0]
            action_name = tool_call["name"]
            action_args = tool_call["arguments"]

            if action_name == "done":
                result_data = action_args.get("result", "")
                success = action_args.get("success", True)
                node.browser_history.append({
                    "step": step,
                    "thought": response["content"],
                    "action": action_name,
                    "params": action_args,
                    "observation": "Task completed",
                })
                return ExecutionResult(
                    data=result_data,
                    success=success,
                    steps_taken=step,
                )

            # Execute action in browser
            observation = await bridge.execute_action(tab_id, action_name, action_args)

            node.browser_history.append({
                "step": step,
                "thought": response["content"],
                "action": action_name,
                "params": action_args,
                "observation": str(observation),
            })
        else:
            # No tool call — LLM just provided text
            node.browser_history.append({
                "step": step,
                "thought": response["content"],
                "action": "none",
                "params": {},
                "observation": "No action taken",
            })

        # Rate limiting
        await asyncio.sleep(settings.timing.action_delay_seconds)

    # Hit max steps
    return ExecutionResult(
        data={"partial": True, "last_observation": node.browser_history[-1] if node.browser_history else None},
        success=False,
        steps_taken=max_steps,
    )


def _format_recent_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for entry in history:
        lines.append(
            f"Step {entry.get('step', '?')}: "
            f"[{entry.get('action', '?')}] "
            f"{json.dumps(entry.get('params', {}))[:100]} "
            f"→ {str(entry.get('observation', ''))[:200]}"
        )
    return "\n".join(lines)


def _compress_history(old_history: list[dict], existing_memory: str) -> str:
    """Compress old history into a summary string."""
    actions = []
    for entry in old_history:
        actions.append(f"Step {entry.get('step', '?')}: {entry.get('action', '?')}")
    summary = "; ".join(actions)
    if existing_memory:
        return f"{existing_memory}\n{summary}"
    return summary
