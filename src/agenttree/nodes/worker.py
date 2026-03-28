"""ReAct browser automation loop for leaf nodes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import structlog

from agenttree.browser.bridge import BrowserBridge
from agenttree.browser.captcha import CaptchaHandler
from agenttree.browser.tools import BROWSER_TOOLS_SCHEMA
from agenttree.config import Settings
from agenttree.engine.tree import TreeNode
from agenttree.llm.client import LLMClient
from agenttree.prompts.executor import format_executor_prompt

logger = structlog.get_logger()

# Actions that involve navigation and should trigger a CAPTCHA check
_NAVIGATION_ACTIONS = {"navigate", "click", "goto", "go_to_url"}

# Actions whose first positional argument is a ref-based element selector
_REF_ACTIONS = {"click", "type_text", "select"}


def _translate_ref(action_args: dict) -> dict:
    """Translate a short ref ID (e.g. 'e5') into a data-agent-ref selector."""
    args = dict(action_args)
    ref = args.get("ref") or args.get("selector") or args.get("element")
    if ref and isinstance(ref, str) and not ref.startswith("["):
        selector = f'[data-agent-ref="{ref}"]'
        # Update whichever key held the ref
        for key in ("ref", "selector", "element"):
            if key in args:
                args[key] = selector
                break
    return args


@dataclass
class ExecutionResult:
    data: Any
    success: bool
    steps_taken: int


async def execute_leaf(
    node: TreeNode,
    tab_id: str | int,
    bridge: BrowserBridge,
    llm: LLMClient,
    settings: Settings,
) -> ExecutionResult:
    """Run the ReAct browser automation loop for a leaf node."""
    max_steps = settings.timing.max_steps_per_leaf
    memory = ""
    captcha_handler = CaptchaHandler(bridge)

    for step in range(1, max_steps + 1):
        # Get DOM state (use the LLM-optimised variant)
        dom = await bridge.get_dom_for_llm(tab_id)

        # Check for CAPTCHA via poll-based handler
        await captcha_handler.wait_if_captcha(tab_id)

        # Compress old history into memory
        recent_history = _format_recent_history(node.browser_history[-5:])
        if len(node.browser_history) > 5:
            memory = _compress_history(node.browser_history[:-5], memory)

        dom_text = dom.get("text", "(no DOM)") if isinstance(dom, dict) else str(dom)

        # Detect silent failures (soft-errors, empty pages, etc.)
        try:
            from agenttree.browser.failure_detector import detect_silent_failure
            failure = await detect_silent_failure(dom_text, node.task)
            if failure:
                logger.warning(
                    "silent_failure_detected",
                    node_id=node.id,
                    tab_id=tab_id,
                    failure=failure,
                )
        except Exception:
            pass

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

            # Translate ref IDs for element-targeting actions
            if action_name in _REF_ACTIONS:
                action_args = _translate_ref(action_args)

            # Execute action in browser
            observation = await bridge.execute_action(tab_id, action_name, action_args)

            node.browser_history.append({
                "step": step,
                "thought": response["content"],
                "action": action_name,
                "params": action_args,
                "observation": str(observation),
            })

            # After navigation-like actions, check for CAPTCHAs
            if action_name in _NAVIGATION_ACTIONS:
                await captcha_handler.wait_if_captcha(tab_id)
        else:
            # No tool call -- LLM just provided text.
            # Treat substantial text content as a completed result (common in mock mode
            # or when the model doesn't support tool calling).
            content = response["content"]
            node.browser_history.append({
                "step": step,
                "thought": content,
                "action": "none",
                "params": {},
                "observation": "No action taken",
            })
            # If we get text without tool calls after step 1, treat it as the result
            if step >= 2 and content and len(content) > 50:
                return ExecutionResult(
                    data=content,
                    success=True,
                    steps_taken=step,
                )

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
            f"-> {str(entry.get('observation', ''))[:200]}"
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
