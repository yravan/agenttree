"""Adaptive redundancy: execute-once → evaluate → escalate → majority vote."""

from __future__ import annotations

import json
from typing import Any

import structlog

from agenttree.browser.bridge import BrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.client import LLMClient
from agenttree.nodes.voter import vote
from agenttree.nodes.worker import ExecutionResult, execute_leaf
from agenttree.prompts.evaluator import format_evaluator_prompt

logger = structlog.get_logger()


async def evaluate_confidence(
    task: str,
    result: Any,
    llm: LLMClient,
    node_id: str = "",
) -> float:
    """Quick LLM evaluation of result confidence."""
    result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
    messages = format_evaluator_prompt(task, result_str)

    response = await llm.call(
        messages,
        node_id=node_id,
        role="evaluator",
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response["content"])
        return float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        return 0.5


async def execute_with_redundancy(
    node: TreeNode,
    tab_pool: TabPool,
    bridge: BrowserBridge,
    llm: LLMClient,
    settings: Settings,
) -> ExecutionResult:
    """Execute a leaf node with adaptive redundancy."""
    high_threshold = settings.redundancy.high_confidence_threshold
    medium_threshold = settings.redundancy.medium_confidence_threshold
    max_runs = settings.redundancy.redundancy_n

    # Run 1
    tab_id = await tab_pool.acquire()
    try:
        result1 = await execute_leaf(node, tab_id, bridge, llm, settings)
    finally:
        await tab_pool.release(tab_id)

    node.redundancy_results.append({
        "run": 1,
        "data": result1.data,
        "success": result1.success,
        "steps": result1.steps_taken,
    })

    if not settings.redundancy.adaptive:
        return result1

    # Evaluate confidence
    confidence = await evaluate_confidence(node.task, result1.data, llm, node_id=node.id)
    node.confidence = confidence
    logger.info("confidence_eval", node_id=node.id, confidence=confidence, run=1)

    # High confidence — accept immediately
    if confidence >= high_threshold:
        return result1

    # Medium confidence — run once more, pick better
    if confidence >= medium_threshold:
        node.status = NodeStatus.VOTING
        tab_id2 = await tab_pool.acquire()
        try:
            result2 = await execute_leaf(node, tab_id2, bridge, llm, settings)
        finally:
            await tab_pool.release(tab_id2)

        node.redundancy_results.append({
            "run": 2,
            "data": result2.data,
            "success": result2.success,
            "steps": result2.steps_taken,
        })

        vote_result = await vote(
            node.task,
            [result1.data, result2.data],
            llm,
            settings,
            node_id=node.id,
        )
        node.vote_outcome = {
            "best_index": vote_result.best_index,
            "confidence": vote_result.confidence,
            "reasoning": vote_result.reasoning,
            "method": vote_result.method,
        }
        node.confidence = vote_result.confidence

        best = [result1, result2][vote_result.best_index]
        return best

    # Low confidence — run max_runs - 1 more times and majority vote
    node.status = NodeStatus.VOTING
    additional_results = [result1]
    for run_num in range(2, max_runs + 1):
        tab_id_n = await tab_pool.acquire()
        try:
            result_n = await execute_leaf(node, tab_id_n, bridge, llm, settings)
        finally:
            await tab_pool.release(tab_id_n)

        additional_results.append(result_n)
        node.redundancy_results.append({
            "run": run_num,
            "data": result_n.data,
            "success": result_n.success,
            "steps": result_n.steps_taken,
        })

    all_data = [r.data for r in additional_results]
    vote_result = await vote(
        node.task,
        all_data,
        llm,
        settings,
        node_id=node.id,
    )
    node.vote_outcome = {
        "best_index": vote_result.best_index,
        "confidence": vote_result.confidence,
        "reasoning": vote_result.reasoning,
        "method": vote_result.method,
    }
    node.confidence = vote_result.confidence

    return additional_results[vote_result.best_index]
