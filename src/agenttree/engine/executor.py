"""Recursive execute_node() algorithm with TaskGroup parallelism."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from agenttree.browser.bridge import BrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.client import BudgetExceededError, LLMClient
from agenttree.llm.cost import CostTracker
from agenttree.nodes.aggregator import aggregate
from agenttree.nodes.decider import decide
from agenttree.redundancy import execute_with_redundancy
from agenttree.store import Store

logger = structlog.get_logger()


async def execute_node(
    node: TreeNode,
    store: Store,
    settings: Settings,
    tab_pool: TabPool,
    bridge: BrowserBridge,
    llm: LLMClient,
    cost_tracker: CostTracker,
    node_counter: dict[str, int] | None = None,
) -> TreeNode:
    """Recursively execute a tree node — branch or execute based on LLM decision."""
    if node_counter is None:
        node_counter = {"count": 0}

    node.started_at = time.time()
    node.status = NodeStatus.DECIDING
    await store.save_node(node)
    await store.log_event({
        "type": "status_change",
        "node_id": node.id,
        "old": "PENDING",
        "new": "DECIDING",
    })

    # Budget check
    if cost_tracker.total >= settings.costs.budget_usd:
        node.status = NodeStatus.FAILED
        node.error = f"Budget exceeded: ${cost_tracker.total:.2f} >= ${settings.costs.budget_usd:.2f}"
        node.completed_at = time.time()
        await store.save_node(node)
        return node

    # Total node cap
    node_counter["count"] += 1
    if node_counter["count"] > settings.tree.max_total_nodes:
        node.status = NodeStatus.FAILED
        node.error = f"Max total nodes ({settings.tree.max_total_nodes}) exceeded"
        node.completed_at = time.time()
        await store.save_node(node)
        return node

    # Decide: branch or execute
    decision = await decide(
        task=node.task,
        depth=node.depth,
        settings=settings,
        llm=llm,
        node_id=node.id,
    )
    node.decision = decision.action
    node.decision_reasoning = decision.reasoning

    if decision.action == "execute":
        # LEAF NODE — run browser automation with adaptive redundancy
        node.status = NodeStatus.EXECUTING
        await store.save_node(node)
        await store.log_event({
            "type": "status_change",
            "node_id": node.id,
            "old": "DECIDING",
            "new": "EXECUTING",
        })

        result = await execute_with_redundancy(node, tab_pool, bridge, llm, settings)
        node.result = result.data
        node.status = NodeStatus.COMPLETE
        node.completed_at = time.time()
        await store.save_node(node)
        await store.log_event({
            "type": "node_complete",
            "node_id": node.id,
            "confidence": node.confidence,
        })

    elif decision.action == "branch":
        # BRANCH NODE — spawn children, wait, aggregate
        node.status = NodeStatus.BRANCHING
        await store.save_node(node)
        await store.log_event({
            "type": "status_change",
            "node_id": node.id,
            "old": "DECIDING",
            "new": "BRANCHING",
        })

        respawn_round = 0
        max_respawn_rounds = 2  # Limit respawn iterations

        while True:
            # Create child nodes
            children: list[TreeNode] = []
            subtasks = decision.subtasks if respawn_round == 0 else decision.subtasks
            for subtask in subtasks[: settings.tree.max_breadth]:
                child = TreeNode(
                    task=subtask,
                    parent_id=node.id,
                    depth=node.depth + 1,
                )
                node.children_ids.append(child.id)
                children.append(child)
                await store.save_node(child)
                await store.log_event({
                    "type": "node_created",
                    "node_id": child.id,
                    "parent_id": node.id,
                    "task": subtask,
                })

            # Execute ALL children in parallel
            node.status = NodeStatus.WAITING_CHILDREN
            await store.save_node(node)

            async with asyncio.TaskGroup() as tg:
                for child in children:
                    tg.create_task(
                        safe_execute_node(
                            child, store, settings, tab_pool, bridge, llm,
                            cost_tracker, node_counter,
                        )
                    )

            # All children done — aggregate
            node.status = NodeStatus.AGGREGATING
            await store.save_node(node)

            child_results = []
            for child_id in node.children_ids:
                child_node = await store.load_node(child_id)
                if child_node:
                    child_results.append({
                        "task": child_node.task,
                        "status": child_node.status.value,
                        "confidence": child_node.confidence,
                        "result": child_node.result,
                        "error": child_node.error,
                    })

            aggregation = await aggregate(
                task=node.task,
                child_results=child_results,
                llm=llm,
                settings=settings,
                node_id=node.id,
            )

            node.result = aggregation.final_result
            node.confidence = aggregation.confidence
            node.completeness = aggregation.completeness

            # Check if we need to spawn more children
            if (
                aggregation.needs_more_work
                and aggregation.additional_subtasks
                and settings.tree.allow_respawn
                and respawn_round < max_respawn_rounds
                and cost_tracker.total < settings.costs.budget_usd
                and node_counter["count"] < settings.tree.max_total_nodes
            ):
                respawn_round += 1
                decision.subtasks = aggregation.additional_subtasks
                logger.info(
                    "respawning_children",
                    node_id=node.id,
                    round=respawn_round,
                    new_subtasks=len(aggregation.additional_subtasks),
                )
                continue

            break

        node.status = NodeStatus.COMPLETE
        node.completed_at = time.time()
        await store.save_node(node)
        await store.log_event({
            "type": "node_complete",
            "node_id": node.id,
            "confidence": node.confidence,
        })

    # Accumulate cost
    node.cost_usd = cost_tracker.total
    node.tokens_used = cost_tracker.total_input_tokens + cost_tracker.total_output_tokens
    await store.save_node(node)

    return node


async def safe_execute_node(
    node: TreeNode,
    store: Store,
    settings: Settings,
    tab_pool: TabPool,
    bridge: BrowserBridge,
    llm: LLMClient,
    cost_tracker: CostTracker,
    node_counter: dict[str, int] | None = None,
) -> TreeNode:
    """Wraps execute_node to prevent one child's failure from cancelling siblings."""
    try:
        return await asyncio.wait_for(
            execute_node(node, store, settings, tab_pool, bridge, llm, cost_tracker, node_counter),
            timeout=settings.timing.timeout_per_node_seconds,
        )
    except asyncio.TimeoutError:
        node.status = NodeStatus.FAILED
        node.error = f"Timed out after {settings.timing.timeout_per_node_seconds}s"
        node.completed_at = time.time()
        await store.save_node(node)
        await store.log_event({
            "type": "node_failed",
            "node_id": node.id,
            "error": node.error,
        })
        return node
    except BudgetExceededError as e:
        node.status = NodeStatus.FAILED
        node.error = str(e)
        node.completed_at = time.time()
        await store.save_node(node)
        return node
    except Exception as e:
        node.status = NodeStatus.FAILED
        node.error = str(e)
        node.completed_at = time.time()
        await store.save_node(node)
        await store.log_event({
            "type": "node_failed",
            "node_id": node.id,
            "error": str(e),
        })
        logger.error("node_execution_error", node_id=node.id, error=str(e))
        return node
