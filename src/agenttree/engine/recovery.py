"""Crash recovery — reconstruct tree from SQLite and resume incomplete nodes."""

from __future__ import annotations

import structlog

from agenttree.browser.bridge import BrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings
from agenttree.engine.executor import safe_execute_node
from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.client import LLMClient
from agenttree.llm.cost import CostTracker
from agenttree.store import Store

logger = structlog.get_logger()


async def recover_and_resume(
    store: Store,
    settings: Settings,
    tab_pool: TabPool,
    bridge: BrowserBridge,
    llm: LLMClient,
    cost_tracker: CostTracker,
) -> TreeNode | None:
    """Resume an incomplete run from SQLite checkpoint.

    Returns the root node after resumption, or None if no incomplete work.
    """
    incomplete = await store.load_incomplete_nodes()
    if not incomplete:
        logger.info("recovery_no_incomplete_nodes")
        return None

    all_nodes = await store.load_all_nodes()
    node_map = {n.id: n for n in all_nodes}

    # Find root
    root = None
    for n in all_nodes:
        if n.parent_id is None:
            root = n
            break

    if not root:
        logger.warning("recovery_no_root_found")
        return None

    logger.info(
        "recovery_starting",
        total_nodes=len(all_nodes),
        incomplete=len(incomplete),
    )

    # Count existing nodes for the counter
    node_counter = {"count": len(all_nodes)}

    # Resume each incomplete node based on its last status
    for node in incomplete:
        status = node.status
        logger.info("recovery_resuming_node", node_id=node.id, status=status.value)

        if status in (NodeStatus.DECIDING, NodeStatus.BRANCHING):
            # Re-run from decision
            node.status = NodeStatus.PENDING
            await store.save_node(node)
            await safe_execute_node(
                node, store, settings, tab_pool, bridge, llm,
                cost_tracker, node_counter,
            )

        elif status == NodeStatus.WAITING_CHILDREN:
            # Check which children completed, re-run incomplete ones
            children_to_resume = []
            for child_id in node.children_ids:
                child = node_map.get(child_id)
                if child and child.status not in (NodeStatus.COMPLETE, NodeStatus.FAILED):
                    child.status = NodeStatus.PENDING
                    await store.save_node(child)
                    children_to_resume.append(child)

            if children_to_resume:
                import asyncio
                async with asyncio.TaskGroup() as tg:
                    for child in children_to_resume:
                        tg.create_task(
                            safe_execute_node(
                                child, store, settings, tab_pool, bridge, llm,
                                cost_tracker, node_counter,
                            )
                        )

            # Now re-aggregate
            from agenttree.nodes.aggregator import aggregate
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

            agg = await aggregate(node.task, child_results, llm, settings, node.id)
            node.result = agg.final_result
            node.confidence = agg.confidence
            node.completeness = agg.completeness
            node.status = NodeStatus.COMPLETE
            await store.save_node(node)

        elif status == NodeStatus.EXECUTING:
            # Re-run browser task from scratch
            node.status = NodeStatus.PENDING
            node.browser_history = []
            await store.save_node(node)
            await safe_execute_node(
                node, store, settings, tab_pool, bridge, llm,
                cost_tracker, node_counter,
            )

        elif status == NodeStatus.AGGREGATING:
            # Re-run aggregation with saved child results
            from agenttree.nodes.aggregator import aggregate
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

            agg = await aggregate(node.task, child_results, llm, settings, node.id)
            node.result = agg.final_result
            node.confidence = agg.confidence
            node.completeness = agg.completeness
            node.status = NodeStatus.COMPLETE
            await store.save_node(node)

        elif status == NodeStatus.VOTING:
            # Re-run from PENDING (voting state is ephemeral)
            node.status = NodeStatus.PENDING
            await store.save_node(node)
            await safe_execute_node(
                node, store, settings, tab_pool, bridge, llm,
                cost_tracker, node_counter,
            )

    # Reload root
    root = await store.load_node(root.id)
    return root
