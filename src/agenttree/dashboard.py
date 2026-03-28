"""Rich terminal dashboard — live tree + stats table."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from agenttree.engine.tree import NodeStatus, TreeNode
from agenttree.llm.cost import CostTracker
from agenttree.store import Store

STATUS_COLORS = {
    NodeStatus.PENDING: "dim",
    NodeStatus.DECIDING: "cyan",
    NodeStatus.BRANCHING: "blue",
    NodeStatus.WAITING_CHILDREN: "yellow",
    NodeStatus.EXECUTING: "green",
    NodeStatus.VOTING: "magenta",
    NodeStatus.AGGREGATING: "blue",
    NodeStatus.COMPLETE: "bold green",
    NodeStatus.FAILED: "bold red",
}

STATUS_ICONS = {
    NodeStatus.PENDING: ".",
    NodeStatus.DECIDING: "?",
    NodeStatus.BRANCHING: "+",
    NodeStatus.WAITING_CHILDREN: "~",
    NodeStatus.EXECUTING: ">",
    NodeStatus.VOTING: "V",
    NodeStatus.AGGREGATING: "A",
    NodeStatus.COMPLETE: "OK",
    NodeStatus.FAILED: "X",
}


class Dashboard:
    def __init__(self, store: Store, cost_tracker: CostTracker):
        self.store = store
        self.cost_tracker = cost_tracker
        self.console = Console()
        self._live: Live | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    def _build_tree_widget(self, nodes: list[TreeNode]) -> Tree:
        """Build a Rich Tree from the node list."""
        if not nodes:
            return Tree("[dim]No nodes yet[/dim]")

        node_map = {n.id: n for n in nodes}
        tree_nodes: dict[str, Any] = {}

        # Find root
        root = None
        for n in nodes:
            if n.parent_id is None:
                root = n
                break
        if not root:
            return Tree("[dim]No root node[/dim]")

        def add_node(parent_tree: Tree, node: TreeNode) -> None:
            color = STATUS_COLORS.get(node.status, "white")
            icon = STATUS_ICONS.get(node.status, "?")
            label = (
                f"[{color}][{icon}] {node.id} "
                f"d={node.depth} "
                f"{node.task[:60]}{'...' if len(node.task) > 60 else ''}"
            )
            if node.confidence > 0:
                label += f" ({node.confidence:.0%})"
            label += f"[/{color}]"

            branch = parent_tree.add(label)
            tree_nodes[node.id] = branch

            # Add children
            for child_id in node.children_ids:
                child = node_map.get(child_id)
                if child:
                    add_node(branch, child)

        tree = Tree(f"[bold]AgentTree[/bold]")
        add_node(tree, root)
        return tree

    def _build_stats_table(self, nodes: list[TreeNode]) -> Table:
        """Build a stats table."""
        table = Table(title="Run Statistics", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        total = len(nodes)
        complete = sum(1 for n in nodes if n.status == NodeStatus.COMPLETE)
        failed = sum(1 for n in nodes if n.status == NodeStatus.FAILED)
        executing = sum(1 for n in nodes if n.status == NodeStatus.EXECUTING)
        branching = sum(1 for n in nodes if n.status in (
            NodeStatus.BRANCHING, NodeStatus.WAITING_CHILDREN, NodeStatus.AGGREGATING
        ))
        pending = sum(1 for n in nodes if n.status in (
            NodeStatus.PENDING, NodeStatus.DECIDING
        ))

        table.add_row("Total nodes", str(total))
        table.add_row("Complete", f"[green]{complete}[/green]")
        table.add_row("Failed", f"[red]{failed}[/red]" if failed else "0")
        table.add_row("Executing", f"[green]{executing}[/green]" if executing else "0")
        table.add_row("Branching", f"[blue]{branching}[/blue]" if branching else "0")
        table.add_row("Pending", str(pending))
        table.add_row("", "")
        table.add_row("LLM calls", str(self.cost_tracker.call_count))
        table.add_row("Total cost", f"[yellow]${self.cost_tracker.total:.4f}[/yellow]")
        table.add_row("Budget left", f"${self.cost_tracker.budget_remaining:.2f}")
        table.add_row("Tokens (in/out)",
                       f"{self.cost_tracker.total_input_tokens:,} / {self.cost_tracker.total_output_tokens:,}")

        return table

    def _render(self, nodes: list[TreeNode]) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(Panel(self._build_tree_widget(nodes), title="Tree"), ratio=3),
            Layout(Panel(self._build_stats_table(nodes), title="Stats"), ratio=1),
        )
        return layout

    async def _update_loop(self) -> None:
        """Periodically refresh the dashboard."""
        with Live(console=self.console, refresh_per_second=2) as live:
            self._live = live
            while self._running:
                try:
                    nodes = await self.store.load_all_nodes()
                    live.update(self._render(nodes))
                except Exception:
                    pass
                await asyncio.sleep(0.5)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def final_summary(self, root: TreeNode | None) -> None:
        """Print a final summary after completion."""
        nodes = await self.store.load_all_nodes()
        self.console.print()
        self.console.print(Panel(self._build_tree_widget(nodes), title="Final Tree"))
        self.console.print(self._build_stats_table(nodes))

        if root and root.result:
            self.console.print()
            self.console.print(Panel(
                str(root.result)[:2000],
                title="Result",
                border_style="green" if root.status == NodeStatus.COMPLETE else "red",
            ))
