"""CLI entry point — python -m agenttree "task"."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv
from rich.console import Console

from agenttree.browser.bridge import BrowserBridge, MockBrowserBridge
from agenttree.browser.tab_pool import TabPool
from agenttree.config import Settings, load_settings
from agenttree.dashboard import Dashboard
from agenttree.engine.executor import execute_node
from agenttree.engine.recovery import recover_and_resume
from agenttree.engine.tree import TreeNode
from agenttree.llm.client import LLMClient
from agenttree.llm.cost import CostTracker
from agenttree.store import Store, create_run_dir

logger = structlog.get_logger()
console = Console()


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level.upper(), structlog.INFO)
        ),
    )


async def run(task: str, settings: Settings, resume_run: str | None = None) -> None:
    """Execute a task through the AgentTree engine."""
    cost_tracker = CostTracker(
        budget_usd=settings.costs.budget_usd,
        warn_at_usd=settings.costs.warn_at_usd,
    )
    llm = LLMClient(settings, cost_tracker)

    # Determine bridge mode
    use_mock = os.environ.get("AGENTTREE_MOCK", "").lower() in ("1", "true", "yes")
    if use_mock:
        bridge = MockBrowserBridge(llm_client=llm)
    else:
        bridge = BrowserBridge(port=settings.extension.websocket_port)

    tab_pool = TabPool(bridge, max_tabs=settings.resources.max_parallel_tabs)

    # Create or resume run
    if resume_run:
        run_dir = Path(settings.store.path) / resume_run
        run_id = resume_run
    else:
        run_id, run_dir = create_run_dir(settings.store.path)

    store = Store(run_dir)
    await store.init()

    dashboard: Dashboard | None = None
    if settings.logging.dashboard:
        dashboard = Dashboard(store, cost_tracker)

    try:
        # Connect to browser
        await bridge.connect()

        # Start dashboard
        if dashboard:
            await dashboard.start()

        if resume_run:
            # Resume from checkpoint
            console.print(f"[yellow]Resuming run: {run_id}[/yellow]")
            root = await recover_and_resume(
                store, settings, tab_pool, bridge, llm, cost_tracker,
            )
        else:
            # Save run metadata
            await store.save_meta({
                "run_id": run_id,
                "task": task,
                "config": {
                    "model": settings.llm.model,
                    "max_depth": settings.tree.max_depth,
                    "max_breadth": settings.tree.max_breadth,
                    "budget_usd": settings.costs.budget_usd,
                },
                "started_at": time.time(),
            })

            # Create root node
            root = TreeNode(task=task, depth=0)
            await store.save_node(root)
            await store.log_event({
                "type": "run_started",
                "run_id": run_id,
                "task": task,
            })

            console.print(f"[bold]Run ID:[/bold] {run_id}")
            console.print(f"[bold]Task:[/bold] {task}")
            console.print(f"[bold]Model:[/bold] {settings.llm.model}")
            console.print(f"[bold]Budget:[/bold] ${settings.costs.budget_usd:.2f}")
            console.print()

            # Execute
            root = await asyncio.wait_for(
                execute_node(
                    root, store, settings, tab_pool, bridge, llm, cost_tracker,
                ),
                timeout=settings.timing.timeout_total_seconds,
            )

        # Save final results
        if root and root.result:
            await store.save_results(root.result)
            await store.log_event({
                "type": "run_complete",
                "run_id": run_id,
                "confidence": root.confidence,
                "cost": cost_tracker.total,
            })

    except asyncio.TimeoutError:
        console.print("[bold red]Run timed out![/bold red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — progress saved to SQLite.[/yellow]")
        console.print(f"[yellow]Resume with: agenttree --resume {run_id}[/yellow]")
    finally:
        if dashboard:
            await dashboard.stop()
            if root:
                await dashboard.final_summary(root)

        await tab_pool.close_all()
        await bridge.disconnect()
        await store.close()

    # Print summary
    console.print()
    console.print(f"[bold]Total cost:[/bold] ${cost_tracker.total:.4f}")
    console.print(f"[bold]LLM calls:[/bold] {cost_tracker.call_count}")
    console.print(f"[bold]Results saved:[/bold] {run_dir / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agenttree",
        description="Recursive tree-based browser automation framework",
    )
    parser.add_argument("task", nargs="?", help="The task to execute")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--resume", "-r", help="Resume a previous run by ID")
    parser.add_argument("--mock", action="store_true", help="Use mock browser (no Chrome extension needed)")
    parser.add_argument("--budget", type=float, help="Override budget in USD")
    parser.add_argument("--max-depth", type=int, help="Override max tree depth")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable live dashboard")

    args = parser.parse_args()

    if not args.task and not args.resume:
        parser.error("Either a task or --resume is required")

    # Load .env
    load_dotenv()

    # Load settings
    config_path = Path(args.config)
    if config_path.exists():
        settings = load_settings(config_path)
    else:
        settings = Settings()

    # Apply CLI overrides
    if args.mock:
        os.environ["AGENTTREE_MOCK"] = "1"
    if args.budget:
        settings.costs.budget_usd = args.budget
    if args.max_depth:
        settings.tree.max_depth = args.max_depth
    if args.no_dashboard:
        settings.logging.dashboard = False

    configure_logging(settings.logging.level)

    # Handle Ctrl-C gracefully
    loop = asyncio.new_event_loop()

    try:
        loop.run_until_complete(run(
            task=args.task or "",
            settings=settings,
            resume_run=args.resume,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
