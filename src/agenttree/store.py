"""SQLite persistence + event log + run directory management."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from agenttree.engine.tree import NodeStatus, TreeNode

logger = structlog.get_logger()

NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    status TEXT NOT NULL,
    depth INTEGER NOT NULL,
    task TEXT,
    decision TEXT,
    decision_reasoning TEXT,
    children_ids TEXT DEFAULT '[]',
    result TEXT,
    confidence REAL DEFAULT 0.0,
    completeness REAL DEFAULT 0.0,
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    created_at REAL,
    started_at REAL,
    completed_at REAL,
    retries INTEGER DEFAULT 0,
    error TEXT,
    browser_history TEXT DEFAULT '[]'
);
"""

LLM_CALLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT,
    role TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    timestamp REAL
);
"""


class Store:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.db_path = run_dir / "tree.db"
        self.events_path = run_dir / "events.jsonl"
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Initialize the database and event log."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute(NODES_SCHEMA)
        await self._db.execute(LLM_CALLS_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save_node(self, node: TreeNode) -> None:
        """Insert or update a node."""
        assert self._db
        data = node.to_dict()
        # Serialize any non-scalar fields to JSON strings for SQLite
        for key, val in data.items():
            if val is not None and not isinstance(val, (str, int, float, bool, type(None))):
                data[key] = json.dumps(val, default=str)

        await self._db.execute(
            """INSERT OR REPLACE INTO nodes
            (id, parent_id, status, depth, task, decision, decision_reasoning,
             children_ids, result, confidence, completeness, tokens_used, cost_usd,
             created_at, started_at, completed_at, retries, error, browser_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["id"], data["parent_id"], data["status"], data["depth"],
                data["task"], data["decision"], data["decision_reasoning"],
                data["children_ids"], data["result"], data["confidence"],
                data["completeness"], data["tokens_used"], data["cost_usd"],
                data["created_at"], data["started_at"], data["completed_at"],
                data["retries"], data["error"], data["browser_history"],
            ),
        )
        await self._db.commit()

    async def load_node(self, node_id: str) -> TreeNode | None:
        """Load a node by ID."""
        assert self._db
        cursor = await self._db.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))
        return TreeNode.from_dict(data)

    async def load_all_nodes(self) -> list[TreeNode]:
        """Load all nodes."""
        assert self._db
        cursor = await self._db.execute("SELECT * FROM nodes ORDER BY created_at")
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [TreeNode.from_dict(dict(zip(columns, row))) for row in rows]

    async def load_incomplete_nodes(self) -> list[TreeNode]:
        """Load nodes that are not COMPLETE or FAILED (for crash recovery)."""
        assert self._db
        cursor = await self._db.execute(
            "SELECT * FROM nodes WHERE status NOT IN (?, ?) ORDER BY depth, created_at",
            (NodeStatus.COMPLETE.value, NodeStatus.FAILED.value),
        )
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [TreeNode.from_dict(dict(zip(columns, row))) for row in rows]

    async def log_llm_call(
        self,
        node_id: str,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        assert self._db
        await self._db.execute(
            """INSERT INTO llm_calls (node_id, role, model, input_tokens, output_tokens, cost_usd, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (node_id, role, model, input_tokens, output_tokens, cost_usd, time.time()),
        )
        await self._db.commit()

    async def log_event(self, event: dict) -> None:
        """Append an event to the JSONL event log."""
        event.setdefault("ts", time.time())
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    async def save_meta(self, meta: dict) -> None:
        """Save run metadata."""
        meta_path = self.run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    async def save_results(self, results: Any) -> None:
        """Save final results."""
        results_path = self.run_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @property
    def node_count(self) -> int:
        """Synchronous approximate count (for dashboard)."""
        # This is a rough count; for exact, use async
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM nodes")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0


def create_run_dir(base_path: str) -> tuple[str, Path]:
    """Create a new run directory and return (run_id, path)."""
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(base_path) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "nodes").mkdir(exist_ok=True)
    return run_id, run_dir
