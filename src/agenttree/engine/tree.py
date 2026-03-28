"""TreeNode dataclass and NodeStatus enum."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class NodeStatus(str, enum.Enum):
    PENDING = "PENDING"
    DECIDING = "DECIDING"
    BRANCHING = "BRANCHING"
    WAITING_CHILDREN = "WAITING_CHILDREN"
    EXECUTING = "EXECUTING"
    VOTING = "VOTING"
    AGGREGATING = "AGGREGATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class TreeNode:
    task: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None
    status: NodeStatus = NodeStatus.PENDING
    depth: int = 0

    # Decision
    decision: str | None = None  # "branch" or "execute"
    decision_reasoning: str = ""

    # Children
    children_ids: list[str] = field(default_factory=list)

    # Results
    result: Any | None = None
    confidence: float = 0.0
    completeness: float = 0.0

    # Redundancy
    redundancy_results: list[dict] = field(default_factory=list)
    vote_outcome: dict | None = None

    # Tracking
    tokens_used: int = 0
    cost_usd: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    retries: int = 0
    error: str | None = None

    # Browser state (for leaves)
    browser_history: list[dict] = field(default_factory=list)
    tab_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "depth": self.depth,
            "task": self.task,
            "decision": self.decision,
            "decision_reasoning": self.decision_reasoning,
            "children_ids": self.children_ids,
            "result": self.result,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retries": self.retries,
            "error": self.error,
            "browser_history": self.browser_history,
            "tab_id": self.tab_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreeNode:
        data = dict(data)
        data["status"] = NodeStatus(data["status"])
        # Handle JSON string fields
        import json
        for list_field in ("children_ids", "browser_history", "redundancy_results"):
            if isinstance(data.get(list_field), str):
                data[list_field] = json.loads(data[list_field])
        if isinstance(data.get("result"), str):
            try:
                data["result"] = json.loads(data["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(data.get("vote_outcome"), str):
            try:
                data["vote_outcome"] = json.loads(data["vote_outcome"])
            except (json.JSONDecodeError, TypeError):
                pass
        # Remove unknown keys
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)
