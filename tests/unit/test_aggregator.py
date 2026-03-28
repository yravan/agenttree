"""Unit tests for the aggregator (combining child results via LLM)."""

from __future__ import annotations

import json

import pytest

from agenttree.config import Settings
from agenttree.nodes.aggregator import aggregate, AggregationResult


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
        self._idx = 0

    async def call(self, messages, *, node_id="", role="", model=None,
                   temperature=None, max_tokens=None, tools=None, response_format=None):
        self.calls.append({"messages": messages, "role": role, "node_id": node_id, "tools": tools})
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
            return resp
        return {"content": "{}", "tool_calls": [], "input_tokens": 10, "output_tokens": 10, "cost": 0.001, "model": "test"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregate_all_successful():
    """Aggregation of 3 successful child results produces a combined output."""
    combined = {
        "final_result": {"products": ["A", "B", "C"]},
        "confidence": 0.9,
        "completeness": 0.95,
        "gaps": [],
        "needs_more_work": False,
        "additional_subtasks": [],
    }
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps(combined),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    child_results = [
        {"task": "Search site A", "status": "COMPLETE", "confidence": 0.9, "result": {"product": "A"}},
        {"task": "Search site B", "status": "COMPLETE", "confidence": 0.85, "result": {"product": "B"}},
        {"task": "Search site C", "status": "COMPLETE", "confidence": 0.92, "result": {"product": "C"}},
    ]

    result = await aggregate(
        task="Compare prices across sites",
        child_results=child_results,
        llm=llm,
        settings=settings,
        node_id="node-a1",
    )

    assert isinstance(result, AggregationResult)
    assert result.final_result == {"products": ["A", "B", "C"]}
    assert result.confidence == 0.9
    assert result.completeness == 0.95
    assert result.gaps == []
    assert result.needs_more_work is False
    assert result.additional_subtasks == []
    assert len(llm.calls) == 1
    assert llm.calls[0]["role"] == "aggregator"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregate_with_failures():
    """Aggregation handles a mix of successful and failed child results."""
    combined = {
        "final_result": {"products": ["A"]},
        "confidence": 0.6,
        "completeness": 0.4,
        "gaps": ["Site B returned an error", "Site C timed out"],
        "needs_more_work": False,
        "additional_subtasks": [],
    }
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps(combined),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    child_results = [
        {"task": "Search site A", "status": "COMPLETE", "confidence": 0.9, "result": {"product": "A"}},
        {"task": "Search site B", "status": "FAILED", "confidence": 0.0, "result": None, "error": "Connection refused"},
        {"task": "Search site C", "status": "FAILED", "confidence": 0.0, "result": None, "error": "Timeout"},
    ]

    result = await aggregate(
        task="Compare prices across sites",
        child_results=child_results,
        llm=llm,
        settings=settings,
        node_id="node-a2",
    )

    assert result.confidence == 0.6
    assert result.completeness == 0.4
    assert len(result.gaps) == 2
    assert result.needs_more_work is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregate_handles_malformed_response():
    """Malformed JSON from the LLM triggers the fallback aggregation path."""
    llm = MockLLMClient(responses=[
        {
            "content": "This is not JSON at all...",
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    child_results = [
        {"task": "Search site A", "status": "COMPLETE", "confidence": 0.9, "result": "data from A"},
        {"task": "Search site B", "status": "COMPLETE", "confidence": 0.8, "result": "data from B"},
    ]

    result = await aggregate(
        task="Combine results",
        child_results=child_results,
        llm=llm,
        settings=settings,
        node_id="node-a3",
    )

    # Fallback combines raw results
    assert result.final_result == ["data from A", "data from B"]
    assert result.confidence == 0.5
    assert result.completeness == 0.5
    assert "Failed to parse" in result.gaps[0]
    assert result.needs_more_work is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregate_triggers_respawn():
    """When needs_more_work is true, the aggregation result reflects that."""
    combined = {
        "final_result": {"partial": "data"},
        "confidence": 0.4,
        "completeness": 0.3,
        "gaps": ["Missing data from category X"],
        "needs_more_work": True,
        "additional_subtasks": ["Search category X on site D"],
    }
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps(combined),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    child_results = [
        {"task": "Search category Y", "status": "COMPLETE", "confidence": 0.7, "result": {"partial": "data"}},
    ]

    result = await aggregate(
        task="Comprehensive category search",
        child_results=child_results,
        llm=llm,
        settings=settings,
        node_id="node-a4",
    )

    assert result.needs_more_work is True
    assert len(result.additional_subtasks) == 1
    assert result.additional_subtasks[0] == "Search category X on site D"
    assert result.completeness == 0.3
