"""Unit tests for the voter (majority voting / LLM pick-best)."""

from __future__ import annotations

import json

import pytest

from agenttree.config import Settings
from agenttree.nodes.voter import vote, VoteResult


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
async def test_single_result_passthrough():
    """With only one result, vote() returns it directly without calling LLM."""
    llm = MockLLMClient(responses=[])
    settings = Settings()

    result = await vote(
        task="Find something",
        results=["the only result"],
        llm=llm,
        settings=settings,
        node_id="node-v1",
    )

    assert isinstance(result, VoteResult)
    assert result.best_result == "the only result"
    assert result.best_index == 0
    assert result.confidence == 1.0
    assert result.method == "single"
    # No LLM call should have been made
    assert len(llm.calls) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_voter_picks_best():
    """The LLM voter should pick the result at the specified index."""
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps({
                "best_index": 1,
                "confidence": 0.95,
                "reasoning": "Result 1 has more complete data.",
            }),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    candidates = [
        "Partial data from site A",
        "Complete data from site B with prices and reviews",
        "Incomplete data from site C",
    ]

    result = await vote(
        task="Find product details",
        results=candidates,
        llm=llm,
        settings=settings,
        node_id="node-v2",
    )

    assert result.best_index == 1
    assert result.best_result == candidates[1]
    assert result.confidence == 0.95
    assert result.method == "llm"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_voter_handles_malformed_llm_response():
    """Invalid JSON from the LLM should fall back to first result."""
    llm = MockLLMClient(responses=[
        {
            "content": "not valid json!!!",
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    candidates = ["result A", "result B"]

    result = await vote(
        task="Pick best",
        results=candidates,
        llm=llm,
        settings=settings,
        node_id="node-v3",
    )

    assert result.best_index == 0
    assert result.best_result == "result A"
    assert result.confidence == 0.5
    assert result.method == "fallback"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_voter_with_2_results():
    """Voting with exactly 2 results should call the LLM and pick based on response."""
    llm = MockLLMClient(responses=[
        {
            "content": json.dumps({
                "best_index": 0,
                "confidence": 0.8,
                "reasoning": "First result is more accurate.",
            }),
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 10,
            "cost": 0.001,
            "model": "test",
        }
    ])
    settings = Settings()

    candidates = ["result alpha", "result beta"]

    result = await vote(
        task="Compare two results",
        results=candidates,
        llm=llm,
        settings=settings,
        node_id="node-v4",
    )

    assert result.best_result == "result alpha"
    assert result.best_index == 0
    assert result.confidence == 0.8
    # Should have called the LLM (either majority or llm method)
    assert result.method in ("majority", "llm")
