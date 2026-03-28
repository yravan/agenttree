"""Unit tests for CostTracker."""

from __future__ import annotations

import asyncio

import pytest

from agenttree.llm.cost import CostTracker


pytestmark = pytest.mark.unit


class TestCostTracker:

    @pytest.mark.asyncio
    async def test_track_cost_accumulation(self, cost_tracker: CostTracker):
        """Adding costs should accumulate total and token counts."""
        await cost_tracker.add(
            cost=0.001, input_tokens=100, output_tokens=50,
            model="test-model", node_id="n1", role="executor",
        )
        await cost_tracker.add(
            cost=0.002, input_tokens=200, output_tokens=100,
            model="test-model", node_id="n2", role="planner",
        )

        assert abs(cost_tracker.total - 0.003) < 1e-9
        assert cost_tracker.total_input_tokens == 300
        assert cost_tracker.total_output_tokens == 150
        assert cost_tracker.call_count == 2
        assert abs(cost_tracker.budget_remaining - 4.997) < 1e-9

    @pytest.mark.asyncio
    async def test_budget_warning_threshold(self):
        """Warning flag should flip when total reaches warn_at_usd."""
        tracker = CostTracker(budget_usd=1.0, warn_at_usd=0.5)
        assert tracker._warned is False

        # Below threshold
        await tracker.add(
            cost=0.4, input_tokens=100, output_tokens=50,
            model="m", node_id="n1",
        )
        assert tracker._warned is False

        # At / above threshold
        await tracker.add(
            cost=0.15, input_tokens=100, output_tokens=50,
            model="m", node_id="n2",
        )
        assert tracker._warned is True
        assert abs(tracker.total - 0.55) < 1e-9

    @pytest.mark.asyncio
    async def test_concurrent_cost_updates(self):
        """Ten coroutines adding costs concurrently must not lose updates."""
        tracker = CostTracker(budget_usd=100.0)

        async def _add(i: int) -> None:
            await tracker.add(
                cost=0.01, input_tokens=10, output_tokens=5,
                model="m", node_id=f"n{i}", role="executor",
            )

        await asyncio.gather(*[_add(i) for i in range(10)])

        assert tracker.call_count == 10
        assert abs(tracker.total - 0.10) < 1e-9
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50

    @pytest.mark.asyncio
    async def test_cost_per_node(self, cost_tracker: CostTracker):
        """Records should track per-node cost correctly."""
        await cost_tracker.add(
            cost=0.005, input_tokens=500, output_tokens=200,
            model="test-model", node_id="nodeA", role="executor",
        )
        await cost_tracker.add(
            cost=0.003, input_tokens=300, output_tokens=100,
            model="test-model", node_id="nodeB", role="planner",
        )
        await cost_tracker.add(
            cost=0.002, input_tokens=200, output_tokens=50,
            model="test-model", node_id="nodeA", role="aggregator",
        )

        node_a_records = [r for r in cost_tracker.records if r.node_id == "nodeA"]
        node_b_records = [r for r in cost_tracker.records if r.node_id == "nodeB"]

        assert len(node_a_records) == 2
        assert len(node_b_records) == 1
        assert abs(sum(r.cost_usd for r in node_a_records) - 0.007) < 1e-9

    def test_estimate_cost_fallback(self, cost_tracker: CostTracker):
        """estimate_cost should return a non-negative fallback value."""
        estimate = cost_tracker.estimate_cost(
            model="unknown/nonexistent-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert estimate >= 0
        # Fallback formula: (input + output) * 1e-6
        expected_fallback = (1000 + 500) * 1e-6
        # Either tokencost kicks in or fallback; both should be > 0
        assert estimate > 0
