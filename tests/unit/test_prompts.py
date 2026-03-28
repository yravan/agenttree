"""Unit tests for prompt template formatters."""

from __future__ import annotations

import pytest

from agenttree.prompts.aggregator import format_aggregation_prompt
from agenttree.prompts.decision import format_decision_prompt
from agenttree.prompts.evaluator import format_evaluator_prompt
from agenttree.prompts.executor import format_executor_prompt
from agenttree.prompts.voter import format_voter_prompt


pytestmark = pytest.mark.unit


class TestDecisionPrompt:

    def test_decision_prompt_format(self):
        """Decision prompt should have system + user messages with expected content."""
        messages = format_decision_prompt(
            task="Find cheapest flights to Paris",
            depth=1,
            max_depth=5,
            max_breadth=10,
            max_steps=30,
            parent_context="User wants budget travel options",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        system = messages[0]["content"]
        user = messages[1]["content"]

        # System prompt should contain key instructions
        assert "branch" in system.lower() or "BRANCH" in system
        assert "execute" in system.lower() or "EXECUTE" in system
        assert "1/5" in system  # depth/max_depth
        assert "10" in system  # max_breadth

        # User prompt should contain task and parent context
        assert "Find cheapest flights to Paris" in user
        assert "User wants budget travel options" in user

    def test_decision_prompt_at_max_depth(self):
        """At max depth, the prompt must force EXECUTE."""
        messages = format_decision_prompt(
            task="Simple lookup",
            depth=5,
            max_depth=5,
            max_breadth=10,
            max_steps=30,
        )

        system = messages[0]["content"]
        assert "MUST" in system
        assert "EXECUTE" in system
        assert "maximum depth" in system.lower()


class TestExecutorPrompt:

    def test_executor_prompt_format(self):
        """Executor prompt should contain task, step info, memory, and DOM."""
        messages = format_executor_prompt(
            task="Click the search button",
            step=3,
            max_steps=10,
            memory="Navigated to google.com. Typed query.",
            recent_history="Step 2: typed 'test query' into search box",
            dom='[e1] <input type="text" />\n[e2] <button>Search</button>',
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        system = messages[0]["content"]
        user = messages[1]["content"]

        assert "browser automation" in system.lower()
        assert "done" in system.lower()

        assert "Click the search button" in user
        assert "3/10" in user
        assert "Navigated to google.com" in user
        assert "[e2] <button>Search</button>" in user


class TestAggregatorPrompt:

    def test_aggregator_prompt_format(self):
        """Aggregation prompt should list all child results."""
        child_results = [
            {"task": "Search site A", "status": "COMPLETE", "confidence": 0.9, "result": "Found 3 items"},
            {"task": "Search site B", "status": "FAILED", "confidence": 0.0, "result": None, "error": "Timeout"},
        ]
        messages = format_aggregation_prompt(
            task="Find products across sites",
            child_results=child_results,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        system = messages[0]["content"]
        user = messages[1]["content"]

        assert "aggregator" in system.lower()
        assert "confidence" in system.lower()
        assert "completeness" in system.lower()

        assert "Find products across sites" in user
        assert "Search site A" in user
        assert "Search site B" in user
        assert "COMPLETE" in user
        assert "Timeout" in user


class TestVoterPrompt:

    def test_voter_prompt_format(self):
        """Voter prompt should present all candidate results for comparison."""
        results = [
            '{"price": 120, "source": "SiteA"}',
            '{"price": 115, "source": "SiteB"}',
            '{"price": 130, "source": "SiteC"}',
        ]
        messages = format_voter_prompt(
            task="Find cheapest widget",
            results=results,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        system = messages[0]["content"]
        user = messages[1]["content"]

        assert "best" in system.lower()
        assert "best_index" in system
        assert "0-indexed" in system

        assert "Find cheapest widget" in user
        assert "RESULT 0" in user
        assert "RESULT 1" in user
        assert "RESULT 2" in user
        assert "SiteB" in user


class TestEvaluatorPrompt:

    def test_evaluator_prompt_format(self):
        """Evaluator prompt should contain the task and result to score."""
        messages = format_evaluator_prompt(
            task="Extract hotel prices",
            result='{"hotels": [{"name": "Hotel A", "price": 99}]}',
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        system = messages[0]["content"]
        user = messages[1]["content"]

        assert "score" in system.lower() or "Score" in system
        assert "0.0" in system
        assert "1.0" in system

        assert "Extract hotel prices" in user
        assert "Hotel A" in user
