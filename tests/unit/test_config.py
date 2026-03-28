"""Unit tests for Settings / config loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agenttree.config import LLMConfig, Settings, load_settings


pytestmark = pytest.mark.unit

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "configs"


class TestConfig:

    def test_load_default_config(self):
        """Loading the fixture default.yaml should produce valid Settings."""
        config_path = FIXTURES_DIR / "default.yaml"
        assert config_path.exists(), f"Fixture not found: {config_path}"
        original_yaml = Settings.model_config.get("yaml_file")
        try:
            settings = load_settings(config_path)

            assert settings.tree.max_depth == 3
            assert settings.tree.max_breadth == 5
            assert settings.costs.budget_usd == 5.0
            assert settings.timing.max_steps_per_leaf == 15
            assert settings.llm.model == "test-model"
        finally:
            Settings.model_config["yaml_file"] = original_yaml

    def test_config_defaults(self):
        """An empty (no-file) Settings object should carry sensible defaults."""
        # Temporarily point yaml_file at a non-existent path so only
        # Pydantic field defaults are used (no file loaded).
        original_yaml = Settings.model_config.get("yaml_file")
        try:
            Settings.model_config["yaml_file"] = "/tmp/_nonexistent_agenttree_test_.yaml"
            with patch.dict(os.environ, {}, clear=False):
                settings = Settings()

            assert settings.tree.max_depth == 5
            assert settings.tree.max_breadth == 10
            assert settings.costs.budget_usd == 10.0
            assert settings.timing.action_delay_seconds == 1.0
            assert settings.redundancy.adaptive is True
        finally:
            Settings.model_config["yaml_file"] = original_yaml

    def test_config_nested_model_override(self, mock_settings: Settings):
        """Overriding nested LLM sub-models should work."""
        settings = Settings(
            llm=LLMConfig(
                model="base-model",
                executor_model="executor-override",
            ),
        )
        assert settings.llm.model == "base-model"
        assert settings.llm.executor_model == "executor-override"
        assert settings.llm.planner_model is None

    def test_api_key_env_resolution(self):
        """LLMConfig.api_key should resolve from the environment variable."""
        with patch.dict(os.environ, {"MY_KEY_ENV": "sk-test-secret-123"}):
            cfg = LLMConfig(api_key_env="MY_KEY_ENV")
            assert cfg.api_key == "sk-test-secret-123"

        # When the env var is absent, api_key should return empty string
        cfg2 = LLMConfig(api_key_env="DEFINITELY_NOT_SET_XYZ")
        assert cfg2.api_key == ""

    def test_model_for_role(self):
        """model_for_role returns role-specific override or falls back to default."""
        cfg = LLMConfig(
            model="default-model",
            planner_model="planner-v2",
            aggregator_model="aggregator-v1",
        )
        assert cfg.model_for_role("planner") == "planner-v2"
        assert cfg.model_for_role("aggregator") == "aggregator-v1"
        assert cfg.model_for_role("executor") == "default-model"  # no override
        assert cfg.model_for_role("unknown_role") == "default-model"
