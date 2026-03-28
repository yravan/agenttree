"""Pydantic settings model — loads config.yaml with env var overrides."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from pydantic_settings import YamlConfigSettingsSource
    HAS_YAML_SOURCE = True
except ImportError:
    HAS_YAML_SOURCE = False


class TreeConfig(BaseModel):
    max_depth: int = 5
    max_breadth: int = 10
    max_total_nodes: int = 500
    allow_respawn: bool = True


class RedundancyConfig(BaseModel):
    redundancy_n: int = 3
    adaptive: bool = True
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.5
    voter_model: str | None = None


class ResourcesConfig(BaseModel):
    max_parallel_tabs: int = 15
    max_concurrent_llm_calls: int = 10


class LLMConfig(BaseModel):
    model: str = "meta-llama/llama-3.3-70b-instruct"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key_env: str = "LLM_API_KEY"
    base_url_env: str = "LLM_BASE_URL"
    planner_model: str | None = None
    executor_model: str | None = None
    aggregator_model: str | None = None

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def base_url(self) -> str:
        return os.environ.get(self.base_url_env, "https://openrouter.ai/api/v1")

    def model_for_role(self, role: str) -> str:
        overrides = {
            "planner": self.planner_model,
            "executor": self.executor_model,
            "aggregator": self.aggregator_model,
        }
        return overrides.get(role) or self.model


class CostsConfig(BaseModel):
    budget_usd: float = 10.00
    warn_at_usd: float = 7.50


class TimingConfig(BaseModel):
    max_steps_per_leaf: int = 30
    timeout_per_node_seconds: int = 300
    timeout_total_seconds: int = 7200
    action_delay_seconds: float = 1.0


class DomConfig(BaseModel):
    max_tokens: int = 16000


class StoreConfig(BaseModel):
    path: str = "./runs"


class MCPConfig(BaseModel):
    base_url: str = "http://127.0.0.1:12306/mcp"
    timeout: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    save_conversations: bool = True
    save_dom_snapshots: bool = False
    dashboard: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        env_prefix="AGENTTREE_",
        env_nested_delimiter="__",
    )

    tree: TreeConfig = Field(default_factory=TreeConfig)
    redundancy: RedundancyConfig = Field(default_factory=RedundancyConfig)
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    dom: DomConfig = Field(default_factory=DomConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        sources = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        if HAS_YAML_SOURCE:
            sources.append(YamlConfigSettingsSource(settings_cls))
        return tuple(sources)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML file with env var overrides."""
    if config_path:
        # Set the yaml file path for pydantic-settings
        Settings.model_config["yaml_file"] = str(config_path)
    return Settings()
