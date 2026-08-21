from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from wica.agent import Agent, build_chat_model
from wica.config import AgentConfig, MissingEnvError, WicaConfig

E2E_DIR = Path(__file__).parent

# One config per provider, named symmetrically (no privileged default) — the parametrized e2e
# tests (smoke + tool-calling round trip) run against each. Each uses api_key_env, so a provider
# whose key env var is unset skips (never fails): the key resolves at build, so MissingEnvError ->
# pytest.skip lives in real_chat_model / real_agent (not the inert load). See specs/project.md
# ("Live/e2e tests") and specs/config.md ("API key").
PROVIDER_CONFIGS = [
    E2E_DIR / "e2e.anthropic.config.json",
    E2E_DIR / "e2e.openai.config.json",
    E2E_DIR / "e2e.huggingface-hub.config.json",
]


def load_agent_config(config_path: Path) -> AgentConfig:
    """Load the config (inert — validates only, reads no env/files). The api key resolves later, at
    build; the skip-when-unset lives in the build helpers below."""
    return WicaConfig.from_json(config_path).agent


def real_chat_model(config_path: Path) -> BaseChatModel:
    """Build the chat model through WICA's own construction path (`build_chat_model`) — the same
    one `Agent.from_config` uses. Going through the shared builder (rather than a separate
    `init_chat_model` call) means the e2e tier exercises the real provider-branching construction,
    `huggingface-hub` included, instead of a divergent path that would build the wrong model."""
    try:
        return build_chat_model(load_agent_config(config_path))
    except MissingEnvError as exc:
        pytest.skip(f"{exc.env_var} not set — skipping e2e test")


def real_agent(config_path: Path, **kwargs: Any) -> Agent:
    """Build an Agent through the full config pipeline (file -> WicaConfig -> Agent), so the e2e
    tier exercises this pipeline against a live provider, not just a model builder. Deliberately
    doesn't call apply_logging: that's already covered deterministically by tests/test_config.py,
    and applying it here would reset the wica logger on every test, fighting any level a developer
    sets by hand while debugging a live e2e run."""
    try:
        return Agent.from_config(load_agent_config(config_path), **kwargs)
    except MissingEnvError as exc:
        pytest.skip(f"{exc.env_var} not set — skipping e2e test")
