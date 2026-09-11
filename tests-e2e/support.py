from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from wica import Wica
from wica.agent import build_chat_model
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
    return WicaConfig.from_json_file(config_path).agent


def real_chat_model(config_path: Path) -> BaseChatModel:
    """Build the chat model through WICA's own construction path (`build_chat_model`) — the same one
    the Agent uses at build. Going through the shared builder (rather than a separate
    `init_chat_model` call) means the e2e tier exercises the real provider-branching construction,
    `huggingface-hub` included, instead of a divergent path that would build the wrong model."""
    try:
        return build_chat_model(load_agent_config(config_path))
    except MissingEnvError as exc:
        pytest.skip(f"{exc.env_var} not set — skipping e2e test")


def real_wica(
    config_path: Path, *, system_prompt: str | None = None, **kwargs: Any
) -> Wica:
    """Stand up a full Wica (its own loop + World + Agent) from a committed config, through the real
    entrypoint — `Wica.init`. This is what makes the live tier meaningful: it drives the whole
    system the way production does, exercising the real per-provider construction (including
    `huggingface-hub`'s dedicated non-`init_chat_model` path). The caller passes code-only wiring
    (`output_sink`, `output_command`, `coalesce_window`) as kwargs. `system_prompt` overrides the
    committed persona when a test needs a specific one (replacing the committed persona; the file
    field is cleared so the exactly-one invariant holds). Skips when the config's api_key_env is
    unset — the key resolves at build inside `Wica.init`, so `MissingEnvError -> pytest.skip` lives
    here."""
    config = WicaConfig.from_json_file(config_path)
    if system_prompt is not None:
        config = WicaConfig(
            agent=dataclasses.replace(
                config.agent, system_prompt=system_prompt, system_prompt_file=None
            )
        )
    try:
        return Wica.init(config, **kwargs)
    except MissingEnvError as exc:
        pytest.skip(f"{exc.env_var} not set — skipping e2e test")
