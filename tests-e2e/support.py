from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, NoReturn

import pytest
from langchain_core.language_models import BaseChatModel

from wica import Wica
from wica.agent import build_chat_model
from wica.config import AgentConfig, MissingEnvError, WicaConfig
from wica.content import Content, TextPart
from wica.testing import CheckSpec, ExpectStep, InputStep, Scenario, WaitStep

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
    E2E_DIR / "e2e.google-genai.config.json",
    E2E_DIR / "e2e.mistralai.config.json",
]


# Set to "1" (CI's live job does), a missing key fails the test instead of skipping it, so a run
# cannot go green with a provider silently untested. Same name and meaning as in wica-interaction.
# See specs/testing.md ("Live tier").
FAIL_ON_MISSING = "WICA_E2E_FAIL_ON_MISSING"


def missing_key(exc: MissingEnvError) -> NoReturn:
    """Skip the calling test for a key that is not set — or fail it under `FAIL_ON_MISSING`."""
    reason = f"{exc.env_var} not set — skipping e2e test"
    if os.environ.get(FAIL_ON_MISSING) == "1":
        pytest.fail(f"{exc.env_var} not set ({FAIL_ON_MISSING}=1)", pytrace=False)
    pytest.skip(reason)


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
        missing_key(exc)


def real_wica(
    config_path: Path,
    *,
    system_prompt: str | None = None,
    agent_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Wica:
    """Stand up a full Wica (its own loop + World + Agent) from a committed config, through the real
    entrypoint — `Wica.init`. This is what makes the live tier meaningful: it drives the whole
    system the way production does, exercising the real per-provider construction (including
    `huggingface-hub`'s dedicated non-`init_chat_model` path). Extra kwargs (`coalesce_window`,
    `loop`) go to `Wica.init`; the output sink / output Command are wired by the caller afterwards
    (`wica.set_output_sink(...)`, `wica.set_output_command(...)`). `system_prompt` overrides the
    committed persona when a test needs a specific one (replacing the committed persona; the file
    field is cleared so the exactly-one invariant holds); `agent_overrides` replaces other
    `AgentConfig` fields (e.g. `history_reactions`). Skips when the config's api_key_env is unset
    (fails under `FAIL_ON_MISSING`) — the key resolves at build inside `Wica.init`, so
    `MissingEnvError -> pytest.skip` lives here."""
    config = WicaConfig.from_json_file(config_path)
    overrides = dict(agent_overrides or {})
    if system_prompt is not None:
        overrides.update(system_prompt=system_prompt, system_prompt_file=None)
    if overrides:
        config = WicaConfig(agent=dataclasses.replace(config.agent, **overrides))
    try:
        return Wica.init(config, **kwargs)
    except MissingEnvError as exc:
        missing_key(exc)


def greeting_probe() -> Scenario:
    """The cheapest probe that the demo runs end to end: one greeting, an expected reply, idle,
    and the session-wide checks. A test fixture, not benchmark data — the benchmark's workload is
    the session scenarios under benchmarks/scenarios/."""
    return Scenario(
        name="greeting",
        timeout_s=60,
        checks=(
            CheckSpec("answers_each_input"),
            CheckSpec("no_failed_reactions"),
            CheckSpec("no_colliding_cancel"),
        ),
        steps=(
            InputStep("speech_input", "Hello there!"),
            ExpectStep("output", {}, within=20),
            WaitStep("idle"),
        ),
    )


def identity_serialize(value: Any, previous: Any) -> Content:
    """The one-line serializer every e2e Input uses: the value, as text."""
    return [TextPart(str(value))]
