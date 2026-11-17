from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest
from langchain_core.language_models import BaseChatModel

from wica import Wica
from wica.agent import build_chat_model
from wica.config import AgentConfig, ConfigError, MissingEnvError, WicaConfig
from wica.content import Content, TextPart
from wica.testing import CheckSpec, ExpectStep, InputStep, Scenario, WaitStep

E2E_DIR = Path(__file__).parent

# One e2e config per provider, named symmetrically (no privileged default), plus the Hub's model a
# second time in tool mode — the parametrized e2e tests run against each. Each uses api_key_env,
# so a provider whose key env var is unset skips (never fails): the key resolves at build, so
# MissingEnvError -> pytest.skip lives in real_chat_model / real_wica (not the inert load). See
# specs/testing.md ("Live tier", "E2e configs") and specs/config.md ("API key").
PROVIDER_CONFIGS = [
    E2E_DIR / "e2e.anthropic.config.json",
    E2E_DIR / "e2e.openai.config.json",
    E2E_DIR / "e2e.huggingface-hub.config.json",
    E2E_DIR / "e2e.huggingface-hub-tool.config.json",
    E2E_DIR / "e2e.gemini.config.json",
    E2E_DIR / "e2e.mistral.config.json",
]

OUTPUT_MODES = ("tag", "tool")

# Overrides every config's output mode when set (`tag` or `tool`), to compare a model in both
# modes by hand; CI never sets it. See specs/testing.md ("E2e configs").
OUTPUT_MODE_OVERRIDE = "WICA_ROBOT_OUTPUT"


@dataclass(frozen=True)
class E2EConfig:
    """An e2e config file: the WICA config the test stands up (its `agent` block, a plain
    `WicaConfig`) and how the test wires it — `output`, the output Command as a tag (`"tag"`)
    or as a tool call (`"tool"`, `set_output_command(..., as_tool=True)`)."""

    wica: WicaConfig
    output: str

    @property
    def output_as_tool(self) -> bool:
        return self.output == "tool"


def load_e2e_config(config_path: Path) -> E2EConfig:
    """Load an e2e config file (inert — validates only, reads no env). Strict like `WicaConfig`:
    `output` is required and is `tag` or `tool`; a key other than `agent` and `output` is a
    `ConfigError`. The `agent` block goes to `WicaConfig.from_dict` as is, with the file's
    directory as `base_dir`."""
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path.name}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path.name}: an e2e config is a JSON object")
    unknown = sorted(set(data) - {"agent", "output"})
    if unknown:
        raise ConfigError(f"{config_path.name}: unknown e2e config keys {unknown}")
    output = data.get("output")
    if output not in OUTPUT_MODES:
        raise ConfigError(
            f"{config_path.name}: output must be 'tag' or 'tool', not {output!r}"
        )
    wica = WicaConfig.from_dict(
        {"agent": data.get("agent")}, base_dir=config_path.parent
    )
    return E2EConfig(wica=wica, output=output)


def output_as_tool(config_path: Path) -> bool:
    """Whether the live tests over *config_path* wire the output Command as a tool call: the
    config's `output`, unless `WICA_ROBOT_OUTPUT` (`tag` or `tool`) overrides it."""
    override = os.environ.get(OUTPUT_MODE_OVERRIDE)
    if override is not None:
        if override not in OUTPUT_MODES:
            raise ValueError(
                f"{OUTPUT_MODE_OVERRIDE} must be 'tag' or 'tool', not {override!r}"
            )
        return override == "tool"
    return load_e2e_config(config_path).output_as_tool


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


def rate_limited(exc: BaseException) -> BaseException | None:
    """The provider error in `exc`'s chain that reports HTTP 429 (a rate limit or an exhausted
    quota), if any. Duck-typed over the SDKs the providers raise through — `status_code` (OpenAI,
    Anthropic), `code` (google-genai's `ClientError`), `response.status_code` (httpx's
    `HTTPStatusError` from Mistral, huggingface_hub's `HfHubHTTPError`) — so no SDK is imported."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if 429 in (
            getattr(current, "status_code", None),
            getattr(current, "code", None),
            getattr(response, "status_code", None),
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


# config path -> the provider's 429 message at preflight, or None when the provider answered (or
# failed some other way, which is the tests' business). One entry per provider per session.
_PREFLIGHT: dict[Path, str | None] = {}


def skip_if_rate_limited(config_path: Path) -> None:
    """Skip the calling test when the provider rate-limits the key, checked with **one direct
    call per provider per session**, outside any `Wica`.

    The per-test 429 check (conftest.py) only sees a 429 that surfaces. Inside a `Wica` it often
    does not: a provider's client retries a 429 with backoff for longer than `model_timeout`, so
    the reaction ends `model_timeout` and the test fails on a missed expectation with no 429
    anywhere. A direct call has no deadline: the client gives up and raises it. A missing key is
    not this function's case and propagates (`MissingEnvError`). See specs/testing.md ("Live
    tier")."""
    if config_path not in _PREFLIGHT:
        config = load_agent_config(config_path)
        reason: str | None = None
        if config.provider != "fake":
            model = build_chat_model(config)
            try:
                model.invoke("Reply with OK.")
            except Exception as exc:
                limited = rate_limited(exc)
                reason = str(limited)[:160] if limited is not None else None
        _PREFLIGHT[config_path] = reason
    reason = _PREFLIGHT[config_path]
    if reason is not None:
        pytest.skip(f"provider rate limit (429) at preflight: {reason}")


def load_agent_config(config_path: Path) -> AgentConfig:
    """Load the e2e config's Agent block (inert — validates only, reads no env/files). The api
    key resolves later, at build; the skip-when-unset lives in the build helpers below."""
    return load_e2e_config(config_path).wica.agent


def real_chat_model(config_path: Path) -> BaseChatModel:
    """Build the chat model through WICA's own construction path (`build_chat_model`) — the same one
    the Agent uses at build. Going through the shared builder (rather than a separate
    `init_chat_model` call) means the e2e tier exercises the real provider-branching construction,
    `huggingface-hub` included, instead of a divergent path that would build the wrong model. Skips
    when the key is unset or the provider rate-limits it (`skip_if_rate_limited`)."""
    try:
        skip_if_rate_limited(config_path)
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
    `loop`) go to `Wica.init`; the output Command is wired by the caller afterwards
    (`wica.set_output_command(...)`). `system_prompt` overrides the
    committed persona when a test needs a specific one (replacing the committed persona; the file
    field is cleared so the exactly-one invariant holds); `agent_overrides` replaces other
    `AgentConfig` fields (e.g. `history_reactions`). Skips when the config's api_key_env is unset
    (fails under `FAIL_ON_MISSING`) — the key resolves at build inside `Wica.init`, so
    `MissingEnvError -> pytest.skip` lives here — and when the provider rate-limits the key
    (`skip_if_rate_limited`)."""
    config = load_e2e_config(config_path).wica
    overrides = dict(agent_overrides or {})
    if system_prompt is not None:
        overrides.update(system_prompt=system_prompt, system_prompt_file=None)
    if overrides:
        config = WicaConfig(agent=dataclasses.replace(config.agent, **overrides))
    try:
        skip_if_rate_limited(config_path)
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
