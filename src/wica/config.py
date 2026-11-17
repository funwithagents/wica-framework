from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised for a structurally invalid framework config (missing/unknown/mistyped key)."""


class MissingEnvError(ConfigError):
    """Raised when a config field references an environment variable that isn't set."""

    def __init__(self, env_var: str) -> None:
        super().__init__(f"environment variable {env_var!r} is not set")
        self.env_var = env_var


_logger = logging.getLogger(__name__)

# The provider for any server implementing OpenAI's Chat Completions API — the route to a local or
# self-hosted model. Named `<origin>-<api>-compatible` after the API it implements, and the only
# provider that takes `base_url`. See specs/config.md ("Providers").
OPENAI_CHAT_COMPLETIONS_COMPATIBLE = "openai-chat-completions-compatible"

# Old provider names → new ones, applied whenever an AgentConfig is created; an old name logs a
# deprecation warning. See specs/config.md ("Provider names").
PROVIDER_RENAMES: dict[str, str] = {
    "google_genai": "gemini",
    "mistralai": "mistral",
}

# The default deadline on one model call, in seconds: tight on purpose, a reactive agent's answer
# later than a few seconds is already a failure. See specs/agent.md ("Model-call deadline").
DEFAULT_MODEL_TIMEOUT = 10.0

# The default number of retries after the response guards reject a response: one second chance,
# since every retry is a full model call the user waits for. See specs/agent.md ("Response
# guards").
DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_NEXT_STEPS = 1


@dataclass(frozen=True)
class AgentConfig:
    """Plain data mirroring the dictionary/JSON ``agent`` block.

    The ``*_env``/``*_file`` fields hold references resolved at Agent build, not during config
    creation — see ``resolve_api_key`` / ``resolve_system_prompt`` and specs/config.md.

    Frozen: derive a variant with ``dataclasses.replace(...)`` rather than assigning. The structural
    invariants are enforced in ``__post_init__``, so direct construction fails the same way the
    loaders do.
    """

    provider: str
    model: str
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_provider: str = "auto"  # only used by provider "huggingface-hub"
    # The server's base URL, `/v1` included: required with "openai-chat-completions-compatible",
    # rejected with any other provider. See specs/config.md ("Providers").
    base_url: str | None = None
    # None = unbounded history. A positive int X keeps the last X reactions in the prompt (the
    # Agent lets history grow to 2X, then cuts back to X). See specs/agent.md ("History window").
    history_reactions: int | None = None
    # None = off. A fraction in (0, 1] of the model's context window the prompt should stay under;
    # a model call reporting more input tokens than that cuts the oldest reactions down to about
    # half of it. Needs a known context window: `context_window`, else the model's LangChain
    # profile, else Agent build fails. See specs/agent.md ("History budget").
    history_budget: float | None = None
    # The context window in tokens for `history_budget` (None = use the model profile's
    # max_input_tokens). Unused without `history_budget`.
    context_window: int | None = None
    # Seconds one whole model call (provider retries included) may take before the step is
    # abandoned with the `model_timeout` outcome; None = no deadline. See specs/agent.md
    # ("Model-call deadline").
    model_timeout: float | None = DEFAULT_MODEL_TIMEOUT
    # How many steps in a row the model's `<next>` note may wake before something other than the
    # Agent's own output triggers one: 0 turns the note off, None removes the cap. See
    # specs/agent.md ("Continuing after an output").
    max_next_steps: int | None = DEFAULT_MAX_NEXT_STEPS
    # How many times one reaction may call the model again after its response guards reject a
    # response (0 = never retry). See specs/agent.md ("Response guards").
    max_retries: int = DEFAULT_MAX_RETRIES
    # Whether an output is delivered while the response streams (True) or only once the whole
    # response has passed the response guards (False); None for the output mode's default: True
    # with tags, False with the output Command as a tool (where True is not supported yet). See
    # specs/agent.md ("When an output is delivered").
    stream_output: bool | None = None

    def __post_init__(self) -> None:
        # The structural invariants the loaders enforce, applied to direct construction too, so
        # every AgentConfig instance is valid. See specs/config.md.
        renamed = PROVIDER_RENAMES.get(self.provider)
        if renamed is not None:
            _logger.warning(
                "agent: provider %r is deprecated, use %r", self.provider, renamed
            )
            object.__setattr__(self, "provider", renamed)
        _validate_base_url(
            self.provider, self.base_url, self.model_kwargs, block="agent"
        )
        has_inline = self.system_prompt is not None
        has_file = self.system_prompt_file is not None
        if has_inline and has_file:
            raise ConfigError(
                "agent: specify at most one of 'system_prompt'/'system_prompt_file', not both"
            )
        if not has_inline and not has_file:
            raise ConfigError(
                "agent: specify one of 'system_prompt'/'system_prompt_file'"
            )
        if self.api_key is not None and self.api_key_env is not None:
            raise ConfigError(
                "agent: specify at most one of 'api_key'/'api_key_env', not both"
            )
        _validate_history_reactions(self.history_reactions, block="agent")
        _validate_history_budget(self.history_budget, block="agent")
        _validate_context_window(self.context_window, block="agent")
        _validate_model_timeout(self.model_timeout, block="agent")
        _validate_max_next_steps(self.max_next_steps, block="agent")
        _validate_max_retries(self.max_retries, block="agent")
        _validate_stream_output(self.stream_output, block="agent")

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, base_dir: str | Path | None = None
    ) -> AgentConfig:
        return cls(**_parse_agent_block(data, base_dir=_as_base_dir(base_dir)))


@dataclass(frozen=True)
class WicaConfig:
    agent: AgentConfig

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, base_dir: str | Path | None = None
    ) -> WicaConfig:
        return cls(**_parse_wica_block(data, base_dir=_as_base_dir(base_dir)))

    @classmethod
    def from_json(cls, text: str, *, base_dir: str | Path | None = None) -> WicaConfig:
        """Parse a WICA config from a JSON **string**. Like ``from_dict`` but for un-parsed text:
        it ``json.loads`` the string, then validates the same shape. A JSON string has no location
        of its own, so it takes the same optional ``base_dir`` as ``from_dict`` for locating a
        relative ``system_prompt_file``; ``from_json_file`` supplies it from the file's directory.
        See specs/config.md ("Plain dataclasses with dictionary and JSON loaders")."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON config: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("config must be a JSON object")
        return cls(**_parse_wica_block(data, base_dir=_as_base_dir(base_dir)))

    @classmethod
    def from_json_file(cls, path: str | Path) -> WicaConfig:
        """Load a WICA config from a dedicated JSON **file**. Reads the file, then parses it via
        ``from_json`` with ``base_dir`` derived from the file's own directory, so a relative
        ``system_prompt_file`` is located relative to the config file and stays CWD-independent.
        See specs/config.md ("System prompt")."""
        config_path = Path(path)
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"could not read config file {config_path}: {exc}"
            ) from exc
        try:
            return cls.from_json(text, base_dir=config_path.parent)
        except ConfigError as exc:
            raise ConfigError(f"in config file {config_path}: {exc}") from exc


_AGENT_REQUIRED = {"provider", "model"}
_AGENT_OPTIONAL_COMMON = {
    "api_key",
    "api_key_env",
    "model_kwargs",
    "hf_provider",
    "base_url",
    "history_reactions",
    "history_budget",
    "context_window",
    "model_timeout",
    "max_next_steps",
    "max_retries",
    "stream_output",
}
_AGENT_PROMPT_KEYS = {"system_prompt", "system_prompt_file"}
_WICA_ALLOWED = {"agent"}


def _as_base_dir(base_dir: str | Path | None) -> Path | None:
    """Normalize the optional `from_dict`/`from_json` base directory (a `str` or `Path`) into the
    `Path | None` the `_parse_*` locate plumbing expects — the same base-dir context `from_json_file`
    derives from the config file's own directory. See specs/config.md ("System prompt")."""
    return Path(base_dir) if base_dir is not None else None


def _validate_base_url(
    provider: str, base_url: Any, model_kwargs: Any, *, block: str
) -> None:
    """`base_url` is required with openai-chat-completions-compatible and rejected with any other
    provider (silently ignored, it would send the requests to the vendor instead of the intended
    server); with that provider `model_kwargs.base_url` is rejected too, so the address has one
    home. See specs/config.md ("Providers")."""
    if provider != OPENAI_CHAT_COMPLETIONS_COMPATIBLE:
        if base_url is not None:
            raise ConfigError(
                f"{block}: 'base_url' is only for provider "
                f"{OPENAI_CHAT_COMPLETIONS_COMPATIBLE!r}, not {provider!r}"
            )
        return
    if base_url is None:
        raise ConfigError(
            f"{block}: provider {OPENAI_CHAT_COMPLETIONS_COMPATIBLE!r} needs 'base_url', "
            "the server's address with its /v1 path (e.g. 'http://127.0.0.1:8080/v1')"
        )
    if not isinstance(base_url, str) or not base_url:
        raise ConfigError(
            f"{block}: 'base_url' must be a non-empty string, got {base_url!r}"
        )
    if isinstance(model_kwargs, dict) and "base_url" in model_kwargs:
        raise ConfigError(
            f"{block}: set the server's address in 'base_url', not in 'model_kwargs'"
        )


def _validate_history_reactions(value: Any, *, block: str) -> int | None:
    """None (unbounded) or a positive int — the number of reactions the history window keeps.
    A bool is rejected even though it is an int subclass. See specs/config.md."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            f"{block}: 'history_reactions' must be a positive integer or null, got {value!r}"
        )
    return value


def _validate_history_budget(value: Any, *, block: str) -> float | None:
    """None (off) or a number in (0, 1] — the fraction of the context window the prompt should stay
    under. See specs/config.md."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= 1
    ):
        raise ConfigError(
            f"{block}: 'history_budget' must be a number in (0, 1] or null, got {value!r}"
        )
    return float(value)


def _validate_context_window(value: Any, *, block: str) -> int | None:
    """None (use the model profile) or a positive int — the context window in tokens."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            f"{block}: 'context_window' must be a positive integer or null, got {value!r}"
        )
    return value


def _validate_model_timeout(value: Any, *, block: str) -> float | None:
    """None (no deadline) or a positive number of seconds. See specs/config.md."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ConfigError(
            f"{block}: 'model_timeout' must be a positive number of seconds or null, got {value!r}"
        )
    return float(value)


def _validate_max_next_steps(value: Any, *, block: str) -> int | None:
    """A non-negative integer, or None for no cap. See specs/config.md."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(
            f"{block}: 'max_next_steps' must be a non-negative integer or null, got {value!r}"
        )
    return value


def _validate_max_retries(value: Any, *, block: str) -> int:
    """A non-negative integer. See specs/config.md."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(
            f"{block}: 'max_retries' must be a non-negative integer, got {value!r}"
        )
    return value


def _validate_stream_output(value: Any, *, block: str) -> bool | None:
    """A boolean, or None for the output mode's default. See specs/config.md."""
    if value is not None and not isinstance(value, bool):
        raise ConfigError(
            f"{block}: 'stream_output' must be true, false or null, got {value!r}"
        )
    return value


def _require_str(data: dict[str, Any], key: str, *, block: str) -> str:
    if key not in data:
        raise ConfigError(f"{block}: missing required key {key!r}")
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(
            f"{block}: {key!r} must be a string, got {type(value).__name__}"
        )
    return value


def _validate_api_key(
    data: dict[str, Any], *, block: str
) -> tuple[str | None, str | None]:
    """Structural check only — no env read. Enforces at-most-one and string types, and passes both
    fields through onto AgentConfig; resolve_api_key does the env read later at build."""
    has_literal = "api_key" in data
    has_env_ref = "api_key_env" in data
    if has_literal and has_env_ref:
        raise ConfigError(
            f"{block}: specify at most one of 'api_key'/'api_key_env', not both"
        )
    api_key = _require_str(data, "api_key", block=block) if has_literal else None
    api_key_env = (
        _require_str(data, "api_key_env", block=block) if has_env_ref else None
    )
    return api_key, api_key_env


def _validate_system_prompt(
    data: dict[str, Any], *, base_dir: Path | None, block: str
) -> tuple[str | None, str | None]:
    """Structural check + *locate* (no read). Enforces exactly-one and string types. For a relative
    `system_prompt_file`, a caller-supplied base_dir (from_json_file derives it from the config
    directory) absolutizes it against that directory so the path is bound to the config's location,
    not the process CWD — a locate, not a file read; the read is deferred to resolve_system_prompt
    at build. With base_dir None the path is stored as given. See specs/config.md ("System
    prompt")."""
    has_inline = "system_prompt" in data
    has_file = "system_prompt_file" in data
    if has_inline and has_file:
        raise ConfigError(
            f"{block}: specify at most one of 'system_prompt'/'system_prompt_file', not both"
        )
    if not has_inline and not has_file:
        raise ConfigError(
            f"{block}: specify one of 'system_prompt'/'system_prompt_file'"
        )
    if has_inline:
        return _require_str(data, "system_prompt", block=block), None
    prompt_file = _require_str(data, "system_prompt_file", block=block)
    if base_dir is not None and not Path(prompt_file).is_absolute():
        # Locate against the config directory (absolutize) — no file *read*; .resolve() only
        # normalizes `..`/symlinks (a stat-level lookup, tolerant of a missing file) so the stored
        # path is stable regardless of the process CWD at read time.
        prompt_file = str((base_dir / prompt_file).resolve())
    return None, prompt_file


def _parse_agent_block(
    data: dict[str, Any], *, base_dir: Path | None
) -> dict[str, Any]:
    block = "agent"
    if not isinstance(data, dict):
        raise ConfigError(f"{block}: must be a JSON object")
    allowed = _AGENT_REQUIRED | _AGENT_OPTIONAL_COMMON | _AGENT_PROMPT_KEYS
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{block}: unknown key(s) {sorted(unknown)}")

    provider = _require_str(data, "provider", block=block)
    model = _require_str(data, "model", block=block)
    system_prompt, system_prompt_file = _validate_system_prompt(
        data, base_dir=base_dir, block=block
    )
    api_key, api_key_env = _validate_api_key(data, block=block)

    model_kwargs = data.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ConfigError(f"{block}: 'model_kwargs' must be an object")

    hf_provider = data["hf_provider"] if "hf_provider" in data else "auto"
    if not isinstance(hf_provider, str):
        raise ConfigError(
            f"{block}: 'hf_provider' must be a string, got {type(hf_provider).__name__}"
        )

    base_url = data.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise ConfigError(
            f"{block}: 'base_url' must be a string, got {type(base_url).__name__}"
        )

    history_reactions = _validate_history_reactions(
        data.get("history_reactions"), block=block
    )
    history_budget = _validate_history_budget(data.get("history_budget"), block=block)
    context_window = _validate_context_window(data.get("context_window"), block=block)
    model_timeout = _validate_model_timeout(
        data.get("model_timeout", DEFAULT_MODEL_TIMEOUT), block=block
    )
    max_next_steps = _validate_max_next_steps(
        data.get("max_next_steps", DEFAULT_MAX_NEXT_STEPS), block=block
    )
    max_retries = _validate_max_retries(
        data.get("max_retries", DEFAULT_MAX_RETRIES), block=block
    )
    stream_output = _validate_stream_output(data.get("stream_output"), block=block)

    return {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "system_prompt_file": system_prompt_file,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "model_kwargs": model_kwargs,
        "hf_provider": hf_provider,
        "base_url": base_url,
        "history_reactions": history_reactions,
        "history_budget": history_budget,
        "context_window": context_window,
        "model_timeout": model_timeout,
        "max_next_steps": max_next_steps,
        "max_retries": max_retries,
        "stream_output": stream_output,
    }


def _parse_wica_block(data: dict[str, Any], *, base_dir: Path | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError("config: must be a JSON object")
    unknown = set(data) - _WICA_ALLOWED
    if unknown:
        raise ConfigError(f"config: unknown key(s) {sorted(unknown)}")

    if "agent" not in data:
        raise ConfigError("config: missing required key 'agent'")
    agent_data = data["agent"]
    if not isinstance(agent_data, dict):
        raise ConfigError("config: 'agent' must be an object")
    agent = AgentConfig(**_parse_agent_block(agent_data, base_dir=base_dir))

    return {"agent": agent}


def resolve_api_key(config: AgentConfig) -> str | None:
    """Resolve the effective api key at Agent build: literal `api_key` if set; else the value of
    the env var named by `api_key_env` (raising MissingEnvError if unset); else None (so the
    provider reads its own standard env var). See specs/config.md ("API key")."""
    if config.api_key is not None:
        return config.api_key
    if config.api_key_env is not None:
        value = os.environ.get(config.api_key_env)
        if not value:
            raise MissingEnvError(config.api_key_env)
        return value
    return None


def resolve_system_prompt(config: AgentConfig) -> str:
    """Resolve the effective system prompt at Agent build: inline `system_prompt` if set, else the
    contents of `system_prompt_file` (already absolutized by from_json_file). A missing/unreadable file
    raises ConfigError naming the path. Load-time validation guarantees exactly one is set. See
    specs/config.md ("System prompt")."""
    if config.system_prompt is not None:
        return config.system_prompt
    if config.system_prompt_file is not None:
        prompt_path = Path(config.system_prompt_file)
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"could not read system_prompt_file {prompt_path}: {exc}"
            ) from exc
    raise ConfigError(
        "agent: no system prompt configured"
    )  # unreachable: __post_init__ guarantees exactly one is set
