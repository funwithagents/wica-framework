from __future__ import annotations

import importlib
import json
import logging
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal

from wica.fields import Field, field_problems, read_fields


class ConfigError(ValueError):
    """Raised for a structurally invalid framework config (missing/unknown/mistyped key)."""


class MissingSecretError(ConfigError):
    """No key under the variable the config names (`api_key_env`, or the provider's default):
    neither in the secrets given in code nor in the environment. Carries the variable, so a
    caller can say which key to set without reading the message. See specs/config.md ("API
    key")."""

    def __init__(self, variable: str, *, label: str = "Agent") -> None:
        super().__init__(
            f"{label}: no key under {variable!r}: set the environment variable, "
            "or pass it in `secrets`"
        )
        self.variable = variable
        self.label = label

    @property
    def env_var(self) -> str:
        """The variable's name, under its old attribute (deprecated: read `variable`)."""
        return self.variable


def __getattr__(name: str) -> type[MissingSecretError]:
    # `MissingEnvError` was the error before keys were resolved with `secrets`: kept one
    # release as the same class, so `except MissingEnvError` still catches it.
    if name == "MissingEnvError":
        warnings.warn(
            "wica.config.MissingEnvError is deprecated: use MissingSecretError",
            DeprecationWarning,
            stacklevel=2,
        )
        return MissingSecretError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    creation — see ``resolve_api_key`` / ``resolve_system_prompt`` and specs/config.md. There is no
    ``api_key``: a config never holds a key (``Wica.init(..., secrets=...)`` takes keys held in
    code).

    Frozen: derive a variant with ``dataclasses.replace(...)`` rather than assigning. The structural
    invariants are enforced in ``__post_init__``, so direct construction fails the same way the
    loaders do.
    """

    provider: str
    model: str
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    # The variable holding the key; None for the provider's default (specs/config.md, "API key").
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
        # The validation the loaders run, over the declared fields, applied to direct construction
        # too, so every AgentConfig instance is valid. See specs/config.md, specs/setup.md.
        renamed = PROVIDER_RENAMES.get(self.provider)
        if renamed is not None:
            _logger.warning(
                "agent: provider %r is deprecated, use %r", self.provider, renamed
            )
            object.__setattr__(self, "provider", renamed)
        # A None where the dataclass's own default is None stands for an absent key; any other
        # None was passed and is validated (`max_retries=None` is refused, as in JSON).
        values = {
            f.name: getattr(self, f.name)
            for f in dataclass_fields(self)
            if getattr(self, f.name) is not None or f.default is not None
        }
        problems = agent_problems(values)
        if problems:
            raise ConfigError(problems[0][1])

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


# --- The fields, declared once ------------------------------------------------------
#
# The `agent` block's fields as data: the loaders, `AgentConfig.__post_init__` and
# `wica.setup.check_config` validate against them, and `wica.setup.config_schema` draws the form
# from them, so the two cannot drift apart. See specs/setup.md ("Fields are declared once").

_LABEL = "agent:"

FAKE_PROVIDER = "fake"

ProviderKind = Literal["hosted", "local", "fixture"]

API_KEY_REFUSED = (
    "agent: does not take 'api_key': keep the key in an environment variable and name it "
    "in 'api_key_env'"
)

PROVIDER_FIELD = Field(
    "provider",
    "string",
    "Which provider runs the model.",
    required=True,
)

COMMON_FIELDS: tuple[Field, ...] = (
    Field(
        "model",
        "string",
        "The model, as the provider names it (for the Hugging Face Hub, its repo id; for a "
        "compatible server, what it expects in a request's 'model').",
        required=True,
    ),
    Field(
        "system_prompt",
        "string",
        "The persona, written here. Give it here or in 'system_prompt_file', not both.",
        allow_empty=True,
    ),
    Field(
        "system_prompt_file",
        "string",
        "The file holding the persona; a relative path is relative to the config's folder. "
        "Give it here or in 'system_prompt', not both.",
    ),
    Field(
        "history_reactions",
        "integer",
        "How many past reactions the prompt always keeps: history grows to twice this, then "
        "is cut back to it. Absent, history is unbounded.",
        minimum=1,
        advanced=True,
    ),
    Field(
        "history_budget",
        "number",
        "The fraction of the model's context window the prompt should stay under: past it, "
        "the oldest reactions are cut. Absent, off.",
        exclusive_minimum=0,
        maximum=1,
        advanced=True,
    ),
    Field(
        "context_window",
        "integer",
        "The model's context window in tokens, for 'history_budget'. Absent, the model's "
        "profile gives it.",
        minimum=1,
        advanced=True,
    ),
    Field(
        "model_timeout",
        "number",
        "Seconds one model call may take before the reaction is abandoned; null for no "
        "deadline.",
        implied=DEFAULT_MODEL_TIMEOUT,
        exclusive_minimum=0,
        advanced=True,
    ),
    Field(
        "max_next_steps",
        "integer",
        "How many steps in a row the model's <next> note may wake: 0 turns it off, null "
        "removes the cap.",
        implied=DEFAULT_MAX_NEXT_STEPS,
        minimum=0,
        advanced=True,
    ),
    Field(
        "max_retries",
        "integer",
        "How many times one reaction may call the model again after its response guards "
        "rejected a response.",
        default=DEFAULT_MAX_RETRIES,
        minimum=0,
        advanced=True,
    ),
    Field(
        "stream_output",
        "boolean",
        "Deliver an output while the response streams (true), or once the response guards "
        "accepted it (false). Absent, the output mode's default.",
        advanced=True,
    ),
)

# `model_kwargs` is a free-form object, which `Field` has no kind for: checked by hand
# (`agent_problems`) and drawn as raw JSON (`wica.setup`).
MODEL_KWARGS_DESCRIPTION = (
    "Extra parameters passed to the provider's model (temperature, an output cap, a "
    "reasoning level), as a JSON object."
)


def key_env_field(title: str, variable: str | None) -> Field:
    """The `api_key_env` field of a provider: the environment variable that holds *title*'s
    key, *variable* by default. With no default (a server that may need no key), the field may
    stay absent. See specs/config.md ("API key")."""
    message = f"{_LABEL} 'api_key_env' must be a non-empty string"
    if variable is None:
        return Field(
            "api_key_env",
            "string",
            f"The environment variable holding the {title} key. Absent, no key is sent.",
            secret="env",
            message=message,
        )
    return Field(
        "api_key_env",
        "string",
        f"The environment variable holding the {title} key.",
        default=variable,
        secret="env",
        message=message,
    )


_HF_PROVIDER = Field(
    "hf_provider",
    "string",
    "The Hub's inference provider that serves the model ('auto' lets the Hub choose).",
    default="auto",
)
_BASE_URL_MESSAGE = (
    f"{_LABEL} provider {OPENAI_CHAT_COMPLETIONS_COMPATIBLE!r} needs 'base_url', the server's "
    "address with its /v1 path (e.g. 'http://127.0.0.1:8080/v1')"
)
_BASE_URL_DESCRIPTION = (
    "The server's address with its /v1 path (e.g. http://127.0.0.1:8080/v1)."
)


@dataclass(frozen=True)
class Provider:
    """A provider WICA knows: what a setup page and the loader need before it is installed.
    See specs/setup.md ("Fields are declared once")."""

    name: str
    title: str
    kind: ProviderKind
    # The extra carrying its package (None: built in), and the modules that extra provides.
    extra: str | None
    packages: tuple[str, ...]
    # Its own fields, beside the common ones: `api_key_env` (with the provider's usual
    # variable as default), `base_url`, `hf_provider`.
    fields: tuple[Field, ...]

    @property
    def key_env(self) -> str | None:
        """The variable its key is read from when the config names none."""
        for own in self.fields:
            if own.secret == "env":
                return own.default
        return None

    @property
    def takes_key(self) -> bool:
        return any(own.secret == "env" for own in self.fields)


def _hosted(name: str, title: str, extra: str, package: str, variable: str) -> Provider:
    return Provider(
        name, title, "hosted", extra, (package,), (key_env_field(title, variable),)
    )


# The providers WICA knows, in the order a form offers them. Any other LangChain id is still
# accepted, with no extra and no default key variable. See specs/config.md ("Providers").
PROVIDERS: dict[str, Provider] = {
    provider.name: provider
    for provider in (
        _hosted(
            "anthropic",
            "Anthropic",
            "anthropic",
            "langchain_anthropic",
            "ANTHROPIC_API_KEY",
        ),
        _hosted("openai", "OpenAI", "openai", "langchain_openai", "OPENAI_API_KEY"),
        _hosted(
            "gemini", "Gemini", "gemini", "langchain_google_genai", "GEMINI_API_KEY"
        ),
        _hosted(
            "mistral", "Mistral", "mistral", "langchain_mistralai", "MISTRAL_API_KEY"
        ),
        Provider(
            "huggingface-hub",
            "Hugging Face Hub",
            "hosted",
            "huggingface-hub",
            ("langchain_huggingface",),
            (key_env_field("Hugging Face", "HF_TOKEN"), _HF_PROVIDER),
        ),
        Provider(
            OPENAI_CHAT_COMPLETIONS_COMPATIBLE,
            "OpenAI-compatible server",
            "local",
            "openai-compatible",
            ("langchain_openai",),
            (
                Field(
                    "base_url",
                    "string",
                    _BASE_URL_DESCRIPTION,
                    required=True,
                    message=_BASE_URL_MESSAGE,
                ),
                key_env_field("server's", None),
            ),
        ),
        Provider(FAKE_PROVIDER, "Scripted fake", "fixture", None, (), ()),
    )
}

# The fields a provider that does not declare them still accepts (and type-checks), as before
# they were declared per provider: they are just not drawn in its branch of the form.
_LOOSE_FIELDS: tuple[Field, ...] = (
    key_env_field("provider's", None),
    _HF_PROVIDER,
    Field("base_url", "string", _BASE_URL_DESCRIPTION),
)

_AGENT_KEYS = frozenset(
    {PROVIDER_FIELD.name, "model_kwargs"}
    | {f.name for f in COMMON_FIELDS}
    | {f.name for f in _LOOSE_FIELDS}
)


def provider_for(name: str) -> Provider | None:
    """The provider *name* selects (an old name resolves quietly), or None for one WICA does
    not know."""
    return PROVIDERS.get(PROVIDER_RENAMES.get(name, name))


def default_key_env(name: str) -> str | None:
    """The variable a provider's key is read from when `api_key_env` names none."""
    provider = provider_for(name)
    return provider.key_env if provider is not None else None


def install_hint(provider: Provider) -> str:
    """What to install for a provider whose package is missing."""
    requirement = f"wica[{provider.extra}]"
    return (
        f"provider {provider.name!r} needs {requirement}, which is not installed: "
        f"install it (pip install '{requirement}', or uv add '{requirement}')"
    )


def import_provider(name: str) -> None:
    """Import the packages the provider's extra provides, raising `ImportError` with the install
    hint when one is missing. `build_chat_model` calls it before its own imports and
    `wica.setup.check_setup` tries it, so the hint has one home. Nothing for `fake` or a
    provider WICA does not know."""
    provider = provider_for(name)
    if provider is None:
        return
    for package in provider.packages:
        try:
            importlib.import_module(package)
        except ImportError as exc:
            raise ImportError(install_hint(provider)) from exc


def agent_fields(provider: str) -> tuple[Field, ...]:
    """The fields an `agent` block selecting *provider* is validated against: the common ones,
    the provider's own, and the loose ones it does not declare."""
    selected = provider_for(provider)
    own = selected.fields if selected is not None else ()
    declared = {f.name for f in own}
    return (
        COMMON_FIELDS + own + tuple(f for f in _LOOSE_FIELDS if f.name not in declared)
    )


def agent_problems(data: Any) -> list[tuple[str, str]]:
    """Every problem that makes an `agent` block invalid, as `(field name, message)` — the name
    is "" for the block as a whole. The loaders and `AgentConfig` raise the first; `wica.setup`
    reports them all. Reads nothing but the block. See specs/setup.md ("Fields are declared
    once")."""
    if not isinstance(data, dict):
        return [("", f"{_LABEL} must be a JSON object")]
    problems: list[tuple[str, str]] = []
    if "api_key" in data:
        problems.append(("api_key", API_KEY_REFUSED))
    for key in sorted(set(data) - _AGENT_KEYS - {"api_key"}, key=str):
        problems.append((str(key), f"{_LABEL} unknown key {key!r}"))
    provider = data.get("provider")
    fields = (PROVIDER_FIELD,) + agent_fields(
        provider if isinstance(provider, str) else ""
    )
    present = []
    for declared in fields:
        if declared.required and declared.name not in data:
            problems.append(
                (
                    declared.name,
                    declared.message
                    or f"{_LABEL} missing required key {declared.name!r}",
                )
            )
        else:
            present.append(declared)
    problems += field_problems(data, present, label=_LABEL)
    if "model_kwargs" in data and not isinstance(data["model_kwargs"], dict):
        problems.append(("model_kwargs", f"{_LABEL} 'model_kwargs' must be an object"))
    if problems:
        return problems
    return _rule_problems(data)


def _rule_problems(data: dict[str, Any]) -> list[tuple[str, str]]:
    """The rules across fields, over a block whose fields are valid."""
    problems: list[tuple[str, str]] = []
    has_inline = data.get("system_prompt") is not None
    has_file = data.get("system_prompt_file") is not None
    if has_inline and has_file:
        problems.append(
            (
                "",
                f"{_LABEL} specify at most one of 'system_prompt'/'system_prompt_file', "
                "not both",
            )
        )
    elif not has_inline and not has_file:
        problems.append(
            ("", f"{_LABEL} specify one of 'system_prompt'/'system_prompt_file'")
        )
    # `base_url` is for the compatible server only (silently ignored, it would send the requests
    # to the vendor instead of the intended server); there, the address has one home. See
    # specs/config.md ("Providers").
    provider = PROVIDER_RENAMES.get(data["provider"], data["provider"])
    if provider != OPENAI_CHAT_COMPLETIONS_COMPATIBLE:
        if data.get("base_url") is not None:
            problems.append(
                (
                    "base_url",
                    f"{_LABEL} 'base_url' is only for provider "
                    f"{OPENAI_CHAT_COMPLETIONS_COMPATIBLE!r}, not {provider!r}",
                )
            )
    elif "base_url" in data.get("model_kwargs", {}):
        problems.append(
            (
                "model_kwargs",
                f"{_LABEL} set the server's address in 'base_url', not in 'model_kwargs'",
            )
        )
    return problems


def _as_base_dir(base_dir: str | Path | None) -> Path | None:
    """Normalize the optional `from_dict`/`from_json` base directory (a `str` or `Path`) into the
    `Path | None` the `_parse_*` locate plumbing expects — the same base-dir context `from_json_file`
    derives from the config file's own directory. See specs/config.md ("System prompt")."""
    return Path(base_dir) if base_dir is not None else None


def _parse_agent_block(
    data: dict[str, Any], *, base_dir: Path | None
) -> dict[str, Any]:
    """Validate an `agent` block (raising its first problem) and read its values, a default
    where one is absent. For a relative `system_prompt_file`, a caller-supplied base_dir
    (from_json_file derives it from the config directory) absolutizes it against that directory,
    so the path is bound to the config's location, not the process CWD — a locate, not a file
    read; the read is deferred to resolve_system_prompt at build. See specs/config.md."""
    problems = agent_problems(data)
    if problems:
        raise ConfigError(problems[0][1])
    values = read_fields(
        data, agent_fields(data["provider"]), label=_LABEL, error=ConfigError
    )
    prompt_file = values["system_prompt_file"]
    if (
        prompt_file is not None
        and base_dir is not None
        and not Path(prompt_file).is_absolute()
    ):
        # Locate against the config directory (absolutize) — no file *read*; .resolve() only
        # normalizes `..`/symlinks (a stat-level lookup, tolerant of a missing file).
        prompt_file = str((base_dir / prompt_file).resolve())
    return {
        "provider": data["provider"],
        "model": values["model"],
        "system_prompt": values["system_prompt"],
        "system_prompt_file": prompt_file,
        # Kept as written: None leaves the variable to the provider's default at build.
        "api_key_env": data.get("api_key_env"),
        "model_kwargs": data.get("model_kwargs", {}),
        "hf_provider": values["hf_provider"],
        "base_url": values["base_url"],
        "history_reactions": values["history_reactions"],
        "history_budget": values["history_budget"],
        "context_window": values["context_window"],
        # Absent takes the default; an explicit null removes the deadline (or the cap).
        "model_timeout": values["model_timeout"]
        if "model_timeout" in data
        else DEFAULT_MODEL_TIMEOUT,
        "max_next_steps": values["max_next_steps"]
        if "max_next_steps" in data
        else DEFAULT_MAX_NEXT_STEPS,
        "max_retries": values["max_retries"],
        "stream_output": values["stream_output"],
    }


def wica_problems(data: Any) -> list[tuple[str, str]]:
    """Every problem of a WICA block (`{"agent": {...}}`), as `(JSON Pointer, message)`: its own
    shape, then the `agent` block's problems under `/agent`. The loaders raise the first;
    `wica.setup.check_config` reports them all. See specs/setup.md."""
    if not isinstance(data, dict):
        return [("", "config: must be a JSON object")]
    problems = [
        (f"/{key}", f"config: unknown key {key!r}")
        for key in sorted(set(data) - {"agent"}, key=str)
    ]
    if "agent" not in data:
        return [*problems, ("/agent", "config: missing required key 'agent'")]
    if not isinstance(data["agent"], dict):
        return [*problems, ("/agent", "config: 'agent' must be an object")]
    return problems + [
        (f"/agent/{name}" if name else "/agent", message)
        for name, message in agent_problems(data["agent"])
    ]


def _parse_wica_block(data: dict[str, Any], *, base_dir: Path | None) -> dict[str, Any]:
    problems = wica_problems(data)
    if problems:
        raise ConfigError(problems[0][1])
    return {
        "agent": AgentConfig(**_parse_agent_block(data["agent"], base_dir=base_dir))
    }


def resolve_api_key(
    config: AgentConfig, secrets: Mapping[str, str] | None = None
) -> str | None:
    """Resolve the key at Agent build: the variable `api_key_env` names, else the provider's
    default; its key from *secrets* (variable name → key, given in code) first, then the
    environment. No key under it raises MissingSecretError. None when there is no variable: the
    `fake` provider, a compatible server or a provider WICA does not know naming none. See
    specs/config.md ("API key")."""
    if config.provider == FAKE_PROVIDER:
        return None
    variable = config.api_key_env or default_key_env(config.provider)
    if variable is None:
        return None
    key = (secrets or {}).get(variable) or os.environ.get(variable)
    if not key:
        raise MissingSecretError(variable)
    return key


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
