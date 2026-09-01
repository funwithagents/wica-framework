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


@dataclass
class AgentConfig:
    """Plain data mirroring the JSON `agent` block. The `*_env`/`*_file` fields hold references
    resolved at Agent build, not at load — see resolve_api_key / resolve_system_prompt and
    specs/config.md."""

    provider: str
    model: str
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_provider: str = "auto"  # only used by provider "huggingface-hub"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        return cls(**_parse_agent_block(data, base_dir=None))


@dataclass
class WicaConfig:
    agent: AgentConfig
    logging: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WicaConfig:
        return cls(**_parse_wica_block(data, base_dir=None))

    @classmethod
    def from_json(cls, path: str | Path) -> WicaConfig:
        config_path = Path(path)
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"could not read config file {config_path}: {exc}"
            ) from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid JSON in config file {config_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config file {config_path} must contain a JSON object")
        return cls(**_parse_wica_block(data, base_dir=config_path.parent))


_AGENT_REQUIRED = {"provider", "model"}
_AGENT_OPTIONAL_COMMON = {"api_key", "api_key_env", "model_kwargs", "hf_provider"}
_AGENT_PROMPT_KEYS = {"system_prompt", "system_prompt_file"}
_WICA_ALLOWED = {"agent", "logging"}


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
    `system_prompt_file`, from_json (base_dir set) absolutizes it against the config directory so
    the path is bound to the config's location, not the process CWD — a locate, not a file read; the
    read is deferred to resolve_system_prompt at build. from_dict (base_dir None) stores it as
    given. See specs/config.md ("System prompt")."""
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
        # Locate against the config directory (absolutize) — no I/O, and .resolve() normalizes
        # `..`/symlinks so the stored path is stable regardless of the process CWD at read time.
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

    return {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "system_prompt_file": system_prompt_file,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "model_kwargs": model_kwargs,
        "hf_provider": hf_provider,
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

    log_level = data.get("logging", "INFO")
    if not isinstance(log_level, str):
        raise ConfigError("config: 'logging' must be a string")

    return {"agent": agent, "logging": log_level}


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
    contents of `system_prompt_file` (already absolutized by from_json). A missing/unreadable file
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
    )  # unreachable given validation


def apply_logging(level: str) -> None:
    """Set the level on the `wica` logger tree, e.g. from a loaded WicaConfig.logging."""
    logging.getLogger("wica").setLevel(level.upper())
