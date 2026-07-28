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
    provider: str
    model: str
    system_prompt: str
    api_key: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

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
            raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in config file {config_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config file {config_path} must contain a JSON object")
        return cls(**_parse_wica_block(data, base_dir=config_path.parent))


_AGENT_REQUIRED = {"provider", "model"}
_AGENT_OPTIONAL_COMMON = {"api_key", "api_key_env", "model_kwargs"}
_AGENT_PROMPT_KEYS = {"system_prompt", "system_prompt_file"}
_WICA_ALLOWED = {"agent", "logging"}


def _require_str(data: dict[str, Any], key: str, *, block: str) -> str:
    if key not in data:
        raise ConfigError(f"{block}: missing required key {key!r}")
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{block}: {key!r} must be a string, got {type(value).__name__}")
    return value


def _resolve_api_key(data: dict[str, Any], *, block: str) -> str | None:
    has_literal = "api_key" in data
    has_env_ref = "api_key_env" in data
    if has_literal and has_env_ref:
        raise ConfigError(f"{block}: specify at most one of 'api_key'/'api_key_env', not both")
    if has_literal:
        return _require_str(data, "api_key", block=block)
    if has_env_ref:
        env_var = _require_str(data, "api_key_env", block=block)
        value = os.environ.get(env_var)
        if not value:
            raise MissingEnvError(env_var)
        return value
    return None


def _resolve_system_prompt(data: dict[str, Any], *, base_dir: Path | None, block: str) -> str:
    has_inline = "system_prompt" in data
    has_file = "system_prompt_file" in data
    if has_inline and has_file:
        raise ConfigError(
            f"{block}: specify at most one of 'system_prompt'/'system_prompt_file', not both"
        )
    if has_inline:
        return _require_str(data, "system_prompt", block=block)
    if has_file:
        if base_dir is None:
            raise ConfigError(
                f"{block}: 'system_prompt_file' requires loading from a file (use "
                "WicaConfig.from_json), not from_dict — there's no base directory to resolve "
                "it against"
            )
        rel_path = _require_str(data, "system_prompt_file", block=block)
        prompt_path = base_dir / rel_path
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"{block}: could not read system_prompt_file {prompt_path}: {exc}"
            ) from exc
    raise ConfigError(f"{block}: specify one of 'system_prompt'/'system_prompt_file'")


def _parse_agent_block(data: dict[str, Any], *, base_dir: Path | None) -> dict[str, Any]:
    block = "agent"
    if not isinstance(data, dict):
        raise ConfigError(f"{block}: must be a JSON object")
    allowed = _AGENT_REQUIRED | _AGENT_OPTIONAL_COMMON | _AGENT_PROMPT_KEYS
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{block}: unknown key(s) {sorted(unknown)}")

    provider = _require_str(data, "provider", block=block)
    model = _require_str(data, "model", block=block)
    system_prompt = _resolve_system_prompt(data, base_dir=base_dir, block=block)
    api_key = _resolve_api_key(data, block=block)

    model_kwargs = data.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ConfigError(f"{block}: 'model_kwargs' must be an object")

    return {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "api_key": api_key,
        "model_kwargs": model_kwargs,
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


def apply_logging(level: str) -> None:
    """Set the level on the `wica` logger tree, e.g. from a loaded WicaConfig.logging."""
    logging.getLogger("wica").setLevel(level.upper())
