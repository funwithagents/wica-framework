"""Setting the agent up from an application, as data (specs/setup.md).

Seven functions over the WICA block (the document `WicaConfig.from_dict` takes,
`{"agent": {...}}`): what its config looks like (`config_schema`), which extra a config needs
(`required_extras`), what is wrong in the config itself (`check_config`), what this machine is
missing to run it (`check_setup`: the provider's extra, its key, the prompt file), nothing to
prepare (`prepare`), no option lists (`options`), and one authenticated, free call to the
provider to find out whether the key and the model are good (`probe`). They implement the
module contract of the `module-setup` project and exchange JSON-compatible data only.

All but `probe` run on the base install, with no network. Nothing here is imported by the
framework itself.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from wica.config import (
    COMMON_FIELDS,
    DEFAULT_MODEL_TIMEOUT,
    FAKE_PROVIDER,
    MODEL_KWARGS_DESCRIPTION,
    OPENAI_CHAT_COMPLETIONS_COMPATIBLE,
    PROVIDERS,
    AgentConfig,
    Provider,
    import_provider,
    provider_for,
    resolve_api_key,
    wica_problems,
)
from wica.fields import fields_schema

__all__ = [
    "check_config",
    "check_setup",
    "config_schema",
    "options",
    "prepare",
    "probe",
    "required_extras",
]

_DISTRIBUTION = "wica"

# What a compatible server is sent when no key variable is named (as the Agent does).
_PLACEHOLDER_API_KEY = "not-needed"


def _branch(provider: Provider) -> dict[str, Any]:
    schema = fields_schema(provider.fields)
    return {
        "title": f"{provider.title} ({provider.kind})",
        "required": ["provider", *schema["required"]],
        "properties": {"provider": {"const": provider.name}, **schema["properties"]},
    }


def config_schema() -> dict[str, Any]:
    common = fields_schema(COMMON_FIELDS)
    return {
        "contract": 1,
        "distribution": _DISTRIBUTION,
        "title": "Agent",
        "schema": {
            "type": "object",
            "required": ["agent"],
            "properties": {
                "agent": {
                    "type": "object",
                    "title": "Provider",
                    "description": "Which provider runs the agent's model: a provider's API "
                    "(hosted), a server of your own speaking OpenAI's Chat Completions API "
                    "(local), or the scripted fake, which needs no key and no network "
                    "(fixture).",
                    "required": common["required"],
                    "properties": {
                        **common["properties"],
                        "model_kwargs": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": MODEL_KWARGS_DESCRIPTION,
                            "x-advanced": True,
                        },
                    },
                    "oneOf": [_branch(provider) for provider in PROVIDERS.values()],
                }
            },
        },
    }


def _selected(config: Any) -> tuple[Provider | None, dict[str, Any]]:
    """The provider the WICA block selects (None for one WICA does not know, or none) and its
    `agent` block."""
    agent = config.get("agent") if isinstance(config, dict) else None
    if not isinstance(agent, dict):
        return None, {}
    name = agent.get("provider")
    return (provider_for(name) if isinstance(name, str) else None), agent


def required_extras(config: dict[str, Any]) -> list[str]:
    provider, _ = _selected(config)
    if provider is None or provider.extra is None:
        return []
    return [f"{_DISTRIBUTION}[{provider.extra}]"]


def check_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": path, "message": message} for path, message in wica_problems(config)
    ]


def _key_variable(provider: Provider | None, agent: dict[str, Any]) -> str | None:
    """The variable the config reads its key from, or None when it reads none."""
    if provider is not None and provider.name == FAKE_PROVIDER:
        return None
    named = agent.get("api_key_env")
    if isinstance(named, str) and named:
        return named
    return provider.key_env if provider is not None else None


def check_setup(config: dict[str, Any]) -> list[dict[str, Any]]:
    provider, agent = _selected(config)
    found: list[dict[str, Any]] = []
    if provider is not None and provider.extra is not None:
        try:
            import_provider(provider.name)
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            status, detail = "missing", str(exc) or type(exc).__name__
        else:
            status, detail = "ready", "Installed."
        found.append(
            {
                "id": f"extra:{_DISTRIBUTION}[{provider.extra}]",
                "title": f"{provider.title} extra",
                "status": status,
                "detail": detail,
                "preparable": False,
            }
        )
    variable = _key_variable(provider, agent)
    if variable is not None:
        present = bool(os.environ.get(variable))
        found.append(
            {
                "id": f"key:{variable}",
                "title": variable,
                "status": "ready" if present else "missing",
                "detail": "Set."
                if present
                else f"Set the environment variable {variable} (named by 'api_key_env').",
                "preparable": False,
            }
        )
    prompt_file = agent.get("system_prompt_file")
    if isinstance(prompt_file, str) and prompt_file:
        # The contract passes no directory: a relative path is looked up against the working
        # directory, as `from_dict` without `base_dir` does (specs/setup.md, Open questions).
        path = Path(prompt_file).expanduser().resolve()
        exists = path.is_file()
        found.append(
            {
                "id": "file:system_prompt_file",
                "title": "System prompt file",
                "status": "ready" if exists else "missing",
                "detail": f"Found at {path}." if exists else f"No file at {path}.",
                "preparable": False,
            }
        )
    return found


def prepare(
    config: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Nothing to prepare: WICA downloads nothing."""


def options(name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """No option list is offered."""
    return []


# --- probe: one authenticated, free call per provider --------------------------------


class _ProbeFailure(Exception):
    """What a probe found wrong, in words for the user."""


def _refusal(exc: Exception, provider: str, model: str) -> str:
    """An SDK's error, in words: the key refused, the model unknown, or its own message."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if status in (401, 403):
        return f"{provider} refused the key ({status})."
    if status == 404:
        return f"{provider} has no model {model!r}."
    return f"{provider}: {exc}"


def _probe_anthropic(config: AgentConfig, key: str | None, timeout: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=0)
    try:
        client.models.retrieve(config.model)
    except Exception as exc:  # noqa: BLE001 - reported in words
        raise _ProbeFailure(_refusal(exc, "Anthropic", config.model)) from exc
    return f"Anthropic accepted the key and serves {config.model!r}."


def _probe_openai(config: AgentConfig, key: str | None, timeout: float) -> str:
    import openai

    client = openai.OpenAI(api_key=key, timeout=timeout, max_retries=0)
    try:
        client.models.retrieve(config.model)
    except Exception as exc:  # noqa: BLE001 - reported in words
        raise _ProbeFailure(_refusal(exc, "OpenAI", config.model)) from exc
    return f"OpenAI accepted the key and serves {config.model!r}."


def _probe_gemini(config: AgentConfig, key: str | None, timeout: float) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
    )
    try:
        client.models.get(model=config.model)
    except Exception as exc:  # noqa: BLE001 - reported in words
        raise _ProbeFailure(_refusal(exc, "Gemini", config.model)) from exc
    return f"Gemini accepted the key and serves {config.model!r}."


_MISTRAL_MODELS_URL = "https://api.mistral.ai/v1/models"


def _probe_mistral(config: AgentConfig, key: str | None, timeout: float) -> str:
    import httpx

    try:
        response = httpx.get(
            _MISTRAL_MODELS_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - reported in words
        raise _ProbeFailure(_refusal(exc, "Mistral", config.model)) from exc
    listed: set[str] = set()
    for model in response.json().get("data", []):
        listed.add(model.get("id", ""))
        listed.update(model.get("aliases") or [])
    if config.model not in listed:
        raise _ProbeFailure(
            f"Mistral accepted the key but does not list {config.model!r}."
        )
    return f"Mistral accepted the key and serves {config.model!r}."


def _served_by(info: Any) -> list[str]:
    """The inference providers a Hub model's info says serve it."""
    mapping = getattr(info, "inference_provider_mapping", None) or []
    if isinstance(mapping, dict):
        return list(mapping)
    return [str(getattr(entry, "provider", entry)) for entry in mapping]


def _probe_huggingface(config: AgentConfig, key: str | None, timeout: float) -> str:
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(
            config.model,
            token=key,
            timeout=timeout,
            expand=["inferenceProviderMapping"],
        )
    except Exception as exc:  # noqa: BLE001 - reported in words
        raise _ProbeFailure(
            _refusal(exc, "The Hugging Face Hub", config.model)
        ) from exc
    served = _served_by(info)
    if not served:
        raise _ProbeFailure(
            f"No inference provider serves {config.model!r} on the Hugging Face Hub."
        )
    if config.hf_provider != "auto" and config.hf_provider not in served:
        raise _ProbeFailure(
            f"{config.hf_provider!r} does not serve {config.model!r} "
            f"(served by: {', '.join(served)})."
        )
    return f"The Hugging Face Hub accepted the token; {config.model!r} is served."


def _probe_compatible(config: AgentConfig, key: str | None, timeout: float) -> str:
    import openai

    client = openai.OpenAI(
        base_url=config.base_url,
        api_key=key if key is not None else _PLACEHOLDER_API_KEY,
        timeout=timeout,
        max_retries=0,
    )
    try:
        listed = [model.id for model in client.models.list()]
    except Exception as exc:  # noqa: BLE001 - reported in words
        raise _ProbeFailure(_refusal(exc, str(config.base_url), config.model)) from exc
    if config.model in listed or not listed:
        return f"The server at {config.base_url} answered."
    # Several servers serve whatever model they loaded, under any name: said, not failed.
    return (
        f"The server at {config.base_url} answered, but does not list {config.model!r} "
        f"(it lists {', '.join(listed)})."
    )


def _probe_unknown(config: AgentConfig, secrets: Mapping[str, str] | None) -> str:
    from wica.agent import build_chat_model

    build_chat_model(config, secrets=secrets)
    return (
        f"Built the {config.provider!r} chat model; nothing was sent: WICA has no free "
        "call to test this provider with."
    )


_PROBES: dict[str, Callable[[AgentConfig, str | None, float], str]] = {
    "anthropic": _probe_anthropic,
    "openai": _probe_openai,
    "gemini": _probe_gemini,
    "mistral": _probe_mistral,
    "huggingface-hub": _probe_huggingface,
    OPENAI_CHAT_COMPLETIONS_COMPATIBLE: _probe_compatible,
}


def _run_probe(config: dict[str, Any], secrets: Mapping[str, str] | None) -> str:
    problems = wica_problems(config)
    if problems:
        raise _ProbeFailure(problems[0][1])
    agent = AgentConfig.from_dict(config["agent"])
    if agent.provider == FAKE_PROVIDER:
        return "The scripted fake needs nothing to run."
    probe_provider = _PROBES.get(agent.provider)
    if probe_provider is None:
        return _probe_unknown(agent, secrets)
    import_provider(agent.provider)
    key = resolve_api_key(agent, secrets)
    timeout = agent.model_timeout or DEFAULT_MODEL_TIMEOUT
    return probe_provider(agent, key, timeout)


def probe(
    config: dict[str, Any], *, secrets: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """One authenticated, free call to the provider: is the key accepted, is the model there?
    Never a chat request. Never raises. `secrets` are keys held in code, by variable name, looked
    up before the environment (as `Wica.init` does)."""
    started = time.monotonic()
    try:
        message, ok = _run_probe(config, secrets), True
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises
        message, ok = str(exc) or type(exc).__name__, False
    return {
        "ok": ok,
        "message": message,
        "duration_s": round(time.monotonic() - started, 3),
    }
