from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wica import agent as agent_module
from wica.config import AgentConfig, ConfigError, MissingEnvError, WicaConfig, apply_logging


def _agent_dict(**overrides: Any) -> dict[str, Any]:
    data = {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "system_prompt": "You are a test assistant.",
    }
    data.update(overrides)
    return data


def _wica_dict(**agent_overrides: Any) -> dict[str, Any]:
    return {"logging": "DEBUG", "agent": _agent_dict(**agent_overrides)}


# --- from_dict happy path -------------------------------------------------------------


def test_wica_config_from_dict_happy_path():
    cfg = WicaConfig.from_dict(_wica_dict(model_kwargs={"temperature": 0.5}))
    assert cfg.logging == "DEBUG"
    assert cfg.agent.provider == "anthropic"
    assert cfg.agent.model == "claude-sonnet-5"
    assert cfg.agent.system_prompt == "You are a test assistant."
    assert cfg.agent.api_key is None
    assert cfg.agent.model_kwargs == {"temperature": 0.5}


def test_wica_config_from_dict_defaults_logging():
    data = _wica_dict()
    del data["logging"]
    cfg = WicaConfig.from_dict(data)
    assert cfg.logging == "INFO"


# --- from_json: inline vs. file-referenced system prompt ------------------------------


def test_from_json_inline_system_prompt(tmp_path: Path):
    config_path = tmp_path / "agent.config.json"
    config_path.write_text(json.dumps(_wica_dict()))

    cfg = WicaConfig.from_json(config_path)
    assert cfg.agent.system_prompt == "You are a test assistant."


def test_from_json_system_prompt_file_resolves_relative_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "prompts").mkdir()
    prompt_path = tmp_path / "prompts" / "persona.md"
    prompt_path.write_text("You are Wica, a friendly robot.")

    config_path = tmp_path / "agent.config.json"
    data = _wica_dict()
    del data["agent"]["system_prompt"]
    data["agent"]["system_prompt_file"] = "prompts/persona.md"
    config_path.write_text(json.dumps(data))

    # Run from a different CWD to prove resolution is config-file-relative, not CWD-relative.
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    cfg = WicaConfig.from_json(config_path)
    assert cfg.agent.system_prompt == "You are Wica, a friendly robot."


def test_system_prompt_file_missing_on_disk_raises(tmp_path: Path):
    config_path = tmp_path / "agent.config.json"
    data = _wica_dict()
    del data["agent"]["system_prompt"]
    data["agent"]["system_prompt_file"] = "does-not-exist.md"
    config_path.write_text(json.dumps(data))

    with pytest.raises(ConfigError, match="does-not-exist.md"):
        WicaConfig.from_json(config_path)


def test_agent_config_from_dict_rejects_system_prompt_file():
    data = _agent_dict()
    del data["system_prompt"]
    data["system_prompt_file"] = "persona.md"
    with pytest.raises(ConfigError, match="system_prompt_file"):
        AgentConfig.from_dict(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["agent"].__setitem__("system_prompt_file", "x.md"),  # both
        lambda d: d["agent"].pop("system_prompt"),  # neither
    ],
)
def test_exactly_one_system_prompt_field_enforced(mutate: Any):
    data = _wica_dict()
    mutate(data)
    with pytest.raises(ConfigError):
        WicaConfig.from_dict(data)


# --- api-key resolution -----------------------------------------------------------------


def test_api_key_literal_passthrough():
    cfg = AgentConfig.from_dict(_agent_dict(api_key="sk-literal"))
    assert cfg.api_key == "sk-literal"


def test_api_key_env_resolves_when_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WICA_TEST_KEY", "sk-from-env")
    cfg = AgentConfig.from_dict(_agent_dict(api_key_env="WICA_TEST_KEY"))
    assert cfg.api_key == "sk-from-env"


def test_api_key_env_unset_raises_missing_env_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WICA_TEST_KEY_UNSET", raising=False)
    with pytest.raises(MissingEnvError) as exc_info:
        AgentConfig.from_dict(_agent_dict(api_key_env="WICA_TEST_KEY_UNSET"))
    assert exc_info.value.env_var == "WICA_TEST_KEY_UNSET"
    assert isinstance(exc_info.value, ConfigError)


def test_api_key_both_literal_and_env_rejected():
    with pytest.raises(ConfigError, match="api_key"):
        AgentConfig.from_dict(_agent_dict(api_key="sk-x", api_key_env="SOME_VAR"))


def test_api_key_neither_given_is_none():
    cfg = AgentConfig.from_dict(_agent_dict())
    assert cfg.api_key is None


# --- strict validation -------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["provider", "model"])
def test_missing_required_agent_key_raises(missing: str):
    data = _agent_dict()
    del data[missing]
    with pytest.raises(ConfigError, match=missing):
        AgentConfig.from_dict(data)


def test_unknown_agent_key_raises():
    with pytest.raises(ConfigError, match="modl"):
        AgentConfig.from_dict(_agent_dict(modl="typo"))


def test_unknown_top_level_key_raises():
    data = _wica_dict()
    data["bogus"] = True
    with pytest.raises(ConfigError, match="bogus"):
        WicaConfig.from_dict(data)


def test_model_kwargs_wrong_type_raises():
    with pytest.raises(ConfigError, match="model_kwargs"):
        AgentConfig.from_dict(_agent_dict(model_kwargs="not-a-dict"))


def test_logging_wrong_type_raises():
    data = _wica_dict()
    data["logging"] = 123
    with pytest.raises(ConfigError, match="logging"):
        WicaConfig.from_dict(data)


def test_missing_agent_block_raises():
    with pytest.raises(ConfigError, match="agent"):
        WicaConfig.from_dict({"logging": "INFO"})


# --- Agent.from_config api_key forwarding -------------------------------------------------


def test_from_config_forwards_api_key_when_set(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, *, model_provider: str, **kwargs: Any) -> object:
        captured["model"] = model
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(
        provider="anthropic", model="claude-sonnet-5", system_prompt="hi", api_key="sk-abc"
    )
    agent_module.Agent.from_config(config)

    assert captured["kwargs"]["api_key"] == "sk-abc"


def test_from_config_omits_api_key_when_unset(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, *, model_provider: str, **kwargs: Any) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(provider="anthropic", model="claude-sonnet-5", system_prompt="hi")
    agent_module.Agent.from_config(config)

    assert "api_key" not in captured["kwargs"]


def test_from_json_then_apply_logging_then_from_config_composes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The blessed pattern (no from_config_file convenience): callers load the file, apply
    logging, then build the Agent themselves — see specs/config.md "Flow into the Agent"."""
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, *, model_provider: str, **kwargs: Any) -> object:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config_path = tmp_path / "agent.config.json"
    config_path.write_text(json.dumps(_wica_dict(api_key="sk-abc")))

    wica_config = WicaConfig.from_json(config_path)
    apply_logging(wica_config.logging)
    agent_module.Agent.from_config(wica_config.agent)

    import logging

    assert logging.getLogger("wica").level == logging.DEBUG
    logging.getLogger("wica").setLevel(logging.WARNING)  # don't leak into other tests

    assert captured["model"] == "claude-sonnet-5"
    assert captured["kwargs"]["api_key"] == "sk-abc"


# --- hf_provider field --------------------------------------------------------------------


def test_hf_provider_defaults_to_auto():
    cfg = AgentConfig.from_dict(_agent_dict())
    assert cfg.hf_provider == "auto"


def test_hf_provider_parsed_when_present():
    cfg = AgentConfig.from_dict(
        _agent_dict(provider="huggingface-hub", model="meta-llama/x", hf_provider="together")
    )
    assert cfg.hf_provider == "together"


def test_hf_provider_wrong_type_raises():
    with pytest.raises(ConfigError, match="hf_provider"):
        AgentConfig.from_dict(_agent_dict(hf_provider=123))


# --- build_chat_model: provider construction branch ---------------------------------------


def test_build_chat_model_openai_routes_through_init_chat_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, *, model_provider: str, **kwargs: Any) -> object:
        captured["model"] = model
        captured["provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(provider="openai", model="gpt-4o", system_prompt="hi", api_key="sk-x")
    agent_module.build_chat_model(config)

    assert captured["model"] == "gpt-4o"
    assert captured["provider"] == "openai"
    assert captured["kwargs"]["api_key"] == "sk-x"


def test_build_chat_model_huggingface_hub_branch(monkeypatch: pytest.MonkeyPatch):
    """huggingface-hub is built directly as ChatHuggingFace(llm=HuggingFaceEndpoint(...)), NOT via
    init_chat_model, and forwards the resolved api_key as huggingfacehub_api_token (its own kwarg
    name) — see specs/agent.md "Provider-agnostic model, from config"."""
    import langchain_huggingface

    captured: dict[str, Any] = {}

    class FakeEndpoint:
        def __init__(self, **kwargs: Any) -> None:
            captured["endpoint"] = kwargs

    class FakeChat:
        def __init__(self, *, llm: Any) -> None:
            captured["llm"] = llm

    # build_chat_model does `from langchain_huggingface import ...` inside its branch, which reads
    # these attributes off the module at call time, so patching them here takes effect.
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEndpoint", FakeEndpoint)
    monkeypatch.setattr(langchain_huggingface, "ChatHuggingFace", FakeChat)
    # Guard against the else-branch: if the provider check regressed, init_chat_model must not run.
    monkeypatch.setattr(
        agent_module,
        "init_chat_model",
        lambda *a, **k: pytest.fail("huggingface-hub must not go through init_chat_model"),
    )

    config = AgentConfig(
        provider="huggingface-hub",
        model="meta-llama/Llama-3.3-70B-Instruct",
        system_prompt="hi",
        api_key="hf-token",
        hf_provider="fireworks-ai",
        model_kwargs={"temperature": 0.3},
    )
    model = agent_module.build_chat_model(config)

    endpoint_kwargs = captured["endpoint"]
    assert endpoint_kwargs["repo_id"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert endpoint_kwargs["provider"] == "fireworks-ai"
    assert endpoint_kwargs["huggingfacehub_api_token"] == "hf-token"
    assert endpoint_kwargs["task"] == "text-generation"
    assert endpoint_kwargs["temperature"] == 0.3
    # HF gets the token kwarg, never the generic api_key the init_chat_model providers receive.
    assert "api_key" not in endpoint_kwargs
    assert isinstance(model, FakeChat)
    assert isinstance(captured["llm"], FakeEndpoint)


def test_build_chat_model_huggingface_hub_omits_token_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    import langchain_huggingface

    captured: dict[str, Any] = {}

    class FakeEndpoint:
        def __init__(self, **kwargs: Any) -> None:
            captured["endpoint"] = kwargs

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEndpoint", FakeEndpoint)
    monkeypatch.setattr(langchain_huggingface, "ChatHuggingFace", lambda *, llm: object())

    config = AgentConfig(
        provider="huggingface-hub", model="meta-llama/x", system_prompt="hi"
    )
    agent_module.build_chat_model(config)

    # No key given: don't pass the token kwarg at all, so HuggingFaceEndpoint reads the standard
    # env var (mirrors how the init_chat_model providers behave when api_key is unset).
    assert "huggingfacehub_api_token" not in captured["endpoint"]
    assert captured["endpoint"]["provider"] == "auto"
