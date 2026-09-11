from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from wica import agent as agent_module
from wica.config import (
    AgentConfig,
    ConfigError,
    MissingEnvError,
    WicaConfig,
    resolve_api_key,
    resolve_system_prompt,
)


def _agent_dict(**overrides: Any) -> dict[str, Any]:
    data = {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "system_prompt": "You are a test assistant.",
    }
    data.update(overrides)
    return data


def _wica_dict(**agent_overrides: Any) -> dict[str, Any]:
    return {"agent": _agent_dict(**agent_overrides)}


# --- from_dict happy path -------------------------------------------------------------


def test_wica_config_from_dict_happy_path():
    cfg = WicaConfig.from_dict(_wica_dict(model_kwargs={"temperature": 0.5}))
    assert cfg.agent.provider == "anthropic"
    assert cfg.agent.model == "claude-sonnet-5"
    assert cfg.agent.system_prompt == "You are a test assistant."
    # Unresolved indirection fields default to None on a plain inline+keyless config.
    assert cfg.agent.system_prompt_file is None
    assert cfg.agent.api_key is None
    assert cfg.agent.api_key_env is None
    assert cfg.agent.model_kwargs == {"temperature": 0.5}


# --- config mirrors the JSON: fields stored verbatim, not resolved at load ------------


def test_agent_config_from_dict_stores_system_prompt_file_verbatim():
    data = _agent_dict()
    del data["system_prompt"]
    data["system_prompt_file"] = "persona.md"
    cfg = AgentConfig.from_dict(data)
    # from_dict has no config directory to locate against, so it stores the path as given (no
    # rejection — the old from_dict-rejects-system_prompt_file carve-out is gone).
    assert cfg.system_prompt is None
    assert cfg.system_prompt_file == "persona.md"


def test_from_dict_locates_system_prompt_file_against_base_dir(tmp_path: Path):
    # A relative system_prompt_file + base_dir absolutizes against that dir — the same locate
    # from_json_file does against the config file's directory (no read).
    data = _agent_dict()
    del data["system_prompt"]
    data["system_prompt_file"] = "prompts/persona.md"

    cfg = AgentConfig.from_dict(data, base_dir=tmp_path)

    assert cfg.system_prompt is None
    assert cfg.system_prompt_file == str(
        (tmp_path / "prompts" / "persona.md").resolve()
    )


def test_from_dict_base_dir_matches_from_json_file_location(tmp_path: Path):
    # from_dict(..., base_dir=<config dir>) yields the same located path as from_json_file for a
    # config file in that dir — the app hands a section-dict + os.path.dirname(path) and gets parity.
    config_dir = tmp_path / "deploy"
    config_dir.mkdir()

    agent = _agent_dict()
    del agent["system_prompt"]
    agent["system_prompt_file"] = "prompts/wica.md"

    config_path = config_dir / "agent.config.json"
    config_path.write_text(json.dumps({"agent": agent}))

    from_json_file_cfg = WicaConfig.from_json_file(config_path)
    from_dict_cfg = WicaConfig.from_dict(
        {"agent": dict(agent)}, base_dir=str(config_dir)
    )

    assert (
        from_dict_cfg.agent.system_prompt_file
        == from_json_file_cfg.agent.system_prompt_file
        == str((config_dir / "prompts" / "wica.md").resolve())
    )


def test_from_dict_without_base_dir_stores_relative_verbatim():
    # Regression guard on the unchanged default: no base_dir keeps the relative path as given.
    data = _agent_dict()
    del data["system_prompt"]
    data["system_prompt_file"] = "prompts/persona.md"

    assert AgentConfig.from_dict(data).system_prompt_file == "prompts/persona.md"


def test_from_dict_base_dir_ignored_for_absolute_system_prompt_file(tmp_path: Path):
    # An absolute path is stored as-is; base_dir is ignored (matches from_json_file).
    absolute = str((tmp_path / "elsewhere" / "persona.md").resolve())
    data = _agent_dict()
    del data["system_prompt"]
    data["system_prompt_file"] = absolute

    cfg = AgentConfig.from_dict(data, base_dir=tmp_path / "deploy")

    assert cfg.system_prompt_file == absolute


def test_agent_config_from_dict_stores_api_key_env_verbatim():
    cfg = AgentConfig.from_dict(_agent_dict(api_key_env="WICA_SOME_KEY"))
    # The env var *name* is kept; nothing is read at load (the var need not even exist).
    assert cfg.api_key is None
    assert cfg.api_key_env == "WICA_SOME_KEY"


# --- from_json: inline vs. located (but unread) system_prompt_file ---------------------


def test_from_json_inline_system_prompt(tmp_path: Path):
    config_path = tmp_path / "agent.config.json"
    config_path.write_text(json.dumps(_wica_dict()))

    cfg = WicaConfig.from_json_file(config_path)
    assert cfg.agent.system_prompt == "You are a test assistant."
    assert cfg.agent.system_prompt_file is None


def test_from_json_locates_system_prompt_file_relative_to_config_but_defers_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "persona.md").write_text("You are Wica, a friendly robot.")

    config_path = tmp_path / "agent.config.json"
    data = _wica_dict()
    del data["agent"]["system_prompt"]
    data["agent"]["system_prompt_file"] = "prompts/persona.md"
    config_path.write_text(json.dumps(data))

    # Run from a different CWD to prove the located path is config-file-relative, not CWD-relative.
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    cfg = WicaConfig.from_json_file(config_path)
    # Located (absolutized against the config dir), inline slot stays empty, file not read yet.
    assert cfg.agent.system_prompt is None
    assert cfg.agent.system_prompt_file == str(
        (tmp_path / "prompts" / "persona.md").resolve()
    )
    # The read is deferred to build:
    assert resolve_system_prompt(cfg.agent) == "You are Wica, a friendly robot."


def test_from_json_does_not_read_system_prompt_file_at_load(tmp_path: Path):
    config_path = tmp_path / "agent.config.json"
    data = _wica_dict()
    del data["agent"]["system_prompt"]
    data["agent"]["system_prompt_file"] = "does-not-exist.md"
    config_path.write_text(json.dumps(data))

    # Loads fine — locating is not reading, so a missing file is not an error until build.
    cfg = WicaConfig.from_json_file(config_path)
    assert Path(cfg.agent.system_prompt_file or "").is_absolute()
    # ...and the error surfaces at resolution (build), naming the path.
    with pytest.raises(ConfigError, match="does-not-exist.md"):
        resolve_system_prompt(cfg.agent)


# --- from_json: parse a JSON string (base_dir optional, like from_dict) ---------------


def test_from_json_parses_string():
    cfg = WicaConfig.from_json(
        json.dumps(_wica_dict(model_kwargs={"temperature": 0.5}))
    )
    assert cfg.agent.provider == "anthropic"
    assert cfg.agent.model == "claude-sonnet-5"
    assert cfg.agent.system_prompt == "You are a test assistant."
    assert cfg.agent.model_kwargs == {"temperature": 0.5}


def test_from_json_string_locates_system_prompt_file_against_base_dir(tmp_path: Path):
    # A JSON string has no location of its own, so from_json takes the same optional base_dir as
    # from_dict and locates a relative system_prompt_file against it (parity with from_dict).
    data = _wica_dict()
    del data["agent"]["system_prompt"]
    data["agent"]["system_prompt_file"] = "prompts/persona.md"

    cfg = WicaConfig.from_json(json.dumps(data), base_dir=tmp_path)

    assert cfg.agent.system_prompt is None
    assert cfg.agent.system_prompt_file == str(
        (tmp_path / "prompts" / "persona.md").resolve()
    )


def test_from_json_invalid_json_raises_config_error():
    with pytest.raises(ConfigError, match="invalid JSON"):
        WicaConfig.from_json("{not valid json")


def test_from_json_non_object_raises_config_error():
    with pytest.raises(ConfigError, match="JSON object"):
        WicaConfig.from_json("[1, 2, 3]")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["agent"].__setitem__("system_prompt_file", "x.md"),  # both
        lambda d: d["agent"].pop("system_prompt"),  # neither
    ],
)
def test_exactly_one_system_prompt_field_enforced_at_load(mutate: Any):
    data = _wica_dict()
    mutate(data)
    with pytest.raises(ConfigError):
        WicaConfig.from_dict(data)


# --- resolve_system_prompt (build-time read) ------------------------------------------


def test_resolve_system_prompt_returns_inline():
    cfg = AgentConfig.from_dict(_agent_dict())
    assert resolve_system_prompt(cfg) == "You are a test assistant."


def test_resolve_system_prompt_reads_file(tmp_path: Path):
    prompt = tmp_path / "persona.md"
    prompt.write_text("You are Wica.")
    cfg = AgentConfig(provider="anthropic", model="m", system_prompt_file=str(prompt))
    assert resolve_system_prompt(cfg) == "You are Wica."


def test_resolve_system_prompt_missing_file_raises(tmp_path: Path):
    cfg = AgentConfig(
        provider="anthropic", model="m", system_prompt_file=str(tmp_path / "nope.md")
    )
    with pytest.raises(ConfigError, match="nope.md"):
        resolve_system_prompt(cfg)


# --- api key: verbatim at load, resolved at build -------------------------------------


def test_api_key_literal_stored():
    cfg = AgentConfig.from_dict(_agent_dict(api_key="sk-literal"))
    assert cfg.api_key == "sk-literal"


def test_resolve_api_key_returns_literal():
    cfg = AgentConfig.from_dict(_agent_dict(api_key="sk-literal"))
    assert resolve_api_key(cfg) == "sk-literal"


def test_resolve_api_key_reads_env_when_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WICA_TEST_KEY", "sk-from-env")
    cfg = AgentConfig.from_dict(_agent_dict(api_key_env="WICA_TEST_KEY"))
    # Not resolved at load...
    assert cfg.api_key is None
    # ...resolved at build.
    assert resolve_api_key(cfg) == "sk-from-env"


def test_resolve_api_key_unset_env_raises_missing_env_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("WICA_TEST_KEY_UNSET", raising=False)
    # Loads fine (verbatim) — the error is deferred to resolution.
    cfg = AgentConfig.from_dict(_agent_dict(api_key_env="WICA_TEST_KEY_UNSET"))
    with pytest.raises(MissingEnvError) as exc_info:
        resolve_api_key(cfg)
    assert exc_info.value.env_var == "WICA_TEST_KEY_UNSET"
    assert isinstance(exc_info.value, ConfigError)


def test_api_key_both_literal_and_env_rejected_at_load():
    with pytest.raises(ConfigError, match="api_key"):
        AgentConfig.from_dict(_agent_dict(api_key="sk-x", api_key_env="SOME_VAR"))


def test_resolve_api_key_none_when_neither_given():
    cfg = AgentConfig.from_dict(_agent_dict())
    assert cfg.api_key is None
    assert resolve_api_key(cfg) is None


# --- strict validation (structural, eager at load) ------------------------------------


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


def test_missing_agent_block_raises():
    with pytest.raises(ConfigError, match="agent"):
        WicaConfig.from_dict({})


# --- Resolution happens at build (build_chat_model / resolve_system_prompt) -------------


def test_build_chat_model_forwards_api_key_when_set(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model: str, *, model_provider: str, **kwargs: Any
    ) -> object:
        captured["model"] = model
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        system_prompt="hi",
        api_key="sk-abc",
    )
    agent_module.build_chat_model(config)

    assert captured["kwargs"]["api_key"] == "sk-abc"


def test_build_chat_model_omits_api_key_when_unset(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model: str, *, model_provider: str, **kwargs: Any
    ) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(
        provider="anthropic", model="claude-sonnet-5", system_prompt="hi"
    )
    agent_module.build_chat_model(config)

    assert "api_key" not in captured["kwargs"]


def test_resolve_system_prompt_reads_file_at_build(tmp_path: Path):
    prompt = tmp_path / "persona.md"
    prompt.write_text("Persona from a file.")

    config = AgentConfig(
        provider="anthropic", model="m", system_prompt_file=str(prompt)
    )
    assert resolve_system_prompt(config) == "Persona from a file."


def test_build_chat_model_raises_missing_env_at_build(monkeypatch: pytest.MonkeyPatch):
    """The behavior change: an unset api_key_env surfaces at Agent build, not at config load."""
    monkeypatch.delenv("WICA_TEST_KEY_UNSET", raising=False)
    monkeypatch.setattr(agent_module, "init_chat_model", lambda *a, **k: object())

    config = AgentConfig.from_dict(
        _agent_dict(api_key_env="WICA_TEST_KEY_UNSET")
    )  # loads fine
    with pytest.raises(MissingEnvError):
        agent_module.build_chat_model(config)


def test_from_json_then_wica_init_composes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The blessed startup shape: load the file, then Wica.init — which builds the Agent (resolution
    at build). See specs/config.md "Flow into the Agent"."""
    import asyncio

    from wica import Wica

    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model: str, *, model_provider: str, **kwargs: Any
    ) -> object:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config_path = tmp_path / "agent.config.json"
    config_path.write_text(json.dumps(_wica_dict(api_key="sk-abc")))

    wica_config = WicaConfig.from_json_file(config_path)
    loop = (
        asyncio.new_event_loop()
    )  # injected + owned here, never started (no model call needed)
    try:
        Wica.init(wica_config, loop=loop)  # builds the model (monkeypatched)
        assert captured["model"] == "claude-sonnet-5"
    finally:
        loop.close()
    assert captured["kwargs"]["api_key"] == "sk-abc"


# --- hf_provider field --------------------------------------------------------------------


def test_hf_provider_defaults_to_auto():
    cfg = AgentConfig.from_dict(_agent_dict())
    assert cfg.hf_provider == "auto"


def test_hf_provider_parsed_when_present():
    cfg = AgentConfig.from_dict(
        _agent_dict(
            provider="huggingface-hub", model="meta-llama/x", hf_provider="together"
        )
    )
    assert cfg.hf_provider == "together"


def test_hf_provider_wrong_type_raises():
    with pytest.raises(ConfigError, match="hf_provider"):
        AgentConfig.from_dict(_agent_dict(hf_provider=123))


# --- build_chat_model: provider construction branch ---------------------------------------


def test_build_chat_model_openai_routes_through_init_chat_model(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model: str, *, model_provider: str, **kwargs: Any
    ) -> object:
        captured["model"] = model
        captured["provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(
        provider="openai", model="gpt-4o", system_prompt="hi", api_key="sk-x"
    )
    agent_module.build_chat_model(config)

    assert captured["model"] == "gpt-4o"
    assert captured["provider"] == "openai"
    assert captured["kwargs"]["api_key"] == "sk-x"


def test_build_chat_model_resolves_api_key_env(monkeypatch: pytest.MonkeyPatch):
    """build_chat_model is the api-key resolution point for the init_chat_model providers: an
    api_key_env config resolves to the env value here, at build."""
    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model: str, *, model_provider: str, **kwargs: Any
    ) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("WICA_TEST_KEY", "sk-resolved")

    config = AgentConfig.from_dict(
        _agent_dict(provider="openai", api_key_env="WICA_TEST_KEY")
    )
    agent_module.build_chat_model(config)

    assert captured["kwargs"]["api_key"] == "sk-resolved"


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
        lambda *a, **k: pytest.fail(
            "huggingface-hub must not go through init_chat_model"
        ),
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
    monkeypatch.setattr(
        langchain_huggingface, "ChatHuggingFace", lambda *, llm: object()
    )

    config = AgentConfig(
        provider="huggingface-hub", model="meta-llama/x", system_prompt="hi"
    )
    agent_module.build_chat_model(config)

    # No key given: don't pass the token kwarg at all, so HuggingFaceEndpoint reads the standard
    # env var (mirrors how the init_chat_model providers behave when api_key is unset).
    assert "huggingfacehub_api_token" not in captured["endpoint"]
    assert captured["endpoint"]["provider"] == "auto"


# --- Direct-construction invariants and frozen configs ---------------------------


def test_direct_construction_with_both_prompt_fields_raises():
    with pytest.raises(ConfigError):
        AgentConfig(
            provider="fake", model="m", system_prompt="a", system_prompt_file="b.md"
        )


def test_direct_construction_with_no_prompt_raises():
    with pytest.raises(ConfigError):
        AgentConfig(provider="fake", model="m")


def test_direct_construction_with_both_key_fields_raises():
    with pytest.raises(ConfigError):
        AgentConfig(
            provider="fake",
            model="m",
            system_prompt="a",
            api_key="k",
            api_key_env="E",
        )


def test_configs_are_frozen():
    cfg = WicaConfig(agent=AgentConfig(provider="fake", model="m", system_prompt="a"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent.system_prompt = "b"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent = cfg.agent  # type: ignore[misc]
    variant = dataclasses.replace(cfg.agent, system_prompt="b")
    assert variant.system_prompt == "b" and cfg.agent.system_prompt == "a"
