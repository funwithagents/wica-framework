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
    MissingSecretError,
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


# --- api key: a variable named at load, resolved at build ----------------------------


@pytest.mark.parametrize("value", ["sk-literal", "", None])
def test_a_key_written_in_the_config_is_refused(value: Any):
    with pytest.raises(ConfigError, match="does not take 'api_key'.*'api_key_env'"):
        AgentConfig.from_dict(_agent_dict(api_key=value))
    with pytest.raises(ConfigError, match="does not take 'api_key'"):
        WicaConfig.from_dict(_wica_dict(api_key="sk-x", api_key_env="SOME_VAR"))


def test_resolve_api_key_reads_the_named_variable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WICA_TEST_KEY", "sk-from-env")
    cfg = AgentConfig.from_dict(_agent_dict(api_key_env="WICA_TEST_KEY"))
    assert resolve_api_key(cfg) == "sk-from-env"


@pytest.mark.parametrize(
    ("provider", "variable"),
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("huggingface-hub", "HF_TOKEN"),
        ("mistralai", "MISTRAL_API_KEY"),
    ],
)
def test_resolve_api_key_defaults_to_the_provider_s_usual_variable(
    monkeypatch: pytest.MonkeyPatch, provider: str, variable: str
):
    monkeypatch.setenv(variable, "sk-usual")
    cfg = AgentConfig.from_dict(_agent_dict(provider=provider))
    assert cfg.api_key_env is None  # the config mirrors the JSON
    assert resolve_api_key(cfg) == "sk-usual"
    monkeypatch.delenv(variable)
    with pytest.raises(MissingSecretError) as exc_info:
        resolve_api_key(cfg)
    assert exc_info.value.variable == variable


def test_secrets_given_in_code_come_before_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    cfg = AgentConfig.from_dict(_agent_dict())
    assert resolve_api_key(cfg, {"ANTHROPIC_API_KEY": "sk-code"}) == "sk-code"
    # A secret under another name does not count; the environment answers.
    assert resolve_api_key(cfg, {"OTHER": "sk-other"}) == "sk-env"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    named = AgentConfig.from_dict(_agent_dict(api_key_env="MY_KEY"))
    assert resolve_api_key(named, {"MY_KEY": "sk-mine"}) == "sk-mine"


def test_resolve_api_key_unset_or_empty_raises_missing_secret_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("WICA_TEST_KEY_UNSET", raising=False)
    # Loads fine (verbatim) — the error is deferred to resolution.
    cfg = AgentConfig.from_dict(_agent_dict(api_key_env="WICA_TEST_KEY_UNSET"))
    with pytest.raises(MissingSecretError) as exc_info:
        resolve_api_key(cfg)
    error = exc_info.value
    assert (error.variable, error.label) == ("WICA_TEST_KEY_UNSET", "Agent")
    assert isinstance(error, ConfigError)
    assert str(error) == (
        "Agent: no key under 'WICA_TEST_KEY_UNSET': set the environment variable, "
        "or pass it in `secrets`"
    )
    monkeypatch.setenv("WICA_TEST_KEY_UNSET", "")
    with pytest.raises(MissingSecretError):
        resolve_api_key(cfg, {"WICA_TEST_KEY_UNSET": ""})


def test_missing_env_error_is_a_deprecated_alias_of_missing_secret_error():
    import wica
    import wica.config

    with pytest.warns(DeprecationWarning, match="MissingSecretError"):
        alias = wica.config.MissingEnvError
    with pytest.warns(DeprecationWarning, match="MissingSecretError"):
        assert wica.MissingEnvError is alias
    assert alias is MissingSecretError
    # Caught by its old name, read through its old attribute.
    try:
        raise MissingSecretError("SOME_VAR")
    except alias as exc:
        assert exc.env_var == "SOME_VAR"


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "fake"},
        {"provider": "fake", "api_key_env": "NEVER_READ"},
        {"provider": "openai-chat-completions-compatible", "base_url": "http://x/v1"},
        {"provider": "ollama"},
    ],
)
def test_resolve_api_key_none_when_no_variable_applies(overrides: dict[str, Any]):
    """The fake reads no key; a compatible server and a provider WICA does not know have no
    default variable."""
    cfg = AgentConfig.from_dict(_agent_dict(**overrides))
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
        provider="anthropic", model="claude-sonnet-5", system_prompt="hi"
    )
    agent_module.build_chat_model(config, secrets={"ANTHROPIC_API_KEY": "sk-abc"})

    assert captured["kwargs"]["api_key"] == "sk-abc"


def test_build_chat_model_omits_api_key_only_for_an_unknown_provider_naming_none(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(
        model: str, *, model_provider: str, **kwargs: Any
    ) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    config = AgentConfig(provider="ollama", model="llama3", system_prompt="hi")
    agent_module.build_chat_model(config)

    assert "api_key" not in captured["kwargs"]


def test_build_chat_model_names_the_extra_of_a_provider_not_installed(
    monkeypatch: pytest.MonkeyPatch,
):
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("langchain_mistralai"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(__import__("sys").modules, "langchain_mistralai", raising=False)
    config = AgentConfig(provider="mistral", model="m", system_prompt="hi")
    with pytest.raises(ImportError, match=r"wica\[mistral\]"):
        agent_module.build_chat_model(config, secrets={"MISTRAL_API_KEY": "k"})


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
    with pytest.raises(MissingSecretError):
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
    config_path.write_text(json.dumps(_wica_dict(api_key_env="MY_KEY")))

    wica_config = WicaConfig.from_json_file(config_path)
    loop = (
        asyncio.new_event_loop()
    )  # injected + owned here, never started (no model call needed)
    try:
        # builds the model (monkeypatched), the key held in code
        Wica.init(wica_config, loop=loop, secrets={"MY_KEY": "sk-abc"})
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


# --- history_reactions field --------------------------------------------------------------


def test_history_reactions_defaults_to_unbounded():
    assert AgentConfig.from_dict(_agent_dict()).history_reactions is None
    assert (
        AgentConfig.from_dict(_agent_dict(history_reactions=None)).history_reactions
        is None
    )


def test_history_reactions_parsed_from_dict_and_json():
    assert (
        AgentConfig.from_dict(_agent_dict(history_reactions=3)).history_reactions == 3
    )
    cfg = WicaConfig.from_json(json.dumps(_wica_dict(history_reactions=12)))
    assert cfg.agent.history_reactions == 12


@pytest.mark.parametrize("value", [0, -1, "3", True, 2.5])
def test_history_reactions_rejects_non_positive_integers(value: Any):
    with pytest.raises(ConfigError, match="history_reactions"):
        AgentConfig.from_dict(_agent_dict(history_reactions=value))
    with pytest.raises(ConfigError, match="history_reactions"):
        AgentConfig(
            provider="fake", model="x", system_prompt="p", history_reactions=value
        )


# --- history_budget / context_window fields -----------------------------------------------


def test_history_budget_and_context_window_default_to_off():
    cfg = AgentConfig.from_dict(_agent_dict())
    assert cfg.history_budget is None and cfg.context_window is None
    cfg = AgentConfig.from_dict(_agent_dict(history_budget=None, context_window=None))
    assert cfg.history_budget is None and cfg.context_window is None


def test_history_budget_and_context_window_parsed():
    cfg = WicaConfig.from_json(
        json.dumps(_wica_dict(history_budget=0.6, context_window=128000))
    ).agent
    assert cfg.history_budget == 0.6 and cfg.context_window == 128000
    assert AgentConfig.from_dict(_agent_dict(history_budget=1)).history_budget == 1.0
    # The window alone is accepted (and unused without a budget).
    assert (
        AgentConfig.from_dict(_agent_dict(context_window=4096)).context_window == 4096
    )


@pytest.mark.parametrize("value", [0, 1.5, -0.2, "0.5", True])
def test_history_budget_rejects_values_outside_zero_one(value: Any):
    with pytest.raises(ConfigError, match="history_budget"):
        AgentConfig.from_dict(_agent_dict(history_budget=value))
    with pytest.raises(ConfigError, match="history_budget"):
        AgentConfig(provider="fake", model="x", system_prompt="p", history_budget=value)


@pytest.mark.parametrize("value", [0, -1, "4096", True, 2.5])
def test_context_window_rejects_non_positive_integers(value: Any):
    with pytest.raises(ConfigError, match="context_window"):
        AgentConfig.from_dict(_agent_dict(context_window=value))
    with pytest.raises(ConfigError, match="context_window"):
        AgentConfig(provider="fake", model="x", system_prompt="p", context_window=value)


# --- model_timeout field ------------------------------------------------------------------


def test_model_timeout_defaults_to_ten_seconds():
    assert AgentConfig.from_dict(_agent_dict()).model_timeout == 10.0
    assert (
        AgentConfig(provider="fake", model="x", system_prompt="p").model_timeout == 10.0
    )


def test_model_timeout_parsed_and_null_removes_it():
    cfg = WicaConfig.from_json(json.dumps(_wica_dict(model_timeout=2.5))).agent
    assert cfg.model_timeout == 2.5
    assert AgentConfig.from_dict(_agent_dict(model_timeout=30)).model_timeout == 30.0
    assert AgentConfig.from_dict(_agent_dict(model_timeout=None)).model_timeout is None


@pytest.mark.parametrize("value", [0, -1, "10", True])
def test_model_timeout_rejects_non_positive_numbers(value: Any):
    with pytest.raises(ConfigError, match="model_timeout"):
        AgentConfig.from_dict(_agent_dict(model_timeout=value))
    with pytest.raises(ConfigError, match="model_timeout"):
        AgentConfig(provider="fake", model="x", system_prompt="p", model_timeout=value)


# --- max_retries field --------------------------------------------------------------------


def test_max_retries_defaults_to_one():
    assert AgentConfig.from_dict(_agent_dict()).max_retries == 1
    assert AgentConfig(provider="fake", model="x", system_prompt="p").max_retries == 1


def test_max_retries_parsed_and_zero_accepted():
    cfg = WicaConfig.from_json(json.dumps(_wica_dict(max_retries=3))).agent
    assert cfg.max_retries == 3
    assert AgentConfig.from_dict(_agent_dict(max_retries=0)).max_retries == 0


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_max_retries_rejects_anything_but_a_non_negative_integer(value: Any):
    with pytest.raises(ConfigError, match="max_retries"):
        AgentConfig.from_dict(_agent_dict(max_retries=value))
    with pytest.raises(ConfigError, match="max_retries"):
        AgentConfig(provider="fake", model="x", system_prompt="p", max_retries=value)


# --- max_next_steps field -----------------------------------------------------------------


def test_max_next_steps_defaults_to_one():
    assert AgentConfig.from_dict(_agent_dict()).max_next_steps == 1
    assert (
        AgentConfig(provider="fake", model="x", system_prompt="p").max_next_steps == 1
    )


def test_max_next_steps_parsed_zero_turns_it_off_and_null_removes_the_cap():
    cfg = WicaConfig.from_json(json.dumps(_wica_dict(max_next_steps=3))).agent
    assert cfg.max_next_steps == 3
    assert AgentConfig.from_dict(_agent_dict(max_next_steps=0)).max_next_steps == 0
    assert (
        AgentConfig.from_dict(_agent_dict(max_next_steps=None)).max_next_steps is None
    )


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_max_next_steps_rejects_anything_but_a_non_negative_integer_or_null(
    value: Any,
):
    with pytest.raises(ConfigError, match="max_next_steps"):
        AgentConfig.from_dict(_agent_dict(max_next_steps=value))
    with pytest.raises(ConfigError, match="max_next_steps"):
        AgentConfig(provider="fake", model="x", system_prompt="p", max_next_steps=value)


def test_force_tool_call_is_no_longer_a_config_key():
    """The forced tool call is gone: the strict loader refuses its key like any unknown one."""
    with pytest.raises(ConfigError, match="force_tool_call"):
        AgentConfig.from_dict(_agent_dict(force_tool_call=True))
    with pytest.raises(ConfigError, match="force_tool_call"):
        WicaConfig.from_json(json.dumps(_wica_dict(force_tool_call=False)))


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

    config = AgentConfig(provider="openai", model="gpt-4o", system_prompt="hi")
    agent_module.build_chat_model(config, secrets={"OPENAI_API_KEY": "sk-x"})

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


def test_build_chat_model_gemini_is_a_pass_through():
    """gemini (Gemini via the Gemini Developer API) takes no WICA branch: build_chat_model hands it
    to init_chat_model under LangChain's id `google_genai`, and the two things that must land on the real model
    do — the resolved api_key on its Google key field, and model_kwargs (thinking_level, the
    knob the committed configs use to minimize Gemini 3's default thinking). Constructing the
    model makes no request — see specs/config.md "Providers"."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    config = AgentConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        system_prompt="hi",
        model_kwargs={"thinking_level": "low"},
    )
    model = agent_module.build_chat_model(
        config, secrets={"GEMINI_API_KEY": "test-key"}
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-3.8-flash")
    assert model.google_api_key is not None
    assert model.google_api_key.get_secret_value() == "test-key"
    assert model.thinking_level == "low"


def test_build_chat_model_mistral_is_a_pass_through():
    """mistral (Mistral via La Plateforme) takes no WICA branch: build_chat_model hands it to
    init_chat_model under LangChain's id `mistralai`, and the two things that must land on the real model do — the
    resolved api_key on its mistral_api_key field, and model_kwargs (temperature). Constructing
    the model makes no request — see specs/config.md "Providers"."""
    from langchain_mistralai import ChatMistralAI

    config = AgentConfig(
        provider="mistral",
        model="mistral-small-latest",
        system_prompt="hi",
        model_kwargs={"temperature": 0.2},
    )
    model = agent_module.build_chat_model(
        config, secrets={"MISTRAL_API_KEY": "test-key"}
    )

    assert isinstance(model, ChatMistralAI)
    assert model.model == "mistral-small-latest"
    assert model.mistral_api_key is not None
    assert model.mistral_api_key.get_secret_value() == "test-key"
    assert model.temperature == 0.2


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
        hf_provider="fireworks-ai",
        model_kwargs={"temperature": 0.3},
    )
    model = agent_module.build_chat_model(config, secrets={"HF_TOKEN": "hf-token"})

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


def test_build_chat_model_huggingface_hub_passes_the_token_from_its_default_variable(
    monkeypatch: pytest.MonkeyPatch,
):
    """The Hub's key is always passed explicitly, read from HF_TOKEN by default: the package is
    never left to find a token of its own. See specs/config.md ("API key")."""
    import langchain_huggingface

    captured: dict[str, Any] = {}

    class FakeEndpoint:
        def __init__(self, **kwargs: Any) -> None:
            captured["endpoint"] = kwargs

    class FakeChat:
        def __init__(self, *, llm: Any) -> None:
            pass

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEndpoint", FakeEndpoint)
    monkeypatch.setattr(langchain_huggingface, "ChatHuggingFace", FakeChat)
    monkeypatch.setenv("HF_TOKEN", "hf-from-env")

    config = AgentConfig(
        provider="huggingface-hub", model="meta-llama/x", system_prompt="hi"
    )
    agent_module.build_chat_model(config)

    assert captured["endpoint"]["huggingfacehub_api_token"] == "hf-from-env"
    assert captured["endpoint"]["provider"] == "auto"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(MissingSecretError, match="HF_TOKEN"):
        agent_module.build_chat_model(config)


def _hub_request_params(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The parameters a real ChatHuggingFace (built offline) sends with each Hub request."""
    from langchain_huggingface import ChatHuggingFace

    model = agent_module.build_chat_model(
        AgentConfig(
            provider="huggingface-hub",
            model="openai/gpt-oss-120b",
            hf_provider="cerebras",
            system_prompt="hi",
            model_kwargs=model_kwargs,
        ),
        secrets={"HF_TOKEN": "hf_test"},
    )
    assert isinstance(model, ChatHuggingFace)
    return model._default_params  # pyright: ignore[reportPrivateUsage]


def test_build_chat_model_huggingface_hub_sends_no_output_cap_by_default():
    """HuggingFaceEndpoint's 512-token default would cut a reasoning model off before its tool
    call; unless the config sets one, the request carries no cap. See specs/config.md."""
    assert "max_tokens" not in _hub_request_params({})


def test_build_chat_model_huggingface_hub_keeps_a_configured_output_cap():
    assert _hub_request_params({"max_new_tokens": 300})["max_tokens"] == 300


# --- Direct-construction invariants and frozen configs ---------------------------


def test_direct_construction_with_both_prompt_fields_raises():
    with pytest.raises(ConfigError):
        AgentConfig(
            provider="fake", model="m", system_prompt="a", system_prompt_file="b.md"
        )


def test_direct_construction_with_no_prompt_raises():
    with pytest.raises(ConfigError):
        AgentConfig(provider="fake", model="m")


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"history_reactions": 0},
        {"model_timeout": -1.0},
        {"max_retries": True},
        {"hf_provider": ""},
        {"api_key_env": ""},
    ],
)
def test_direct_construction_validates_like_the_loader(overrides: dict[str, Any]):
    """One validation over the declared fields: what the loader refuses, the constructor
    refuses, in the same words. See specs/setup.md ("Fields are declared once")."""
    data = _agent_dict(**overrides)
    with pytest.raises(ConfigError) as loaded:
        AgentConfig.from_dict(data)
    with pytest.raises(ConfigError) as direct:
        AgentConfig(**data)
    assert str(direct.value) == str(loaded.value)


def test_configs_are_frozen():
    cfg = WicaConfig(agent=AgentConfig(provider="fake", model="m", system_prompt="a"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent.system_prompt = "b"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent = cfg.agent  # type: ignore[misc]
    variant = dataclasses.replace(cfg.agent, system_prompt="b")
    assert variant.system_prompt == "b" and cfg.agent.system_prompt == "a"


# --- provider names and base_url --------------------------------------------------------

_COMPATIBLE = "openai-chat-completions-compatible"


@pytest.mark.parametrize(
    ("old", "new"), [("google_genai", "gemini"), ("mistralai", "mistral")]
)
def test_an_old_provider_name_loads_as_the_new_one_with_a_warning(
    old: str, new: str, caplog: pytest.LogCaptureFixture
):
    """Old names keep working: the stored provider is the new name, whichever way the config was
    created, and each read logs a deprecation warning naming the new one — see specs/config.md
    ("Provider names")."""
    with caplog.at_level("WARNING", logger="wica.config"):
        loaded = WicaConfig.from_dict(_wica_dict(provider=old))
        direct = AgentConfig(provider=old, model="m", system_prompt="hi")

    assert loaded.agent.provider == new
    assert direct.provider == new
    warnings = [r.getMessage() for r in caplog.records if r.name == "wica.config"]
    assert len(warnings) == 2
    assert all(old in w and new in w and "deprecated" in w for w in warnings)


def test_a_current_provider_name_logs_nothing(caplog: pytest.LogCaptureFixture):
    with caplog.at_level("WARNING", logger="wica.config"):
        cfg = WicaConfig.from_dict(_wica_dict(provider="gemini"))
        dataclasses.replace(cfg.agent, model="other")
    assert cfg.agent.provider == "gemini"
    assert [r for r in caplog.records if r.name == "wica.config"] == []


def test_base_url_is_loaded_for_the_compatible_provider():
    cfg = WicaConfig.from_json(
        json.dumps(
            _wica_dict(
                provider=_COMPATIBLE, model="local", base_url="http://127.0.0.1:8080/v1"
            )
        )
    )
    assert cfg.agent.provider == _COMPATIBLE
    assert cfg.agent.base_url == "http://127.0.0.1:8080/v1"


def test_the_compatible_provider_needs_a_base_url():
    with pytest.raises(ConfigError, match="needs 'base_url'"):
        WicaConfig.from_dict(_wica_dict(provider=_COMPATIBLE, model="local"))
    with pytest.raises(ConfigError, match="needs 'base_url'"):
        AgentConfig(provider=_COMPATIBLE, model="local", system_prompt="hi")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "fake", "huggingface-hub"])
def test_base_url_is_rejected_with_any_other_provider(provider: str):
    """Silently ignored, a base_url would send the requests to the vendor instead of the intended
    server — see specs/config.md ("Providers")."""
    with pytest.raises(ConfigError, match="only for provider"):
        WicaConfig.from_dict(
            _wica_dict(provider=provider, base_url="http://localhost/v1")
        )
    with pytest.raises(ConfigError, match="only for provider"):
        AgentConfig(
            provider=provider,
            model="m",
            system_prompt="hi",
            base_url="http://localhost/v1",
        )


@pytest.mark.parametrize("bad", ["", 8080, ["http://x/v1"]])
def test_base_url_must_be_a_non_empty_string(bad: Any):
    with pytest.raises(ConfigError, match="base_url"):
        WicaConfig.from_dict(
            _wica_dict(provider=_COMPATIBLE, model="local", base_url=bad)
        )


def test_the_compatible_provider_rejects_base_url_in_model_kwargs():
    with pytest.raises(ConfigError, match="not in 'model_kwargs'"):
        WicaConfig.from_dict(
            _wica_dict(
                provider=_COMPATIBLE,
                model="local",
                base_url="http://127.0.0.1:8080/v1",
                model_kwargs={"base_url": "http://elsewhere/v1"},
            )
        )


def test_openai_keeps_base_url_in_model_kwargs():
    """The vendor provider stays a pass-through: a proxy address in its model_kwargs still loads."""
    cfg = WicaConfig.from_dict(
        _wica_dict(provider="openai", model_kwargs={"base_url": "http://proxy/v1"})
    )
    assert cfg.agent.model_kwargs == {"base_url": "http://proxy/v1"}


# --- stream_output field ------------------------------------------------------------------


def test_stream_output_defaults_to_the_mode_and_parses_a_boolean():
    # Unset: the output mode's default (streamed with tags, not with the output as a tool).
    assert AgentConfig.from_dict(_agent_dict()).stream_output is None
    assert (
        AgentConfig(provider="fake", model="x", system_prompt="p").stream_output is None
    )
    cfg = WicaConfig.from_json(json.dumps(_wica_dict(stream_output=False))).agent
    assert cfg.stream_output is False
    assert AgentConfig.from_dict(_agent_dict(stream_output=True)).stream_output is True
    assert AgentConfig.from_dict(_agent_dict(stream_output=None)).stream_output is None


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_stream_output_rejects_anything_but_a_boolean(value: Any):
    with pytest.raises(ConfigError, match="stream_output"):
        AgentConfig.from_dict(_agent_dict(stream_output=value))
    with pytest.raises(ConfigError, match="stream_output"):
        AgentConfig(provider="fake", model="x", system_prompt="p", stream_output=value)
