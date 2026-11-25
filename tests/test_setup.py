"""`wica.setup`: setting the agent up from an application, as data.

See specs/setup.md.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest
from app_setup.contract import check_contract

from wica import setup
from wica.config import PROVIDERS, ConfigError, WicaConfig

_ROOT = Path(__file__).resolve().parent.parent


def _examples() -> dict[str, dict[str, Any]]:
    """Every example config, by file name."""
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((_ROOT / "examples" / "configs").glob("*.json"))
    }


def _block(**agent: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "system_prompt": "You are terse.",
    }
    base.update(agent)
    return {"agent": {k: v for k, v in base.items() if v is not ...}}


# --- The description --------------------------------------------------------------------


def test_the_contract_s_conformance_check_passes_over_the_examples() -> None:
    check_contract(setup, list(_examples().values()))


def test_there_is_an_example_for_every_known_provider() -> None:
    selected = {block["agent"]["provider"] for block in _examples().values()}
    assert selected == set(PROVIDERS)


def test_config_schema_is_stable_and_offers_one_branch_per_known_provider() -> None:
    description = setup.config_schema()

    assert description == setup.config_schema()
    assert (description["contract"], description["distribution"]) == (1, "wica")
    agent = description["schema"]["properties"]["agent"]
    assert agent["title"] == "Provider"
    branches = agent["oneOf"]
    assert [b["properties"]["provider"]["const"] for b in branches] == list(PROVIDERS)
    by_name = {b["properties"]["provider"]["const"]: b for b in branches}
    assert by_name["anthropic"]["title"] == "Anthropic (hosted)"
    assert by_name["fake"]["title"] == "Scripted fake (fixture)"
    key = by_name["huggingface-hub"]["properties"]["api_key_env"]
    assert (key["default"], key["x-secret-env"]) == ("HF_TOKEN", True)
    assert by_name["huggingface-hub"]["properties"]["hf_provider"]["default"] == "auto"
    compatible = by_name["openai-chat-completions-compatible"]
    assert compatible["required"] == ["provider", "model", "base_url"]
    assert "default" not in compatible["properties"]["api_key_env"]
    assert "api_key_env" not in by_name["fake"]["properties"]
    assert agent["properties"]["model_timeout"]["default"] == 10.0
    assert agent["properties"]["model_kwargs"]["additionalProperties"] is True
    assert "api_key" not in json.dumps(description).replace("api_key_env", "")


def test_model_belongs_to_each_provider_s_branch() -> None:
    """A model id means nothing to another provider: `model` is declared in every branch
    (required, no default), not among the fields common to every branch."""
    agent = setup.config_schema()["schema"]["properties"]["agent"]

    assert "model" not in agent["properties"] and "model" not in agent["required"]
    for branch in agent["oneOf"]:
        assert "model" in branch["required"]
        assert "default" not in branch["properties"]["model"]


def test_switching_provider_in_the_form_drops_the_model() -> None:
    from app_setup.contract import parse_description
    from app_setup.form import FormChoice, FormGroup, layout, select_branch

    schema = parse_description(setup.config_schema()).schema
    document = {
        "agent": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "api_key_env": "MY_KEY",
            "system_prompt": "You are terse.",
            "max_retries": 2,
        }
    }
    agent = next(
        item
        for item in layout(schema, document).items
        if isinstance(item, FormGroup) and item.path == ("agent",)
    )
    choice = next(item for item in agent.items if isinstance(item, FormChoice))

    switched = select_branch(document, ("agent",), choice.choice, "openai")

    # The common fields stay; the old provider's model and key variable go.
    assert switched["agent"] == {
        "provider": "openai",
        "system_prompt": "You are terse.",
        "max_retries": 2,
    }
    assert [p["path"] for p in setup.check_config(switched)] == ["/agent/model"]
    assert setup.check_config(switched)[0]["message"] == (
        "agent: missing required key 'model'"
    )


# --- required_extras ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "extra"),
    [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("gemini", "gemini"),
        ("mistral", "mistral"),
        ("mistralai", "mistral"),
        ("huggingface-hub", "huggingface-hub"),
        ("openai-chat-completions-compatible", "openai-compatible"),
    ],
)
def test_required_extras_names_the_provider_s_extra(provider: str, extra: str) -> None:
    assert setup.required_extras({"agent": {"provider": provider}}) == [
        f"wica[{extra}]"
    ]


@pytest.mark.parametrize(
    "config",
    [
        {},
        [],
        {"agent": "anthropic"},
        {"agent": {}},
        {"agent": {"provider": 3}},
        {"agent": {"provider": "fake"}},
        {"agent": {"provider": "ollama"}},
    ],
)
def test_required_extras_is_empty_when_nothing_is_needed_or_selected(
    config: Any,
) -> None:
    assert setup.required_extras(config) == []


def test_every_extra_a_provider_names_is_declared() -> None:
    declared = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]
    assert {p.extra for p in PROVIDERS.values() if p.extra} <= set(declared)


# --- check_config ---------------------------------------------------------------------------


PROBLEMS = [
    ([], ""),
    ({}, "/agent"),
    ({"agent": 3}, "/agent"),
    ({**_block(), "bogus": 1}, "/bogus"),
    (_block(api_key="sk-x"), "/agent/api_key"),
    (_block(modl="typo"), "/agent/modl"),
    (_block(provider=...), "/agent/provider"),
    (_block(provider=""), "/agent/provider"),
    (_block(model=...), "/agent/model"),
    (_block(history_reactions=0), "/agent/history_reactions"),
    (_block(model_timeout=0), "/agent/model_timeout"),
    (_block(max_retries=None), "/agent/max_retries"),
    (_block(model_kwargs=[]), "/agent/model_kwargs"),
    (_block(api_key_env=""), "/agent/api_key_env"),
    (_block(system_prompt_file="p.md"), "/agent"),
    (_block(system_prompt=...), "/agent"),
    (_block(base_url="http://x/v1"), "/agent/base_url"),
    (_block(provider="openai-chat-completions-compatible"), "/agent/base_url"),
    (
        _block(
            provider="openai-chat-completions-compatible",
            base_url="http://x/v1",
            model_kwargs={"base_url": "http://y/v1"},
        ),
        "/agent/model_kwargs",
    ),
]


@pytest.mark.parametrize(("config", "path"), PROBLEMS)
def test_a_problem_is_reported_at_its_path_in_the_loader_s_words(
    config: Any, path: str
) -> None:
    problems = setup.check_config(config)

    assert [p["path"] for p in problems] == [path]
    with pytest.raises(ConfigError) as refused:
        WicaConfig.from_dict(config)
    assert str(refused.value) == problems[0]["message"]


def test_several_problems_are_reported_at_once() -> None:
    problems = setup.check_config(
        _block(api_key="k", model=3, max_retries=-1, extra=True)
    )

    assert [p["path"] for p in problems] == [
        "/agent/api_key",
        "/agent/extra",
        "/agent/model",
        "/agent/max_retries",
    ]
    assert problems[0]["message"] == (
        "agent: does not take 'api_key': keep the key in an environment variable and "
        "name it in 'api_key_env'"
    )


@pytest.mark.parametrize(
    "config",
    [
        _block(),
        _block(provider="ollama", api_key_env="OLLAMA_KEY"),
        _block(provider="google_genai"),
        _block(provider="fake", api_key_env="IGNORED"),
        _block(hf_provider="together"),
        _block(model_timeout=None, max_next_steps=None, history_budget=1),
    ],
)
def test_a_config_the_loader_accepts_has_no_problem(config: dict[str, Any]) -> None:
    WicaConfig.from_dict(config)
    assert setup.check_config(config) == []


def test_every_example_config_loads_from_its_file() -> None:
    for path in sorted((_ROOT / "examples" / "configs").glob("*.json")):
        config = WicaConfig.from_json_file(path)
        assert config.agent.system_prompt_file is not None
        assert Path(config.agent.system_prompt_file).is_file(), path.name


# --- No provider imported --------------------------------------------------------------------


_BLOCKED = textwrap.dedent(
    """
    import importlib.abc, json, sys

    BLOCKED = ("langchain_anthropic", "langchain_openai", "langchain_google_genai",
               "langchain_mistralai", "langchain_huggingface", "anthropic", "openai",
               "google.genai", "huggingface_hub")

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in {b.split(".")[0] for b in BLOCKED}:
                raise ImportError(f"blocked: {name}")
            return None

    sys.meta_path.insert(0, Block())
    from wica import setup

    out = {"schema": bool(setup.config_schema()), "setup": {}}
    for provider in ("anthropic", "openai", "gemini", "mistral", "huggingface-hub",
                     "openai-chat-completions-compatible"):
        config = {"agent": {"provider": provider, "model": "m", "system_prompt": "p",
                            "base_url": "http://x/v1"}}
        if provider != "openai-chat-completions-compatible":
            del config["agent"]["base_url"]
        assert setup.check_config(config) == [], provider
        setup.required_extras(config)
        out["setup"][provider] = setup.check_setup(config)
    out["loaded"] = sorted(m for m in sys.modules if m.split(".")[0] in
                           {b.split(".")[0] for b in BLOCKED})
    print(json.dumps(out))
    """
)


def test_the_base_install_describes_and_checks_without_any_provider() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )
    out = json.loads(result.stdout)

    assert out["schema"] and out["loaded"] == []
    for provider, items in out["setup"].items():
        extra = items[0]
        assert extra["id"] == f"extra:wica[{PROVIDERS[provider].extra}]"
        assert extra["status"] == "missing"
        assert f"wica[{PROVIDERS[provider].extra}]" in extra["detail"]
        assert all(not item["id"].startswith("file:") for item in items)


# --- check_setup ------------------------------------------------------------------------------


def test_check_setup_reports_the_extra_the_key_and_the_prompt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prompt = tmp_path / "persona.md"
    config = _block(system_prompt=..., system_prompt_file=str(prompt))

    items = setup.check_setup(config)

    assert [(i["id"], i["status"], i["preparable"], i["required"]) for i in items] == [
        ("extra:wica[anthropic]", "ready", False, True),
        ("key:ANTHROPIC_API_KEY", "missing", False, True),
        ("file:system_prompt_file", "missing", False, True),
    ]
    assert "ANTHROPIC_API_KEY" in items[1]["detail"]
    assert str(prompt) in items[2]["detail"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert setup.check_setup(config)[1]["status"] == "missing"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-set")
    prompt.write_text("You are terse.", encoding="utf-8")
    assert [i["status"] for i in setup.check_setup(config)] == ["ready"] * 3


def test_a_relative_prompt_file_is_found_against_base_dir_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "persona.md").write_text("You are terse.", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = _block(provider="fake", system_prompt=..., system_prompt_file="persona.md")

    found = setup.check_setup(config, base_dir=str(folder))
    assert [(i["id"], i["status"]) for i in found] == [
        ("file:system_prompt_file", "ready")
    ]
    assert str(folder / "persona.md") in found[0]["detail"]
    # The loader locates it the same way.
    loaded = WicaConfig.from_dict(config, base_dir=folder)
    assert loaded.agent.system_prompt_file == str((folder / "persona.md").resolve())
    # Without a folder, the working directory: not there.
    assert setup.check_setup(config)[0]["status"] == "missing"
    monkeypatch.chdir(folder)
    assert setup.check_setup(config)[0]["status"] == "ready"


def test_the_example_configs_find_their_persona_against_their_own_folder() -> None:
    folder = str(_ROOT / "examples" / "configs")
    for name, config in _examples().items():
        items = setup.check_setup(config, base_dir=folder)
        prompt = [i for i in items if i["id"] == "file:system_prompt_file"]
        assert [i["status"] for i in prompt] == ["ready"], name


def test_the_machine_functions_take_base_dir() -> None:
    config = _examples()["fake.json"]
    folder = str(_ROOT / "examples" / "configs")
    with pytest.raises(ValueError, match="file:system_prompt_file"):
        setup.prepare(config, "file:system_prompt_file", base_dir=folder)
    assert setup.options("models", config, base_dir=folder) == []
    assert setup.probe(config, base_dir=folder)["ok"] is True


def test_check_setup_reads_the_variable_the_config_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WICA_OPENAI_API_KEY", "sk-set")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    items = setup.check_setup(
        _block(provider="openai", api_key_env="WICA_OPENAI_API_KEY")
    )
    assert items[1]["id"] == "key:WICA_OPENAI_API_KEY"
    assert items[1]["status"] == "ready"


@pytest.mark.parametrize(
    ("config", "ids"),
    [
        (_block(provider="fake", api_key_env="IGNORED"), []),
        (
            _block(
                provider="openai-chat-completions-compatible", base_url="http://x/v1"
            ),
            ["extra:wica[openai-compatible]"],
        ),
        (
            _block(
                provider="openai-chat-completions-compatible",
                base_url="http://x/v1",
                api_key_env="SERVER_KEY",
            ),
            ["extra:wica[openai-compatible]", "key:SERVER_KEY"],
        ),
        (_block(provider="ollama"), []),
    ],
)
def test_check_setup_has_no_key_item_where_no_key_is_read(
    config: dict[str, Any], ids: list[str]
) -> None:
    assert [item["id"] for item in setup.check_setup(config)] == ids


_ITEM_KEYS = {"id", "title", "status", "detail", "preparable", "required"}


@pytest.mark.parametrize(
    "provider", ["anthropic", "openai-chat-completions-compatible", "ollama"]
)
def test_every_item_is_required_and_none_is_preparable(
    provider: str, tmp_path: Path
) -> None:
    config = _block(
        provider=provider,
        api_key_env="SOME_KEY",
        system_prompt=...,
        system_prompt_file=str(tmp_path / "persona.md"),
        **(
            {"base_url": "http://x/v1"}
            if provider == "openai-chat-completions-compatible"
            else {}
        ),
    )

    items = setup.check_setup(config)

    assert items
    for item in items:
        assert set(item) == _ITEM_KEYS
        assert (item["required"], item["preparable"]) == (True, False), item["id"]


@pytest.mark.parametrize(
    "item",
    [
        "extra:wica[anthropic]",
        "key:ANTHROPIC_API_KEY",
        "file:system_prompt_file",
        "weights:nothing",
    ],
)
def test_prepare_refuses_every_item_naming_it(item: str) -> None:
    config = _block(system_prompt=..., system_prompt_file="persona.md")
    with pytest.raises(ValueError, match=re.escape(repr(item))):
        setup.prepare(
            config, item, on_progress=lambda progress: pytest.fail(str(progress))
        )


def test_options_have_nothing_to_offer() -> None:
    assert setup.options("models", _block()) == []


# --- probe ------------------------------------------------------------------------------------


def test_probe_on_the_fake_is_ok_at_once() -> None:
    result = setup.probe(_examples()["fake.json"])

    assert result["ok"] is True
    assert result["duration_s"] < 1


@pytest.mark.parametrize(
    ("config", "words"),
    [
        (_block(api_key="sk-x"), "does not take 'api_key'"),
        (_block(api_key_env="WICA_SETUP_TEST_UNSET"), "no key under"),
    ],
)
def test_probe_reports_a_refused_config_or_a_missing_key_without_raising(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], words: str
) -> None:
    monkeypatch.delenv("WICA_SETUP_TEST_UNSET", raising=False)
    result = setup.probe(config)
    assert result["ok"] is False
    assert words in result["message"]


class _Models:
    """A stand-in for an SDK's `models` resource: knows some models, records each call."""

    def __init__(self, known: set[str], calls: list[str], *, refuse_key: bool) -> None:
        self.known, self.calls, self.refuse_key = known, calls, refuse_key

    def retrieve(self, model: str) -> object:
        self.calls.append(f"retrieve {model}")
        if self.refuse_key:
            raise _StatusError(401)
        if model not in self.known:
            raise _StatusError(404)
        return object()


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("sdk", ["anthropic", "openai"])
@pytest.mark.parametrize(
    ("model", "refuse_key", "ok", "words"),
    [
        ("known-model", False, True, "serves 'known-model'"),
        ("other-model", False, False, "has no model 'other-model'"),
        ("known-model", True, False, "refused the key (401)"),
    ],
)
def test_a_hosted_probe_retrieves_the_model_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
    sdk: str,
    model: str,
    refuse_key: bool,
    ok: bool,
    words: str,
) -> None:
    module = pytest.importorskip(sdk)
    calls: list[str] = []
    seen: dict[str, Any] = {}

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.models = _Models({"known-model"}, calls, refuse_key=refuse_key)

    monkeypatch.setattr(module, "Anthropic" if sdk == "anthropic" else "OpenAI", Client)
    result = setup.probe(
        _block(provider=sdk, model=model), secrets={f"{sdk.upper()}_API_KEY": "sk-test"}
    )

    assert (result["ok"], calls) == (ok, [f"retrieve {model}"])
    assert words in result["message"]
    assert seen["api_key"] == "sk-test"


def test_the_mistral_probe_finds_the_model_under_an_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "ministral-3b-2512", "aliases": ["ministral-3b-latest"]}
                ]
            },
        )

    real_get = httpx.get

    def get(url: str, **kwargs: Any) -> httpx.Response:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.get(url, headers=kwargs.get("headers"))

    monkeypatch.setattr(httpx, "get", get)
    try:
        found = setup.probe(
            _block(provider="mistral", model="ministral-3b-latest"),
            secrets={"MISTRAL_API_KEY": "k"},
        )
        missing = setup.probe(
            _block(provider="mistral", model="mistral-large"),
            secrets={"MISTRAL_API_KEY": "k"},
        )
    finally:
        monkeypatch.setattr(httpx, "get", real_get)

    assert found["ok"] is True
    assert missing["ok"] is False and "'mistral-large'" in missing["message"]
    assert [r.method for r in requests] == ["GET", "GET"]
    assert requests[0].headers["authorization"] == "Bearer k"


def test_the_compatible_probe_lists_the_server_s_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    seen: dict[str, Any] = {}

    class Model:
        def __init__(self, id: str) -> None:
            self.id = id

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)
            self.models = self

        def list(self) -> list[Model]:
            return [Model("loaded-model")]

    monkeypatch.setattr(openai, "OpenAI", Client)
    config = _block(
        provider="openai-chat-completions-compatible",
        model="something-else",
        base_url="http://127.0.0.1:8080/v1",
    )
    result = setup.probe(config)

    assert result["ok"] is True
    assert "does not list 'something-else'" in result["message"]
    assert (seen["base_url"], seen["api_key"]) == (
        "http://127.0.0.1:8080/v1",
        "not-needed",
    )


def test_probe_reports_a_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import wica.config

    def missing(name: str) -> None:
        raise ImportError(
            "provider 'gemini' needs wica[gemini], which is not installed"
        )

    monkeypatch.setattr(wica.setup, "import_provider", missing)
    result = setup.probe(_block(provider="gemini"), secrets={"GEMINI_API_KEY": "k"})

    assert result["ok"] is False
    assert "wica[gemini]" in result["message"]
    assert wica.config.import_provider is not missing


def test_probe_of_an_unknown_provider_only_builds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wica import agent

    built: list[dict[str, Any]] = []
    monkeypatch.setattr(
        agent, "init_chat_model", lambda *a, **k: built.append(k) or object()
    )
    result = setup.probe(_block(provider="ollama", model="llama3"))

    assert result["ok"] is True and len(built) == 1
    assert "nothing was sent" in result["message"]
