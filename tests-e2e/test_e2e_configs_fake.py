"""The e2e configs themselves, key-less: every committed config loads, states its output mode,
and the loader is as strict as `WicaConfig`. No provider is called. See specs/testing.md
("E2e configs")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wica.config import ConfigError

from support import PROVIDER_CONFIGS, load_e2e_config, output_as_tool

AGENT = {"provider": "fake", "model": "scripted", "system_prompt": "Test."}


def write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "e2e.test.config.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_every_committed_config_loads_with_its_output_mode(config_path: Path) -> None:
    config = load_e2e_config(config_path)
    assert config.output in ("tag", "tool")
    assert config.wica.agent.api_key_env  # a committed config never carries a key


def test_the_modes_the_live_set_covers() -> None:
    """Mistral runs in tool mode only; the Hub's model runs in both (specs/testing.md)."""
    modes = {p.stem: load_e2e_config(p).output for p in PROVIDER_CONFIGS}
    assert modes["e2e.mistral.config"] == "tool"
    assert modes["e2e.huggingface-hub.config"] == "tag"
    assert modes["e2e.huggingface-hub-tool.config"] == "tool"
    hub = [
        load_e2e_config(p).wica.agent
        for p in PROVIDER_CONFIGS
        if "huggingface" in p.stem
    ]
    assert len({(a.model, a.hf_provider) for a in hub}) == 1  # the same model, twice


def test_output_selects_the_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WICA_ROBOT_OUTPUT", raising=False)
    assert output_as_tool(write(tmp_path, {"output": "tool", "agent": AGENT}))
    assert not output_as_tool(write(tmp_path, {"output": "tag", "agent": AGENT}))


def test_the_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path, {"output": "tag", "agent": AGENT})
    monkeypatch.setenv("WICA_ROBOT_OUTPUT", "tool")
    assert output_as_tool(path)
    monkeypatch.setenv("WICA_ROBOT_OUTPUT", "both")
    with pytest.raises(ValueError, match="WICA_ROBOT_OUTPUT"):
        output_as_tool(path)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"agent": AGENT}, "output must be"),
        ({"output": "sink", "agent": AGENT}, "output must be"),
        ({"output": "tag", "agent": AGENT, "as_tool": True}, "unknown e2e config keys"),
        ({"output": "tag", "agent": {**AGENT, "output": "tool"}}, ""),
        ({"output": "tag"}, ""),
    ],
    ids=["no output", "bad output", "unknown key", "output inside agent", "no agent"],
)
def test_the_loader_is_strict(tmp_path: Path, data: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_e2e_config(write(tmp_path, data))
