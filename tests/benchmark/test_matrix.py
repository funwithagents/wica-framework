"""The matrix file (specs/benchmark.md, "The matrix"): defaults merge under each entry, the
selection flags pick by id glob and tag, the live gate is a reason string, faults name the entry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wica.benchmark import MatrixError, load_matrix, select_entries

DOC: dict[str, Any] = {
    "_comment": "test matrix",
    "agent_defaults": {"system_prompt_file": "prompts/persona.md"},
    "entries": [
        {
            "id": "fake",
            "agent": {"provider": "fake", "model": "scripted", "system_prompt": "hi"},
            "enabled": False,
            "tags": ["fixture"],
        },
        {
            "id": "openai/gpt-4o",
            "agent": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "T_OPENAI",
            },
            "tags": ["api", "default"],
            "notes": "the default",
            "body": "reachy",
        },
        {
            "id": "openai/gpt-4o-mini",
            "agent": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_env": "T_OPENAI",
            },
            "tags": ["api"],
            "requires_env": ["T_EXTRA"],
        },
    ],
}


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document))
    return path


def test_entries_load_with_defaults_merged_and_extra_keys_kept(tmp_path: Path):
    matrix = load_matrix(_write(tmp_path, DOC))
    assert matrix.comment == "test matrix"
    assert matrix.ids == ["fake", "openai/gpt-4o", "openai/gpt-4o-mini"]
    gpt = matrix.entry("openai/gpt-4o")
    assert gpt.agent == {
        "system_prompt_file": "prompts/persona.md",  # the default, as written
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "T_OPENAI",
    }
    assert gpt.base_dir == tmp_path  # a relative prompt path resolves from here
    # The fixture's inline prompt replaces the defaulted prompt file: the pair is either-or.
    assert "system_prompt_file" not in matrix.entry("fake").agent
    assert matrix.entry("fake").agent["system_prompt"] == "hi"
    assert gpt.tags == ("api", "default") and gpt.enabled and gpt.notes == "the default"
    assert gpt.extra == {"body": "reachy"}  # an application's own key
    assert not matrix.entry("fake").enabled
    with pytest.raises(KeyError):
        matrix.entry("nope")


def test_selection_by_id_glob_and_tag(tmp_path: Path):
    entries = load_matrix(_write(tmp_path, DOC)).entries
    ids = lambda selected: [e.id for e in selected]  # noqa: E731
    assert ids(select_entries(entries)) == [
        "fake",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]
    assert ids(select_entries(entries, only=["openai/*"])) == [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]
    assert ids(select_entries(entries, only=["openai/*"], exclude=["*-mini"])) == [
        "openai/gpt-4o"
    ]
    assert ids(select_entries(entries, tags=["default", "fixture"])) == [
        "fake",
        "openai/gpt-4o",
    ]
    assert ids(select_entries(entries, exclude_tags=["fixture"])) == [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]
    assert ids(select_entries(entries, only=["nothing"])) == []


def test_skip_reason_is_the_live_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    matrix = load_matrix(_write(tmp_path, DOC))
    monkeypatch.delenv("T_OPENAI", raising=False)
    monkeypatch.delenv("T_EXTRA", raising=False)
    assert matrix.entry("fake").skip_reason() is None  # no key variable: always runs
    assert (
        matrix.entry("openai/gpt-4o").skip_reason()
        == "environment variable 'T_OPENAI' is not set"
    )
    monkeypatch.setenv("T_OPENAI", "sk-test")
    assert matrix.entry("openai/gpt-4o").skip_reason() is None
    assert (
        matrix.entry("openai/gpt-4o-mini").skip_reason()
        == "environment variable 'T_EXTRA' is not set"
    )
    monkeypatch.setenv("T_EXTRA", "1")
    assert matrix.entry("openai/gpt-4o-mini").skip_reason() is None


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d["entries"][1].pop("id"), "entry 1: 'id' must be"),
        (lambda d: d["entries"][2].update(id="fake"), "entry 'fake': duplicate id"),
        (
            lambda d: d["entries"][1].update(agent="gpt"),
            "entry 'openai/gpt-4o': 'agent' must be an object",
        ),
        (
            lambda d: d["entries"][1].update(tags="api"),
            "'tags' must be a list of strings",
        ),
        (
            lambda d: d["entries"][1].update(enabled="yes"),
            "'enabled' must be a boolean",
        ),
        (lambda d: d.update(entries=[]), "'entries' must be a non-empty list"),
        (lambda d: d.update(configs=[]), "unknown key(s) ['configs']"),
    ],
)
def test_faults_name_the_file_and_the_entry(tmp_path: Path, mutate: Any, fragment: str):
    document = json.loads(json.dumps(DOC))
    mutate(document)
    path = _write(tmp_path, document)
    with pytest.raises(MatrixError) as excinfo:
        load_matrix(path)
    assert str(excinfo.value).startswith(str(path)) and fragment in str(excinfo.value)
