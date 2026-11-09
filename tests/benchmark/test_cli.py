"""The shared `cli` (specs/benchmark.md, "The runner"): the matrix and its selection flags, the
disabled gate, the batch folder, `--result`, and argument errors."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from bench_support import NOOP, SAY_HI, FakeTarget

from wica.benchmark import Entry, cli


def _step(command: str, /, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": command, "args": args}]}


GREETING = {
    "name": "greeting",
    "requires": {"inputs": ["speech_input"], "output_command": True},
    "timeout_s": 15,
    "checks": ["answers_each_input", "no_noop_on_input"],
    "script": [
        _step("set_speech_input", value="hello"),
        _step("expect_output", within=3),
        _step("finish", verdict="pass"),
    ],
}

MATRIX = {
    "entries": [
        {
            "id": "good/a",
            "agent": {"provider": "fake", "model": "a", "script": SAY_HI},
            "tags": ["api", "default"],
        },
        {
            "id": "good/b",
            "agent": {"provider": "fake", "model": "b", "script": SAY_HI},
            "tags": ["api"],
        },
        {
            "id": "mute",
            "agent": {"provider": "fake", "model": "mute", "script": [NOOP]},
            "tags": ["api"],
        },
        {
            "id": "fixture",
            "agent": {"provider": "fake", "model": "fixture", "script": SAY_HI},
            "enabled": False,
            "tags": ["fixture"],
        },
    ]
}


def _factory(entry: Entry) -> FakeTarget:
    target = FakeTarget(entry.agent["script"], label=entry.id)
    target.agent_config = {k: v for k, v in entry.agent.items() if k != "script"}
    return target


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(MATRIX))
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "greeting.json").write_text(json.dumps(GREETING))
    return {
        "matrix": matrix,
        "scenarios": scenarios,
        "runs": tmp_path / "runs",
        "results": tmp_path / "results",
    }


def _cli(layout: dict[str, Path], argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = cli(
        argv,
        target_factory=_factory,
        default_matrix=layout["matrix"],
        default_scenarios=layout["scenarios"],
        default_runs=layout["runs"],
        default_results=layout["results"],
        prog="test-bench",
        out=out,
    )
    return code, out.getvalue()


def test_cli_runs_the_enabled_matrix_and_writes_the_batch(layout: dict[str, Path]):
    code, text = _cli(layout, ["--k", "2", "--out", str(layout["runs"] / "b1")])
    assert (
        code == 0
    )  # a benchmark compares, it does not gate — even with `mute` failing
    assert "good/a / greeting #1: ok" in text and "good/b / greeting #2: ok" in text
    assert (
        "mute / greeting #1: FAIL ['expect_output()', 'answers_each_input', 'no_noop_on_input']"
        in text
    )
    assert "fixture: skipped (disabled (use --include-disabled))" in text
    summary = json.loads((layout["runs"] / "b1" / "summary.json").read_text())
    by_label = {c["label"]: c for c in summary["cells"]}
    assert set(by_label) == {"good/a", "good/b", "mute"}
    assert by_label["good/a"]["pass_k"] == 1 and by_label["mute"]["pass_k"] == 0
    assert summary["skips"] == [
        {
            "label": "fixture",
            "scenario": "*",
            "reason": "disabled (use --include-disabled)",
        }
    ]
    assert (
        "| fixture | skipped: disabled"
        in (layout["runs"] / "b1" / "summary.md").read_text()
    )
    assert (
        layout["runs"] / "b1" / "good" / "a" / "greeting" / "2.md"
    ).exists()  # the id nests


def test_cli_selects_by_glob_and_tag_and_opts_a_disabled_row_in(
    layout: dict[str, Path],
):
    code, text = _cli(
        layout, ["--only", "good/*", "--exclude", "*/b", "--scenario", "greeting"]
    )
    assert code == 0 and "good/a / greeting #1: ok" in text
    assert "good/b" not in text and "mute" not in text
    (batch,) = list(layout["runs"].iterdir())  # <runs>/<YYYYMMDD-HHMM>
    assert len(batch.name) == 13 and batch.name[8] == "-"
    assert f"Batch written to {batch}" in text

    code, text = _cli(
        layout,
        ["--tags", "fixture", "--include-disabled", "--out", str(layout["runs"] / "f")],
    )
    assert code == 0 and "fixture / greeting #1: ok" in text and "skipped" not in text

    code, text = _cli(
        layout, ["--exclude-tags", "api", "--out", str(layout["runs"] / "g")]
    )
    assert code == 0 and "fixture: skipped" in text and "greeting #1" not in text


def test_cli_keeps_a_result_on_request(layout: dict[str, Path]):
    code, text = _cli(layout, ["--tags", "default", "--result", "baseline"])
    assert code == 0
    (result,) = list(layout["results"].iterdir())
    assert result.name.endswith("_baseline")
    assert (result / "description.md").exists() and (result / "runs.json").exists()
    assert "- `good/a`: fake / a" in (result / "description.md").read_text()
    assert "finish its description.md and findings.md" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["--only", "nope"],
        ["--scenario", "nope"],
        ["--k", "0"],
        ["--matrix", "missing.json"],
        ["--scenarios", "missing"],
    ],
)
def test_cli_rejects_bad_arguments(layout: dict[str, Path], argv: list[str]):
    with pytest.raises(SystemExit) as excinfo:
        _cli(layout, argv)
    assert excinfo.value.code == 2
