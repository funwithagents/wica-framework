"""The scenario file loader (specs/benchmark.md, "The scenario file"): a valid file round-trips,
every shape fault names the file and the step, and `TESTER_COMMANDS` is exactly what the scripted
tester registers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from wica import Wica
from wica.testing import (
    TESTER_COMMANDS,
    Requires,
    ScenarioError,
    TestRun,
    build_tester,
    load_scenario,
    load_scenarios,
)


def _step(command: str, /, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": command, "args": args}]}


VALID: dict[str, Any] = {
    "name": "greeting",
    "description": "says hi",
    "requires": {"inputs": ["speech_input"], "output_command": True},
    "timeout_s": 30,
    "checks": ["answers_each_input", "no_failed_reactions"],
    "params": {"motors": "enabled"},
    "script": [
        _step("set_speech_input", value="hi"),
        _step("expect_output", within=5),
        _step("finish", verdict="pass"),
    ],
}


def _write(directory: Path, name: str, document: dict[str, Any]) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_valid_scenario_round_trips(tmp_path: Path):
    scenario = load_scenario(_write(tmp_path, "greeting", VALID))
    assert scenario.name == "greeting"
    assert scenario.description == "says hi"
    assert scenario.requires == Requires(inputs=("speech_input",), output_command=True)
    assert scenario.timeout_s == 30.0
    assert scenario.checks == ("answers_each_input", "no_failed_reactions")
    assert scenario.params == {"motors": "enabled"}
    assert scenario.script == VALID["script"]  # verbatim: the fake model's script
    assert scenario.path == tmp_path / "greeting.json"


def test_defaults_when_optional_keys_are_absent(tmp_path: Path):
    minimal = {"name": "s", "script": [_step("finish", verdict="pass")]}
    scenario = load_scenario(_write(tmp_path, "s", minimal))
    assert scenario.requires == Requires()
    assert scenario.timeout_s == 60.0
    assert scenario.checks is None  # ⇒ every check the target satisfies
    assert scenario.params == {}


@pytest.mark.parametrize(
    ("mutate", "fragment", "step"),
    [
        (lambda d: d.update(name="other"), "must equal the file stem", None),
        (
            lambda d: d["script"].__setitem__(
                0,
                {
                    "tool_calls": [
                        {"name": "wait", "args": {"seconds": 1}},
                        {"name": "finish", "args": {}},
                    ]
                },
            ),
            "exactly one call",
            0,
        ),
        (
            lambda d: d["script"].__setitem__(1, _step("expect_miracle")),
            "not a tester Command",
            1,
        ),
        (
            lambda d: d["script"].__setitem__(
                2, {"tool_calls": [{"name": "finish", "args": "pass"}]}
            ),
            "'args' must be an object",
            2,
        ),
        (lambda d: d.update(checks=["no_such_check"]), "unknown check(s)", None),
        (lambda d: d.update(requires={"outputs": True}), "unknown key(s)", None),
        (lambda d: d.update(timeout_s=0), "positive number", None),
        (lambda d: d.update(script=[]), "non-empty list", None),
    ],
)
def test_shape_faults_name_the_file_and_the_step(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    fragment: str,
    step: int | None,
):
    document = json.loads(json.dumps(VALID))
    mutate(document)
    path = _write(tmp_path, "greeting", document)
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert message.startswith(str(path))
    assert fragment in message
    if step is not None:
        assert f"step {step}" in message


def test_load_scenarios_orders_by_name(tmp_path: Path):
    for name in ("zeta", "alpha", "mid"):
        _write(tmp_path, name, {**VALID, "name": name})
    (tmp_path / "notes.txt").write_text("ignored")
    assert [s.name for s in load_scenarios(tmp_path)] == ["alpha", "mid", "zeta"]


def test_tester_commands_match_what_the_scripted_tester_registers(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory([{"text": ""}])
    run = TestRun(sut)
    tester = build_tester(run, [])
    try:
        registered = {
            name
            for name in tester.agent.commands
            if name not in ("noop", "cancel_command") and not name.startswith("set_")
        }
        assert registered == TESTER_COMMANDS
        assert "set_prompt" in tester.agent.commands  # the one open-ended family
    finally:
        tester.close()
        run.close()
