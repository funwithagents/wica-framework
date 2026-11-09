"""Scenarios (specs/wica-tester.md, "Scenarios"): the four step kinds load and validate, the
requirements are derived from the steps, the steps translate into the scripted tester's script,
and `run_scenario` plays them on a fake SUT — completed (checks judged), failed on an expect,
stalled on a wait."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sut_support import ADD_THEN_TEXT

from wica import Wica
from wica.testing import (
    CheckSpec,
    CheckStep,
    ExpectStep,
    InputStep,
    Requires,
    Scenario,
    ScenarioError,
    TestRun,
    WaitStep,
    load_scenario,
    load_scenarios,
    run_scenario,
    to_script,
)

VALID: dict[str, Any] = {
    "name": "dance",
    "description": "a dance, stopped",
    "checks": ["no_failed_reactions"],
    "timeout_s": 90,
    "params": {"motors": "enabled"},
    "steps": [
        {"input": {"speech_input": "Hi!"}},
        {"expect": {"output": {"contains": "hi"}}, "within": 10},
        {"wait": "speaking"},
        {"input": {"speech_input": "Dance!"}},
        {"check": {"acts_when_asked": "dance"}},
        {"wait": {"running": "dance"}, "within": 20},
        {"wait": {"seconds": 2}},
        {"input": {"speech_input": "Stop!"}},
        {"check": "cancels_when_asked"},
        {"expect": {"no_action": "dance"}, "for": 1},
        {"wait": "idle"},
    ],
}


def _write(directory: Path, name: str, document: dict[str, Any]) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _copy(**changes: Any) -> dict[str, Any]:
    return {**json.loads(json.dumps(VALID)), **changes}


def test_a_valid_scenario_loads_into_typed_steps(tmp_path: Path):
    scenario = load_scenario(_write(tmp_path, "dance", VALID))
    assert scenario.name == "dance" and scenario.timeout_s == 90.0
    assert scenario.params == {"motors": "enabled"}
    assert scenario.checks == (CheckSpec("no_failed_reactions"),)
    assert scenario.steps[:6] == (
        InputStep("speech_input", "Hi!"),
        ExpectStep("output", {"contains": "hi"}, within=10.0),
        WaitStep("speaking"),
        InputStep("speech_input", "Dance!"),
        CheckStep(CheckSpec("acts_when_asked", "dance")),
        WaitStep({"running": "dance"}, within=20.0),
    )
    assert scenario.steps[9] == ExpectStep("no_action", "dance", for_=1.0)
    assert scenario.check_names == (
        "no_failed_reactions",
        "acts_when_asked",
        "cancels_when_asked",
    )


def test_requirements_are_derived_from_the_steps(tmp_path: Path):
    scenario = load_scenario(_write(tmp_path, "dance", VALID))
    # cancel_command is WICA's own and never required; dance comes from the wait, the expect and
    # the check; the output Command from the output expect and the "speaking" wait.
    assert scenario.requires == Requires(
        inputs=("speech_input",), commands=("dance",), output_command=True
    )


@pytest.mark.parametrize(
    ("mutate", "fragment", "step"),
    [
        (lambda d: d.update(name="other"), "must equal the file stem", None),
        (lambda d: d.update(steps=[]), "non-empty list", None),
        (lambda d: d.update(script=[]), "unknown key(s) ['script']", None),
        (
            lambda d: d["steps"].__setitem__(0, {"input": {"a": 1, "b": 2}}),
            "exactly one Input key",
            0,
        ),
        (
            lambda d: d["steps"].__setitem__(0, {"input": {"a": 1}, "wait": "idle"}),
            "exactly one of",
            0,
        ),
        (lambda d: d["steps"].__setitem__(2, {"wait": "soon"}), "'wait' is", 2),
        (
            lambda d: d["steps"].__setitem__(1, {"expect": {"miracle": {}}}),
            "exactly one of",
            1,
        ),
        (
            lambda d: d["steps"].__setitem__(1, {"expect": {"no_action": "say"}}),
            'needs "for"',
            1,
        ),
        (
            lambda d: d["steps"].__setitem__(1, {"expect": {"output": {}}, "for": 2}),
            "only a 'no_action'",
            1,
        ),
        (
            lambda d: d["steps"].__setitem__(1, {"expect": {"world": {"key": "k"}}}),
            "'world'",
            1,
        ),
        (
            lambda d: d["steps"].__setitem__(4, {"check": "no_such_check"}),
            "unknown check",
            4,
        ),
        (
            lambda d: d["steps"].__setitem__(4, {"check": {"acts_when_asked": 3}}),
            "acts_when_asked",
            4,
        ),
        (
            lambda d: d["steps"].__setitem__(8, {"check": {"cancels_when_asked": "x"}}),
            "takes no argument",
            8,
        ),
        (
            lambda d: d["steps"].insert(0, {"check": "cancels_when_asked"}),
            "must follow an input",
            0,
        ),
        (lambda d: d.update(checks=["cancels_when_asked"]), "write it as a step", None),
        (
            lambda d: d["steps"].__setitem__(2, {"wait": "idle", "within": 0}),
            "positive number",
            2,
        ),
    ],
)
def test_faults_name_the_file_and_the_step(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    fragment: str,
    step: int | None,
):
    document = _copy()
    mutate(document)
    path = _write(tmp_path, "dance", document)
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(path)
    message = str(excinfo.value)
    assert message.startswith(str(path)) and fragment in message
    if step is not None:
        assert f"step {step}" in message


def test_load_scenarios_orders_by_name(tmp_path: Path):
    for name in ("zeta", "alpha", "mid"):
        _write(tmp_path, name, _copy(name=name))
    (tmp_path / "notes.txt").write_text("ignored")
    assert [s.name for s in load_scenarios(tmp_path)] == ["alpha", "mid", "zeta"]


def test_to_script_plays_inputs_waits_and_expects_then_finishes(tmp_path: Path):
    scenario = load_scenario(_write(tmp_path, "dance", VALID))
    names = [step["tool_calls"][0]["name"] for step in to_script(scenario)]
    assert names == [
        "set_speech_input",
        "expect_output",
        "wait_for",
        "set_speech_input",
        "wait_for",
        "wait_for",
        "set_speech_input",
        "expect_no_action",
        "wait_for",
        "finish",
    ]  # check steps are judged after the run, not played
    assert to_script(scenario)[1]["tool_calls"][0]["args"] == {
        "contains": "hi",
        "within": 10.0,
    }


# --- run_scenario on a fake SUT (`prompt` Input, `add` Command) -----------------------------


def _scenario(*steps: Any, checks: tuple[CheckSpec, ...] = ()) -> Scenario:
    return Scenario(name="s", steps=tuple(steps), checks=checks, timeout_s=30)


def _play(
    sut_factory: Callable[..., Wica], script: list[dict[str, Any]], scenario: Scenario
):
    sut = sut_factory(script)
    sut.start()
    run = TestRun(sut, scenario="s")
    try:
        return run_scenario(run, scenario)
    finally:
        run.close()
        sut.close()


def test_a_completed_run_judges_its_checks_over_their_windows(
    sut_factory: Callable[..., Wica],
):
    report = _play(
        sut_factory,
        ADD_THEN_TEXT,
        _scenario(
            InputStep("prompt", "add 2 and 2"),
            CheckStep(CheckSpec("acts_when_asked", "add")),
            ExpectStep("action", "add", within=5),
            WaitStep("idle", within=5),
            checks=(CheckSpec("no_failed_reactions"),),
        ),
    )
    assert report.outcome == "completed" and report.stopped_at is None
    assert report.ok and report.verdict == "pass"
    checks = report.checks()
    assert [f.expectation for f in checks] == ["no_failed_reactions", "acts_when_asked"]
    assert checks[0].detail.startswith("whole run: ")
    assert checks[1].detail.startswith("on input 1 (prompt='add 2 and 2'): add(")


def test_a_failed_expect_stops_the_run_and_skips_the_checks(
    sut_factory: Callable[..., Wica],
):
    report = _play(
        sut_factory,
        [{"text": "no tools today"}],
        _scenario(
            InputStep("prompt", "add 2 and 2"),
            CheckStep(CheckSpec("acts_when_asked", "add")),
            ExpectStep("action", "add", within=0.5),
            InputStep("prompt", "never written"),
            checks=(CheckSpec("no_failed_reactions"),),
        ),
    )
    assert report.outcome == "failed" and not report.ok
    assert report.stopped_at == "step 2: expect_action(name='add')"
    assert report.checks() == []  # a stopped run is not judged
    writes = [e for e in report.timeline if e.kind == "action" and e.name == "set"]
    assert len(writes) == 1  # nothing was written past the stop


def test_a_wait_that_times_out_stalls_the_run(sut_factory: Callable[..., Wica]):
    report = _play(
        sut_factory,
        [{"text": "no tools today"}],
        _scenario(
            InputStep("prompt", "add 2 and 2"),
            WaitStep({"running": "add"}, within=0.5),
        ),
    )
    assert report.outcome == "stalled" and not report.ok
    assert report.stopped_at is not None and report.stopped_at.startswith(
        "step 1: wait_for("
    )
    stall = [f for f in report.findings if f.kind == "wait"][0]
    assert "add has not started since the last input" in stall.detail
