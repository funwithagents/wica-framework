"""Scenarios: a fixed user written as data — a flat list of `input`, `wait`, `expect` and `check`
steps. Implements specs/wica-tester.md, "Scenarios".

- `input` writes one SUT Input.
- `wait` paces the conversation ("idle", "speaking", {"running": name}, {"seconds": n}); a timeout
  stops the run as **failed**, like an expect.
- `expect` asserts; a miss stops the run as **failed**.
- `check` is a rule judged after a completed run, over the window of the input it follows (from
  that input to the next); top-level `checks` cover the whole run.

`run_scenario(run, scenario)` plays the steps on a `TestRun` directly (the pytest and benchmark
path); `to_script(scenario)` translates them into the scripted tester's fake-model script.
Requirements are derived from the steps, not written.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wica.testing.checks import (
    CHECKS,
    CANCEL,
    NOOP,
    Check,
    CheckContext,
    CheckSpec,
    command_arg,
    input_windows,
    run_check,
)
from wica.testing.harness import TestRun
from wica.testing.report import Finding, Report
from wica.testing.tester import SET_PREFIX, set_command_name

_TOP_KEYS = {"name", "description", "checks", "steps", "timeout_s", "params"}
_DEFAULT_TIMEOUT_S = 300.0
_WAIT_WITHIN_S = 60.0
_EXPECT_WITHIN_S = 30.0
_NATIVE_COMMANDS = frozenset({NOOP, CANCEL})
_WAIT_WORDS = frozenset({"idle", "speaking"})
_EXPECT_KINDS = ("output", "action", "text", "no_action", "world", "latency")
_EXPECT_EXTRAS = {"action": {"where", "state", "result_contains"}}
_OUTPUT_CHECKS = frozenset({"answers_each_input", "quiet_on_background_change"})


class ScenarioError(ValueError):
    """A scenario file that does not load: the message names the file (and the step)."""


@dataclass(frozen=True)
class Requires:
    """What a SUT must expose for a scenario to run."""

    inputs: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    output_command: bool = False

    def missing(
        self,
        *,
        inputs: Mapping[str, Any] | set[str] | frozenset[str],
        commands: Mapping[str, Any] | set[str] | frozenset[str],
        output_command_name: str | None,
    ) -> list[str]:
        """The requirements the given SUT surface does not satisfy, as reasons."""
        reasons = [f"input {key!r}" for key in self.inputs if key not in inputs]
        reasons += [
            f"command {name!r}" for name in self.commands if name not in commands
        ]
        if self.output_command and output_command_name is None:
            reasons.append("an output Command")
        return reasons


@dataclass(frozen=True)
class InputStep:
    key: str
    value: Any


@dataclass(frozen=True)
class WaitStep:
    condition: str | dict[str, Any]
    within: float = _WAIT_WITHIN_S


@dataclass(frozen=True)
class ExpectStep:
    """`what` is one of output, action, text, no_action, world, latency; `spec` its value, `extra`
    the action's where / state / result_contains."""

    what: str
    spec: Any
    within: float = _EXPECT_WITHIN_S
    for_: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckStep:
    spec: CheckSpec


Step = InputStep | WaitStep | ExpectStep | CheckStep


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[Step, ...]
    description: str = ""
    checks: tuple[CheckSpec, ...] = ()  # top-level: judged over the whole run
    timeout_s: float = _DEFAULT_TIMEOUT_S
    params: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def requires(self) -> Requires:
        """Derived from the steps: the Inputs written, the Commands named (waits, expects, check
        arguments; WICA's native `noop` / `cancel_command` excepted), and whether an output Command
        is needed (a "speaking" wait, an output expect, a check that reads it)."""
        inputs: list[str] = []
        commands: list[str] = []
        output = False

        def command(name: str) -> None:
            if name not in _NATIVE_COMMANDS and name not in commands:
                commands.append(name)

        for step in self.steps:
            if isinstance(step, InputStep) and step.key not in inputs:
                inputs.append(step.key)
            elif isinstance(step, WaitStep):
                if step.condition == "speaking":
                    output = True
                elif isinstance(step.condition, dict) and "running" in step.condition:
                    command(step.condition["running"])
            elif isinstance(step, ExpectStep):
                if step.what == "output":
                    output = True
                elif step.what in ("action", "no_action"):
                    command(step.spec)
            elif isinstance(step, CheckStep):
                if step.spec.name == "acts_when_asked":
                    command(command_arg(step.spec.arg)[0])
                output = output or step.spec.name in _OUTPUT_CHECKS
        output = output or any(c.name in _OUTPUT_CHECKS for c in self.checks)
        return Requires(tuple(inputs), tuple(commands), output)

    @property
    def check_names(self) -> tuple[str, ...]:
        """Every check the scenario applies, top-level first, then in step order."""
        on_steps = [s.spec.name for s in self.steps if isinstance(s, CheckStep)]
        return tuple(c.name for c in self.checks) + tuple(on_steps)


# --- loading -------------------------------------------------------------------------------------


def load_scenario(
    path: str | Path, *, registry: Mapping[str, Check] | None = None
) -> Scenario:
    """Load and validate one scenario file; check names resolve against *registry* (the built-in
    `CHECKS` when None). Every fault names the file, and the step index when it has one."""
    path = Path(path)
    registry = registry if registry is not None else CHECKS

    def fail(what: str, step: int | None = None) -> ScenarioError:
        where = f"{path}: step {step}: " if step is not None else f"{path}: "
        return ScenarioError(where + what)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise fail(f"could not read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise fail(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise fail("the document must be a JSON object")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise fail(f"unknown key(s) {sorted(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise fail("'name' must be a non-empty string")
    if name != path.stem:
        raise fail(f"'name' {name!r} must equal the file stem {path.stem!r}")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise fail("'description' must be a string")
    timeout_s = _positive(
        data.get("timeout_s", _DEFAULT_TIMEOUT_S), "'timeout_s'", fail
    )
    params = data.get("params", {})
    if not isinstance(params, dict):
        raise fail("'params' must be an object")

    raw_checks = data.get("checks", [])
    if not isinstance(raw_checks, list):
        raise fail("'checks' must be a list")
    checks = tuple(
        _check_spec(item, registry, top_level=True, fail=lambda m: fail(m))
        for item in raw_checks
    )

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise fail("'steps' must be a non-empty list")
    steps: list[Step] = []
    seen_input = False
    for index, raw in enumerate(raw_steps):
        step = _step(raw, registry, lambda m, i=index: fail(m, i))
        if isinstance(step, InputStep):
            seen_input = True
        elif isinstance(step, CheckStep) and not seen_input:
            raise fail(
                "a check step must follow an input (top-level 'checks' cover the run)",
                index,
            )
        steps.append(step)

    return Scenario(
        name=name,
        steps=tuple(steps),
        description=description,
        checks=checks,
        timeout_s=timeout_s,
        params=dict(params),
        path=path,
    )


def load_scenarios(
    directory: str | Path, *, registry: Mapping[str, Check] | None = None
) -> list[Scenario]:
    """Every `*.json` of *directory*, loaded, in name order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ScenarioError(f"{directory}: not a directory")
    return [
        load_scenario(path, registry=registry)
        for path in sorted(directory.glob("*.json"), key=lambda p: p.name)
    ]


def _step(raw: Any, registry: Mapping[str, Check], fail: Any) -> Step:
    if not isinstance(raw, dict):
        raise fail("a step must be an object")
    kinds = [k for k in ("input", "wait", "expect", "check") if k in raw]
    if len(kinds) != 1:
        raise fail("a step has exactly one of 'input', 'wait', 'expect', 'check'")
    kind = kinds[0]
    if kind == "input":
        _only(raw, {"input"}, fail)
        value = raw["input"]
        if not isinstance(value, dict) or len(value) != 1:
            raise fail("'input' must be an object with exactly one Input key")
        ((key, val),) = value.items()
        return InputStep(str(key), val)
    if kind == "wait":
        _only(raw, {"wait", "within"}, fail)
        condition = raw["wait"]
        valid = (isinstance(condition, str) and condition in _WAIT_WORDS) or (
            isinstance(condition, dict)
            and len(condition) == 1
            and (
                isinstance(condition.get("running"), str)
                or _is_number(condition.get("seconds"))
            )
        )
        if not valid:
            raise fail(
                '\'wait\' is "idle", "speaking", {"running": <Command>} or '
                '{"seconds": <n>}'
            )
        return WaitStep(
            condition, _positive(raw.get("within", _WAIT_WITHIN_S), "'within'", fail)
        )
    if kind == "expect":
        _only(raw, {"expect", "within", "for"}, fail)
        return _expect(raw, fail)
    _only(raw, {"check"}, fail)
    return CheckStep(_check_spec(raw["check"], registry, top_level=False, fail=fail))


def _expect(raw: dict[str, Any], fail: Any) -> ExpectStep:
    body = raw["expect"]
    if not isinstance(body, dict):
        raise fail("'expect' must be an object")
    present = [k for k in _EXPECT_KINDS if k in body]
    if len(present) != 1:
        raise fail(f"'expect' names exactly one of {list(_EXPECT_KINDS)}")
    what = present[0]
    extras = {k: v for k, v in body.items() if k != what}
    allowed = _EXPECT_EXTRAS.get(what, set())
    if set(extras) - allowed:
        raise fail(f"'expect' {what!r} takes no {sorted(set(extras) - allowed)}")
    spec = body[what]
    if what in ("action", "no_action") and not isinstance(spec, str):
        raise fail(f"'expect' {what!r} is a Command name")
    if what in ("output", "text") and not (
        isinstance(spec, dict) and not set(spec) - {"contains", "arg"}
    ):
        raise fail(f"'expect' {what!r} is an object with optional 'contains' / 'arg'")
    if what == "world" and not (
        isinstance(spec, dict) and set(spec) == {"key", "equals"}
    ):
        raise fail("'expect' 'world' is {\"key\": …, \"equals\": …}")
    if what == "latency" and not (
        isinstance(spec, dict) and set(spec) == {"metric", "p95_below"}
    ):
        raise fail("'expect' 'latency' is {\"metric\": …, \"p95_below\": …}")
    for_ = raw.get("for")
    if what == "no_action":
        if for_ is None:
            raise fail("'expect' 'no_action' needs \"for\" (seconds)")
        for_ = _positive(for_, "'for'", fail)
    elif for_ is not None:
        raise fail("only a 'no_action' expect takes \"for\"")
    within = _positive(raw.get("within", _EXPECT_WITHIN_S), "'within'", fail)
    return ExpectStep(what, spec, within, for_, extras)


def _check_spec(
    raw: Any, registry: Mapping[str, Check], *, top_level: bool, fail: Any
) -> CheckSpec:
    if isinstance(raw, str):
        name, arg = raw, None
    elif isinstance(raw, dict) and len(raw) == 1:
        ((name, arg),) = raw.items()
    else:
        raise fail("a check is a name or {<name>: <argument>}")
    if name not in registry:
        raise fail(f"unknown check {name!r}; known: {sorted(registry)}")
    check = registry[name]
    if top_level and check.placement == "step":
        raise fail(
            f"check {name!r} concerns one input: write it as a step after that input"
        )
    if check.arg == "none" and arg is not None:
        raise fail(f"check {name!r} takes no argument")
    if check.arg == "command":
        try:
            command_arg(arg)
        except ValueError as exc:
            raise fail(f"check {name!r}: {exc}") from exc
    return CheckSpec(name, arg)


def _only(raw: dict[str, Any], allowed: set[str], fail: Any) -> None:
    extra = set(raw) - allowed
    if extra:
        raise fail(f"unknown key(s) {sorted(extra)} in this step")


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    )


def _positive(value: Any, what: str, fail: Any) -> float:
    if not _is_number(value) or value <= 0:
        raise fail(f"{what} must be a positive number")
    return float(value)


# --- running -------------------------------------------------------------------------------------


def run_scenario(
    run: TestRun, scenario: Scenario, *, registry: Mapping[str, Check] | None = None
) -> Report:
    """Play *scenario*'s steps on *run* (the SUT already started), stop on a failed expect or a
    timed-out wait, then judge the checks if the run completed. Returns the report."""
    deadline = time.monotonic() + scenario.timeout_s
    for index, step in enumerate(scenario.steps):
        if time.monotonic() > deadline:
            run.stop(f"step {index}: scenario timeout {scenario.timeout_s}s")
            break
        if isinstance(step, InputStep):
            run.set(step.key, step.value)
        elif isinstance(step, WaitStep):
            finding = run.wait_for(step.condition, within=step.within)
            if not finding.ok:
                run.stop(f"step {index}: {finding.expectation}")
                break
        elif isinstance(step, ExpectStep):
            finding = _play_expect(run, step)
            if not finding.ok:
                run.stop(f"step {index}: {finding.expectation}")
                break
    else:
        run.finish("pass")
    if run.outcome == "completed":
        evaluate_checks(run, scenario, registry=registry)
    return run.report()


def _play_expect(run: TestRun, step: ExpectStep) -> Finding:
    spec, within = step.spec, step.within
    if step.what == "output":
        return run.expect_output(
            contains=spec.get("contains"), arg=spec.get("arg"), within=within
        )
    if step.what == "action":
        return run.expect_action(
            spec,
            where=step.extra.get("where"),
            state=step.extra.get("state"),
            result_contains=step.extra.get("result_contains"),
            within=within,
        )
    if step.what == "text":
        return run.expect_text(contains=spec.get("contains"), within=within)
    if step.what == "no_action":
        return run.expect_no_action(spec, for_=step.for_ or 0.0)
    if step.what == "world":
        return run.expect_world(spec["key"], equals=spec["equals"], within=within)
    return run.expect_latency(spec["metric"], p95_below=spec["p95_below"])


def check_context(run: TestRun, scenario: Scenario) -> CheckContext:
    """What the checks (and the measures) need beyond the timeline, read from the Probe."""
    inputs = run.probe.inputs
    timeline = run.timeline()
    return CheckContext(
        output_command_name=run.probe.output_command_name,
        inputs=tuple(inputs),
        triggering_inputs=tuple(k for k, c in inputs.items() if c.triggers_llm_call),
        commands=tuple(run.probe.commands),
        scenario_inputs=scenario.requires.inputs,
        origin=timeline[0].t if timeline else 0.0,
    )


def evaluate_checks(
    run: TestRun, scenario: Scenario, *, registry: Mapping[str, Check] | None = None
) -> list[Finding]:
    """Judge the scenario's checks over the finished timeline and add them to the run's findings:
    the top-level ones over the whole run, each step check over the window of the input it
    follows."""
    registry = registry if registry is not None else CHECKS
    timeline = run.timeline()
    windows = input_windows(timeline)
    context = check_context(run, scenario)
    findings = [
        run_check(registry[spec.name], spec, timeline, context, label="whole run")
        for spec in scenario.checks
    ]
    ordinal = 0
    current: InputStep | None = None
    for step in scenario.steps:
        if isinstance(step, InputStep):
            ordinal += 1
            current = step
        elif isinstance(step, CheckStep) and current is not None:
            start, end = windows[ordinal - 1] if ordinal - 1 < len(windows) else (0, 0)
            label = f"on input {ordinal} ({current.key}={current.value!r})"
            findings.append(
                run_check(
                    registry[step.spec.name],
                    step.spec,
                    timeline[start:end],
                    context,
                    label=label,
                )
            )
    for finding in findings:
        run.add_finding(finding)
    return findings


def to_script(scenario: Scenario) -> list[dict[str, Any]]:
    """The scripted tester's fake-model script for *scenario*: one tester Command per input, wait
    and expect step, then `finish`. Check steps are not played — `evaluate_checks` judges them
    after the run."""

    def call(command: str, /, **args: Any) -> dict[str, Any]:
        return {"tool_calls": [{"name": command, "args": args}]}

    script: list[dict[str, Any]] = []
    for step in scenario.steps:
        if isinstance(step, InputStep):
            script.append(call(set_command_name(step.key), value=step.value))
        elif isinstance(step, WaitStep):
            script.append(
                call("wait_for", condition=step.condition, within=step.within)
            )
        elif isinstance(step, ExpectStep):
            script.append(_expect_call(step, call))
    script.append(call("finish", verdict="pass"))
    return script


def _expect_call(step: ExpectStep, call: Any) -> dict[str, Any]:
    spec, within = step.spec, step.within
    if step.what == "output":
        return call("expect_output", within=within, **spec)
    if step.what == "action":
        return call("expect_action", name=spec, within=within, **step.extra)
    if step.what == "text":
        return call("expect_text", within=within, **spec)
    if step.what == "no_action":
        return call("expect_no_action", name=spec, for_=step.for_)
    if step.what == "world":
        return call(
            "expect_world", key=spec["key"], equals=spec["equals"], within=within
        )
    return call("expect_latency", metric=spec["metric"], p95_below=spec["p95_below"])


__all__ = [
    "CheckStep",
    "ExpectStep",
    "InputStep",
    "Requires",
    "SET_PREFIX",
    "Scenario",
    "ScenarioError",
    "Step",
    "WaitStep",
    "check_context",
    "evaluate_checks",
    "load_scenario",
    "load_scenarios",
    "run_scenario",
    "to_script",
]
