"""The scenario file: a fixed user as a scripted-tester script plus the checks to run over the
result. Implements specs/benchmark.md, "The scenario file".

A scenario is one JSON file named by its stem. Its `script` is **verbatim** the fake model's
script (one `tool_calls` list per step, each exactly one call to a tester Command), so a recorded
trace drops in unchanged; `requires` says what a target must expose for the scenario to apply, and
`checks` names the primer-level (or application) checks to run over the finished timeline. The
loader validates everything it can without a target: shape, Command names, check names against a
registry — every error names the file and the step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wica.testing.checks import Check

SET_PREFIX = "set_"

# The tester's fixed Command set (specs/wica-tester.md, "The shared Command set"); `set_<key>` is
# the one open-ended family, one per settable SUT Input. `tests/testing/test_scenario.py` pins
# this set against what `build_tester` actually registers, so the two cannot drift.
TESTER_COMMANDS = frozenset(
    {
        "wait",
        "expect_action",
        "expect_output",
        "expect_text",
        "expect_no_action",
        "expect_idle",
        "expect_world",
        "expect_latency",
        "note",
        "report_issue",
        "finish",
    }
)

_TOP_KEYS = {
    "name",
    "description",
    "requires",
    "timeout_s",
    "checks",
    "params",
    "script",
}
_REQUIRES_KEYS = {"inputs", "commands", "output_command"}
_DEFAULT_TIMEOUT_S = 60.0


class ScenarioError(ValueError):
    """A scenario file that does not load: the message names the file (and the step)."""


@dataclass(frozen=True)
class Requires:
    """What a target must expose for a scenario (or a check) to apply."""

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
        """The requirements the given target surface does not satisfy, as reasons; empty when
        every one holds."""
        reasons = [f"input {key!r}" for key in self.inputs if key not in inputs]
        reasons += [
            f"command {name!r}" for name in self.commands if name not in commands
        ]
        if self.output_command and output_command_name is None:
            reasons.append("an output Command")
        return reasons


@dataclass(frozen=True)
class Scenario:
    """One loaded scenario file. `checks` is None when the file names none: the runner then applies
    every registered check whose own `requires` the target satisfies."""

    name: str
    description: str = ""
    requires: Requires = field(default_factory=Requires)
    timeout_s: float = _DEFAULT_TIMEOUT_S
    checks: tuple[str, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    script: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None


def load_scenario(
    path: str | Path, *, registry: Mapping[str, Check] | None = None
) -> Scenario:
    """Load and validate one scenario file. Check names resolve against *registry* (the built-in
    `CHECKS` when None); an unknown one is a load error, as is any shape fault."""
    path = Path(path)
    if registry is None:
        from wica.testing.checks import CHECKS

        registry = CHECKS

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

    raw_requires = data.get("requires", {})
    if not isinstance(raw_requires, dict):
        raise fail("'requires' must be an object")
    unknown = set(raw_requires) - _REQUIRES_KEYS
    if unknown:
        raise fail(f"'requires' has unknown key(s) {sorted(unknown)}")
    requires = Requires(
        inputs=tuple(
            _str_list(raw_requires.get("inputs", []), "requires.inputs", fail)
        ),
        commands=tuple(
            _str_list(raw_requires.get("commands", []), "requires.commands", fail)
        ),
        output_command=_bool(
            raw_requires.get("output_command", False), "requires.output_command", fail
        ),
    )

    timeout_s = data.get("timeout_s", _DEFAULT_TIMEOUT_S)
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or timeout_s <= 0
    ):
        raise fail("'timeout_s' must be a positive number")

    checks: tuple[str, ...] | None = None
    if "checks" in data:
        names = _str_list(data["checks"], "checks", fail)
        unknown_checks = [n for n in names if n not in registry]
        if unknown_checks:
            raise fail(f"unknown check(s) {unknown_checks}; known: {sorted(registry)}")
        checks = tuple(names)

    params = data.get("params", {})
    if not isinstance(params, dict):
        raise fail("'params' must be an object")

    script = data.get("script")
    if not isinstance(script, list) or not script:
        raise fail("'script' must be a non-empty list of steps")
    for index, step in enumerate(script):
        _validate_step(step, index, fail)

    return Scenario(
        name=name,
        description=description,
        requires=requires,
        timeout_s=float(timeout_s),
        checks=checks,
        params=dict(params),
        script=[dict(step) for step in script],
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


def is_tester_command(name: str) -> bool:
    return name in TESTER_COMMANDS or (
        name.startswith(SET_PREFIX) and len(name) > len(SET_PREFIX)
    )


def _validate_step(step: Any, index: int, fail: Any) -> None:
    if not isinstance(step, dict):
        raise fail("a step must be an object", index)
    if set(step) != {"tool_calls"}:
        raise fail("a step has exactly one key, 'tool_calls'", index)
    calls = step["tool_calls"]
    if not isinstance(calls, list) or len(calls) != 1:
        raise fail("'tool_calls' must hold exactly one call", index)
    call = calls[0]
    if not isinstance(call, dict) or set(call) - {"name", "args"} or "name" not in call:
        raise fail("a call is an object with 'name' and optional 'args'", index)
    name = call["name"]
    if not isinstance(name, str) or not is_tester_command(name):
        raise fail(
            f"{name!r} is not a tester Command (one of {sorted(TESTER_COMMANDS)} or "
            f"'{SET_PREFIX}<key>')",
            index,
        )
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise fail("'args' must be an object", index)


def _str_list(value: Any, what: str, fail: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise fail(f"'{what}' must be a list of strings")
    return list(value)


def _bool(value: Any, what: str, fail: Any) -> bool:
    if not isinstance(value, bool):
        raise fail(f"'{what}' must be a boolean")
    return value
