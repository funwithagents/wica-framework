"""The benchmark runner: a `Target` opened per run, the scenario's script driven through
`wica.testing`, the checks and measures applied, one report written per run. Implements
specs/benchmark.md, "The target" and "The runner".

The runner never builds a `Wica` itself: the application's `Target` does, exactly as its entry
point would, so what is benchmarked is the real system. `open` is called **per run** — every
repetition is a fresh system — and a `Skip` (key unset, extra absent) is a summary entry, never
an error. Runs are sequential: parallel runs would confound the latency measures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wica.benchmark.matrix import Entry
from wica.benchmark.summary import Summary, aggregate, git_commit
from wica.instrumentation import now
from wica.testing.checks import (
    Check,
    CheckContext,
    apply_checks,
    build_registry,
    measures,
)
from wica.testing.harness import TestRun
from wica.testing.report import Report, metric_samples
from wica.testing.scenario import Scenario
from wica.testing.tester import run_script
from wica.wica import Wica


class Skip(Exception):
    """The target cannot run this configuration: key unset, extra absent, requirement missing."""


class Target(Protocol):
    """An application stood up as a running `Wica` for one benchmark configuration."""

    label: str  # the configuration's label: the config file's stem
    agent_config: dict[str, Any]  # the `agent` block, recorded in the summary header

    def open(self, scenario: Scenario) -> Wica:
        """Build, wire and start the system for *scenario* — or raise `Skip(reason)`."""
        ...

    def close(self) -> None:
        """The application's own teardown; called after every run, a skipped one included."""
        ...


TargetFactory = Callable[[Entry], Target]  # one matrix entry → one Target


@dataclass(frozen=True)
class Skipped:
    label: str
    scenario: str
    reason: str


@dataclass(frozen=True)
class Completed:
    """One finished run: the report (checks appended to its findings), the per-run measures and
    the raw metric samples the summary pools."""

    report: Report
    measures: dict[str, float]
    samples: dict[str, list[float]]
    repetition: int
    checks: tuple[str, ...] = ()  # the check findings' names, for the compliance rate

    @property
    def label(self) -> str:
        return self.report.sut_label

    @property
    def scenario(self) -> str:
        return self.report.scenario


def run_once(
    target: Target,
    scenario: Scenario,
    *,
    repetition: int,
    out_dir: Path,
    extra_checks: Iterable[Check] | None = None,
) -> Completed | Skipped:
    """`target.open(scenario)` → requirements check → `run_script` → checks + measures → write
    `<out_dir>/<label>/<scenario>/<repetition>.json` and `.md` → `run.close()`, `target.close()`."""
    registry = build_registry(extra_checks)
    run: TestRun | None = None
    try:
        try:
            wica = target.open(scenario)
        except Skip as exc:
            return Skipped(
                target.label, scenario.name, str(exc) or "skipped by the target"
            )
        run = TestRun(wica, scenario=scenario.name, sut_label=target.label)
        probe = run.probe
        inputs = probe.inputs
        commands = probe.commands
        output = probe.output_command_name
        missing = scenario.requires.missing(
            inputs=inputs, commands=commands, output_command_name=output
        )
        if missing:
            return Skipped(
                target.label, scenario.name, "the target lacks " + ", ".join(missing)
            )
        report = run_script(run, scenario.script, timeout=scenario.timeout_s)
        context = CheckContext(
            scenario=scenario,
            output_command_name=output,
            inputs=tuple(inputs),
            triggering_inputs=tuple(
                key for key, config in inputs.items() if config.triggers_llm_call
            ),
            commands=tuple(commands),
        )
        report = apply_checks(report, scenario.checks, context, registry)
        check_names = tuple(
            f.expectation for f in report.findings if f.expectation in registry
        )
        run_measures = measures(report.timeline, context)
        samples = metric_samples(report.timeline, output_command_name=output)
        base = Path(out_dir) / target.label / scenario.name
        report.write(
            base / f"{repetition}.json",
            extra={
                "bench": {
                    "repetition": repetition,
                    "checks": list(check_names),
                    "measures": run_measures,
                }
            },
        )
        report.write(base / f"{repetition}.md")
        return Completed(report, run_measures, samples, repetition, check_names)
    finally:
        if run is not None:
            run.close()
        target.close()


def run_matrix(
    targets: Sequence[Target],
    scenarios: Sequence[Scenario],
    *,
    k: int,
    out_dir: Path,
    only_configs: Iterable[str] = (),
    only_scenarios: Iterable[str] = (),
    extra_checks: Iterable[Check] | None = None,
    on_outcome: Callable[[Completed | Skipped], None] | None = None,
    skipped: Iterable[Skipped] = (),
) -> Summary:
    """Every (configuration, scenario) pair *k* times, sequentially; a pair skipped once is not
    retried. *on_outcome* sees each run as it ends (the CLI's progress line); *skipped* are the
    entries the caller set aside before building a target (disabled ones), scenario `"*"`."""
    if k < 1:
        raise ValueError("k must be at least 1")
    labels = set(only_configs)
    names = set(only_scenarios)
    chosen_targets = [t for t in targets if not labels or t.label in labels]
    chosen_scenarios = [s for s in scenarios if not names or s.name in names]
    started_at = now()
    completed: list[Completed] = []
    all_skipped: list[Skipped] = list(skipped)
    for target in chosen_targets:
        for scenario in chosen_scenarios:
            for repetition in range(1, k + 1):
                outcome = run_once(
                    target,
                    scenario,
                    repetition=repetition,
                    out_dir=out_dir,
                    extra_checks=extra_checks,
                )
                if on_outcome is not None:
                    on_outcome(outcome)
                if isinstance(outcome, Skipped):
                    all_skipped.append(outcome)
                    break
                completed.append(outcome)
    return aggregate(
        completed,
        all_skipped,
        k=k,
        configs={t.label: dict(t.agent_config) for t in chosen_targets},
        scenarios=[s.name for s in chosen_scenarios],
        started_at=started_at.isoformat(),
        finished_at=now().isoformat(),
        git_commit=git_commit(),
    )
