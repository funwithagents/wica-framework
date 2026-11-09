"""Test tooling for a running `Wica`: the Probe, the Harness (`TestRun`), the report, the scripted
tester, scenarios (`input` / `wait` / `expect` / `check` steps) and the checks they judge.

Imported explicitly (``from wica.testing import TestRun``), never re-exported from ``wica`` — it is
test tooling, like ``wica.fake_model``. See specs/wica-tester.md; the benchmark that compares models
over scenarios is ``wica.benchmark``.
"""

from wica.testing.checks import (
    CHECKS,
    Check,
    CheckContext,
    CheckFn,
    CheckSpec,
    answer_latencies,
    build_registry,
    input_windows,
    measures,
    run_check,
)
from wica.testing.harness import TestRun
from wica.testing.probe import Probe
from wica.testing.report import (
    Finding,
    FindingKind,
    Issue,
    MetricSummary,
    Report,
    RunOutcome,
    SutEvent,
    SutEventKind,
    compute_metrics,
    metric_samples,
    summarize,
)
from wica.testing.scenario import (
    CheckStep,
    ExpectStep,
    InputStep,
    Requires,
    Scenario,
    ScenarioError,
    Step,
    WaitStep,
    check_context,
    evaluate_checks,
    load_scenario,
    load_scenarios,
    run_scenario,
    to_script,
)
from wica.testing.tester import TESTER_COMMANDS, build_tester, run_script

__all__ = [
    "CHECKS",
    "Check",
    "CheckContext",
    "CheckFn",
    "CheckSpec",
    "CheckStep",
    "ExpectStep",
    "Finding",
    "FindingKind",
    "InputStep",
    "Issue",
    "MetricSummary",
    "Probe",
    "Report",
    "Requires",
    "RunOutcome",
    "Scenario",
    "ScenarioError",
    "Step",
    "SutEvent",
    "SutEventKind",
    "TESTER_COMMANDS",
    "TestRun",
    "WaitStep",
    "answer_latencies",
    "build_registry",
    "build_tester",
    "check_context",
    "compute_metrics",
    "evaluate_checks",
    "input_windows",
    "load_scenario",
    "load_scenarios",
    "measures",
    "metric_samples",
    "run_check",
    "run_scenario",
    "run_script",
    "summarize",
    "to_script",
]
