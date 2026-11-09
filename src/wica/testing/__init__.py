"""Test tooling for a running `Wica`: Probe, Harness, report, scripted drivers, scenario files and
primer-level checks (the per-run pieces of specs/benchmark.md; the runner is `wica.benchmark`).

Imported explicitly (``from wica.testing import TestRun``), never re-exported from ``wica`` — it is
test tooling, like ``wica.fake_model``. See specs/wica-tester.md.
"""

from wica.testing.checks import (
    CHECKS,
    Check,
    CheckContext,
    CheckFn,
    apply_checks,
    build_registry,
    measures,
    resolve_checks,
)
from wica.testing.harness import TestRun
from wica.testing.probe import Probe
from wica.testing.report import (
    Finding,
    Issue,
    MetricSummary,
    Report,
    SutEvent,
    SutEventKind,
    compute_metrics,
    metric_samples,
    summarize,
)
from wica.testing.scenario import (
    TESTER_COMMANDS,
    CheckSpec,
    Requires,
    Scenario,
    ScenarioError,
    load_scenario,
    load_scenarios,
)
from wica.testing.tester import build_tester, run_script

__all__ = [
    "CHECKS",
    "Check",
    "CheckContext",
    "CheckFn",
    "CheckSpec",
    "Finding",
    "Issue",
    "MetricSummary",
    "Probe",
    "Report",
    "Requires",
    "Scenario",
    "ScenarioError",
    "SutEvent",
    "SutEventKind",
    "TESTER_COMMANDS",
    "TestRun",
    "apply_checks",
    "build_registry",
    "build_tester",
    "compute_metrics",
    "load_scenario",
    "load_scenarios",
    "measures",
    "metric_samples",
    "resolve_checks",
    "run_script",
    "summarize",
]
