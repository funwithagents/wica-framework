"""Test tooling for a running `Wica`: Probe, Harness, report, scripted drivers.

Imported explicitly (``from wica.testing import TestRun``), never re-exported from ``wica`` — it is
test tooling, like ``wica.fake_model``. See specs/wica-tester.md.
"""

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
    summarize,
)
from wica.testing.tester import build_tester, run_script

__all__ = [
    "Finding",
    "Issue",
    "MetricSummary",
    "Probe",
    "Report",
    "SutEvent",
    "SutEventKind",
    "TestRun",
    "build_tester",
    "compute_metrics",
    "run_script",
    "summarize",
]
