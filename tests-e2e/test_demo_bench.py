"""The demo benchmark against each live provider of the matrix (specs/benchmark.md, "Testing"):
one greeting (a fixture, `support.greeting_probe`) per enabled `benchmarks/matrix.json` entry through
`DemoTarget`. A target's `Skip`
(its `api_key_env` unset) is this tier's `pytest.skip`. It asserts the provider drove the demo (a
reaction ended `ok`), not the model's behavior. One sample is a smoke test; the benchmark
is `python -m benchmarks.bench --k N` over the session workload."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.bench import MATRIX, DemoTarget
from wica.benchmark import Completed, Entry, Skipped, load_matrix, run_once

from support import greeting_probe

ENABLED = [e for e in load_matrix(MATRIX).entries if e.enabled]


@pytest.mark.parametrize("entry", ENABLED, ids=[e.id for e in ENABLED])
def test_greeting_once_per_enabled_entry(entry: Entry, tmp_path: Path):
    greeting = greeting_probe()
    outcome = run_once(DemoTarget(entry), greeting, repetition=1, out_dir=tmp_path)
    if isinstance(outcome, Skipped):
        pytest.skip(outcome.reason)
    assert isinstance(outcome, Completed)
    # A probe that the provider drives the demo at all: the model call completed and the run
    # produced its report. Whether the model then *behaved* (answered, no noop) is the
    # benchmark's business, measured with k — one sample must not turn `main` red.
    reactions = [e.reaction for e in outcome.report.timeline if e.reaction is not None]
    assert any(r.outcome == "ok" for r in reactions), [r.outcome for r in reactions]
    assert outcome.samples["model_latency"]
    assert any(f.expectation == "expect_output()" for f in outcome.report.findings)
    assert (tmp_path / entry.id / "greeting" / "1.json").exists()
