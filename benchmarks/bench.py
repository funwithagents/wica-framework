"""The conversation demo's benchmark: `DemoTarget` and the `wica-demo-bench` entry point.
Implements specs/benchmark.md, "The reference benchmark: the conversation demo".

Run from the repo root (it imports the demo, `examples.conversation_demo.app`, and needs the
`demo` group for the presenters the demo wires):

    uv run --group demo python -m benchmarks.bench --k 5
    uv run --group demo python -m benchmarks.bench --only 'openai/*' --scenario greeting --k 3
    uv run --group demo python -m benchmarks.bench --only fake --include-disabled --scenario conversation

The matrix is `benchmarks/matrix.json` (one entry per model, selected by id glob and tag; the
`fake` fixture is its opt-in key-less row). The workload is `benchmarks/scenarios/` (two session
scenarios). Batches land in `benchmarks/runs/<stamp>/`;
`--result NAME` keeps one under `benchmarks/results/`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from examples.conversation_demo.app import build_system, display_entry, wire
from examples.conversation_demo.speaking import SpeakingSlot
from wica import Wica
from wica.benchmark import Entry, Skip, cli
from wica.config import WicaConfig
from wica.contrib.gradio import TranscriptLog
from wica.testing import Scenario

BENCH_DIR = Path(__file__).parent
MATRIX = BENCH_DIR / "matrix.json"
SCENARIOS_DIR = BENCH_DIR / "scenarios"  # the workload: the session scenarios
RUNS_DIR = BENCH_DIR / "runs"
RESULTS_DIR = BENCH_DIR / "results"


class DemoTarget:
    """The demo stood up as its `main()` does minus the page: `build_system`, `wire` over a
    transcript and a Speaking slot of our own, `start`. `label` is the matrix entry's id."""

    def __init__(self, entry: Entry) -> None:
        self.entry = entry
        self.label = entry.id
        self.agent_config: dict[str, Any] = dict(entry.agent)
        self._wica: Wica | None = None

    def open(self, scenario: Scenario) -> Wica:
        # The demo degrades to an explore-only World when the key is unset; a benchmark has no
        # use for that, so the matrix's gate is a Skip here — checked before any loop starts.
        reason = self.entry.skip_reason()
        if reason is not None:
            raise Skip(reason)
        config = WicaConfig.from_dict(
            {"agent": self.entry.agent}, base_dir=self.entry.base_dir
        )
        wica, _world, config_error = build_system(config)
        if wica is None:
            raise Skip(config_error or "the demo could not build its Wica")
        self._wica = wica
        wire(wica, TranscriptLog(wica, display_entry), SpeakingSlot())
        wica.start()
        return wica

    def close(self) -> None:
        if self._wica is not None:
            self._wica.close()
            self._wica = None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    return cli(
        argv,
        target_factory=DemoTarget,
        default_matrix=MATRIX,
        default_scenarios=SCENARIOS_DIR,
        default_runs=RUNS_DIR,
        default_results=RESULTS_DIR,
        prog="wica-demo-bench",
    )


if __name__ == "__main__":
    sys.exit(main())
