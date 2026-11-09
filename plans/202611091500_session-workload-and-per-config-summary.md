# Session workload, per-turn measures and the per-configuration summary

**Status:** Done

Turns the reference benchmark from a repeated e2e suite into a benchmark ([specs/benchmark.md](../specs/benchmark.md), revised 2026-11-09): the seven one-rule scenarios were each a short e2e test, and running them *k* times gave pass rates per rule but no result per model. A benchmark wants one or two **session scenarios** as the workload, the same for every model, run *k* times, read as **one row per model** — as `tts-engine`'s summary is one row per entry.

Follows plan `202611091100` the same day; the format, checks, runner, matrix and CLI are unchanged in shape.

## Scope

- `src/wica/testing/checks.py` — `measures` gains `turns`, `turns_answered`, `noops_on_input`.
- `src/wica/benchmark/runner.py` — `Completed.checks` names the check findings (for the compliance rate).
- `src/wica/benchmark/summary.py` — per-run derived rates (`compliance`, `answer_rate`, `input_tokens_per_turn`, `output_tokens_per_turn`, `words_per_turn`); `Summary.totals`, one `Cell` per configuration over every scenario (scenario `"*"`); the markdown opens with the per-configuration table; `summary.json` gains `totals`.
- `benchmarks/scenarios/` — the workload: `conversation` (generic, seven paced turns with a barge-in and two quick utterances) and `commands` (the demo's own actions). The seven former one-rule scenarios are deleted (kept for a few hours as `benchmarks/pins/`, then dropped: the session reports name the broken clause already).
- `benchmarks/matrix.json` — the `fake` fixture's script plays the whole `conversation` session (the long story's `say` carries a scripted tool-call id so `cancel_command` can name it).
- `benchmarks/README.md` — the workload, the per-model row, the cost of a batch.
- Tests: `tests/testing/test_checks.py` (turn measures), `tests/benchmark/test_summary.py` (totals, rates, the table order), `test_runner.py` (`Completed.checks`); `tests-e2e/test_demo_bench_fake.py` runs `conversation` and a greeting probe through the fake; `tests-e2e/test_demo_bench.py` runs the greeting probe live (`tests-e2e/support.greeting_probe`, a fixture).
- Docs: the spec's Purpose, Vocabulary, Measures, Summary, reference-benchmark and Testing sections; AGENTS.md's `benchmarks/` row and Testing paragraph; conversation-demo.md's Benchmark paragraph; the sibling `wica-interaction` spec's pointer to the generic scenarios.

## Steps

1. Turn measures in `measures`, over `turns(timeline, context.turn_inputs)`; unit tests.
2. `Completed.checks`; derived per-run rates and `totals` in `aggregate`; the per-configuration table first in `to_markdown`; unit tests for the rates, the totals and the table order.
3. Write `conversation` and `commands`; drop the seven one-rule files; script the fake through `conversation` and run it three times for determinism (the cancel's two completion triggers coalesce into one step; the two quick utterances are spaced so each `say` completion lands in its own step).
4. Update the e2e smokes, the README, AGENTS.md, the specs; verify.

## Decisions

- **No pins.** The one-rule scenarios were first kept as `benchmarks/pins/` for debugging, then dropped: a live CI test of one rule on one sample would fail `main` whenever a model has a bad day (the very variance the benchmark measures with *k*), and as a hand tool they add little, since the session reports name the broken clause. The live CI smoke is a one-utterance greeting probe per enabled entry, a test fixture asserting that the provider drives the demo at all, not a verdict on its behavior.
- **Compliance is checks passed over checks run**, not findings passed over findings: the script's expectations sequence the turns, the checks are the rules.
- **A confirmation after a cancel counts as unprompted output**: it is issued on the step the cancel's completion wakes, which has no external trigger. Consistent across models, and named in the spec so a reader does not misread the column.
- **Cost per model is still out**: it needs a price per entry in the matrix; open question 6.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- `uv run pytest tests-e2e/test_fake_flows.py tests-e2e/test_example_flow.py tests-e2e/test_tester_flows.py tests-e2e/test_demo_bench_fake.py` — the e2e-fake job, key-less.
- `uv run pytest tests-e2e/test_demo_bench.py` with the keys at hand — the live smoke.
- `uv run --group demo python -m benchmarks.bench --only fake --include-disabled --scenario conversation --out <scratch>` three times — `ok` each time, seven turns answered.
