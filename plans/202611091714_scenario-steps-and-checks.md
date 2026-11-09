# Scenario steps: input, wait, expect, check

**Status:** Done

Implements the "Scenarios" section of [specs/wica-tester.md](../specs/wica-tester.md) and the matching revision of [specs/benchmark.md](../specs/benchmark.md) (status `Updated`). Replaces the first cut of PR #8, which kept the fake-model script as the scenario format and put the check arguments in a separate list.

**The design in one paragraph.** A scenario is a flat list of steps, each one of four kinds: `input` (the user writes an Input), `wait` (pacing: `idle`, `speaking`, `{running: name}`, `{seconds: n}`), `expect` (an assertion), `check` (a rule judged after the run, over the window from the input it follows to the next input; top-level `checks` cover the whole run). A failed expect stops the run as **failed**, a timed-out wait as **stalled**; checks are evaluated only on a completed run. In a benchmark, a stopped run counts as failed for pass^k and in no other number.

## Scope

In `wica.testing` (the tester's):

- `scenario.py` — the step format: `Scenario(name, description, steps, checks, timeout_s, params)`, step dataclasses (`Input`, `Wait`, `Expect`, `Check` steps), `load_scenario` / `load_scenarios` validating every step (errors name file and step index), derived requirements (`Scenario.requires`: Inputs from input steps, Commands named in waits, expects and check args, output Command when a step needs it), `to_script(scenario)` translating steps into the scripted tester's internal fake-model script, and `run_scenario(run, scenario) -> Report` (the pytest entry).
- `harness.py` — `wait_for(condition, within)` replaces `expect_idle` (`"idle"` keeps its timeline rule; `"speaking"`, `{"running": name}`, `{"seconds": n}` added); a stall is its own finding kind. The run's **outcome** (`completed` / `failed` / `stalled`, with the step) is recorded; `Report` gains `outcome` and `stopped_at`, and `report.ok` becomes *completed and every check kept*.
- `tester.py` — the `wait_for` Command; a failed expect or a stall makes the next step `finish` (the chain ends at once); `TESTER_COMMANDS` moves here from `scenario.py`.
- `checks.py` — checks run over **windows**: a step check over its input's window, a top-level check over the whole run; arguments are the step's (`acts_when_asked: "dance"` or `{"switch_user_tracking": {"user_id": "u2"}}`); `cancels_when_asked` and `quiet_on_background_change` take none. The `requests` / `keys` arguments of the first cut go away. Evaluated only on a completed run.

In `wica.benchmark`:

- `runner.py` — runs a scenario through `run_scenario`; a stopped run is recorded with its outcome and step, scored only when completed.
- `summary.py` — `runs`, `scored`, `failed`, `stalled` columns; pass^k over completed runs that kept every check; compliance, latency, tokens and measures over scored runs only; a "stopped at" column; the checks-by-configuration table over scored runs.

Data and docs:

- `benchmarks/scenarios/conversation.json`, `commands.json` rewritten in the step format, with no expect.
- `benchmarks/matrix.json` — the fake fixture's notes; its script unchanged if `conversation` still drives the same model calls.
- `tests-e2e/support.greeting_probe` and `tests-e2e/test_tester_flows.py` — rewritten as step scenarios.
- Tests: the step loader (each kind, each error), `wait_for` conditions (with the existing idle tests moved over), the stop semantics (failed / stalled end the chain, checks skipped), window evaluation per check, derived requirements, and the summary's scored-only pooling.
- AGENTS.md rows for `testing/` and `benchmark/`; `benchmarks/README.md`; the e2e flows' docstrings.

## Steps

1. Step format and loader, derived requirements, `to_script`, unit tests.
2. `wait_for` in the Harness and the tester; outcome and stop in the report; `run_scenario`; unit and fake-flow tests.
3. Checks over windows; the first cut's argument forms removed; unit tests.
4. Runner and summary: scored-only pooling, stopped columns; unit tests.
5. Rewrite the two sessions and the probe; run the fake `conversation` three times for determinism.
6. Docs; specs back to `Implemented` (benchmark) and phase notes (tester); this plan `Done`.

## Done 2026-11-09 — what moved during implementation

- `run_scenario` plays the steps **directly** on the `TestRun` (driver (a)), for pytests and the benchmark alike; the scripted tester plays `to_script`'s translation. No second `Wica` per benchmark run.
- `"speaking"` and `{"running": name}` mean "has started since the last input", not "is running now": decidable from the timeline even for a Command short enough to end before the wait looks.
- The benchmark's `Completed` is renamed `RunResult` (`scored` when its outcome is `completed`), since "completed" is now an outcome.
- `TestRun.set` is refused after a stop, so a scripted tester whose chain is still unwinding cannot write past it.
- The scripted tester's failed expect or stall ends the chain (`run.stop`), where it used to continue.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier: 486 passed.
- The e2e-fake set (`test_fake_flows`, `test_example_flow`, `test_tester_flows`, `test_demo_bench_fake`) — key-less, 14 passed.
- `uv run --group demo python -m benchmarks.bench --only fake --include-disabled --scenario conversation --out <scratch>` — completed, every check kept, three runs alike.
