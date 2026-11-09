# Rules, not expectations: what the benchmark measures

**Status:** Done

Revises [specs/benchmark.md](../specs/benchmark.md) after the second batch (`benchmarks/results/20261109-1553_second-look`). Its failures mixed two things: script expectations that only sequence the conversation (`expect_output` after "Stop!", which the persona never promises) and real rule breaks. A benchmark should measure the second only: a **fixed list of generic rules**, checked per run, compliance per model and per rule.

## Scope

- `src/wica/testing/scenario.py` — `CheckSpec(name, args)`; a scenario's `checks` entry is a name or `{"name", "args"}`; duplicates are a load error; `Scenario.check_names`.
- `src/wica/testing/checks.py` — `CheckContext.args`; `Check.takes_args` (never applied by default); `resolve_checks` / `apply_checks` take specs and hand each rule its args. `no_colliding_cancel` allows the output Command alongside `cancel_command` (saying "okay, stopping" while cancelling). Three new rules over the script's writes: `cancels_when_asked`, `acts_when_asked`, `quiet_on_background_change`.
- `src/wica/benchmark/summary.py` — a run passes when every rule held; script expectation misses are counted apart (`Cell.script_missed`); `Cell.rules` (held, applied) per rule; the markdown gains a rules-by-configuration table.
- `benchmarks/scenarios/` — both sessions drive only (`expect_idle` between turns, `expect_output` / `expect_action` before a barge-in) and list their rules with args.
- Tests: rule args parsing, the three rules, the relaxed cancel rule, rule-only pass and the rules table.
- Docs: the spec's Vocabulary, scenario file, Rules, Summary and workload sections; the index row; AGENTS.md's `testing/` row.

## Decisions

- **The script drives, the rules judge.** A script expectation is never a failure; whatever is worth judging becomes a rule. "Stop!" getting a spoken acknowledgement is judged by `answers_each_input` (a turn opened by "Stop!" must be answered), not by a script step.
- **Cancel plus speech is fine.** `no_colliding_cancel` breaks only on `cancel_command` with another action.
- **Rules that need to know which write is which take args** from the scenario rather than guessing from text: generic rules, scenario-specific targets.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — 473 passed.
- The e2e-fake set (`test_fake_flows`, `test_example_flow`, `test_tester_flows`, `test_demo_bench_fake`) — 14 passed.
