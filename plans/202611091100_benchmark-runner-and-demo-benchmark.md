# Benchmark runner, primer checks, scenario format, and the demo reference benchmark

**Status:** Done

Builds [specs/benchmark.md](../specs/benchmark.md) whole: the scenario file format and loader, the primer-level checks and the open registry, the `wica.benchmark` subpackage (the `Target` protocol, the runner with its per-run reports, the summary and the kept results, the shared CLI), the root `benchmarks/` folder (the conversation demo's target and entry point, the matrix, the fake config, the first scenario set, the README) and the tests of both tiers. It is phases 3 and 4 of [specs/wica-tester.md](../specs/wica-tester.md) under one spec; trace export (`to_scenario()`) stays with phase 2 and is out of scope here, as is any judge.

Everything it needs from the toolkit exists since plan `202611061400` (`TestRun`, `run_script`, `Report`, `SutEvent`, `compute_metrics`, `summarize`). **This plan changes no code under `src/wica/` outside `src/wica/testing/` and the new `src/wica/benchmark/`**, and nothing in the demo (`build_system` / `wire` stay as they are; the demo's configs are copied, not moved).

## Scope

In the package:

- `src/wica/testing/scenario.py` — `Scenario`, `Requires`, `load_scenario`, `load_scenarios`, `TESTER_COMMANDS`, validation errors naming file and step.
- `src/wica/testing/checks.py` — `CheckContext`, `CheckFn`, `Check` (fn + requires), the four built-ins, `CHECKS`, `resolve_checks(names, registry, context)`, `apply_checks(report, names, context, registry)`, `measures(timeline, context)`.
- `src/wica/testing/__init__.py` — export `Scenario`, `load_scenario`, `load_scenarios`, `CheckContext`, `CHECKS`, `apply_checks`, `measures`.
- `src/wica/testing/report.py` — `Report.write` learns an optional `extra: dict` merged into the JSON (the `"bench"` key); nothing else.
- `src/wica/benchmark/matrix.py` — `Entry`, `Matrix`, `load_matrix`, `select_entries`, `MatrixError` (one file for every configuration, `tts-engine`'s shape).
- `src/wica/benchmark/runner.py` — `Target`, `Skip`, `Skipped`, `TargetFactory` (over an `Entry`), `run_once`, `run_matrix`.
- `src/wica/benchmark/summary.py` — `Summary` / `aggregate` (with the header: start time, git commit, *k*, scenarios, agent blocks), `write_summary`, `write_result` (copies the summaries, writes `runs.json`, creates `description.md` / `findings.md` only when absent).
- `src/wica/benchmark/cli.py` — `cli` (`--matrix`, the selection flags `--only` / `--exclude` / `--tags` / `--exclude-tags` / `--include-disabled`, `--out` defaulting to a dated batch folder, `--result`, `extra_checks`), returning 0.
- `src/wica/benchmark/__init__.py` — re-exports the three modules' public names; not re-exported from `wica`.

At the root:

- `benchmarks/bench.py` — `DemoTarget` (over a matrix `Entry`, label = id), `main()` through `cli`; run as `uv run --group demo python -m benchmarks.bench`.
- `benchmarks/matrix.json` — the matrix: `agent_defaults` with the demo's persona, one `default`-tagged entry per provider with the demo's agent block, and the opt-in `fake` fixture (the key-less path).
- `benchmarks/scenarios/*.json` — the seven scenarios of the spec.
- `benchmarks/README.md` — how to run a batch and keep a result; the list of results (empty at first).
- `benchmarks/runs/` git-ignored (`.gitignore` gains `benchmarks/runs/`); `benchmarks/results/` committed, created by the first kept result.

Tests, tooling, docs:

- `tests/testing/test_scenario.py`, `test_checks.py`; `tests/benchmark/test_matrix.py`, `test_runner.py`, `test_summary.py`, `test_cli.py` (fast tier, over an in-test fake target); `tests-e2e/test_demo_bench_fake.py` (always-run) and `tests-e2e/test_demo_bench.py` (live), both importing `benchmarks.bench`.
- `tests/test_project_map.py` — `benchmark/__init__.py` joins `_NON_CONCEPT_MODULES`.
- `pyproject.toml` — `benchmarks` added to pyright's `include`.
- `.github/workflows/ci.yml` — `tests-e2e/test_demo_bench_fake.py` added to the e2e-fake job's file list and `tests-e2e/test_demo_bench.py` to the `live` job's (by file, never `-k` — [specs/ci.md](../specs/ci.md)); ci.md's jobs table updated.
- `AGENTS.md` — project map: a `benchmarks/` row in the top-level layout, the two new `testing/` modules on the `testing/` row, a `benchmark/` row among the modules (`runner.py`, `summary.py`, `cli.py`) pointing at the spec; Testing section: a paragraph on running the demo benchmark and keeping a result.
- `specs/benchmark.md` — frontmatter `code:` / `tests:` extended to the new files; `specs/wica-tester.md` — "Placement" and "Phases" already name the layout; `specs/testing.md` — the live-tier section names the benchmark smoke; `specs/conversation-demo.md` — a short "Benchmark" paragraph linking the spec; `specs/_index.md` / `plans/_index.md` statuses.

## Steps

0. **Design gate.** Done 2026-11-09: the spec is `Stable` (`no_double_reply` demoted to the `output_calls` / `unprompted_output` measures, `answers_each_input` defined over observed triggers, the quiet scenario reframed as a re-trigger test, tests split by file, the runner in `wica.benchmark` and the reference benchmark in a root `benchmarks/` folder with batches ignored and results committed). Before coding, only re-check the `CheckContext` fields and how `requires` is spelled in JSON against the shipped toolkit.

1. **Scenario** (`scenario.py`, `test_scenario.py`). Frozen dataclasses; `TESTER_COMMANDS` as the fixed set plus the `set_` prefix rule (mirrors `tester.py`'s registrations — import the names from there if they are exposed, else define them once in `scenario.py` and have `tester.py` import them, so the two cannot drift). Errors: `ValueError(f"{path}: step {i}: …")`.

2. **Checks** (`checks.py`, `test_checks.py`). Timeline helpers first: `commands_by_reaction` (join `reaction_ended.command_call_ids` with `command_issued.call_id`), `turns(timeline, input_keys)` (slices between observed-trigger writes of triggering inputs), `output_issued(timeline, output_name)`. Then the four built-ins with their `requires`, `CHECKS`, `apply_checks` returning `dataclasses.replace(report, findings=[*report.findings, *new])`, `measures`. Tests build timelines with a `_ev(kind, **fields)` helper and real `ReactionTrace` / `TriggerTrace` values.

3. **Runner and summary** (`src/wica/benchmark/`, `report.py`, `tests/benchmark/`). `run_once` exactly as the spec's sequence; the requirement check reads `run.probe.inputs` / `.commands` / `.output_command_name`; `Skipped(label, scenario, reason)`. `aggregate` pools samples through `summarize` and fills the header (`git rev-parse HEAD` through `subprocess`, `None` outside a repository or without git); `Summary.to_markdown()` / `to_dict()`; `write_summary`; `write_result` with the two hand-finished files created once. `cli` with argparse, the filters, `--configs` as paths and `--result`, returning 0. Tests as the spec's Testing section lists, over an in-test fake target (a `Wica` with `speech_input` + `say`, like `tests-e2e/test_tester_flows.py`'s SUT); `write_result` tested in a `tmp_path`, including that a second call leaves `description.md` alone. Add `benchmark/__init__.py` to `_NON_CONCEPT_MODULES` and the AGENTS.md rows so the drift guard passes.

4. **The `benchmarks/` folder** (`bench.py`, `matrix.json`, `README.md`, `.gitignore`, `pyproject.toml`, `tests-e2e/test_demo_bench_fake.py`, `tests-e2e/test_demo_bench.py`, CI). `DemoTarget` as specified; `main()` through `cli`. The live test parametrized over the matrix's enabled entries, turning the target's `Skip` into `pytest.skip`; the always-run case over the `fake` entry. Run the fake path locally and the live path with the keys at hand.

5. **Scenarios** (`benchmarks/scenarios/*.json`). Write the seven files; budgets `within ≥ 15` for model steps; `background-change-is-quiet` writes the demo's `emotion` entry right after `expect_output`, while `say` is still speaking. Run `greeting` against the `fake` entry (the only scenario its script covers — the others would fail, not skip, since the demo satisfies their `requires`) and the whole set once against one live provider with `--k 2`; fix budgets from what the batch shows. Keep that batch as the first result (`--result baseline`), finish its `description.md` and `findings.md`, list it in the README.

6. **Docs and statuses**: AGENTS.md, ci.md, testing.md, conversation-demo.md paragraph, the spec's frontmatter, `benchmark.md` → `Implemented`, this plan → `Done`, both indexes.

## Done 2026-11-09 — what moved during implementation

- `run_once` returns `Completed | Skipped` rather than a bare `Report`: the summary pools raw metric samples, which a `Report` does not hold, so `report.metric_samples` was factored out of `compute_metrics` (behavior unchanged) and `Completed` carries them with the measures.
- `answers_each_input` opens turns on the scenario's `requires.inputs` that trigger (not on every triggering input): a sensor entry such as the demo's `closest_user` is not addressed to the agent. Coalesced writes form one turn through the reaction that saw them.
- `DemoTarget` checks `api_key_env` itself before `build_system`, so an unset key is a `Skip` without the demo's explore-only loop thread being started.
- `run_matrix` does not retry a (configuration, scenario) pair after its first `Skip`; the CLI prints one progress line per run and accepts a scenario file as well as a directory.
- **Workload revision, same day** — plan `202611091500`: the seven scenarios were a repeated e2e suite; they became `benchmarks/pins/` and two session scenarios became the workload, with a per-configuration table at the top of the summary.
- **Matrix revision, same day.** The first cut had one config file per model (`benchmarks/configs/<label>.json` plus an off-matrix `fake.json`). A benchmark meant to compare a long list of models wants one file to add a row to and tag, as `tts-engine`'s `matrix.json`, so the configuration layer became `wica.benchmark.matrix` (`Entry` / `load_matrix` / `select_entries`), `TargetFactory` takes an `Entry`, the CLI selects by id glob and tag with `--include-disabled` for opt-in rows, and the fake fixture is a disabled entry of the one `benchmarks/matrix.json`.
- The first live batch (`--result baseline`) is left for a later session: the four providers with keys at hand pass `greeting` in the live smoke, but a full matrix is the reader's decision to spend.

## Verification

- `uv run ruff check .`, `uv run ruff format .`, `uv run pyright` — clean, `benchmarks/` included.
- `uv run pytest` — the fast tier, `tests/testing/` and `tests/benchmark/` included, and the drift guard with the new modules mapped.
- `uv run pytest tests-e2e -k "tester or bench"` — key-less, passes.
- `uv run pytest tests-e2e/test_demo_bench_fake.py` — passes, key-less; `uv run pytest tests-e2e/test_demo_bench.py` — passes for every provider whose key is set, skips the rest.
- `uv run --group demo python -m benchmarks.bench --only fake --include-disabled --scenario greeting --k 2 --out <scratch>` — writes two reports and a summary with pass^k 1 for the fake config, no network; with `--result smoke`, the five result files appear (run, then removed: not a result worth keeping).
- Mark this plan `Done` (here and in [_index.md](_index.md)) and the spec `Implemented` only once all of the above pass.
