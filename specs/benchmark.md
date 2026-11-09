---
code:
  - src/wica/testing/scenario.py
  - src/wica/testing/checks.py
  - src/wica/testing/report.py
  - src/wica/benchmark/__init__.py
  - src/wica/benchmark/matrix.py
  - src/wica/benchmark/runner.py
  - src/wica/benchmark/summary.py
  - src/wica/benchmark/cli.py
  - benchmarks/bench.py
  - benchmarks/matrix.json
  - benchmarks/scenarios
tests:
  - tests/testing/test_scenario.py
  - tests/testing/test_checks.py
  - tests/benchmark/test_matrix.py
  - tests/benchmark/test_runner.py
  - tests/benchmark/test_summary.py
  - tests/benchmark/test_cli.py
  - tests-e2e/test_demo_bench_fake.py
  - tests-e2e/test_demo_bench.py
---

# WICA benchmark: scenarios, checks and the runner

**Status:** Implemented

## Purpose

[wica-tester.md](wica-tester.md) gives one run against one `Wica`: a timeline, measures and a report. A **benchmark** is many such runs compared — the same scenarios, against the same application, with a different model behind the Agent each time, repeated enough to see past model variance. This spec adds the three pieces that turn runs into a comparison, all of them application-agnostic:

- a **scenario file**: a fixed user as a scripted tester script plus the checks to run over the result — for a benchmark, a **session** of several turns, the same workload for every model;
- **primer-level checks**: the rules *every* WICA agent is under because the runtime primer states them, applied to a finished timeline;
- the **runner**: a config matrix × a scenario set × *k* repetitions, one report per run, one summary with **one row per model** pooled over every run (pass^k, rule compliance, turns answered, latency, tokens and words per turn) and then the same per scenario.

It also ships the framework's own **reference benchmark**: the conversation demo ([conversation-demo.md](conversation-demo.md)) as the system under test, a matrix of models, and a workload of two session scenarios written against the demo's `speech_input` + `say` shape, run *k* times per model. That benchmark answers "how well does model X drive a WICA agent at all"; an application built on WICA (the first is `wica-interaction`, `wica-interaction/specs/benchmark.md`) reuses the runner, the format and the primer checks for its own target, and adds the checks its persona and body demand. The generic layer lives in the package (`wica.testing` for the per-run pieces, `wica.benchmark` for the runner) so that no application re-implements a runner, and so that two applications' numbers are produced the same way; the reference benchmark — target, matrix, scenarios, results — lives in a root `benchmarks/` folder outside the package, as nothing imports it (the layout `tts-engine` uses for its own benchmark).

This spec absorbs phases 3 and 4 of [wica-tester.md](wica-tester.md) ("Phases"): the runner and the demo scenario. Trace export (`report.to_scenario()`) stays with the live-LLM tester (phase 2), since it is the live tester's calls it exports; the scenario format below is what it will emit.

## Settled

### Vocabulary

- **Target** — an application stood up as a running `Wica` for one benchmark configuration, plus how to shut it down. The runner never builds a `Wica` itself; the application does, exactly as its entry point would (the demo's `build_system` + `wire` + `start`, the interaction app's `build_system` + `wire` + `_start`), so what is benchmarked is the real system with its real persona, Commands and seeding. Only the peripherals the application chooses to bypass are doubled, by the application.
- **Configuration** — one model under test: an **entry** of the matrix file, whose `agent` block (provider, model, `hf_provider`, `api_key_env`, history settings) is what differs between entries; the persona is shared through the file's `agent_defaults`. Its **label** in every report is the entry's `id`. The file is the **matrix**.
- **Scenario** — a fixed user: a tester script (driver (b) of the tester spec) and a list of check names, in one JSON file, named by its stem. A benchmark's scenarios are **sessions** — several user turns, paced as a person would pace them — so one run yields a whole conversation's worth of turns to measure; a scenario that pins one rule in a few steps is a regression probe, not a workload.
- **Run** — one (configuration, scenario, repetition): one `TestRun`, one `Report`.
- **Batch** — one invocation of the runner: a matrix × a scenario set × *k*, written to one dated folder of per-run reports plus a summary. Working data, git-ignored.
- **Result** — a batch worth keeping: its summary and manifest copied under `benchmarks/results/<YYYYMMDD-HHMM>_<name>/` with a hand-written description and findings. Committed, and the only thing a spec or plan cites.
- **Check** — a pure function over a finished timeline producing one `Finding`. Hard: a failed check fails the run.
- **Measure** — a number taken from a run, never thresholded by the runner (the tester spec's "measures, not thresholds").

### The scenario file

`src/wica/testing/scenario.py` — `Scenario` (frozen dataclass), `load_scenario(path)`, `load_scenarios(directory)` (every `*.json`, sorted by name), `TESTER_COMMANDS`.

```json
{
  "name": "stop-while-dancing",
  "description": "A dance is asked for, then stopped mid-way: cancel_command, then a spoken confirmation.",
  "requires": {"inputs": ["speech_input"], "commands": ["dance"], "output_command": true},
  "timeout_s": 60,
  "checks": ["answers_each_input", "no_failed_reactions", "no_colliding_cancel"],
  "params": {},
  "script": [
    {"tool_calls": [{"name": "set_speech_input", "args": {"value": "Please dance for me!"}}]},
    {"tool_calls": [{"name": "expect_action", "args": {"name": "dance", "within": 15}}]},
    {"tool_calls": [{"name": "set_speech_input", "args": {"value": "Stop!"}}]},
    {"tool_calls": [{"name": "expect_action", "args": {"name": "cancel_command", "within": 15}}]},
    {"tool_calls": [{"name": "expect_output", "args": {"within": 15}}]},
    {"tool_calls": [{"name": "finish", "args": {"verdict": "pass"}}]}
  ]
}
```

- `name` — must equal the file stem; the report's `scenario`.
- `requires` — what the target must expose for the scenario to apply: input keys, Command names, and whether an output Command must be set. The runner **skips** a (configuration, scenario) pair whose target lacks any of them and says so in the summary, so one scenario directory can mix generic and application-specific scenarios. All three default to empty / `false`.
- `timeout_s` — `run_script`'s budget; the tester spec's "did not call finish" finding applies.
- `checks` — names resolved against the runner's check registry (below); an unknown name is a load error. Absent ⇒ every registered check whose own `requires` the target satisfies.
- `params` — an opaque object handed to the target when it opens for this scenario (an application reads what it needs: the interaction app's initial motor state, for instance). The runner never reads it.
- `script` — **verbatim** the fake model's script: one `tool_calls` list per step, each exactly one call to a tester Command (`set_<key>`, `expect_action`, `expect_output`, `expect_text`, `expect_no_action`, `expect_world`, `expect_latency`, `wait`, `note`, `report_issue`, `finish`), with JSON-object args. Kept verbatim rather than flattened so a recorded trace (phase 2's `to_scenario()`) drops in unchanged. Validation errors name the file and the step index.

**Matchers stay loose.** One script runs against every model, so a step asserts that the right kind of thing happened — a `say` happened (`expect_output` without `contains`), a Command by name, a silence the primer asks for — not its wording. `within` budgets include the model's think time and whatever duration the target gives its Commands. Wording and relevance are a judge's business (tester spec, open question 2).

### Primer-level checks

`src/wica/testing/checks.py` — `CheckFn = Callable[[list[SutEvent], CheckContext], Finding]`, where `CheckContext` carries what a check may need beyond the timeline: the scenario, the output Command's name and the target's surface (its Inputs, which of them trigger, its Commands). Each check is one `Finding` (`expectation=<name>`, `ok`, a `detail` citing the offending event's `t` and name). A check never waits: it is a function of the finished timeline, run by `apply_checks(report, names, context) -> Report` after `run_script` returns, which appends the findings to the report's.

The built-in checks enforce the **runtime primer** (`agent.md`, "System prompt"), nothing an application's persona adds:

| Check | Requires | Primer clause | Fails when |
|---|---|---|---|
| `answers_each_input` | output Command | "anything addressed to you should be answered" / "must be said by calling `<output>`" | a **turn** — from a write to one of the scenario's `requires.inputs` that triggers a step, whose trigger was *observed* (it appears among a reaction's triggers), to the next such write, or the end — in which the output Command is never issued. Writes one reaction saw together (coalesced) open one turn, at the earliest; a write whose trigger was dropped opens no turn of its own: it belongs to the step that saw it, so `two-quick-utterances` is measured (`dropped_inputs`, `reactions_per_input`), not failed. Scenario-selected like `no_noop_on_input`: a sensor entry is not addressed to the agent |
| `no_noop_on_input` | — | "never use `noop` to skip an expected response" | a reaction triggered by a write to one of the scenario's `requires.inputs` ends in `noop` only (scenario-selected, since a passive entry may legitimately be ignored) |
| `no_failed_reactions` | — | — | any `reaction_ended` with `outcome != "ok"` (model error, empty response) |
| `no_colliding_cancel` | — | "When one action must finish before another starts, do not issue both together" | a reaction issues `cancel_command` together with any other non-`noop` Command |

**A chained reply is a measure, not a failure.** The output Command's completion re-triggers a step so the model *may* chain a second utterance ([commands.md](commands.md), "The output Command"); a check failing it would penalize designed behavior. The runner counts it (`output_calls`, `unprompted_output` below), and a scenario that wants silence after the reply says so with `expect_no_action(<output>, for_=…)`, as `greeting` does.

**The registry is open.** `CHECKS: dict[str, Check]` holds the built-ins, each with its `requires`; `build_registry(extra_checks)` adds an application's persona checks under their own names (the interaction app's `no_collision`, `spoken_style`, …) — the runner and the CLI take `extra_checks`, the loader takes the resulting `registry` — and a scenario file selects them like any other. A name clash with a built-in is an error.

**Measures.** `measures(timeline, context) -> dict[str, float]` adds per run, next to the framework's metrics: `output_calls` (how many times the output Command was issued), `unprompted_output` (issued by a reaction with no external trigger — the chained reply after the output Command's completion counts here, and so does a spoken confirmation issued on the step a cancel's completion wakes), `noop_steps`, `failed_reactions`, `output_words` (summed words of the output Command's string args — a cost of the person's time), and the **turn numbers** a session is read by: `turns` (opened by the scenario's triggering inputs, as `answers_each_input` defines them), `turns_answered` (the output Command issued within the turn), `noops_on_input` (reactions that answered a write to a scenario input with `noop` alone). Written into the run's JSON under `"bench"`; pooled into the summary.

### The matrix

`src/wica/benchmark/matrix.py` — `Entry`, `Matrix`, `load_matrix(path)`, `select_entries(entries, *, only, exclude, tags, exclude_tags)`, `MatrixError`. One JSON file lists every configuration, the way `tts-engine`'s `benchmarks/matrix.json` does, because a benchmark that compares many models wants one place to add a row, tag it and select on it — not a file per model:

```json
{
  "_comment": "...",
  "agent_defaults": {"system_prompt_file": "../examples/conversation_demo/prompts/wica.md"},
  "entries": [
    {"id": "openai/gpt-4o", "agent": {"provider": "openai", "model": "gpt-4o", "api_key_env": "WICA_OPENAI_API_KEY"},
     "tags": ["api", "default"], "notes": "the demo's committed model"},
    {"id": "fake", "agent": {"provider": "fake", "model": "scripted", "system_prompt": "…", "model_kwargs": {"script": ["…"]}},
     "enabled": false, "tags": ["fixture"]}
  ]
}
```

- `id` — unique; the configuration's label everywhere (reports, summary rows, the batch folder `<out>/<id>/<scenario>/`, where a `/` in the id nests). The convention is `<provider>/<model>`, with a suffix when the same model is reached several ways (`huggingface-hub/gpt-oss-120b@cerebras`).
- `agent` — the config's agent block, merged over `agent_defaults` (entry keys win; an either-or pair of [config.md](config.md), `system_prompt` / `system_prompt_file` or `api_key` / `api_key_env`, merges as alternatives, so a fixture with an inline `system_prompt` sits in a matrix whose defaults carry the shared prompt file). A relative `system_prompt_file` resolves against the matrix file's directory when the target builds its `WicaConfig` (`WicaConfig.from_dict({"agent": entry.agent}, base_dir=entry.base_dir)`), so the summary shows the path as written, not an absolute one.
- `tags` — the selection vocabulary; the reference matrix uses `default` (the demo's committed model for that provider), `api` (hosted) and `fixture` (no network). `enabled: false` marks an opt-in row (`--include-disabled`). `requires_env` names extra variables that gate the entry beyond `agent.api_key_env`; `notes` is for the reader. Any other key is kept in `Entry.extra` for an application's target factory (an embodiment, an initial state).
- `Entry.skip_reason()` is the live tier's gate as a reason string — the key variable or one of `requires_env` unset — for the target to raise as `Skip`; an entry without any always runs.
- Validation names the file and the entry (`id`): a missing or duplicate id, a non-object `agent`, a malformed `tags` / `enabled` / `requires_env`, an unknown top-level key.

### The target

`src/wica/benchmark/runner.py` (the `wica.benchmark` subpackage: `runner.py`, `summary.py`, `cli.py`, re-exported from its `__init__`; like `wica.testing`, imported explicitly and never re-exported from `wica`):

```python
class Target(Protocol):
    label: str                      # the configuration's label: the config file's stem
    agent_config: dict[str, Any]    # the `agent` block, recorded in the summary header
    def open(self, scenario: Scenario) -> Wica: ...   # built, wired, started — or raise Skip(reason)
    def close(self) -> None: ...                       # the application's own teardown

class Skip(Exception): ...          # the target cannot run this configuration: key unset, extra absent, requirement missing

TargetFactory = Callable[[Entry], Target]  # one matrix entry → one Target
```

`open` is called **per run**: every repetition is a fresh system, so no history, World state or timers leak between repetitions. A `Skip` (an unset `api_key_env`, a provider extra not installed — the usual degraded starts, which the application already turns into a reason string) is recorded in the summary, never an error: like the live tier, a matrix runs with the keys at hand.

### The runner

`run_once(target, scenario, *, repetition, out_dir, extra_checks) -> Completed | Skipped` (`Completed` = the `Report` with the checks appended to its findings, the run's measures and the raw metric samples the summary pools — `report.metric_samples`, factored out of `compute_metrics`): `target.open(scenario)` → requirements check (`Probe.inputs` / `Probe.commands` / `output_command_name` against `scenario.requires`) → `TestRun(wica, scenario=name, sut_label=label)` → `run_script(run, scenario.script, timeout=scenario.timeout_s)` → `apply_checks` + `measures` → write `<out>/<label>/<scenario>/<i>.json` and `<i>.md` through `Report.write(path, extra={"bench": …})` — `write` gains an optional `extra` mapping merged into the JSON document (the Markdown is unchanged) → `finally: run.close(); target.close()`. Runs are **sequential**: parallel runs would confound the latency measures.

`run_matrix(targets, scenarios, *, k, out_dir, only_configs=(), only_scenarios=(), extra_checks=None, on_outcome=None) -> Summary` (`runner.py`; a pair skipped once is not retried for the remaining repetitions, `on_outcome` is the CLI's progress line, `skipped` the entries the caller set aside before building a target — disabled rows, recorded with scenario `"*"`); `write_summary(summary, out_dir)` → `summary.json` and `summary.md`, and `write_result(summary, result_dir)` (`summary.py`, below).

`cli(argv, *, target_factory, default_matrix, default_scenarios, default_runs, default_results, prog, extra_checks=None) -> int` (`cli.py`) — the one argparse definition every benchmark entry point reuses:

```
<prog> [--matrix FILE] [--only GLOB ...] [--exclude GLOB ...] [--tags TAG ...] [--exclude-tags TAG ...] [--include-disabled]
       [--scenarios DIR|FILE] [--scenario NAME ...] [--out DIR] [--k N] [--result NAME]
```

The selection flags are `select_entries`' (`tts-engine`'s vocabulary); an entry marked `enabled: false` is listed as skipped unless `--include-disabled`, and an empty selection is an argument error. `--out` is the batch folder, by default `<default_runs>/<YYYYMMDD-HHMM>/`. `--result NAME` keeps the batch as a result (below). Exit code 0 whatever the results: a benchmark compares, it does not gate.

### The summary

**One row per configuration first** — every run of every scenario pooled, the row a reader compares models by (`Summary.totals`, scenario `"*"`):

| Column | Meaning |
|---|---|
| runs, pass^k | runs over every scenario; 1 when every one of them passed |
| compliance | median over runs of *checks passed ÷ checks run* — how well the model keeps the primer's rules |
| answered | median over runs of *turns answered ÷ turns* |
| `reaction_latency`, `model_latency` p50 / p95 | pooled over every reaction |
| tokens in / out per turn, words per turn | per-run totals divided by the run's turns, median over runs |
| unprompted, noops on input, dropped | per-run counts, median over runs |
| failed | what failed, with counts |

The per-run rates (`compliance`, `answer_rate`, `input_tokens_per_turn`, `output_tokens_per_turn`, `words_per_turn`) are derived when pooling, from the run's check findings (`Completed.checks` names them), its measures and its token samples; they appear in `per_run` beside the measures.

Then **one table per scenario**, one row per configuration:

| Column | Meaning |
|---|---|
| runs / passed | *k* and how many runs were `report.ok` (every finding passed, verdict not `fail`) |
| pass rate | passed ÷ runs |
| pass^k | 1 when every run passed, else 0 — the "reliably follows the rules" bit |
| failed | the expectations and checks that failed, with counts |
| `reaction_latency`, `model_latency` | p50 / p95 pooled over the runs' samples |
| `reactions_per_input` p50, `unprompted_output`, `dropped_inputs` | the re-trigger chain's cost |
| `input_tokens`, `output_tokens` per run (summed per run, median over runs); cache-read share over the reactions that report one | money |
| `output_words` per run | the person's time |

A header lists each configuration's full `agent` block: the configuration under test is more than a model name (provider quirks, `hf_provider`, history settings all move the numbers). **No thresholds anywhere**: the summary is a comparison, the reader draws the line; `expect_latency` remains a scenario's to use.

`summary.json` holds the same numbers (`totals` and `cells`: pooled `MetricSummary` per key, counts, the skips with reasons) under a header: the start time, the git commit the batch ran at (when the working tree is a repository), *k*, the scenario names and each configuration's `agent` block.

**Batches are working data; results are kept.** The batch folder (`--out`: the per-run `.json` / `.md` and the two summary files) is git-ignored, like `tests-e2e/reports/`. A batch that marks a decision — a model chosen, a primer change measured, a regression confirmed — is kept as a **result** with `--result <name>`: `write_result(summary, results_dir / f"{stamp}_{name}")` copies `summary.md` and `summary.json`, writes `runs.json` (the manifest: the batch folder, the header above, the skips) and creates, when absent, two files the runner never rewrites: `description.md` (the header filled in — date, commit, matrix, scenarios, *k* — then a `TODO` for what the runner cannot know: the machine, the network, why this batch was run) and `findings.md` (empty headings: what the numbers say, which decision they fed, what remains). Both are finished by hand before the result is committed. A spec or plan that quotes a benchmark number quotes the **result id** (the folder name) and reads it from the per-configuration table, never a batch; `benchmarks/README.md` lists every result, one line each. The per-run reports stay in the batch folder — a result is the reading, not the raw data (open question 5).

### The reference benchmark: the conversation demo

`benchmarks/` — a root folder outside the package, the framework's own benchmark whole: the demo target and entry point, the matrix file, the scenario set, the batches and the results. Nothing imports it but its tests, so none of it is installed; it needs the `demo` group (the demo's presenters), which the package never requires.

- `benchmarks/bench.py` — `DemoTarget(entry)`: `label` is the entry's id; `entry.skip_reason()` ⇒ `Skip` before anything starts (the demo's explore-only degrade is of no use to a benchmark); `WicaConfig.from_dict({"agent": entry.agent}, base_dir=entry.base_dir)`, `build_system(config)`; `wica is None` ⇒ `Skip(config_error)`; `wire(wica, TranscriptLog(wica, display_entry), SpeakingSlot())` (the presenters without the page, as the demo's own tests do); `wica.start()`. `close()`: `wica.close()`. `agent_config` is the file's `agent` block. `main()`: `cli(argv, target_factory=DemoTarget, default_matrix="benchmarks/matrix.json", default_scenarios="benchmarks/scenarios", default_runs="benchmarks/runs", default_results="benchmarks/results", prog="wica-demo-bench")`. Run from the repo root: `uv run --group demo python -m benchmarks.bench --k 5` (it imports the demo's `examples.conversation_demo.app`, which resolves from the root the way the tests' `pythonpath` does).
- `benchmarks/matrix.json` — the matrix: `agent_defaults` carries the demo's persona (`../examples/conversation_demo/prompts/wica.md`, so the persona under test is the demo's own), and one `default`-tagged entry per provider carries the agent block of the demo's `agent.<provider>.config.json` (a copy of five short blocks, accepted so the matrix is one file and the demo's configs stay the demo's; the live smoke verifies the ids). More models are rows added here, `<provider>/<model>`, tagged to select on — the README's "Model catalog" says how. The `fake` entry is the key-less path: a scripted fake that greets once, `enabled: false` and tagged `fixture` so it never joins a default batch (it passes `greeting` only): `--only fake --include-disabled --scenario greeting`.
- `benchmarks/scenarios/` — the scenario set below; `benchmarks/runs/` (ignored), `benchmarks/results/` and `benchmarks/README.md` — the batches and results, as above.

**The workload**, `benchmarks/scenarios/`: two session scenarios, run *k* times per matrix entry, each a conversation paced like a person's — after an utterance the tester waits for the reply to be issued (`expect_output`) and then for the agent to be **idle** (`expect_idle`: the reply spoken, the step its completion wakes ended — no timer), so the next utterance is never dropped as busy; only the barge-in turns talk over the robot. A failed expectation does not end the run, so a missed turn is a count in the summary, not a lost session.

| Scenario | Session | Requires |
|---|---|---|
| `conversation` | "Hello there!" → "What can you do?" → a request for a long story, interrupted with "Stop!" as soon as the reply is issued (`cancel_command`, then a spoken confirmation) → "Hi, I'm Alex." and "Can you hear me?" 0.5 s apart (busy handling) → "Thanks, that's all for now. Goodbye!" → silence for 4 s. Seven user turns; all four primer checks | `speech_input`, an output Command — generic, so it runs unchanged against any application of that shape |
| `commands` | "Hi!" then `set_emotion("happy")` while the robot still speaks → no second `say` for 4 s (the `noop` clause on the re-trigger step) → "Please dance for me!" → `dance`, then "Stop!" two seconds in → `cancel_command`, a spoken confirmation → `set_closest_user("u2")` → `switch_user_tracking(user_id="u2")` → `set_closest_user("")` → `switch_user_tracking`. Three primer checks (`answers_each_input` left out: a sensor write is not addressed to the agent) | `speech_input`, `emotion`, `closest_user`; `dance`, `switch_user_tracking`; the output Command — the demo's own |

**No rule-by-rule micro scenarios ship.** A one-rule scenario run *k* times is a repeated e2e test, not a benchmark, and the session reports already name the clause a model broke — the failed check, the failed expectation, the timeline beside them. When a clean-context check of one rule is wanted (does a model that breaks a rule mid-session also break it cold?), a fifteen-line scenario file written for that model and run with `--scenarios <file>` does it; nothing needs committing.

### Testing

Fast tier, `tests/testing/`:

- `test_scenario.py` — a scenario round-trips; `name` ≠ stem, a step with two calls, an unknown tester Command, non-object args, an unknown check name each raise with the file and step index; `load_scenarios` orders by name.
- `test_checks.py` — one passing and one failing hand-built timeline per built-in check (real `ReactionTrace` / `TriggerTrace` instances for the trigger-kind logic); `measures` on a known timeline, the turn numbers included; registering an extra check under a built-in's name raises.
- `tests/benchmark/test_matrix.py` — defaults merge under each entry, extra keys are kept, selection by id glob and tag, `skip_reason` against the environment, faults name the entry.
- `tests/benchmark/test_runner.py`, `test_summary.py`, `test_cli.py` — the chain over the fake provider, key-less (`test_summary.py` also pins the totals, `compliance`, `answer_rate` and the per-turn numbers, and that the markdown opens with the per-configuration table): a target built in-test over a fake SUT (`speech_input` + `say`) with a script that satisfies `greeting`, `run_once` twice → both reports `ok`, the files exist, `aggregate` gives pass rate 1.0 and pass^k 1; a scripted model that `noop`s the greeting fails `no_noop_on_input` and pass^k is 0, while one that replies twice stays `ok` and moves `output_calls` to 2; a target raising `Skip` is a summary skip, not an error; a scenario whose `requires` the target lacks is skipped with the reason; `cli` selects entries by glob and tag, lists a disabled row as skipped unless opted in, and writes the summary.

Full-loop tier, two files, selected **by file** as [ci.md](ci.md) requires:

- `tests-e2e/test_demo_bench_fake.py` — always-run, key-less: the whole `conversation` session through `DemoTarget` over the matrix's `fake` entry (its script plays every turn, the long story cancelled by a scripted tool-call id), asserting the report is `ok`, seven turns all answered, one unprompted output (the confirmation on the cancel's completion step) and both files written; a one-utterance greeting probe (`tests-e2e/support.greeting_probe`, a fixture, not benchmark data) likewise; the committed matrix loads with one enabled `default` per provider and the disabled fixture; the workload loads; a one-run `run_matrix` batch kept with `write_result` produces the five files, a `compliance` of 1 in the total, and leaves an existing `description.md` untouched. Joins the e2e-fake job.
- `tests-e2e/test_demo_bench.py` — live: the greeting probe once through `DemoTarget` per enabled matrix entry (the cheapest probe that the provider drives the demo at all), asserting a reaction ended `ok` and `model_latency` has a sample — **not** that the model answered: one sample of a model's behavior is the benchmark's to measure with *k*, and must not turn `main` red. A `Skip` from the target (its `api_key_env` unset) becomes `pytest.skip`, the tier's rule ([testing.md](testing.md)). Joins the `live` job.

Both import `benchmarks.bench` as the demo tests import `examples.conversation_demo.app` (the repo root on `sys.path`). One pytest run is a smoke test; the benchmark is the entry point with `--k`.

### Placement

The split follows one rule: **what another repo imports is in the package; what only this benchmark needs is not.**

| Piece | Home | Why |
|---|---|---|
| scenario format and loader, primer checks, measures | `src/wica/testing/scenario.py`, `checks.py` | per-run concerns of the toolkit; `to_scenario()` (phase 2) emits the format |
| the matrix format, `Target`, `Skip`, `run_once`, `run_matrix`, `Summary`, `write_result`, `cli` | `src/wica/benchmark/` (`matrix.py`, `runner.py`, `summary.py`, `cli.py`) | what an application's benchmark imports; a subpackage of its own because it is a comparison tool, not a test tool |
| `DemoTarget` and the entry point, the matrix file, the workload, the batches, the results, the README | `benchmarks/` at the root | this repo's own benchmark; nobody imports it, so it is not installed and may depend on the `demo` group |

Neither subpackage is re-exported from `wica`, and neither adds a dependency: the Gradio presenters the demo target wires are the `demo` group's, imported only by `benchmarks/bench.py`. `benchmarks/` joins pyright's `include` and is never run by CI beyond the two smoke tests above. The plan adds the modules to the AGENTS.md project map (a `benchmarks/` row in the top-level layout, a `benchmark/` row among the modules) and to this spec's `code:` / `tests:` (the frontmatter above).

## Open questions

Deferrals, not blockers.

1. **Pass^k with n > k.** With *k* repetitions the summary reports "all passed". Estimating pass^k from more runs than *k* (C(c,k)/C(n,k)) is a later refinement once runs are cheap enough to oversample.
2. **Comparing two summaries.** Where summaries accumulate and how two runs of the matrix are diffed (a model upgrade, a primer change) is the tester spec's open question 4, unchanged.
3. **A judge pass** over the saved reports for relevance and tone — the tester spec's open question 2; the runner leaves room for it (one JSON per run) and adds nothing.
4. **Cost ceilings.** A matrix of five models × two sessions × five runs is a few hundred model calls and about an hour; the CLI has filters but no budget flag. Add one when a batch first surprises someone.
5. **Raw data in results.** A result keeps the summary, not the per-run reports; re-deriving a new measure over an old result is therefore impossible. `tts-engine` keeps its raw records compressed in the result folder for that reason. Add `reports.jsonl.gz` to `write_result` when a measure is first added after a result exists. Merging batches into one result and resuming an interrupted batch (also `tts-engine`'s) are left out for the same reason: a WICA batch is minutes to an hour, re-running is cheaper.
6. **Cost per model.** The per-configuration row has tokens per turn but no money column; that needs a price per entry in the matrix (`price_in`, `price_out` per million tokens, in `Entry.extra` or as defined keys). Add it with the first result whose reading turns on cost.
