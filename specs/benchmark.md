---
code:
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
  - tests/benchmark/test_matrix.py
  - tests/benchmark/test_runner.py
  - tests/benchmark/test_summary.py
  - tests/benchmark/test_cli.py
  - tests-e2e/test_demo_bench_fake.py
---

# WICA benchmark: comparing models over scenarios

**Status:** Implemented

## Purpose

[wica-tester.md](wica-tester.md) gives one run against one `Wica`: a timeline, measures and a report. A **benchmark** is many such runs compared — the same scenarios, against the same application, with a different model behind the Agent each time, repeated enough to see past model variance. The scenarios it runs and the checks that judge them are the tester's ([wica-tester.md](wica-tester.md), "Scenarios"): a scenario is a fixed user as `input` / `wait` / `expect` / `check` steps, and a check is a rule judged after the run. A test asserts them; a benchmark **measures** them — how often each model keeps each check. This spec adds what turns runs into that comparison, application-agnostic:

- the **workload**: scenarios written as **sessions** of several turns, the same for every model, with few expects and many checks;
- the **runner**: a config matrix × a scenario set × *k* repetitions, one report per run, one summary with **one row per model** pooled over every run (pass^k, rule compliance, turns answered, latency, tokens and words per turn) and then the same per scenario.

It also ships the framework's own **reference benchmark**: the conversation demo ([conversation-demo.md](conversation-demo.md)) as the system under test, a matrix of models, and a workload of two session scenarios written against the demo's `speech_input` + `say` shape, run *k* times per model. That benchmark answers "how well does model X drive a WICA agent at all"; an application built on WICA (the first is `wica-interaction`, `wica-interaction/specs/benchmark.md`) reuses the runner, the scenario format and the built-in checks for its own target, and adds the checks its persona and body demand. The generic layer lives in the package (`wica.testing` for scenarios and checks, `wica.benchmark` for the comparison) so that no application re-implements a runner, and so that two applications' numbers are produced the same way; the reference benchmark — target, matrix, scenarios, results — lives in a root `benchmarks/` folder outside the package, as nothing imports it (the layout `tts-engine` uses for its own benchmark).

This spec absorbs phases 3 and 4 of [wica-tester.md](wica-tester.md) ("Phases"): the runner and the demo scenario. Trace export (`report.to_scenario()`) stays with the live-LLM tester (phase 2), since it is the live tester's calls it exports.

## Settled

### Vocabulary

- **Target** — an application stood up as a running `Wica` for one benchmark configuration, plus how to shut it down. The runner never builds a `Wica` itself; the application does, exactly as its entry point would (the demo's `build_system` + `wire` + `start`, the interaction app's `build_system` + `wire` + `_start`), so what is benchmarked is the real system with its real persona, Commands and seeding. Only the peripherals the application chooses to bypass are doubled, by the application.
- **Configuration** — one model under test: an **entry** of the matrix file, whose `agent` block (provider, model, `hf_provider`, `api_key_env`, history settings) is what differs between entries; the persona is shared through the file's `agent_defaults`. Its **label** in every report is the entry's `id`. The file is the **matrix**.
- **Scenario** — a fixed user as `input` / `wait` / `expect` / `check` steps, one JSON file named by its stem ([wica-tester.md](wica-tester.md), "Scenarios"). A benchmark's scenarios are **sessions** — several user turns, paced as a person would pace them — so one run yields a whole conversation's worth of turns to measure; a scenario that pins one rule in a few steps is a regression probe, not a workload.
- **Run** — one (configuration, scenario, repetition): one `TestRun`, one `Report`. Its outcome is `completed` or `failed` (an expect did not hold, or a wait timed out); only a completed run is **scored**.
- **Batch** — one invocation of the runner: a matrix × a scenario set × *k*, written to one dated folder of per-run reports plus a summary. Working data, git-ignored.
- **Result** — a batch worth keeping: its summary and manifest copied under `benchmarks/results/<YYYYMMDD-HHMM>_<name>/` with a hand-written description and findings. Committed, and the only thing a spec or plan cites.
- **Check** — a rule judged after the run (the tester's). What a benchmark measures is a model's **compliance**: the share of checks it kept over its scored runs, overall and per check.
- **Measure** — a number taken from a run, never thresholded by the runner (the tester spec's "measures, not thresholds").

### Scenarios and checks, as a benchmark uses them

The format, the four step kinds, the built-in checks and their windows are the tester's ([wica-tester.md](wica-tester.md), "Scenarios"); a benchmark adds three rules of use:

- **Expects are few.** An expect stops the run, and a stopped run is not scored, so a benchmark scenario keeps them for what the session cannot continue without. What is compared between models is written as checks.
- **Pacing is explicit and generous.** `wait` steps carry `within` budgets that include the slowest model's think time; a timeout fails the run like a missed expect.
- **Never write over a reaction.** An input written while a reaction is in flight is dropped as busy ([agent.md](agent.md), "Concurrency and long-running Commands in v1"), and a benchmark that drops inputs measures the framework's drop rule, not the model. Every input that talks over the robot follows a `{"wait": "ready"}`: the Agent can take an input, while its Commands (a speech, a dance) may still run.
- **Checks sit on the input they judge**, and the session-wide ones (`no_failed_reactions`, `no_colliding_cancel`, …) at the top level.

**An answer may take several steps.** A model may issue a side Command first (an emotion change) and speak on the step its completion wakes; the checks judge a whole turn, so that is one answer, and the measures time it to its first word. **A second utterance in the same turn is a measure, not a failure.** The output Command's completion re-triggers a step so the model *may* chain one ([commands.md](commands.md), "The output Command"); a rule breaking on it would penalize designed behavior. The runner counts it (`extra_output` below), next to the checks.

**Application checks.** The runner and the CLI take `extra_checks`, the tester's `build_registry` adds them beside the built-ins (the interaction app's `no_collision`, `spoken_style`, …), and its scenarios select them like any other.

**Measures.** `measures(timeline, context) -> dict[str, float]` adds per run, next to the framework's metrics: `output_calls` (how many times the output Command was issued), `extra_output` (issued in a turn after the turn's first one — `output_calls − turns_answered`; a turn's first utterance is its answer whichever step issued it, so a confirmation spoken on the step a cancel's completion wakes is "Stop!"'s answer, not an extra), `noop_steps`, `failed_reactions`, `output_words` (summed words of the output Command's string args — a cost of the person's time), and the **turn numbers** a session is read by: `turns` (opened by writes to triggering Inputs, as `answers_each_input` defines them), `turns_answered` (the output Command issued within the turn), `noops_on_input` (reactions that answered a write to a scenario input with `noop` alone). Written into the run's JSON under `"bench"`; pooled into the summary. Beside them, `answer_latencies(timeline, context)` gives one sample per answered turn — the write that opened it → its first output Command — pooled with the framework's latency samples as `answer_latency`: the time to the first word, the latency a person hears, however many steps the answer took.

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

**Failed runs are counted, never scored.** A run that failed — an expect did not hold or a wait timed out — lowers the **completed** rate, counts against pass^k and is listed with the step that stopped it, but contributes to **no other metric**: no compliance, no latency, no tokens, no measure. Its numbers describe a session that did not happen as written, and mixing them in would compare models on different conversations.

**Read completed before compliance.** Compliance is computed over completed runs only, so a model that fails hardest — never answers, returns empty responses, malformed tool calls — completes few runs and keeps a flattering compliance on the rest. The completed rate sits right before compliance in every table so the two are never read apart (the first batch's gpt-oss rows: 100% compliance, half the runs completed).

**One row per configuration first** — every run of every scenario pooled, the row a reader compares models by (`Summary.totals`, scenario `"*"`):

| Column | Meaning |
|---|---|
| runs, completed | runs over every scenario; completed ÷ runs (the rest failed) |
| pass^k | 1 when every run completed and kept every check, else 0 |
| compliance | median over completed runs of *checks kept ÷ checks applied* |
| answered | median over runs of *turns answered ÷ turns* |
| `answer_latency` p50 / p95 | pooled over every answered turn of the completed runs: input → first word |
| `reaction_latency`, `model_latency` p50 / p95 | pooled over every reaction of the completed runs |
| tokens in / out per turn, words per turn | per-run totals divided by the run's turns, median over runs |
| extra replies, noops on input, dropped | per-run counts, median over runs |
| checks broken | which checks broke, with counts |
| stopped at | the steps runs stopped at, with counts — read it before the rest, since a model that stops often is scored on few runs |

Then **one table of checks**, one row per configuration and one column per check any completed run applied (the built-ins first): *kept / applied* over the configuration's scored runs. It answers "which rule does this model break" at a glance.

The per-run rates (`compliance`, `answer_rate`, `input_tokens_per_turn`, `output_tokens_per_turn`, `words_per_turn`) are derived when pooling, over completed runs only, from the run's check findings, its measures and its token samples; they appear in `per_run` beside the measures.

Then **one table per scenario**, one row per configuration:

| Column | Meaning |
|---|---|
| runs, completed | *k*, and completed ÷ runs |
| passed | runs that completed and kept every check |
| pass rate | passed ÷ runs |
| pass^k | 1 when every run passed, else 0 — the "reliably follows the rules" bit |
| checks broken | the checks that broke over the completed runs, with counts |
| `answer_latency`, `reaction_latency`, `model_latency` | p50 / p95 pooled over the runs' samples |
| `reactions_per_input` p50, `extra_output`, `dropped_inputs` | the re-trigger chain's cost |
| `input_tokens`, `output_tokens` per run (summed per run, median over runs); cache-read share over the reactions that report one | money |
| `output_words` per run | the person's time |

A header lists each configuration's full `agent` block: the configuration under test is more than a model name (provider quirks, `hf_provider`, history settings all move the numbers). **No thresholds anywhere**: the summary is a comparison, the reader draws the line; `expect_latency` remains a scenario's to use.

`summary.json` holds the same numbers (`totals` and `cells`: pooled `MetricSummary` per key, counts, the skips with reasons) under a header: the start time, the git commit the batch ran at (when the working tree is a repository), *k*, the scenario names and each configuration's `agent` block.

**Batches are working data; results are kept.** The batch folder (`--out`: the per-run `.json` / `.md` and the two summary files) is git-ignored, like `tests-e2e/reports/`. A batch that marks a decision — a model chosen, a primer change measured, a regression confirmed — is kept as a **result** with `--result <name>`: `write_result(summary, results_dir / f"{stamp}_{name}")` copies `summary.md` and `summary.json`, writes `runs.json` (the manifest: the batch folder, the header above, the skips) and creates, when absent, two files the runner never rewrites: `description.md` (the header filled in — date, commit, matrix, scenarios, *k* — then a `TODO` for what the runner cannot know: the machine, the network, why this batch was run) and `findings.md` (empty headings: what the numbers say, which decision they fed, what remains). Both are finished by hand before the result is committed. A spec or plan that quotes a benchmark number quotes the **result id** (the folder name) and reads it from the per-configuration table, never a batch; `benchmarks/README.md` lists every result, one line each. The per-run reports stay in the batch folder — a result is the reading, not the raw data (open question 5).

### The reference benchmark: the conversation demo

`benchmarks/` — a root folder outside the package, the framework's own benchmark whole: the demo target and entry point, the matrix file, the scenario set, the batches and the results. Nothing imports it but its tests, so none of it is installed; it needs the `demo` group (the demo's presenters), which the package never requires.

- `benchmarks/bench.py` — `DemoTarget(entry)`: `label` is the entry's id; `entry.skip_reason()` ⇒ `Skip` before anything starts (the demo's explore-only degrade is of no use to a benchmark); `WicaConfig.from_dict({"agent": entry.agent}, base_dir=entry.base_dir)`, `build_system(config)`; `wica is None` ⇒ `Skip(config_error)`; `wire(wica, TranscriptLog(wica, display_entry), SpeakingSlot())` (the presenters without the page, as the demo's own tests do); `wica.start()`. `close()`: `wica.close()`. `agent_config` is the file's `agent` block. `main()`: `cli(argv, target_factory=DemoTarget, default_matrix="benchmarks/matrix.json", default_scenarios="benchmarks/scenarios", default_runs="benchmarks/runs", default_results="benchmarks/results", prog="wica-demo-bench")`. Run from the repo root: `uv run --group demo python -m benchmarks.bench --k 5` (it imports the demo's `examples.conversation_demo.app`, which resolves from the root the way the tests' `pythonpath` does).
- `benchmarks/matrix.json` — the matrix: `agent_defaults` carries the demo's persona (`../examples/conversation_demo/prompts/wica.md`, so the persona under test is the demo's own), and one `default`-tagged entry per provider carries the agent block of the demo's `agent.<provider>.config.json` (a copy of five short blocks, accepted so the matrix is one file and the demo's configs stay the demo's; the live smoke verifies the ids). More models are rows added here, `<provider>/<model>`, tagged to select on — the README's "Model catalog" says how. The `fake` entry is the key-less path: a scripted fake that plays the `conversation` session, `enabled: false` and tagged `fixture` so it never joins a default batch: `--only fake --include-disabled --scenario conversation`.
- `benchmarks/scenarios/` — the scenario set below; `benchmarks/runs/` (ignored), `benchmarks/results/` and `benchmarks/README.md` — the batches and results, as above.

**The workload**, `benchmarks/scenarios/`: two session scenarios, run *k* times per matrix entry, each a conversation paced like a person's — after an utterance a `{"wait": "idle"}` holds the next one until the reply is spoken and the step its completion wakes has ended, so nothing is dropped as busy; only the barge-in inputs talk over the robot, after a `{"wait": "speaking"}` or `{"wait": {"running": "dance"}}` and then a `{"wait": "ready"}`, so they too land while the Agent can take them. Neither scenario has an expect: everything compared is a check.

| Scenario | Session (inputs, with the checks on each) | Top-level checks |
|---|---|---|
| `conversation` | "Hello there!" · "What can you do?" · a request for a long story · "Stop!" as soon as it is spoken (`cancels_when_asked`) · "Hi, I'm Alex." and "Can you hear me?" as soon as the Agent is ready, while the first answer may still be spoken · "Thanks, that's all for now. Goodbye!" — generic: it needs only `speech_input` and an output Command | `answers_each_input` (so "Stop!" gets a spoken acknowledgement too), `no_noop_on_input`, `no_failed_reactions`, `no_colliding_cancel` |
| `commands` | "Hi!" · `emotion = "happy"` while it speaks (`quiet_on_background_change`) · "Please dance for me!" (`acts_when_asked: dance`) · "Stop!" two seconds into the dance, once the Agent is ready (`cancels_when_asked`, `answers_each_input`) · `closest_user = "u2"` (`acts_when_asked: switch_user_tracking(user_id=u2)`) · `closest_user = ""` (`acts_when_asked: switch_user_tracking`) — the demo's own | `no_noop_on_input`, `no_failed_reactions`, `no_colliding_cancel` |

**No rule-by-rule micro scenarios ship.** A one-rule scenario run *k* times is a repeated e2e test, not a benchmark, and the session reports already name the clause a model broke — the failed check, the failed expectation, the timeline beside them. When a clean-context check of one rule is wanted (does a model that breaks a rule mid-session also break it cold?), a fifteen-line scenario file written for that model and run with `--scenarios <file>` does it; nothing needs committing.

### Testing

Fast tier, `tests/testing/`:

- The scenario format and the checks are tested with the tester ([wica-tester.md](wica-tester.md)): `tests/testing/test_scenario.py`, `test_checks.py`.
- `tests/benchmark/test_matrix.py` — defaults merge under each entry, extra keys are kept, selection by id glob and tag, `skip_reason` against the environment, faults name the entry.
- `tests/benchmark/test_runner.py`, `test_summary.py`, `test_cli.py` — the chain over the fake provider, key-less (`test_summary.py` also pins the totals, `compliance`, `answer_rate` and the per-turn numbers, and that the markdown opens with the per-configuration table): a target built in-test over a fake SUT (`speech_input` + `say`) with a script that satisfies a greeting scenario, `run_once` twice → both reports `ok`, the files exist, `aggregate` gives pass rate 1.0 and pass^k 1; a scripted model that `noop`s the greeting breaks `no_noop_on_input` and pass^k is 0, while one that replies twice stays `ok` and moves `output_calls` to 2; a run stopped by an expect or a wait counts in pass^k and the completed rate and in no other number; a target raising `Skip` is a summary skip, not an error; a scenario whose `requires` the target lacks is skipped with the reason; `cli` selects entries by glob and tag, lists a disabled row as skipped unless opted in, and writes the summary.

Full-loop tier, one key-less file plus a rule, selected **by file** as [ci.md](ci.md) requires:

- `tests-e2e/test_demo_bench_fake.py` — always-run, key-less: the whole `conversation` session through `DemoTarget` over the matrix's `fake` entry (its script plays every turn, the long story cancelled by a scripted tool-call id), asserting the run completed with every check kept, seven turns all answered with no extra reply (the confirmation on the cancel's completion step is "Stop!"'s answer), seven answer-latency samples, no dropped input and both files written; a one-utterance greeting probe (`tests-e2e/support.greeting_probe`, a fixture, not benchmark data) likewise; the committed matrix loads with one enabled `default` per provider and the disabled fixture; the workload loads; a one-run `run_matrix` batch kept with `write_result` produces the five files, a `compliance` of 1 in the total, and leaves an existing `description.md` untouched. Joins the e2e-fake job.
- **No live test over the matrix.** The matrix is the benchmark's own list — models are added, disabled and retired as the benchmark's questions change, and each one carries its provider's quotas and latencies — so no test is parametrized over it: a matrix edit must never change what the test suite runs, and a model the key's plan cannot call must never turn `main` red. The live e2e set stays the fixed, scoped `PROVIDER_CONFIGS` list of [testing.md](testing.md); the matrix's models are exercised when the benchmark is run.

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
