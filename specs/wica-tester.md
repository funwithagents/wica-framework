---
code:
  - src/wica/testing/__init__.py
  - src/wica/testing/probe.py
  - src/wica/testing/harness.py
  - src/wica/testing/report.py
  - src/wica/testing/tester.py
  - src/wica/wica.py
  - src/wica/world.py
  - src/wica/agent.py
  - src/wica/command.py
  - src/wica/instrumentation.py
  - tests-e2e/support.py
  - tests-e2e/conftest.py
tests:
  - tests/testing/test_probe.py
  - tests/testing/test_harness.py
  - tests/testing/test_report.py
  - tests/testing/test_scripted_tester.py
  - tests-e2e/test_tester_flows.py
  - tests-e2e/test_wica.py
  - tests-e2e/test_fake_flows.py
---

# WicaTester

**Status:** Stable

## Purpose

A toolkit for testing a `Wica` instance **through its own World**: a **Probe** that connects to the system under test, a **Harness** that records a timestamped timeline and derives measurements from it, and **drivers** that decide what happens next — a Python test, a scripted fake model, or a live model playing a user. The name comes from the fullest form: the **WicaTester**, a *second* `Wica` whose shape is the mirror image of the one under test, acting on it through Commands and perceiving it through Inputs.

Today the full-loop tier ([testing.md](testing.md)) hand-rolls its observation each time — `wait_until`, `find_command_entry`, per-key listeners, a `RecordingSink` copied across three files — and records nothing about *when* things happened. The toolkit replaces that boilerplate with one observation layer and adds what a hand-written test never has: reaction latencies, dropped-input detection, cost per turn and a report, with no change to what the test asserts.

The background survey, the comparison with the field and the brainstorm minutes live in [_wica-tester-study.md](_wica-tester-study.md); this spec records the settled design. The framework changes the toolkit needed shipped first (plans `202609131341` and `202609141552`), so everything below builds on the public surface as it is.

## Settled

### Vocabulary: SUT, mirror, drivers

The `Wica` under test is the **SUT**. Its World has *Inputs* (entries the application updates) and its Agent issues *Commands*. The tester's shape is the SUT's mirror image, expressed with nothing but WICA's own Inputs and Commands vocabulary:

| SUT side | Tester side |
|---|---|
| Input entry `speech_input` (application writes it) | Command `set_speech_input(value)` → `sut.world.update("speech_input", value)` |
| A Command the model issues (e.g. an output Command) | Input entry `sut:action` / `sut:output`, fed from the SUT's Events |
| A Command outcome (`agent:command:<call_id>` going terminal) | Input entry update the tester observes |
| Free text to the output sink | Input entry `sut:reasoning`, visible, non-triggering |

Nothing is needed on the SUT side beyond what every application already registers. Names are **read from the SUT** (`world.keys()`, `agent.commands`, `agent.output_command_name`), never assumed: the toolkit knows no application Command by name.

Three **drivers** sit on top of one wiring (see "Drivers"): a Python test calling the toolkit directly, a scripted-fake tester `Wica`, and a live-LLM tester `Wica`. The first two are deterministic regression tools; the third is exploratory.

### What the framework provides (and what it deliberately does not)

The Probe connects through public surface only:

- `sut.world.update(key, value)` is thread-safe from any thread; listeners and `on_trigger` dispatch on the SUT's loop. Two `Wica` instances therefore talk in both directions with no bridge: a tester listener runs on the SUT loop and calls `tester.world.update(...)`, which hops to the tester loop itself. WicaTester is the validating case for multi-instance ([wica.md](wica.md) open question #1); the first `Done` plan closes that question.
- **Schema enumeration**: `world.keys()` and `world.get_config(key)` (`WorldEntryConfig`: declared `type`, `triggers_llm_call`, serializers) give one `set_<key>` Command per SUT Input with the right argument type; `agent.commands` (read-only mapping) plus `Command.tool` give each Command's description and args schema; `agent.output_command_name` names the conversational channel.
- **Instrumentation Events** on the facade — `on_world_trigger` (raw), `on_agent_trigger` (observed), `on_agent_prompt` (rendered messages), `on_agent_command` (`CommandIssued(name, args, call_id)`, fired for every issuance including `noop`, once the `agent:command:<call_id>` entry is registered so a per-key listener can be attached immediately), `on_agent_text` (the step's free text, before the sink receives it).
- **Per-Command re-trigger control**: `Command(fn, triggers_on_completion=False)`.
- `provider: "fake"` ([fake-provider.md](fake-provider.md)) scripts either side.

Three things stay on the tester's side, by design:

| Need | Why no framework change |
|---|---|
| The SUT's transcript (`Agent._history` is private) | `on_agent_prompt` yields the exact rendered messages of every step; the Harness records them. |
| Timestamps on Events | The **framework** stamps them ([instrumentation.md](instrumentation.md)): every reaction ends with an `on_agent_reaction_ended` record carrying the phase stamps, every Command with an `on_agent_command_ended` record, and a dropped trigger fires `on_agent_trigger_dropped`. The Harness reads those stamps for every latency measure and stamps only its *own* actions on arrival. (Reversed on 2026-09-16 — before that the Harness stamped `time.monotonic()` itself and inferred step boundaries from event order.) |
| Detecting an input dropped while a step was in flight ([_analysis.md](_analysis.md) P1) | That is a *finding*, not a blocker: the framework's `on_agent_trigger_dropped` ([instrumentation.md](instrumentation.md)) names each one, and the Harness reports the count as **dropped triggers**. |

The toolkit never touches the application's output sink. Free text is observed through `on_agent_text`; the sink stays the single replaceable slot the application owns.

### Architecture: three layers, one wiring

```
                     ┌──────────────────────── WicaTester ────────────────────────┐
                     │                                                             │
   scenario ──────►  │  Driver          Harness                     Probe          │
   (script / text)   │  ─ Python test   ─ timeline recorder         ─ mirror       │        ┌───── SUT ─────┐
                     │  ─ fake Wica     ─ latency metrics           ─ Commands ────┼──update─►  world        │
                     │  ─ live-LLM Wica ─ expectations / issues     ─ Inputs   ◄───┼──events──  agent       │
                     │                  ─ report                                   │        └───────────────┘
                     └─────────────────────────────────────────────────────────────┘
```

The three are separable and stay separate: the Probe is policy-free wiring, the Harness is measurement and expectations, and a driver only decides what to do next.

### Probe — the connection

`Probe(sut: Wica)` inspects the SUT once and exposes:

- `inputs`: the SUT entries the tester may write, each as `set(key, value)`; validation against the declared `type` is `sut.world.update`'s own.
- `commands`: the SUT's registered Commands (name, description, args schema), minus the WICA-native `noop`/`cancel_command`, with the output Command flagged.
- One normalized stream of **`SutEvent`** records from subscriptions on the facade Events and on each `agent:command:*` key as it appears: `input_written`, `trigger_raw`, `trigger_observed`, `trigger_dropped`, `prompt` (message count and rendered size, optionally the text), `command_issued`, `command_state` (running / complete / failed / cancelled, with result or error), `command_ended` (carrying the framework's `CommandTrace`), `output_text`, `reaction_ended` (carrying the framework's `ReactionTrace`). Every record is stamped on arrival with the framework's wall clock (`wica.instrumentation.now()`, as POSIX seconds), so a Harness action and a framework record subtract.

**The stream is causally ordered.** Every Probe handler runs on the SUT's loop thread, the per-key `agent:command:*` listener included — it is registered as an *async* listener, which the World runs as a loop task queued before the trigger the same write emits, whereas a sync listener is offloaded to the World's thread-pool executor and can be delivered after the events its state change caused. So a Command's `complete` record always precedes the step its completion triggers, which the Harness watermark relies on: an expectation on that state followed by one on the step's text never skips the text.

The Probe does not know whether a script or a model is driving.

### Harness — timeline, metrics, expectations, report

`TestRun` owns a **timeline**: every `SutEvent` and every tester action, stamped on arrival. The measurements come from the framework's reaction and Command records ([instrumentation.md](instrumentation.md), "Metrics"), which the Probe forwards as `reaction_ended` / `trigger_dropped` / `command_ended` events; the Harness pairs them with its own actions and adds the distributions. The "derived from" column below names the pairing, no model involved:

| Metric | Derived from |
|---|---|
| **Reaction latency** — input written → first Command or output text | `input_written` → first `command_issued` / `output_text` of the step it triggered |
| **Think time** — trigger observed → prompt → first Command | `trigger_observed` → `prompt` → `command_issued`; the `prompt` → `command_issued` slice is pure model latency |
| **Coalescing delay** | `trigger_raw` → `trigger_observed` for the same `(key, id)` |
| **Dropped inputs** | `trigger_raw` with no matching `trigger_observed` |
| **Command duration** | `command_state(running)` → terminal |
| **Barge-in responsiveness** | new `input_written` while the output Command runs → `cancel_command` issued |
| **Steps per user turn** (cost) | `prompt` count between two `input_written` |
| **Token usage** (cost) | `ReactionTrace.usage`, input and output tokens per reaction |

The metric keys of the report are core's measure names (`reaction_latency`, `model_latency`, `coalescing_wait`, `busy_time`, `command_duration`, `reactions_per_input`, `input_tokens`, `output_tokens`), plus the two the Harness alone can pair with its own actions (`steps_per_turn`, `barge_in`), and the dropped count.

**Latency is a measure, not a threshold.** The Harness records distributions (min, median, p95 per metric) for analytics and model-to-model comparison; a scenario *may* add a threshold as an expectation (`expect_latency`), and no threshold is ever a built-in default.

**Expectations wait, with a timeout**, over the timeline and the mirror World. They scan forward from a **watermark**: `set()` moves it to the end of the timeline as it stands *before* the write (the SUT may react before `set()` returns), and a matched expectation moves it past the match, so a sequence of expectations never re-matches an earlier event and repeating one means "the next one". On `TestRun` they are plain blocking calls (a pytest calls them from its own thread, never from the SUT's loop thread); as tester Commands they are async wrappers that run the same blocking call off the tester's loop (`asyncio.to_thread`), so a waiting expectation never blocks the tester's own World. The family: `expect_action(name, where=…, within=…)`, `expect_output(contains=…, within=…)`, `expect_no_action(name, for_=…)`, `expect_world(key, equals=…)`, `expect_latency(metric, p95_below=…)`. Each completes with a pass/fail **finding**. `expect_output` matches the output Command whatever its name and cannot assume a `text` argument (the output Command's signature is application-defined), so it matches against the call's string-valued args unless the scenario names the arg. The same functions serve every driver: a pytest calls them directly on `TestRun`, a fake script or a live model calls them as tester Commands.

The **report** is a dataclass serializable to JSON and Markdown: scenario, both sides' configs, the timeline, the metrics summary, findings (hard, from expectations), issues (soft, model-reported), and a verdict. `report.ok` is what a pytest asserts.

### The shared Command set

The tester's Command set *is* the Harness API: `set_<key>(value)` per SUT Input (from the Probe), `wait(seconds)`, the `expect_*` family, `report_issue(severity, what, evidence)`, `note(text)` and `finish(verdict, summary)`. They are plain async functions; registering them on a tester `Wica` is what turns them into Commands. A tester never issues `noop` in scripted mode (it would end the chain); in live mode `noop` is "keep observing".

### Drivers

**(a) Python test — the debugging path.** A pytest awaits the functions on a `TestRun` directly: `await run.set("speech_input", "hello")`, `await run.expect_output(contains="hello")`. No tester `Wica`, no second loop, failures as tracebacks. This is not the WicaTester; it is the toolkit the WicaTester stands on, and where a broken SUT is most pleasant to debug. It is also what today's e2e tests become (see "Relation to existing tests").

**(b) Scripted-fake tester `Wica` — the deterministic driver.** The tester is a real `Wica` over `provider: "fake"` and the fake's script *is the scenario*, an ordered list of tester Commands, JSON-expressible:

```json
[{"tool_calls": [{"name": "set_speech_input", "args": {"value": "hello"}}]},
 {"tool_calls": [{"name": "expect_output",     "args": {"contains": "hello", "within": 5}}]},
 {"tool_calls": [{"name": "set_closest_user",  "args": {"value": "u2"}}]},
 {"tool_calls": [{"name": "expect_world",      "args": {"key": "tracked_user", "equals": "u2"}}]},
 {"tool_calls": [{"name": "finish",            "args": {"verdict": "pass"}}]}]
```

Two rules make it deterministic:

- **The mirror entries do not trigger.** `sut:action`, `sut:output` and the silence heartbeat are registered `triggers_llm_call=False`; only the tester's own Command completions wake its next step (the uniform re-trigger model, `triggers_on_completion` left `True`). Script step *k* runs exactly when Command *k−1* completes: a linear chain, immune to when or how often the SUT reacts. `finish` ends the run.
- **Waiting lives in the Commands, not the script.** `expect_*` and `wait` poll the mirror World until their condition holds or their timeout expires. The tester runs `coalesce_window=0` and `delay_s=0`.

Determinism holds with the SUT on `provider: "fake"`; against a live SUT the same script is a *fixed-user regression* whose expectations must tolerate model variance. Its honest costs: one tester loop cycle per step, and a failed expectation surfaces in the report, not as a traceback.

**(c) Live-LLM tester `Wica` — the exploratory driver.** The same tester `Wica` with a real model and the mirror entries **triggering**:

- **Inputs.** `sut:action` — the latest Command the SUT issued, `name(args) [state]`; triggers on state changes only (`trigger_condition_fn`), so a `running` → `running` progress update does not fire. `sut:output` — the latest call of the SUT's output Command, if configured; kept separate because it is the channel the tester "hears"; triggers. `sut:reasoning` — the SUT's free text; visible, non-triggering. `sut:state` — the SUT's other prompt entries rendered with a *generic* serializer (`key = repr(value)`), never the SUT's own `serialize_fn`s, which speak in the SUT's first person and would confuse the tester; non-triggering. `clock` — a heartbeat the Harness updates on **silence** ("Nothing happened for N s"); triggers, with a per-scenario period, so the tester can decide "it should have answered by now". `scenario` — the macro direction, static. `metrics` — the Harness's rolling numbers, so the tester reports "slow" with a number.
- **Commands.** The shared set. `wait` lets the tester let a long Command finish, or deliberately not (barge-in); `report_issue` is the in-loop flag; the Harness also ends a run on a step or wall-clock budget.
- **System prompt.** A fixed tester persona, the scenario, the SUT's Command descriptions from the Probe (real names), and rules of engagement: be a realistic user, do not reveal you are testing, do not cooperate with vague answers, every `report_issue` must cite an observed `sut:action` / `sut:output` or a metric. The WICA runtime primer is appended as for any Agent.
- **Re-trigger chain.** The tester registers its `set_*` Commands with `triggers_on_completion=False`, so its next step is woken by what the SUT does or by silence, not by its own near-instant writes. Scripted mode (b) wants the opposite, so there the same functions are registered with the flag left `True`.
- **Perception distortion is bounded.** The tester runs `coalesce_window=0` plus the heartbeat.

**Two visibility presets**, chosen per scenario: **"user"** shows only what a person could perceive (`sut:output`, visible actions); **"QA engineer"** adds `sut:state` and `sut:reasoning`.

**In-loop judging is part of the design; a post-hoc judge is not.** The live tester's `report_issue` calls are flags with evidence, graded by nobody else in v1. A separate judging pass over the recorded trace is a later addition (see open questions). Hard invariants are always Python expectations over the timeline, never a model judgement.

### Recording closes the loop: trace → regression

A live run records the tester model's tool calls; that list is, verbatim, a script for driver (b). The Harness exports a timeline as a scripted scenario (`report.to_scenario()`): the tester's own calls become the steps, and the SUT's observed Commands become interleaved `expect_action(...)` steps with **loosened matchers** (name and arg keys, not exact text). Exploratory findings turn into deterministic regressions over `provider: "fake"` with no translation step — the "recorded cassette" of [testing.md](testing.md) open question #2, at scenario level rather than provider level, which is the level that is stable across model versions.

### Relation to existing tests

The toolkit is adopted **by test kind, not by era**. Nothing is migrated wholesale.

| Tests | What they pin | With the toolkit |
|---|---|---|
| `tests-e2e/test_fake_flows.py` (framework loop over the fake) | Loop correctness: the `agent:command` lifecycle, re-triggering, `noop`, the direct-construction seam | **Stay Python and off the toolkit.** They guard the very seams the Probe stands on; running them through it would let a toolkit bug mask a framework bug. One case is migrated to driver (a) as proof the Harness carries its weight — the rest keep their independence. The direct-construction case has no `Wica` for a Probe to attach to. |
| `tests-e2e/test_smoke.py` | Provider construction, no `Wica` | Unchanged. |
| `tests-e2e/test_wica.py` (live providers, parametrized over `PROVIDER_CONFIGS`) | Per-provider construction and tool calling end to end, coarse assertions | **Become driver (a) consumers.** Loop correctness is already pinned deterministically by the fake flows, so this costs no independence, and it removes the duplicated helpers (`wait_until`, `find_command_entry`, `RecordingSink`), lets the `noop` case wait on a `command_issued` record instead of reading `Agent._history`, and records each provider's reaction latency and pure model latency as a by-product. Assertions stay behavioral; no latency threshold is added. |
| `tests-e2e/test_example_flow.py` (the demo over the fake) | The example's wiring and its Gradio presenters | Partly a scenario (phase 4); the presenter assertions (transcript items, prompt log, Speaking slot) stay Python, since the Harness does not observe them. |

**Reports are kept, not discarded.** `tests-e2e/support.py` gains a report directory (a pytest option or fixture) so every toolkit-driven live run writes `<test>-<provider>.json`, one comparable artifact per provider per run. A single pytest run is a **smoke test, not a benchmark**: one sample per provider. Provider comparison belongs to a **scenario runner** that repeats a scenario *k* times, aggregates the distributions per provider, and reports pass^k — parametrized over `PROVIDER_CONFIGS` the same way the live set is, and skipping a provider whose key is unset.

### Placement and dependencies

- `src/wica/testing/` — a subpackage (`probe.py`, `harness.py`, `tester.py`, `report.py`; a `judge.py` later), imported explicitly like `wica.fake_model` and **not** re-exported from `wica/__init__.py`: it is test tooling. It depends only on core `wica` and adds no dependency, hence no extra. A live tester or judge builds its model through `build_chat_model` from an `AgentConfig`, the same path as everything else.
- Scenarios live in `tests-e2e/scenarios/`; the first one targets the conversation demo (`build_system` + `wire`, over fake or live) as the reference for writing more.
- The plan that creates the subpackage adds its modules to the AGENTS.md project map and to this spec's `code:` frontmatter, and its tests to `tests:`, in the same change — the frontmatter above names only the framework surface the toolkit consumes, because the drift guard requires every listed path to exist.

### Phases

Each phase is its own plan:

1. **Probe + Harness + shared Command set + report**, driven from Python (a) and by a scripted-fake tester `Wica` (b). Fast-tier unit tests over a fake SUT; a `tests-e2e` scenario (always-run, `-k tester`) of fake tester vs fake SUT; one `test_fake_flows.py` case migrated; the live set migrated with the report directory. **Built** by plan `202611061400` (2026-11-06); the spec stays `Stable` until phases 2–4 are built.
2. **Live-LLM tester** (c): triggering mirror entries, silence heartbeat, prompt assembly, visibility presets. Tests: live tester vs fake SUT, parametrized over `PROVIDER_CONFIGS`.
3. **Trace export and the scenario runner**: `report.to_scenario()`, repeated runs, pass^k and per-provider distributions.
4. **Demo scenario** in `tests-e2e/scenarios/` against the conversation demo.

Cost note for live-vs-live: with the output-Command re-trigger chain on both sides, one tester utterance costs about two SUT calls plus two or three tester calls; a ten-turn scenario is roughly fifty calls. Fine for an opt-in tier, never for the default run.

## Open questions

Deferrals, not blockers to phase 1.

1. **Scenario file format for the live driver.** For scripted drivers a scenario is the JSON script or a Python test function. For the LLM driver it is the macro direction plus fixed parameters (persona, visibility preset, step and time budget); its file format is left open until phase 2 is designed.
2. **Post-hoc judge.** A separate grading pass over the saved trace, with a rubric, possibly a different model, run *k* times, is the field's consistent practice (actor flags, harness measures, judge grades). Deferred past phase 3; the in-loop `report_issue` is v1's only model-side judgement.
3. **Testing the software with itself.** The tester is a `Wica`, chosen over importing another agentic framework. A framework regression can then break the SUT and the tester at once and produce confusing failures. The mitigation is structural — the fake-flow tests stay independent of the toolkit — but whether the toolkit needs its own smoke test against a hand-rolled SUT double is unexplored.
4. **Report persistence.** Where reports accumulate across runs (a git-ignored directory, an artifact store) and how two runs are compared is unspecified beyond "one JSON per run".
5. **Dropped-input reporting vs. the P1 fix.** If [_analysis.md](_analysis.md) P1 (run one more step when a trigger was dropped while busy) is adopted, the "dropped inputs" metric changes meaning: the input is no longer lost, only late. The metric then reports the delay instead.
