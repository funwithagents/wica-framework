# WicaTester phase 1: Probe, Harness, report, Python and scripted-fake drivers

**Status:** Done

> **Re-dated 2026-11-06** (first written 2026-09-17 as `202609170900`, never started). [specs/instrumentation.md](../specs/instrumentation.md) has since shipped (`Implemented`): `on_agent_reaction_ended` (`ReactionTrace`), `on_agent_trigger_dropped`, `on_agent_command_ended` (`CommandTrace`) and the per-reaction measures in core. The plan below is written against that surface: **step 2**'s `compute_metrics` derives `reaction_latency`, `model_latency`, `coalescing_wait`, `busy_time`, `command_duration`, `reactions_per_input` and the dropped count from the forwarded records and core's measure functions, keeping only the tester-action pairings (`steps_per_turn`, `barge_in`) as timeline heuristics; `SutEventKind` carries `reaction_ended`, `trigger_dropped`, `command_ended`; **step 3**'s Probe subscribes to all eight `Wica` Events. The live set gained `test_output_command_live` meanwhile; step 6 migrates it with the rest.

Builds phase 1 of [specs/wica-tester.md](../specs/wica-tester.md): the `wica.testing` subpackage (Probe + Harness + report + shared Command set), the two deterministic drivers (a Python test calling `TestRun` directly, and a scripted-fake tester `Wica`), one migrated fake-flow case, the live e2e set migrated onto the Harness with a report directory, and an always-run fake-tester-vs-fake-SUT e2e scenario.

This plan is written to be executed step by step, in order, by an agent with no other context. Read [AGENTS.md](../AGENTS.md) first (verification and status rules), then the spec sections named in each step. Everything the toolkit needs from the framework already exists (`World.keys()`/`get_config()`, `Agent.commands`/`output_command_name`, the five `Wica` Events, `Command.triggers_on_completion`, `provider: "fake"`). **This plan changes no code under `src/wica/` outside the new `src/wica/testing/` package.**

Out of scope (later plans): the live-LLM tester (spec phase 2), `report.to_scenario()` export and the repeated-run scenario runner (phase 3), the demo scenario directory (phase 4), a post-hoc judge.

## Goal

When this plan is `Done`:

1. `from wica.testing import Probe, SutEvent, TestRun, Report, Finding, Issue, MetricSummary, build_tester, run_script` works; nothing from it is re-exported by `wica/__init__.py`.
2. A pytest can do `run = TestRun(sut); run.set("prompt", "hi"); run.expect_text(contains="hi"); run.report()` against any running `Wica`, and get a timeline, metrics and a JSON/Markdown report.
3. `run_script(run, script)` drives the same `TestRun` from a second `Wica` over `provider: "fake"` whose script is a list of tester Commands, deterministically.
4. `tests-e2e/test_wica.py` uses `TestRun` and writes one report per test per provider when `--wica-report-dir` is given; `tests-e2e/test_fake_flows.py::test_scripted_flow_through_wica_init` uses `TestRun`; the other fake flows are untouched.
5. `uv run pytest tests-e2e -k tester` runs a key-less fake-tester-vs-fake-SUT scenario.

## What the code looks like today

- [src/wica/wica.py](../src/wica/wica.py): `Wica.init(config, *, coalesce_window=0.2, loop=None)`, `wica.world`, `wica.agent`, `start()`/`stop()`/`close()`, `is_running`, `register_command(fn)`, `set_output_sink`, `set_output_command`, and the Events `on_world_trigger: Event[WorldEntry]`, `on_agent_trigger: Event[WorldEntry]`, `on_agent_prompt: Event[list[BaseMessage]]`, `on_agent_command: Event[CommandIssued]`, `on_agent_text: Event[str]`. `Event.subscribe(handler)` / `unsubscribe(handler)`. `Wica` does **not** keep its `WicaConfig`.
- [src/wica/world.py](../src/wica/world.py): `world.keys()` (registration order), `world.get_config(key) -> WorldEntryConfig` (fields `type`, `serialize_fn`, `include_in_prompt`, `triggers_llm_call`, …), `world.get_entry(key) -> WorldEntry` (`key`, `current.id`, `current.value`), `world.update(key, value)` (thread-safe; raises `RuntimeError` when the World is not running, `TypeError` on a wrong type), `world.add_listener(key, callback)` / `remove_listener` (callback gets a `WorldEntry` snapshot; runs on that World's loop thread). `world.register(key, type, *, serialize_fn=..., include_in_prompt=..., triggers_llm_call=..., trigger_condition_fn=...)`.
- [src/wica/agent.py](../src/wica/agent.py): `CommandIssued(name, args, call_id)` — emitted for every issuance **including `noop`**; for a real Command it fires once the World entry `agent:command:<call_id>` is registered, so `add_listener` on that key immediately sees `running` then a terminal `CommandExecution(name, args, state, result, error)` (`state` in `running|complete|failed|cancelled`; `is_terminal()`). After the terminal value the Agent **unregisters** the key on its next step, so read the terminal value from the listener's snapshot, never by `get_entry` later. `agent.commands` is a live read-only mapping that includes `noop`/`cancel_command` and the output Command **only after `start()`**. `agent.output_command_name` is `str | None`.
- [src/wica/command.py](../src/wica/command.py): `Command(fn_or_BaseTool, *, name=None, description=None, triggers_on_completion=True)`, `.tool`, `.name`.
- [src/wica/fake_model.py](../src/wica/fake_model.py): `provider: "fake"`, `model_kwargs={"script": [...], "default": {...}, "loop": False, "delay_s": 0.2}`; a step is `{"text": "...", "tool_calls": [{"name": ..., "args": {...}}]}`; tool-call ids are `fake_call_<call_index>_<i>`; `bind_tools` validates that every scripted tool name is a bound Command name on the first call.
- Today's e2e helpers to replace: `wait_until`, `identity_serialize`, `find_command_entry`, `RecordingSink` copied in [tests-e2e/test_fake_flows.py](../tests-e2e/test_fake_flows.py), [tests-e2e/test_wica.py](../tests-e2e/test_wica.py), [tests-e2e/test_example_flow.py](../tests-e2e/test_example_flow.py). [tests-e2e/support.py](../tests-e2e/support.py) holds `PROVIDER_CONFIGS`, `real_wica(config_path, *, system_prompt=None, **kwargs)` (skips when the key env var is unset). [tests-e2e/conftest.py](../tests-e2e/conftest.py) exists and only forces the tier serial (`pytest_cmdline_main`); step 6 extends it.
- [src/wica/instrumentation.py](../src/wica/instrumentation.py): `ReactionTrace` (stamps, `outcome`, `command_call_ids`, `noop`, `usage: TokenUsage | None`; properties `coalescing_wait`, `model_latency`, `busy_time`), `CommandTrace` (`duration`), `now()` (wall-clock UTC), and the measures `reaction_latency(reaction)`, `reactions_per_input(reactions)`. `Wica` surfaces `on_agent_reaction_ended`, `on_agent_trigger_dropped`, `on_agent_command_ended` next to the five Events above.
- [tests/test_project_map.py](../tests/test_project_map.py) enforces: every `src/wica/**/*.py` is linked from the AGENTS.md project map and vice-versa; every non-glue module is named in some spec's `code:` frontmatter; every frontmatter path exists. `_NON_CONCEPT_MODULES` lists the `__init__.py` glue files.

## Ground rules

- Work the steps in order. After every code step run `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`, and (from step 6 on) `uv run pytest tests-e2e -k "fake or example or tester"`. Do not move on with a failure.
- **No new dependencies.** `pydantic` and `langchain_core` are already transitive/direct dependencies; nothing else may be added.
- **Threading model, read this twice.** Every `Wica` Event handler and every World listener runs on **that `Wica`'s loop thread**. Handlers in this package must only append to a list under a lock and notify a `threading.Condition`; they must never block, sleep, or call an expectation. `TestRun`'s expectation methods **block** (they wait on the Condition) and are called from the pytest thread, or from a worker thread via `asyncio.to_thread` inside a tester Command. Never call them from an Event handler.
- Test style (AGENTS.md): functional tests through the public API; fixtures for a fake SUT follow `tests/test_wica.py`'s `wica_factory` (a `Wica.init` over `provider: "fake"`, `coalesce_window=0.0`, `delay_s: 0`, closed on teardown).
- Every new `src/wica/testing/*.py` module gets a module docstring that names the spec section it implements.

## Step 1 — Package skeleton, project map, frontmatter

1. Create `src/wica/testing/__init__.py` with a docstring ("Test tooling for a running `Wica`: Probe, Harness, report, scripted drivers. Imported explicitly, never re-exported from `wica`. See specs/wica-tester.md.") and the re-exports listed in Goal 1 (add them as the modules appear in later steps; start with an empty `__all__`).
2. Create empty-bodied modules `probe.py`, `harness.py`, `report.py`, `tester.py` (docstring only) so the map and drift guard can be updated now.
3. [AGENTS.md](../AGENTS.md) project map: add one row after the `contrib/` row:

   `testing/` ([__init__.py](src/wica/testing/__init__.py)) — [probe.py](src/wica/testing/probe.py), [harness.py](src/wica/testing/harness.py), [report.py](src/wica/testing/report.py), [tester.py](src/wica/testing/tester.py) | Test tooling for a running `Wica` (never re-exported from `wica`; reached as `wica.testing`): the `Probe` (mirror connection + normalized `SutEvent` stream), the `TestRun` Harness (timeline, metrics, blocking expectations), the `Report`, and the scripted-fake tester `Wica` driver (`build_tester`/`run_script`) | [wica-tester.md](specs/wica-tester.md)

   Also add `tests/testing/` to the `tests/` row's description ("`tests/testing/` for the test tooling").
4. [tests/test_project_map.py](../tests/test_project_map.py): add `"testing/__init__.py"` to `_NON_CONCEPT_MODULES`.
5. [specs/wica-tester.md](../specs/wica-tester.md) frontmatter: add `src/wica/testing/probe.py`, `harness.py`, `report.py`, `tester.py` under `code:` and `tests-e2e/conftest.py` too; add `tests/testing/test_probe.py`, `tests/testing/test_harness.py`, `tests/testing/test_report.py`, `tests/testing/test_scripted_tester.py`, `tests-e2e/test_tester_flows.py` under `tests:` (create the files in later steps; create them as empty test modules now if you want the drift guard green immediately).
6. Create `tests/testing/` (no `__init__.py`, like `tests/contrib/`).

Verify: `uv run pytest tests/test_project_map.py` passes.

## Step 2 — `report.py`: the data model

File: `src/wica/testing/report.py`. Spec: "Harness — timeline, metrics, expectations, report".

Frozen dataclasses (all JSON-serializable through `dataclasses.asdict`):

```python
SutEventKind = Literal[
    "input_written", "trigger_raw", "trigger_observed", "trigger_dropped", "prompt",
    "command_issued", "command_state", "command_ended", "output_text", "reaction_ended",
    "action",
]

@dataclass(frozen=True)
class SutEvent:
    kind: SutEventKind
    t: float                              # wall-clock POSIX seconds at arrival (wica.instrumentation.now())
    key: str | None = None                # World key (inputs, triggers, agent:command:<id>)
    version_id: int | None = None         # WorldEntry.current.id for input_written/trigger_*
    name: str | None = None               # Command name, or the tester action name for "action"
    call_id: str | None = None
    args: dict[str, Any] | None = None    # Command args, or the action's kwargs for "action"
    state: str | None = None              # running/complete/failed/cancelled
    result: str | None = None
    error: str | None = None
    text: str | None = None               # output_text; prompt text when recorded
    message_count: int | None = None      # prompt only
    reaction: ReactionTrace | None = None # reaction_ended: the framework's record, carried whole
    command: CommandTrace | None = None   # command_ended: likewise

@dataclass(frozen=True)
class Finding:
    expectation: str      # e.g. "expect_text(contains='hi')"
    ok: bool
    detail: str           # what matched, or why it failed (timeout, wrong value…)
    t: float
    event: SutEvent | None = None   # the matching event, when there is one

@dataclass(frozen=True)
class Issue:
    severity: str         # "low" | "medium" | "high" — free string, validated only by the tester prompt later
    what: str
    evidence: str
    t: float

@dataclass(frozen=True)
class MetricSummary:
    count: int
    min: float
    median: float
    p95: float
    max: float

@dataclass(frozen=True)
class Report:
    scenario: str
    sut_label: str
    tester_label: str | None
    started_at: str        # ISO-8601 wall clock
    finished_at: str
    timeline: list[SutEvent]
    metrics: dict[str, MetricSummary]
    dropped_inputs: int
    findings: list[Finding]
    issues: list[Issue]
    notes: list[str]
    verdict: str | None    # "pass" | "fail" | None when finish() was never called
    summary: str | None

    @property
    def ok(self) -> bool: ...   # every finding ok and verdict != "fail"
    def to_dict(self) -> dict[str, Any]: ...   # dataclasses.asdict
    def to_json(self, *, indent: int = 2) -> str: ...
    def to_markdown(self) -> str: ...
    def write(self, path: Path) -> None: ...   # .json or .md by suffix; creates parent dirs
```

Also in this module:

```python
def summarize(samples: list[float]) -> MetricSummary | None   # None when empty; p95 = sorted[ceil(0.95*n)-1]
def compute_metrics(timeline: list[SutEvent], *, output_command_name: str | None) -> tuple[dict[str, MetricSummary], int]
```

`compute_metrics` returns `(metrics, dropped_inputs)` using exactly these definitions (each metric is omitted from the dict when it has no sample). The framework-stamped ones come from the forwarded records through core's own definitions ([specs/instrumentation.md](../specs/instrumentation.md), "Metrics"); only the two that pair a *tester action* with the SUT's reply are timeline heuristics:

| Metric key | Samples |
|---|---|
| `reaction_latency` | per `reaction_ended`: `wica.instrumentation.reaction_latency(reaction)` when not `None` |
| `model_latency` | per `reaction_ended`: `reaction.model_latency` when not `None` |
| `coalescing_wait` | per `reaction_ended`: `reaction.coalescing_wait` |
| `busy_time` | per `reaction_ended`: `reaction.busy_time` |
| `input_tokens` / `output_tokens` | per `reaction_ended` with `usage`: the reported counts (cost per reaction) |
| `command_duration` | per `command_ended`: `command.duration` |
| `reactions_per_input` | `wica.instrumentation.reactions_per_input(<the reactions in order>)` |
| `steps_per_turn` | for each `input_written`: the number of `prompt` events before the next `input_written` (or the end) |
| `barge_in` | for each `input_written` that occurs after a `command_state(running)` whose `name == output_command_name` and before that call's terminal state: the next `command_issued` named `cancel_command`, minus `input_written.t` |
| `dropped_inputs` (the int) | count of `trigger_dropped` |

`to_markdown()` renders: a title line with scenario and labels, a metrics table (metric, count, min, median, p95, max), the dropped-input count, a findings list (✅/❌ + expectation + detail), issues, notes, and the timeline as a table (`t` relative to the first event with 3 decimals, kind, key/name, state, short text).

Tests — `tests/testing/test_report.py`:

- `test_summarize_percentiles`: `summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])` → `count=10, min=1, median=5.5, p95=10, max=10`; `summarize([])` is `None`.
- `test_compute_metrics_from_a_hand_built_timeline`: build a small timeline by hand — a `reaction_ended` carrying a `ReactionTrace` with known stamps (external trigger written at 0.0, window 0.1 → 0.3, model 0.35 → 1.0, ended 1.05, one Command, usage 120/8), a `command_ended` carrying a `CommandTrace` of 0.5 s, two `prompt`s between one `input_written` and the end, and one `trigger_dropped` — and assert `reaction_latency.min == 1.05`, `model_latency.min == 0.65`, `coalescing_wait.min == 0.2`, `command_duration.min == 0.5`, `input_tokens.min == 120`, `steps_per_turn.min == 2`, `dropped_inputs == 1`.
- `test_report_roundtrips_to_json_and_renders_markdown`: `Report.to_json()` parses back with the same finding count; `to_markdown()` contains the metric name and the expectation string; `write(tmp_path / "r.json")` creates the file.

## Step 3 — `probe.py`: the connection

File: `src/wica/testing/probe.py`. Spec: "Probe — the connection", "What the framework provides".

```python
class Probe:
    def __init__(self, sut: Wica, *, record_prompt_text: bool = False) -> None
    @property
    def inputs(self) -> dict[str, WorldEntryConfig]   # every sut.world key not starting with "agent:", via keys()+get_config(); live (re-read each call)
    @property
    def commands(self) -> dict[str, Command]           # sut.agent.commands minus "noop" and "cancel_command"; live
    @property
    def output_command_name(self) -> str | None        # sut.agent.output_command_name
    def subscribe(self, listener: Callable[[SutEvent], None]) -> None
    def unsubscribe(self, listener: Callable[[SutEvent], None]) -> None
    def set(self, key: str, value: Any) -> None        # records input_written (key, new version id) THEN calls sut.world.update
    def attach(self) -> None                            # subscribe to the eight Events (idempotent)
    def detach(self) -> None                            # unsubscribe everything, incl. per-key listeners still attached
```

Behavior of the eight handlers (each builds a `SutEvent` stamped `t` with `wica.instrumentation.now().timestamp()` — the framework's wall clock, so a Harness action and a framework record subtract — and calls every subscribed listener, isolating each in `try/except Exception` so one bad listener does not break the others):

- `on_world_trigger(entry)` → `trigger_raw(key=entry.key, version_id=entry.current.id)`.
- `on_agent_trigger(entry)` → `trigger_observed(...)` likewise.
- `on_agent_prompt(messages)` → `prompt(message_count=len(messages), text=...)`; `text` is the concatenated `str(m.content)` of the messages only when `record_prompt_text` is true, else `None`.
- `on_agent_command(issued)` → `command_issued(name, call_id, args)`; then, **if `name != "noop"`**, `sut.world.add_listener(f"agent:command:{issued.call_id}", handler)` where `handler(entry)` reads `entry.current.value` (a `CommandExecution`), emits `command_state(key, name, call_id, args, state, result, error)` and, when the value `is_terminal()`, removes itself with `remove_listener` (wrap in `try/except KeyError` — the Agent may already have unregistered the key). Keep the attached `(key, handler)` pairs in a set so `detach()` can remove the rest.
- `on_agent_text(text)` → `output_text(text=text)`.
- `on_agent_trigger_dropped(entry)` → `trigger_dropped(key, version_id)`.
- `on_agent_reaction_ended(trace)` → `reaction_ended(reaction=trace)`, the record carried whole.
- `on_agent_command_ended(trace)` → `command_ended(name, call_id, state, command=trace)`.

`set()` computes the version id **before** updating (`sut.world.get_entry(key).current.id + 1`) so the `input_written` event can be paired with the `trigger_raw` that follows. Validation (`TypeError`, `KeyError`, `RuntimeError`) is `world.update`'s own; do not pre-check.

Tests — `tests/testing/test_probe.py` (fake SUT fixture as in `tests/test_wica.py`, script one `add` tool call then a text; register one `prompt` input with `triggers_llm_call=True`; collect events into a list under a lock, wait with a `threading.Condition`/`wait_until` loop):

- `test_inputs_and_commands_read_the_sut_schema`: `probe.inputs` has `prompt` and not the `agent:command:*` keys; after `start()`, `probe.commands` has `add` and not `noop`/`cancel_command`; `output_command_name` reflects `set_output_command`.
- `test_stream_follows_one_step_end_to_end`: after `probe.set("prompt", "Add 2 and 2.")` the collected kinds contain, in order, `input_written`, `trigger_raw`, `trigger_observed`, `prompt`, `command_issued`, `command_state(running)`, `command_state(complete)` with `result` containing `"4"`, and later `output_text`; `input_written.version_id == trigger_raw.version_id`.
- `test_noop_is_reported_as_a_command_issued_without_state`: script a `noop`; the stream has `command_issued(name="noop")` and no `command_state`.
- `test_detach_stops_the_stream`: after `detach()`, a further `set()` adds no `trigger_*` events.

## Step 4 — `harness.py`: `TestRun`

File: `src/wica/testing/harness.py`. Spec: "Harness", "The shared Command set".

```python
class TestRun:
    def __init__(self, sut: Wica, *, scenario: str = "", sut_label: str = "", record_prompt_text: bool = False) -> None
        # creates Probe(sut, record_prompt_text=...), subscribes its recorder, attaches; starts the clock
    probe: Probe
    finished: threading.Event          # set by finish()
    def timeline(self) -> list[SutEvent]           # snapshot copy under the lock
    # --- the shared Command set (blocking; see Ground rules) ---
    def set(self, key: str, value: Any) -> None                     # records action("set") then probe.set; moves the watermark to the end
    def wait(self, seconds: float) -> None                          # records action("wait"); time.sleep
    def expect_action(self, name: str, *, where: dict[str, Any] | None = None, state: str | None = None, result_contains: str | None = None, within: float = 5.0) -> Finding
    def expect_output(self, *, contains: str | None = None, arg: str | None = None, within: float = 5.0) -> Finding
    def expect_text(self, *, contains: str | None = None, within: float = 5.0) -> Finding
    def expect_no_action(self, name: str, *, for_: float = 2.0) -> Finding
    def expect_world(self, key: str, *, equals: Any, within: float = 5.0) -> Finding
    def expect_latency(self, metric: str, *, p95_below: float) -> Finding
    def note(self, text: str) -> None
    def report_issue(self, severity: str, what: str, evidence: str) -> None
    def finish(self, verdict: str, summary: str = "") -> None       # records, sets finished
    # --- results ---
    def metrics(self) -> tuple[dict[str, MetricSummary], int]       # compute_metrics(timeline, output_command_name=probe.output_command_name)
    def report(self) -> Report
    def close(self) -> None                                         # probe.detach(); idempotent
```

Matching rules, implemented once in a private `_wait_for(predicate, within) -> SutEvent | None`:

- The timeline is a list guarded by a `threading.Condition`; the Probe recorder appends and `notify_all()`s.
- `TestRun` keeps a **watermark** index. `set()` moves it to the end of the timeline **before** writing the World (the SUT may react before `set()` returns, so moving it afterwards could skip the very events the next expectation waits for). A successful `expect_*` moves it to the matched event's index + 1. Expectations scan only events at index ≥ watermark, so a sequence of expectations never re-matches an earlier event, and the same expectation can be repeated to mean "the next one".
- `_wait_for` uses `condition.wait_for(lambda: scan() is not None, timeout=within)`; on timeout it returns `None`.
- `expect_action(name, where, state, result_contains)`: with `state is None` matches a `command_issued` with that `name` whose `args` contain every `where` item (equality); with `state` given matches a `command_state` with that `name`, `state`, and, when `result_contains` is given, that substring in `result`.
- `expect_output(contains, arg)`: like `expect_action(probe.output_command_name)`; fails immediately (no wait) when no output Command is set. Matches `contains` against `args[arg]` when `arg` is given, else against every `str` value in `args`.
- `expect_text(contains)`: matches `output_text` whose `text` is non-empty and contains `contains` when given.
- `expect_no_action(name, for_)`: waits the full `for_` and passes only if no `command_issued` with that `name` arrived at index ≥ watermark; the watermark is not moved.
- `expect_world(key, equals)`: polls `sut.world.get(key) == equals` every 20 ms until `within`; no timeline event to match, watermark unchanged.
- `expect_latency(metric, p95_below)`: computes metrics now; fails when the metric has no samples or `p95 >= p95_below`.
- Every expectation records an `action` event (`name` = method name, `args` = the kwargs) **before** waiting, and appends a `Finding` to the run's findings whether it passed or failed. It never raises; the pytest asserts `finding.ok` (or `report.ok`).

`report()` builds the `Report` with `verdict`/`summary` from `finish()` (or `None`), ISO timestamps, and the label given at construction; `tester_label` is `None` here (the tester driver fills it, step 5).

Tests — `tests/testing/test_harness.py` (fake SUT as above):

- `test_set_expect_action_and_text_in_sequence`: script `add(2, 2)` then text `"The sum is 4."`; `run.set("prompt", "Add 2 and 2.")`; `run.expect_action("add", where={"a": 2})` ok with `event.call_id` set; `run.expect_action("add", state="complete", result_contains="4")` ok; `run.expect_text(contains="sum is 4")` ok; `run.report().ok` true and `metrics()[0]` has `command_duration` and `reaction_latency` with `count == 1`.
- `test_failed_expectation_is_a_finding_not_an_exception`: `run.expect_action("never", within=0.2)` returns `Finding(ok=False)`, `report().ok` is false, no exception.
- `test_watermark_prevents_rematching`: script two `add` calls in two steps (second step triggered by the first's completion); `expect_action("add")` twice returns two different `call_id`s; a third with `within=0.2` fails.
- `test_expect_output_uses_the_output_command_whatever_its_name`: `set_output_command(Command(voice, name="talk"))` where `voice(utterance: str)`; script `talk(utterance="hello")` then `noop`; `run.expect_output(contains="hello")` ok without naming the arg; `expect_output(contains="hello", arg="utterance")` ok too (after re-running or with a fresh run).
- `test_expect_no_action_and_expect_world`: script `noop`; `run.set(...)`; `run.expect_no_action("add", for_=0.3)` ok; register a `str` entry `mood` the SUT updates from a Command in another case, or simply `sut.world.update("mood", "happy")` from the test and assert `expect_world("mood", equals="happy")` ok and `expect_world("mood", equals="sad", within=0.2)` fails.
- `test_dropped_input_is_counted`: `coalesce_window=0`, script one text with `delay_s: 0.3` so the step is in flight; `run.set` twice quickly; wait for the text; `run.metrics()[1] >= 1` (the second raw trigger was dropped while busy). If this proves timing-flaky, replace the second `set` with three rapid ones and assert `>= 1`.

## Step 5 — `tester.py`: the scripted-fake tester `Wica` (driver b)

File: `src/wica/testing/tester.py`. Spec: "Drivers (b)", "The shared Command set".

```python
def build_tester(run: TestRun, script: list[dict[str, Any]], *, persona: str = "You are a scripted tester.") -> Wica
def run_script(run: TestRun, script: list[dict[str, Any]], *, timeout: float = 30.0) -> Report
```

`build_tester`:

1. Config: `WicaConfig.from_dict({"agent": {"provider": "fake", "model": "tester", "system_prompt": persona, "model_kwargs": {"script": script, "delay_s": 0, "default": {"tool_calls": [{"name": "noop", "args": {}}]}}}})`; `tester = Wica.init(config, coalesce_window=0.0)`.
2. Mirror entries on `tester.world`, all `str`, `include_in_prompt=True`, `triggers_llm_call=False`, `serialize_fn=lambda v, p: [TextPart(str(v))]`: `sut:action`, `sut:output`, `sut:reasoning`. Plus `scenario` (`str`, `triggers_llm_call=True`) — the kick-off entry. Subscribe to `run.probe` a listener that, **only when `tester.is_running`**, updates `sut:action` on `command_issued`/`command_state` with `f"{name}({args}) [{state or 'issued'}]"`, `sut:output` on a `command_issued` whose name is `run.probe.output_command_name`, and `sut:reasoning` on `output_text`. (These are visible, non-triggering context for phase 2's live tester; in phase 1 nothing reads them, but registering them now proves the two-loop wiring the spec calls out.)
3. Commands, registered on the tester with `tester.register_command(...)`:
   - One `set_<key>` per `run.probe.inputs` entry whose declared `type` is exactly one of `str`, `int`, `float`, `bool` (other types are reachable only from Python, `run.set`). The tool name is the key with every character not in `[A-Za-z0-9_]` replaced by `_`. Build it as a `StructuredTool` so the arg type is right: `args_schema = pydantic.create_model(f"Set_{tool_name}", value=(entry_type, ...))`, `StructuredTool.from_function(coroutine=fn, name=tool_name, description=f"Write the SUT's '{key}' input.", args_schema=args_schema)`, where `fn` is `async def fn(value): await asyncio.to_thread(run.set, key, value); return "ok"`; then `Command(tool)`. Import `StructuredTool` from `langchain_core.tools`.
   - `wait(seconds: float)`, `expect_action(name: str, where: dict | None = None, state: str | None = None, result_contains: str | None = None, within: float = 5.0)`, `expect_output(contains: str | None = None, arg: str | None = None, within: float = 5.0)`, `expect_text(contains: str | None = None, within: float = 5.0)`, `expect_no_action(name: str, for_: float = 2.0)`, `expect_world(key: str, equals: str, within: float = 5.0)` (string-typed `equals` in the Command form; the Python form stays `Any`), `expect_latency(metric: str, p95_below: float)`, `note(text: str)`, `report_issue(severity: str, what: str, evidence: str)`: each an `async def` with a docstring, awaiting `asyncio.to_thread(run.<method>, ...)` and returning `"ok"` or `f"failed: {finding.detail}"`.
   - `finish(verdict: str, summary: str = "")`: registered as `Command(fn, triggers_on_completion=False)` so its completion wakes **no** further tester step — this is what ends the chain. It calls `run.finish` (through `to_thread` too, for uniformity).
4. All other Commands keep `triggers_on_completion=True`: step *k* of the script runs exactly when Command *k−1* completes (spec rule 1). The fake's `default` is `noop`, so a script that forgets `finish` ends quietly and `run.finished` stays unset — `run_script` then times out and reports `verdict=None`.
5. `tester.set_output_sink` is left unset (the tester's free text is irrelevant).

`run_script`: `tester = build_tester(run, script)`; `tester.start()`; `tester.world.update("scenario", run.scenario or "scripted")` to fire step 1; `run.finished.wait(timeout)`; `tester.close()`; return `run.report()` with `tester_label="fake"` (add a `tester_label` keyword to `TestRun.report()` for this). If the wait timed out, first append a failed `Finding("run_script", ok=False, detail="script did not call finish within …")`.

Tests — `tests/testing/test_scripted_tester.py` (fake SUT as above, SUT script `add(2, 2)` then text):

- `test_script_drives_the_sut_and_yields_a_passing_report`: script `[set_prompt("Add 2 and 2."), expect_action(name="add", state="complete", result_contains="4"), expect_text(contains="sum"), finish(verdict="pass")]` → `report.ok`, `report.verdict == "pass"`, the tester's fake model made exactly 4 calls (`tester.agent.model.calls` — build the tester with `build_tester` in the test to reach it, then drive it like `run_script` does).
- `test_only_settable_types_get_set_commands`: register a `dict` input and a `str` input on the SUT; `build_tester(...).agent.commands` (after `start()`) has `set_<str key>` and not `set_<dict key>`; a key like `speech:input` becomes `set_speech_input`.
- `test_a_failed_expectation_lands_in_the_report_and_the_script_continues`: script `expect_action(name="ghost", within=0.2)` then `finish(verdict="fail")` → the run reaches `finish` (so `report.verdict == "fail"`), `report.findings` holds exactly one finding and it is the failed `expect_action`, `report.ok` is false.
- `test_missing_finish_times_out_with_a_failed_finding`: script `[set_prompt(...)]`, `run_script(..., timeout=1.0)` → `report.verdict is None` and the last finding is the timeout.
- `test_mirror_entries_follow_the_sut`: after the run, `tester.world.get("sut:action")` mentions `add` and `complete` (build/start the tester in the test to read its World before `close()`).

## Step 6 — `tests-e2e/conftest.py`, the report directory, and the live set on the Harness

Spec: "Relation to existing tests".

1. Create `tests-e2e/conftest.py`:
   - `pytest_addoption`: `--wica-report-dir` (default `None`, help: "directory where toolkit-driven e2e tests write one JSON report per test per provider").
   - fixture `report_dir(request) -> Path | None` returning the option as a `Path` (created) or `None`.
   - fixture `write_report(request, report_dir) -> Callable[[Report], None]`: when `report_dir` is set, writes `report_dir / f"{request.node.name}.json"` (the node name already includes the provider id, e.g. `test_plain_text_round_trip[e2e.openai.config]`; replace `[`, `]` and `/` with `_`); otherwise a no-op.
2. Add `tests-e2e/reports/` to `.gitignore` (the documented default location to pass).
3. Rewrite the three tests in `tests-e2e/test_wica.py` over `TestRun` and delete the local `wait_until`, `find_command_entry`, `RecordingSink` (keep `identity_serialize` for the `prompt` Input; move it to `support.py` and import it — `test_fake_flows.py` still needs its own until step 7):
   - `test_plain_text_round_trip`: `run = TestRun(wica, scenario="plain text", sut_label=config_path.stem)`; the Agent needs no sink for `on_agent_text` to fire (a missing sink is a no-op internally), so set none; `run.set("prompt", "Say hello in one short sentence.")`; `assert run.expect_text(within=15).ok`; `write_report(run.report())`; `run.close()` in `finally` before `wica.close()`.
   - `test_real_tool_calling_round_trip`: `run.expect_action("add", within=15)` then `run.expect_action("add", state="complete", result_contains="4", within=15)`; keep the docstring's note about not asserting a follow-up text.
   - `test_output_command_live`: `run.expect_output(within=15)` (the output Command, whatever its name, called with any text) — the failure message still quotes the free text the sink received, which the timeline's `output_text` events now provide.
   - `test_noop_empty_reaction_live`: `assert run.expect_action("noop", within=15).ok` — no more `wica.agent._history` read; keep the persona and docstring.
   - Keep all assertions behavioral; add **no** latency threshold.
4. [AGENTS.md](../AGENTS.md) "Testing" → "Live/e2e tests": add one paragraph — the live set runs on `wica.testing.TestRun`; pass `--wica-report-dir tests-e2e/reports` to keep one JSON report per test per provider (reaction and model latency, timeline); a single run is one sample, not a benchmark.
5. [specs/testing.md](../specs/testing.md): in "Live tier", one sentence pointing to [wica-tester.md](../specs/wica-tester.md) "Relation to existing tests" for the report option. Status stays `Implemented` (the behavior is specified by wica-tester.md; this is a cross-reference).

6. [.github/workflows/ci.yml](../.github/workflows/ci.yml) `e2e-fake` job and [specs/ci.md](../specs/ci.md) "The e2e tier, split by its two kinds": the scripted-fake job selects by file, so add `tests-e2e/test_tester_flows.py` (created in step 7) to its `pytest` line and to the spec's list (editorial, status unchanged).

Verify: `uv run pytest tests-e2e -k "fake or example"` still passes (the live tests skip without keys — confirm they **skip**, not error, with `uv run pytest tests-e2e/test_wica.py -rs`). If you hold a key, run one provider per AGENTS.md: `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k anthropic --wica-report-dir tests-e2e/reports'` and open a written report.

## Step 7 — Migrate one fake-flow case; add the fake-tester e2e scenario

1. `tests-e2e/test_fake_flows.py`: rewrite **only** `test_scripted_flow_through_wica_init` over `TestRun` (same script, same assertions expressed as `expect_action("add")`, `expect_action("add", state="complete", result_contains="4")`, `expect_text(contains="The sum is 4.")`, and keep the `model.calls` introspection). Leave `test_scripted_flow_direct_construction` (no `Wica`), `test_output_command_flow_through_the_setters` and `test_noop_flow_takes_no_action` **untouched**, including their local helpers — the spec wants them independent of the toolkit. Update the module docstring: one case demonstrates `wica.testing`, the rest deliberately stay hand-rolled.
2. New `tests-e2e/test_tester_flows.py` (module docstring: always-run, key-less, selected by `-k tester`; fake tester vs fake SUT; see specs/wica-tester.md phase 1):
   - `test_fake_tester_drives_a_fake_sut`: SUT = `Wica.init` over a fake script with an output Command `say(text)` (script: `say("hi there")`, `noop`), inputs `speech_input: str` (`triggers_llm_call=True`) and `closest_user: str` (non-triggering); tester script `[set_speech_input("hello"), expect_output(contains="hi", within=5), set_closest_user("u2"), expect_world(key="closest_user", equals="u2"), expect_no_action(name="say", for_=0.5), finish(verdict="pass")]` via `run_script`; assert `report.ok`, `report.verdict == "pass"`, `metrics["reaction_latency"].count == 1`, `report.to_markdown()` mentions `say`.
   - `test_report_is_written_when_a_directory_is_given`: same run with `write_report` and `--wica-report-dir` simulated by calling `report.write(tmp_path / "x.json")`; assert the file parses and `["verdict"] == "pass"`.
3. [AGENTS.md](../AGENTS.md) "Testing": add `-k tester` next to `-k fake` / `-k example` as the third always-run set; [specs/testing.md](../specs/testing.md) "Always-run scripted-fake flows": one sentence naming `test_tester_flows.py` (editorial, status unchanged).

## Step 8 — Statuses and final verification

1. [specs/wica-tester.md](../specs/wica-tester.md): in "Phases", mark phase 1 "(built by plan `202609170900`)". Status: promote to `Stable` if it is still `Draft` and the human owner has agreed (see the note below); it does **not** become `Implemented` — phases 2–4 are unbuilt. Keep the index row in sync.
2. [specs/wica.md](../specs/wica.md) open question #1 (multiple `Wica` instances): append "Validated by `wica.testing`'s tester `Wica` driving a SUT `Wica` on two loops (`tests/testing/test_scripted_tester.py`)." — editorial, status unchanged.
3. This plan: `Todo` → `Done`; its row in [plans/_index.md](_index.md).

Final verification: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`, and `uv run pytest tests-e2e -k "fake or example or tester"` all pass; `uv run pytest tests-e2e/test_wica.py -rs` shows skips (not errors) without keys.

**Note on the spec status.** AGENTS.md's lifecycle expects a plan to build a `Stable` spec. `wica-tester.md` was `Draft` when this plan was first written because phase 2's scenario file format is open; phase 1 is fully settled. The owner gave the go-ahead to start on 2026-11-06, so the spec was promoted to `Stable` in the same change that set this plan `In progress` (its remaining open questions are deferrals).
