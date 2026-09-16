# Instrumentation: reaction trace, three new Events, OpenTelemetry spans

**Status:** Done

Builds [specs/instrumentation.md](../specs/instrumentation.md) (Stable) in full: the typed per-reaction record and its three Events, wall-clock stamps through the loop, OpenTelemetry spans with context propagation, the per-reaction measures, the transcript's reaction spinner + duration, and a console exporter for the demo. It lands **before** the WicaTester phase 1 plan (`202609162130`), which consumes what this one adds.

This plan is written to be executed step by step, in order, by an agent with no other context. Read [AGENTS.md](../AGENTS.md) first (verification and status rules), then the spec sections named in each step. Every code snippet below is the intended shape; keep names exactly as written, since later steps and the tester plan refer to them.

Out of scope: a metrics panel for the Gradio contrib, `reaction_id` on the four existing Events, the OpenTelemetry Metrics API, an origin timestamp on `update()` (spec open questions 1–4), and any change to what the model sees.

## Goal

When this plan is `Done`:

1. `from wica import ReactionTrace, TriggerTrace, CommandTrace, TokenUsage` works, and `from wica.instrumentation import now, parent_trigger, reaction_latency, reactions_per_input` works.
2. Every `Wica` has `on_agent_reaction_ended: Event[ReactionTrace]`, `on_agent_trigger_dropped: Event[WorldEntry]`, `on_agent_command_ended: Event[CommandTrace]`, each the same object as the Agent's `on_reaction_ended` / `on_trigger_dropped` / `on_command_ended`.
3. `on_reaction_ended` fires once at the end of **every** reaction that started: outcome `ok`, `empty`, `model_error` or `cancelled`.
4. With `opentelemetry-sdk` installed and a `TracerProvider` set by the application, a trace shows `wica.world.update` → `wica.agent.reaction` → `wica.agent.model` / `wica.agent.sink` / `wica.agent.command`, a span opened inside a Command body nests under `wica.agent.command`, and the follow-up reaction triggered by a Command's completion is that command span's child. Without the SDK nothing changes and no import fails.
5. The transcript's reaction group shows a spinner while the model thinks and a duration when the reaction ends.
6. `uv run pytest`, `uv run pytest tests-e2e -k "fake or example"`, `uv run ruff check .`, `uv run ruff format .`, `uv run pyright` all pass.

## What the code looks like today

Read these before step 1; the plan refers to them by name.

- [src/wica/world.py](../src/wica/world.py): `_now()` (line ~25) returns `datetime.now(timezone.utc)` and is used for every `WorldEntryVersion.timestamp`. `_update()` (line ~262) computes `should_trigger` under `self._lock`, builds `new_entry`, then schedules callbacks: `self._loop.call_soon_threadsafe(self._dispatch, listener, copy.deepcopy(new_entry))` per listener, and `self._loop.call_soon_threadsafe(self.on_trigger.emit, copy.deepcopy(new_entry))` when `should_trigger`. `_dispatch()` (line ~363) runs an async listener with `self._loop.create_task(...)` and a sync one with `self._loop.run_in_executor(None, callback, entry)`.
- [src/wica/agent.py](../src/wica/agent.py): `Agent.__init__` declares the four Events `on_trigger`, `on_prompt`, `on_command`, `on_text` and the window state `_window_batch: list[WorldEntry]`, `_window_timer`, `_busy`. `start()` subscribes the shim `schedule_trigger(entry)` → `self._track_task(self._handle_trigger(entry))` on `world.on_trigger`. `_handle_trigger()` drops the trigger with an INFO log when `self._busy`, else appends to `_window_batch` and either flushes or opens the window timer. `_flush_window()` sets `_busy = True` and calls `self._track_task(self._run_batch(batch))`. `_run_batch()` awaits `_run_step(batch)` and clears `_busy` in `finally`. `_run_step()` emits `on_trigger` per batched entry, appends the observation, renders, emits `on_prompt`, awaits `self._bound_model.ainvoke(messages)` inside `try/except Exception` (logs and `return`s on failure), then handles `response.text` (emits `on_text`, awaits the sink), then loops over `response.tool_calls` (a `noop` emits `on_command` and records `NoReactionRecord`; anything else records `CommandRecord` and calls `_dispatch_command`). `_dispatch_command()` registers the `agent:command:<call_id>` entry, emits `on_command`, writes the `running` value, and creates the task `self._track_task(self._run_command(key, call_id, name, args))`. `_run_command()` awaits `command.tool.ainvoke(args)` and writes the terminal state through `_write_terminal()` in its `except CancelledError` (then re-raises), `except Exception` and `else` branches. `_track_task()` wraps `self._loop.create_task(coroutine)`.
- [src/wica/wica.py](../src/wica/wica.py): `Wica.__init__` assigns `self.on_world_trigger = world.on_trigger`, `self.on_agent_trigger = agent.on_trigger`, `self.on_agent_prompt`, `self.on_agent_command`, `self.on_agent_text` the same way.
- [src/wica/__init__.py](../src/wica/__init__.py): the public re-exports and `__all__`.
- [src/wica/contrib/gradio/transcript.py](../src/wica/contrib/gradio/transcript.py): `TranscriptLog.__init__` subscribes `on_agent_trigger`, `on_agent_prompt`, `on_agent_command`; `on_prompt()` increments `_reaction_count`, sets `_current_reaction_id = f"reaction-{n}"` and puts a parent item `{"role": "assistant", "content": "", "metadata": {"id": ..., "title": "💬 reaction N · <label>"}}` on `self._items`; `on_command_update()` shows how an item is edited in place under `self._lock` and how `metadata["status"]` / `metadata["duration"]` are used; `snapshot()`'s signature covers content, title and status.
- Tests: [tests/test_agent.py](../tests/test_agent.py) has `ProgrammableChatModel` (an injected model whose `respond` callable returns an `AIMessage`; `bind_tools` returns itself), helpers `text_response`, `tool_call_response`, `sequence`, and the fixtures `loop`, `world`, `sink`, `make_agent(model, *, world, loop, ...)`. [tests/test_wica.py](../tests/test_wica.py) has `fake_config(script, delay_s=...)`, `RecordingSink`, `wait_until`, `identity_serialize`, and the `wica_factory` fixture (closes every Wica on teardown). [tests/test_world.py](../tests/test_world.py) has `loop` and `world` fixtures. [tests/conftest.py](../tests/conftest.py) holds shared fixtures.
- [tests/test_project_map.py](../tests/test_project_map.py) requires every `src/wica/**/*.py` to be linked in the AGENTS.md project map and named in some spec's `code:` frontmatter.
- [pyproject.toml](../pyproject.toml): `dependencies` is `langchain`, `langchain-core`; `[dependency-groups] dev` and `demo` groups. There is no OpenTelemetry package installed today (`import opentelemetry` fails).
- Python is 3.12: `loop.call_soon_threadsafe(cb, *args, context=ctx)` and `loop.create_task(coro, context=ctx)` both accept a `contextvars.Context`, and a task created without `context=` copies the current context.

## Ground rules

- Work the steps in order. After every code step run `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`; from step 4 on also `uv run pytest tests-e2e -k "fake or example"`. Do not move on with a failure.
- **Instrumentation only.** No step may change what the model receives, what is stored in the World or in history, when a reaction runs, or how Commands are dispatched. If a test in `tests/` or `tests-e2e/` that existed before this plan needs its *assertion* changed, stop and re-read the spec: you have changed behavior.
- **Threading.** Everything in `agent.py` touched here runs on the loop thread. Event emissions stay synchronous and cheap (build a frozen dataclass, call `emit`). Never `await` inside an Event handler and never block the loop.
- **Clock.** Every stamp is `wica.instrumentation.now()` (wall-clock UTC `datetime`). Do not introduce `time.monotonic()` anywhere in `src/wica/`.
- **OpenTelemetry.** Only `opentelemetry-api` may be imported under `src/wica/`. Never import `opentelemetry.sdk` there. Get the tracer with `opentelemetry.trace.get_tracer("wica")`; it is safe to call at import time (a proxy that follows a `TracerProvider` set later).
- Every new module or function gets a docstring that names the spec section it implements.

## Step 1 — Dependency and the leaf module

Spec: "Layer 1 — the reaction record and its Events", "Metrics: per-reaction measures in core", "Which trigger is the reaction's parent".

1. [pyproject.toml](../pyproject.toml): add `"opentelemetry-api>=1.20"` to `[project] dependencies`. Add `"opentelemetry-sdk>=1.20"` to the `dev` group **and** the `demo` group (tests verify parenting through the SDK's in-memory exporter; the demo's console exporter needs it). Run `uv sync --dev` (and `uv sync --group demo` if needed) and confirm `uv run python -c "import opentelemetry.trace, opentelemetry.sdk.trace"` succeeds.
2. Create `src/wica/instrumentation.py`. It is a **dependency leaf**: it imports only the standard library and `opentelemetry` — **no `wica.*` import** (so `world.py` can import it). Contents:

```python
"""Latency and reactivity instrumentation: the reaction/trigger/Command records, the shared
wall-clock, the parent-trigger rule and the per-reaction measures. See specs/instrumentation.md."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from opentelemetry import trace

TRACER_NAME = "wica"
COMMAND_KEY_PREFIX = "agent:command:"   # duplicated from agent.py on purpose: this module must not import it


def now() -> datetime:
    """The one clock every stamp uses: wall-clock UTC, the World's own. See "One clock"."""
    return datetime.now(timezone.utc)


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


@dataclass(frozen=True)
class TriggerTrace:
    key: str
    version_id: int
    written_at: datetime
    arrived_at: datetime
    is_command_completion: bool

    @property
    def hop(self) -> float:
        """Seconds from the World write to the trigger reaching the loop."""
        return (self.arrived_at - self.written_at).total_seconds()


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None


ReactionOutcome = Literal["ok", "empty", "model_error", "cancelled"]


@dataclass(frozen=True)
class ReactionTrace:
    reaction_id: int
    triggers: tuple[TriggerTrace, ...]
    window_opened_at: datetime
    window_closed_at: datetime
    prompt_ready_at: datetime | None
    model_started_at: datetime | None
    model_ended_at: datetime | None
    outcome: ReactionOutcome
    error: str | None
    text_length: int
    sink_duration: float | None
    command_call_ids: tuple[str, ...]
    noop: bool
    usage: TokenUsage | None
    ended_at: datetime
    trace_id: str | None
    span_id: str | None

    @property
    def coalescing_wait(self) -> float: ...        # window_closed_at - window_opened_at, seconds
    @property
    def render_time(self) -> float | None: ...     # prompt_ready_at - window_closed_at; None if prompt_ready_at is None
    @property
    def model_latency(self) -> float | None: ...   # model_ended_at - model_started_at; None if either is None
    @property
    def busy_time(self) -> float: ...              # ended_at - window_closed_at


@dataclass(frozen=True)
class CommandTrace:
    call_id: str
    name: str
    reaction_id: int
    started_at: datetime
    ended_at: datetime
    state: Literal["complete", "failed", "cancelled"]

    @property
    def duration(self) -> float: ...               # ended_at - started_at, seconds


def parent_trigger(triggers: Sequence[TriggerTrace]) -> TriggerTrace:
    """The trigger a reaction is parented to: the earliest-arrived trigger that is not a Command
    completion; if all are completions, the earliest one. `triggers` must be non-empty."""


def reaction_latency(reaction: ReactionTrace) -> float | None:
    """written_at of the earliest external trigger -> ended_at, for a reaction that issued at least
    one Command or delivered text (command_call_ids non-empty or text_length > 0); else None.
    A reaction whose every trigger is a Command completion has no external trigger -> None."""


def reactions_per_input(reactions: Sequence[ReactionTrace]) -> list[int]:
    """For each reaction whose parent trigger is external, the number of reactions from it (inclusive)
    up to the next such reaction (exclusive) — the re-trigger chain's cost. Reactions before the
    first external one are ignored."""
```

   `prompt_ready_at` and `model_started_at` are `datetime | None` (not the spec's bare `datetime`) because a reaction cancelled by `stop()` while rendering never reaches them; the spec's intent (a record for *every* reaction) wins over its field sketch. Note this deviation in the module docstring and, in step 8, in the spec (editorial).

   Note `datetime` arithmetic: `(a - b).total_seconds()` gives a float in seconds.
3. [src/wica/__init__.py](../src/wica/__init__.py): import and add to `__all__`: `CommandTrace`, `ReactionTrace`, `TokenUsage`, `TriggerTrace` (from `wica.instrumentation`). Keep `__all__` sorted.
4. [AGENTS.md](../AGENTS.md) project map: add a row after the `fake_model.py` row:

   `| [instrumentation.py](src/wica/instrumentation.py) | Latency/reactivity instrumentation leaf (zero project imports): the `ReactionTrace`/`TriggerTrace`/`CommandTrace`/`TokenUsage` records, the shared wall-clock `now()`, the OpenTelemetry tracer accessor, the parent-trigger rule and the per-reaction measures | [instrumentation.md](specs/instrumentation.md) |`

5. [specs/instrumentation.md](../specs/instrumentation.md) frontmatter: add `src/wica/instrumentation.py` and `src/wica/contrib/gradio/transcript.py` under `code:`; add `tests/test_instrumentation.py`, `tests/contrib/test_gradio_transcript.py` under `tests:`.
6. Tests — new `tests/test_instrumentation.py`, hand-building records with `datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=...)`:
   - `test_now_is_timezone_aware_utc`: `now().tzinfo is timezone.utc`.
   - `test_reaction_properties_are_differences_in_seconds`: a reaction with window 0.0→0.2, prompt 0.25, model 0.3→1.3, ended 1.4 → `coalescing_wait == 0.2`, `render_time == 0.05`, `model_latency == 1.0`, `busy_time == 1.2` (use `pytest.approx`).
   - `test_parent_trigger_prefers_the_earliest_external_trigger`: triggers `[completion@0.0, input@0.1, input@0.05]` → the input at 0.05; `[completion@0.1, completion@0.0]` → the completion at 0.0.
   - `test_reaction_latency_counts_from_the_earliest_external_write`: two external triggers written at 0.0 and 0.1, ended at 1.0, one command → `1.0`; same with `command_call_ids=()` and `text_length=0` → `None`; only-completion triggers → `None`.
   - `test_reactions_per_input_counts_the_retrigger_chain`: reactions with parent kinds `[external, completion, completion, external, completion]` → `[3, 2]`.
   - `test_command_trace_duration`: 1.0 → 1.5 → `0.5`.

## Step 2 — World: `now()`, context capture and propagation, the update span

Spec: "Layer 2 — OpenTelemetry spans and context propagation" (the `wica.world.update` row and the first propagation bullet), "One clock".

File: [src/wica/world.py](../src/wica/world.py).

1. Replace the module's `_now()` with `from wica.instrumentation import now, tracer` and use `now()` at the three former `_now()` call sites (`register`, `_update`, and the TTL restore in `start()`). Delete `_now`.
2. In `_update()`, after `should_trigger` and `new_entry` are computed and **before** the scheduling loop (still under the lock), capture the context to run the callbacks in:

```python
            if should_trigger:
                with tracer().start_as_current_span(
                    "wica.world.update",
                    attributes={"wica.key": key, "wica.version_id": new_id},
                ):
                    ctx = contextvars.copy_context()
            else:
                ctx = contextvars.copy_context()
```

   The span is short (it ends before the `with` exits) but the context captured *inside* it holds it as the current span, so the reaction span opened later will have it as parent. Then pass `context=ctx` to **both** `call_soon_threadsafe` calls (listeners and `self.on_trigger.emit`). `import contextvars` at the top.
3. In `_dispatch()`, the sync branch must carry the context onto the executor thread (executors do not inherit it). Replace `self._loop.run_in_executor(None, callback, entry)` with `self._loop.run_in_executor(None, contextvars.copy_context().run, callback, entry)`. `_dispatch` itself runs under `ctx` (it was scheduled with `context=ctx`), so the copy it takes is the writer's context. The async branch is unchanged: `create_task` copies the current context.
4. [specs/world.md](../specs/world.md): in "The shared event loop", add one paragraph: `update()` captures the caller's `contextvars` context and runs listeners and the trigger emission in it (sync listeners too, on the executor), so a span or context variable set by the writer is visible in every callback; a triggering update opens a short `wica.world.update` span so the reaction is traced as its child (see instrumentation.md). Also change the `timestamp` row of `WorldEntryVersion` to say the clock is `wica.instrumentation.now()`. Status stays `Implemented` (the code lands in this step).
5. Tests — add to `tests/test_world.py` (use the existing `world` fixture, a `contextvars.ContextVar("who", default="nobody")`, a `threading.Event` to wait):
   - `test_update_runs_listeners_in_the_writers_context`: set the var to `"writer"` (`token = var.set("writer")`), register a key, add one **sync** listener and one **async** listener that each record `var.get()` into a list and set the event; `update`; wait; both recorded `"writer"`. Reset the var with `var.reset(token)` at the end.
   - `test_trigger_subscribers_see_the_writers_context`: same with an `on_trigger.subscribe` handler on a `triggers_llm_call=True` key.
   - `test_triggering_update_opens_an_update_span`: uses the `spans` fixture from step 3 below — **write this test in step 3**, after the fixture exists. Assert one finished span named `wica.world.update` with attributes `wica.key` and `wica.version_id`, and none for a non-triggering update.

## Step 3 — Test fixture for spans

The SDK's global `TracerProvider` can be set only once per process, so it is set once for the whole fast tier.

1. [tests/conftest.py](../tests/conftest.py): add

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """Finished spans of the current test (cleared before each test). `spans.get_finished_spans()`
    returns ReadableSpan objects: `.name`, `.attributes`, `.parent` (a SpanContext or None),
    `.context.span_id`, `.links`."""
    _EXPORTER.clear()
    return _EXPORTER
```

   Because the provider is set at conftest import, every test in the tier records spans; only tests that use the `spans` fixture look at them.
2. Write `test_triggering_update_opens_an_update_span` from step 2.5 now.

## Step 4 — Agent: stamps, the reaction record, the three Events, the spans

Spec: "Vocabulary", "Layer 1", "Layer 2" (all rows and bullets), "Always on".

File: [src/wica/agent.py](../src/wica/agent.py). Import `contextvars`, `from dataclasses import dataclass, field`, and from `wica.instrumentation`: `CommandTrace, ReactionOutcome, ReactionTrace, TokenUsage, TriggerTrace, now, parent_trigger, tracer`; from `opentelemetry` import `context as otel_context, trace as otel_trace`.

1. **Private helpers** (module level, near the record dataclasses):

```python
@dataclass
class _PendingTrigger:
    """A trigger waiting in the coalescing window, with what the reaction record needs."""
    entry: WorldEntry
    arrived_at: datetime
    context: contextvars.Context      # the writer's context, captured in _handle_trigger

    def trace(self) -> TriggerTrace:
        return TriggerTrace(
            key=self.entry.key,
            version_id=self.entry.current.id,
            written_at=self.entry.current.timestamp,
            arrived_at=self.arrived_at,
            is_command_completion=self.entry.key.startswith(_COMMAND_KEY_PREFIX),
        )


@dataclass
class _ReactionBuilder:
    """Mutable per-reaction scratch filled in as the phases pass; frozen into a ReactionTrace at
    the end (see _run_batch)."""
    reaction_id: int
    triggers: tuple[TriggerTrace, ...]
    window_opened_at: datetime
    window_closed_at: datetime
    span: otel_trace.Span
    prompt_ready_at: datetime | None = None
    model_started_at: datetime | None = None
    model_ended_at: datetime | None = None
    outcome: ReactionOutcome = "cancelled"     # overwritten by _run_step; stays "cancelled" if it never finishes
    error: str | None = None
    text_length: int = 0
    sink_duration: float | None = None
    command_call_ids: list[str] = field(default_factory=list)
    noop: bool = False
    usage: TokenUsage | None = None
```

   and a helper `_span_ids(span) -> tuple[str | None, str | None]` returning `(otel_trace.format_trace_id(sc.trace_id), otel_trace.format_span_id(sc.span_id))` when `sc = span.get_span_context()` has `sc.is_valid`, else `(None, None)`. And `_token_usage(response: AIMessage) -> TokenUsage | None`: `None` when `response.usage_metadata` is `None`; else `input_tokens`, `output_tokens`, and `cache_read_tokens = usage.get("input_token_details", {}).get("cache_read")`.

2. **`__init__`**: add the three Events next to the existing four, with the same comment style:

```python
        self.on_reaction_ended: Event[ReactionTrace] = Event()
        self.on_trigger_dropped: Event[WorldEntry] = Event()
        self.on_command_ended: Event[CommandTrace] = Event()
```

   Change `self._window_batch: list[WorldEntry]` to `list[_PendingTrigger]`; add `self._window_opened_at: datetime | None = None`, `self._reaction_counter = 0`, `self._current_reaction: _ReactionBuilder | None = None`, and `self._command_started: dict[str, tuple[datetime, int]] = {}` (key → (running timestamp, reaction_id)).

3. **`_handle_trigger(entry)`**: first line `arrived_at = now()`. In the busy branch, after the INFO log, `self.on_trigger_dropped.emit(entry)` then `return`. Otherwise `self._window_batch.append(_PendingTrigger(entry, arrived_at, contextvars.copy_context()))` and, when this is the first trigger of the batch (`len(self._window_batch) == 1`), `self._window_opened_at = arrived_at`. The task runs in the writer's context because the World scheduled `on_trigger.emit` with it and the shim's `create_task` copied it — nothing else to do.

4. **`_cancel_window()`**: also reset `self._window_opened_at = None`.

5. **`_flush_window()`**: after taking `batch` and clearing `_window_batch`, build the reaction:

```python
        window_closed_at = now()
        window_opened_at = self._window_opened_at or window_closed_at
        self._window_opened_at = None
        self._reaction_counter += 1
        traces = tuple(pending.trace() for pending in batch)
        parent = parent_trigger(traces)
        parent_pending = batch[traces.index(parent)]
        parent_ctx = parent_pending.context.run(otel_context.get_current)
        links = []
        for pending in batch:
            if pending is parent_pending:
                continue
            sc = pending.context.run(otel_trace.get_current_span).get_span_context()
            if sc.is_valid:
                links.append(otel_trace.Link(sc))
        span = tracer().start_span(
            "wica.agent.reaction",
            context=parent_ctx,
            links=links,
            attributes={"wica.reaction_id": self._reaction_counter, "wica.trigger_count": len(batch)},
        )
        builder = _ReactionBuilder(
            reaction_id=self._reaction_counter, triggers=traces,
            window_opened_at=window_opened_at, window_closed_at=window_closed_at, span=span,
        )
        self._busy = True
        self._track_task(self._run_batch([p.entry for p in batch], builder))
```

   `_run_batch` and `_run_step` now take the builder as a second argument.

6. **`_run_batch(batch, builder)`**:

```python
        self._current_reaction = builder
        try:
            with otel_trace.use_span(builder.span, end_on_exit=False):
                await self._run_step(batch, builder)
        finally:
            self._busy = False
            self._current_reaction = None
            ended_at = now()
            builder.span.set_attributes({
                "wica.outcome": builder.outcome, "wica.text_length": builder.text_length,
                "wica.command_count": len(builder.command_call_ids), "wica.noop": builder.noop,
            })
            builder.span.end()
            trace_id, span_id = _span_ids(builder.span)
            self.on_reaction_ended.emit(ReactionTrace(... every builder field ..., ended_at=ended_at,
                                                      trace_id=trace_id, span_id=span_id))
```

   `use_span` makes the reaction span current for everything awaited inside, so the model, sink and command spans nest under it and Command tasks created inside inherit it. The `finally` runs on cancellation too (`CancelledError` propagates after it), which is what gives the `cancelled` outcome. Emit **after** `_busy = False`.

7. **`_run_step(batch, builder)`**: keep every existing line and add stamps:
   - after `self.on_prompt.emit(...)`: `builder.prompt_ready_at = now()`.
   - wrap the model call:

```python
        builder.model_started_at = now()
        try:
            with tracer().start_as_current_span(
                "wica.agent.model",
                attributes={"gen_ai.operation.name": "chat", "gen_ai.provider.name": self._provider,
                            "gen_ai.request.model": self._model_name},
            ) as model_span:
                response = await self._bound_model.ainvoke(messages)
                builder.usage = _token_usage(response)
                if builder.usage is not None:
                    model_span.set_attributes({"gen_ai.usage.input_tokens": builder.usage.input_tokens,
                                               "gen_ai.usage.output_tokens": builder.usage.output_tokens})
        except Exception as exc:
            builder.model_ended_at = now()
            builder.outcome = "model_error"
            builder.error = str(exc)
            ... existing _logger.exception(...) and return ...
        builder.model_ended_at = now()
```

     Store `self._provider = config.provider` and `self._model_name = config.model` in `__init__` (they are read from `AgentConfig`).
   - after the response: `builder.text_length = len(text) if text else 0`; `builder.outcome = "empty" if not text and not response.tool_calls else "ok"`.
   - around the sink await (inside the existing `if text:`): `sink_started = now()`; `with tracer().start_as_current_span("wica.agent.sink"):` around the `try: await self._output_sink(text) except Exception: ...` block; after it `builder.sink_duration = (now() - sink_started).total_seconds()` (set it whether the sink raised or not).
   - in the tool-call loop: for `noop` set `builder.noop = True`; for a real Command, after `self._dispatch_command(...)`, `builder.command_call_ids.append(call_id)`.

8. **`_dispatch_command(call_id, name, args)`**: after the `running` write and before creating the task:

```python
        started_at = self._world.get_entry(key).current.timestamp
        reaction_id = self._current_reaction.reaction_id if self._current_reaction else 0
        self._command_started[key] = (started_at, reaction_id)
        span = tracer().start_span(
            "wica.agent.command",
            attributes={"wica.command.name": name, "wica.call_id": call_id},
        )
        task = self._track_task(self._run_command(key, call_id, name, args, span))
```

   `start_span` here (no `context=`) parents to the current span, which is the reaction span because `_run_step` runs inside `use_span`.

9. **`_run_command(key, call_id, name, args, span)`**: wrap the whole existing body in `with otel_trace.use_span(span, end_on_exit=True):` so any span the Command body opens nests under it and the terminal `world.update` captures it. In each of the three branches, right after the `_write_terminal(...)` call, call `self._emit_command_ended(key, call_id, name, "cancelled" | "failed" | "complete")` and, in the `cancelled` branch, keep the `raise` after it. Before ending, `span.set_attribute("wica.command.state", state)`.

   `_emit_command_ended(key, call_id, name, state)`:

```python
        started = self._command_started.pop(key, None)
        if started is None:
            return
        try:
            ended_at = self._world.get_entry(key).current.timestamp
        except KeyError:
            return   # entry already gone (World stopped and the write was dropped)
        started_at, reaction_id = started
        self.on_command_ended.emit(CommandTrace(call_id=call_id, name=name, reaction_id=reaction_id,
                                                started_at=started_at, ended_at=ended_at, state=state))
```

   `_write_terminal` swallows the `RuntimeError` of a stopped World; in that case the entry's current version is still `running`, so `ended_at` would be wrong. Make `_write_terminal` return `bool` (True when the update happened) and only call `_emit_command_ended` when it returned True.

10. **Docs, same step**: [specs/agent.md](../specs/agent.md) "Instrumentation: observability Events": add three rows to the Event table (`on_reaction_ended`, `on_trigger_dropped`, `on_command_ended`, one line each, pointing at instrumentation.md for the record fields) and one sentence that the loop opens the `wica.agent.reaction` / `model` / `sink` / `command` spans. Mark open question #5 (generic command-activity listening) as **Resolved by `on_command_ended`** (strike it through like #1). Status stays `Implemented`.

11. Tests — add to `tests/test_agent.py`, using `make_agent`, `ProgrammableChatModel`, the `world`/`loop`/`sink` fixtures and a `threading.Event` set from an `on_reaction_ended` subscriber (pattern: `traces: list[ReactionTrace] = []`; `agent.on_reaction_ended.subscribe(lambda t: (traces.append(t), done.set()))`). Register `prompt` (`str`, `triggers_llm_call=True`, `identity_serialize`) as today's tests do.
    - `test_reaction_ended_carries_the_phase_stamps_and_usage`: `respond` returns `AIMessage(content="hi", usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})`; after `world.update("prompt", "x")` and the event: one trace, `reaction_id == 1`, `outcome == "ok"`, `text_length == 2`, `usage == TokenUsage(3, 2, None)`, `noop is False`, `command_call_ids == ()`, one trigger with `key == "prompt"`, `is_command_completion is False`, and the stamps are ordered: `triggers[0].written_at <= triggers[0].arrived_at <= window_opened_at <= window_closed_at <= prompt_ready_at <= model_started_at <= model_ended_at <= ended_at`; `sink_duration is not None and >= 0`; `trace_id` is a 32-char hex string (the conftest provider is set).
    - `test_reaction_ended_reports_an_empty_response`: `respond` returns `AIMessage(content="")` → `outcome == "empty"`, `sink_duration is None`.
    - `test_reaction_ended_reports_a_model_error`: `respond` raises `RuntimeError("boom")` → `outcome == "model_error"`, `error == "boom"`, `model_ended_at is not None`, and `agent` is free again (a second update produces a second trace).
    - `test_reaction_ended_reports_cancellation_on_stop`: `respond` awaits an `asyncio.Event` that is never set; trigger, wait until `on_prompt` fired, call `agent.stop()` from the test thread → a trace with `outcome == "cancelled"`, `model_ended_at is None`.
    - `test_dropped_trigger_fires_on_trigger_dropped`: same blocking `respond`; trigger twice → `on_trigger_dropped` receives the second entry (`current.id` of the second update); then release the event.
    - `test_coalesced_reaction_lists_every_trigger`: `coalesce_window=0.2`, two updates within the window → one trace with two triggers and `coalescing_wait >= 0.15`.
    - `test_command_ended_carries_reaction_id_and_duration`: register `async def add(a: int, b: int) -> int` (docstring required), `respond` returns one tool call → `on_command_ended` gets `CommandTrace(name="add", reaction_id=1, state="complete")` with `duration >= 0`; the follow-up reaction's trace (from the completion re-trigger) has `triggers[0].is_command_completion is True` and `reaction_latency(trace) is None`.
    - `test_span_tree_follows_the_reaction`, using the `spans` fixture: same `add` flow; after the follow-up reaction ended, collect finished spans by name. Assert: `wica.agent.reaction` (first) has parent `== wica.world.update`'s context; `wica.agent.model` and `wica.agent.command` have the first reaction span as parent; the **second** `wica.agent.reaction`'s parent is the second `wica.world.update` span (the completion write), and *that* span's parent is the `wica.agent.command` span — the follow-up reaction is the Command's grandchild through its completion write; the first reaction's `trace_id` in the emitted `ReactionTrace` equals `format_trace_id(span.context.trace_id)`.
    - `test_spans_opened_inside_a_command_nest_under_the_command_span`: the `add` body does `with trace.get_tracer("app").start_as_current_span("tts.synthesize"): ...` → the `tts.synthesize` span's parent is the `wica.agent.command` span.
    - `test_coalesced_reaction_links_the_other_triggers`: `coalesce_window=0.2`, two triggering updates → the reaction span has exactly one link, and its parent is the update span of the **first** update.

## Step 5 — Facade

Spec: "Layer 1" (the facade names).

1. [src/wica/wica.py](../src/wica/wica.py): in `__init__`, after `self.on_agent_text`, add `self.on_agent_reaction_ended = agent.on_reaction_ended`, `self.on_agent_trigger_dropped = agent.on_trigger_dropped`, `self.on_agent_command_ended = agent.on_command_ended` with their `Event[...]` annotations; import the record types.
2. [specs/wica.md](../specs/wica.md): add the three rows to the member table and to the "`Event`s are surfaced, not adapted" table. Status stays `Implemented`.
3. Tests — add to `tests/test_wica.py`:
   - `test_facade_surfaces_the_instrumentation_events`: identity checks (`wica.on_agent_reaction_ended is wica.agent.on_reaction_ended`, etc.).
   - `test_fake_flow_yields_reaction_and_command_traces`: `wica_factory(fake_config([{"tool_calls": [{"name": "add", "args": {"a": 2, "b": 2}}]}, {"text": "done"}]), coalesce_window=0)` with `add` registered; subscribe both Events; write `prompt`; wait for the second reaction → two `ReactionTrace`s (`reaction_id` 1 and 2) and one `CommandTrace(reaction_id=1, state="complete")`.

## Step 6 — Transcript: spinner while thinking, duration when the reaction ends

Spec: "Consumers → Transcript". File: [src/wica/contrib/gradio/transcript.py](../src/wica/contrib/gradio/transcript.py).

1. In `__init__`, add `self._reaction_items: dict[int, dict[str, Any]] = {}` and subscribe `wica.on_agent_reaction_ended.subscribe(self.on_reaction_ended)` when `wica is not None`.
2. In `on_prompt()`, put the parent item with `"status": "pending"` in its metadata and remember it: `self._reaction_items[self._reaction_count] = item` (under `self._lock`).
3. Add:

```python
    def on_reaction_ended(self, reaction: ReactionTrace) -> None:
        """The reaction finished: drop the spinner and show how long the Agent was busy. Matched by
        reaction_id, which counts reactions exactly as on_prompt does (both start at 1 and advance
        once per reaction). Runs on the agent loop."""
        with self._lock:
            item = self._reaction_items.pop(reaction.reaction_id, None)
            if item is None:
                return
            item["metadata"].pop("status", None)
            item["metadata"]["duration"] = round(reaction.busy_time, 1)
```

   The snapshot signature already covers `status`, so the change is pushed to the UI.
4. [specs/gradio-contrib.md](../specs/gradio-contrib.md) "Component 3": in the assistant-side bullet, say the reaction group opens *pending* (spinner) and, when `on_agent_reaction_ended` fires, loses the spinner and shows the reaction's busy time as its duration; add `on_agent_reaction_ended` to the constructor's subscription comment in the signature block. [specs/conversation-demo.md](../specs/conversation-demo.md) "What the user sees" 1.: one sentence that each reaction group shows a spinner while the robot thinks and its duration once it has reacted. Both statuses stay `Implemented`.
5. Tests — add to `tests/contrib/test_gradio_transcript.py` (drive the log directly, as the existing tests do; build a `ReactionTrace` by hand with `reaction_id=1`, any consistent stamps, `busy_time` ≈ 1.2):
   - `test_reaction_group_is_pending_until_the_reaction_ends`: after `on_prompt`, the group item has `status == "pending"`; after `on_reaction_ended(trace)`, no `status` and `duration == 1.2`.
   - `test_reaction_ended_for_an_unknown_id_is_ignored`: `on_reaction_ended` with `reaction_id=7` before any prompt changes nothing.
   - Update `test_on_prompt_opens_a_reaction_group_titled_by_the_trigger` only if it asserts the exact metadata dict (the `status` key is new); the title assertion is unchanged.

## Step 7 — Demo: an opt-in console exporter

Spec: "Consumers → Exporters". File: [examples/conversation_demo/app.py](../examples/conversation_demo/app.py).

1. In `main()`, before `Wica.init`, add: when the environment variable `WICA_DEMO_TRACES` is `"console"`, install `TracerProvider()` + `SimpleSpanProcessor(ConsoleSpanExporter())` via `trace.set_tracer_provider(...)` (imports from `opentelemetry.sdk.trace` / `opentelemetry.sdk.trace.export`, inside the `if` so the SDK is only imported when asked). Log one INFO line saying traces go to the console. Any other value or unset: nothing.
2. [specs/conversation-demo.md](../specs/conversation-demo.md) "Configuration": one bullet describing `WICA_DEMO_TRACES=console`. [INTEGRATING.md](../INTEGRATING.md): a short section "Latency and traces" after recipe 4: subscribe to `wica.on_agent_reaction_ended` for per-reaction numbers (`reaction_latency`, `model_latency`, `busy_time`), and install an OpenTelemetry SDK provider to export spans; mention that a span opened inside a Command nests under `wica.agent.command`.
3. Run the demo once with the fake config if one exists, or skip the manual check; `tests-e2e/test_example_flow.py` (`-k example`) must still pass.

## Step 8 — Statuses and final verification

1. [specs/instrumentation.md](../specs/instrumentation.md): change `prompt_ready_at` and `model_started_at` in the `ReactionTrace` sketch to `datetime | None` with a short note (a reaction cancelled while rendering never reaches them) — editorial. Then `Stable` → `Implemented`, and the row in [specs/_index.md](../specs/_index.md).
2. [specs/_analysis.md](../specs/_analysis.md) P1: append one sentence that `on_trigger_dropped` and `ReactionTrace.busy_time` now make the drop and the busy interval measurable.
3. This plan: `Todo` → `Done`, and its row in [plans/_index.md](_index.md).
4. Final verification: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`, `uv run pytest tests-e2e -k "fake or example"` all pass, and `uv run pytest tests/test_project_map.py` is green.
