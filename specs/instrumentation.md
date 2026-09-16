---
code:
  - src/wica/agent.py
  - src/wica/world.py
  - src/wica/wica.py
  - src/wica/instrumentation.py
  - src/wica/contrib/gradio/transcript.py
  - pyproject.toml
tests:
  - tests/test_agent.py
  - tests/test_world.py
  - tests/test_wica.py
  - tests/test_instrumentation.py
  - tests/contrib/test_gradio_transcript.py
---

# Instrumentation

**Status:** Implemented

## Purpose

Measure and understand WICA's **latency and reactivity**: how long a reaction takes from the moment an input is written to the moment the person perceives the response, where that time goes inside the framework (coalescing, rendering, the model call, the output sink, Command dispatch), and when the Agent is busy and drops an input. Three questions, in order of importance:

1. **User-perceived latency** — input written → the output Command starts (or the free text is delivered). The number a person feels.
2. **The internal breakdown** — which phase of a reaction costs what, so coalescing, rendering, the sink and the provider can be tuned separately.
3. **The global measure across components outside WICA** — a speech recognizer that finishes *before* it writes the input, a TTS engine that synthesizes *inside* the output Command before audio plays. WICA only sees the middle of that chain; the measurement has to let the ends join it without those components knowing WICA.

The Agent already exposes instrumentation `Event`s (`on_trigger`, `on_prompt`, `on_command`, `on_text` — [agent.md](agent.md), "Instrumentation"), the World stamps every version with a wall-clock `timestamp` ([world.md](world.md)), and the Gradio transcript computes Command durations from its own monotonic stamps ([gradio-contrib.md](gradio-contrib.md)). What is **not** observable today, and cannot be reconstructed from outside the loop:

- **The end of a reaction.** A reaction whose model call raises, or whose response carries neither text nor tool calls, emits nothing after `on_prompt`. Pure model latency and the *busy* interval are invisible, so a dropped input cannot be explained.
- **A dropped trigger** is only an INFO log line.
- **Reaction identity.** Consumers correlate `on_trigger` → `on_prompt` → `on_command` by call order on the loop thread; nothing on the payloads says which reaction they belong to.
- **The sink's cost.** The output sink is awaited *before* the response's Commands are dispatched (`_run_step`), so a slow sink delays every Command of that reaction; nobody measures it.
- **Token usage.** LangChain returns `usage_metadata` on the response; the Agent drops it.

This spec adds the measurement. It is **instrumentation only**: nothing here changes what the loop does, what the model sees, or World state — a reaction does not become a World entry (that is the deferred `agent:reaction` design in [agent.md](agent.md), "Future improvements"; the `reaction_id` introduced here is the identity such an entry would reuse).

## Settled

### Vocabulary: a reaction and its phases

A **reaction** is one turn of the Agent's loop — the trigger(s) that woke it, the model call, and the Commands it dispatches — what the transcript calls `reaction N`. The code calls running one reaction a *step* (`_run_step`, "single in flight"); this spec uses *reaction* for the concept and *step* only when naming that code. Its phases, and the stamp each one leaves:

| Phase | Where it happens | Stamp |
|---|---|---|
| Input written | `world.update()` on the caller's thread | the version's `timestamp` (`written_at`) |
| Raw trigger reaches the loop | `on_trigger` emission after the `call_soon_threadsafe` hop | `arrived_at` |
| Coalescing window | `_handle_trigger` → `_flush_window` | `window_opened_at`, `window_closed_at` |
| Observation + rendering | `_run_step` before `on_prompt` | `prompt_ready_at` |
| Model call | `ainvoke` | `model_started_at`, `model_ended_at` |
| Free text to the sink | `await self._output_sink(text)` | `sink_duration` |
| Command dispatch, reaction end | end of `_run_step` | `ended_at` |
| Command running → terminal | the Command task | the entry's `running` and terminal version timestamps |
| Follow-up reaction (re-trigger) | a new reaction | its own record |

**Reaction latency** (the primary number) is `written_at` of the earliest *external* trigger of the reaction (a Command-completion trigger is not an input) → `ended_at` of a reaction that issued at least one Command or delivered text. With an output Command, `ended_at` is when the utterance starts; what the TTS does after that is the span layer's business (below).

### Two layers from one measurement

The loop keeps **one** internal trace object per reaction, filled in as the phases pass, and derives two outputs from it at the reaction's end:

1. A **typed reaction record**, emitted on an `Event` — for in-process consumers that read results back as data: the transcript, a metrics panel, the WicaTester Harness.
2. **OpenTelemetry spans**, opened live while the phases run — for merging with components outside WICA and for export to a tracing backend.

Both come from the same stamps, so they cannot drift. Neither can be turned off (see "Always on").

### Layer 1 — the reaction record and its Events

Three frozen dataclasses in a new leaf module (`src/wica/instrumentation.py`, zero project imports beyond `world.py`'s types, so the Agent can import it):

```python
@dataclass(frozen=True)
class TriggerTrace:
    key: str
    version_id: int
    written_at: datetime          # the World version's timestamp
    arrived_at: datetime          # on_trigger emission on the loop
    is_command_completion: bool   # an agent:command:<call_id> going terminal

@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None  # provider-dependent; None when not reported

@dataclass(frozen=True)
class ReactionTrace:
    reaction_id: int                              # 1, 2, 3 … per Agent — the transcript's "reaction N"
    triggers: tuple[TriggerTrace, ...]        # in arrival order; [0] opened the window
    window_opened_at: datetime
    window_closed_at: datetime                # = reaction start
    prompt_ready_at: datetime | None          # None when cancelled before rendering completed
    model_started_at: datetime | None         # None when cancelled before the model call started
    model_ended_at: datetime | None           # None when cancelled before the response
    outcome: Literal["ok", "empty", "model_error", "cancelled"]
    error: str | None
    text_length: int
    sink_duration: float | None               # seconds; None when no text
    command_call_ids: tuple[str, ...]         # dispatched Commands, in response order (noop excluded)
    noop: bool
    usage: TokenUsage | None                  # from AIMessage.usage_metadata; None when absent
    ended_at: datetime
    trace_id: str | None                      # OpenTelemetry ids when an SDK is installed, else None
    span_id: str | None

@dataclass(frozen=True)
class CommandTrace:
    call_id: str
    name: str
    reaction_id: int
    started_at: datetime                      # the running version's timestamp
    ended_at: datetime                        # the terminal version's timestamp
    state: Literal["complete", "failed", "cancelled"]
```

Derived durations (`coalescing_wait`, `render_time`, `model_latency`, `busy_time`, `duration` on `CommandTrace`) are read-only properties over the stamps, so a record stays raw and a consumer never recomputes a subtraction differently.

The Agent gains three `Event`s, surfaced on the facade like the existing ones ([wica.md](wica.md), "`Event`s are surfaced, not adapted"):

| Event | Type | Emitted |
|---|---|---|
| `on_reaction_ended` / `wica.on_agent_reaction_ended` | `Event[ReactionTrace]` | **once at the end of every reaction that started**, whatever its outcome — a normal one, an empty response, a model error, a cancellation by `stop()`. Emitted from `_run_batch`'s `finally`, after `_busy` is cleared, so a subscriber sees "the Agent is free again" and the record at the same moment |
| `on_trigger_dropped` / `wica.on_agent_trigger_dropped` | `Event[WorldEntry]` | once per trigger dropped because a reaction was in flight — the counterpart of `on_trigger`, which only fires for triggers a reaction observed. Replaces inference (pairing raw against observed triggers with a timeout) with a signal |
| `on_command_ended` / `wica.on_agent_command_ended` | `Event[CommandTrace]` | once per dispatched Command when its terminal state is written, with the reaction that issued it. This is the generic "any Command finished" signal that [agent.md](agent.md) open question #5 asks for, without a broadcast World entry: a consumer no longer needs a per-`call_id` listener just to learn when and how a Command ended |

All three follow the existing instrumentation rules: observation only, defensive payloads (the dataclasses are frozen), `Event.emit`'s per-subscriber isolation, emitted on the loop thread.

**One clock: wall-clock UTC.** Every stamp is a `datetime` from the World's own `_now()`, not `time.monotonic()`. The reason is correlation: the input-written stamp *is* the World version's timestamp, OpenTelemetry stamps spans in wall-clock time, and the log lines are wall-clock — a monotonic stamp could not be subtracted from any of them. The accepted cost is that a clock adjustment during a reaction distorts that one reaction's numbers; for latencies measured in tens of milliseconds to seconds this is the right trade.

### Layer 2 — OpenTelemetry spans and context propagation

WICA depends on **`opentelemetry-api`** (core dependency, `pyproject.toml`) and instruments itself the way the OpenTelemetry project asks libraries to: the API package is small and pure Python, and every span is a **no-op until the application installs the SDK** and an exporter. WICA never imports the SDK; the tests and the demo do not need it.

The spans, all under the tracer named `wica`:

| Span | Parent | Attributes |
|---|---|---|
| `wica.world.update` | the **caller's** current context — a speech recognizer writing from inside its own span makes that span the ancestor of the whole reaction | `wica.key`, `wica.version_id`. Opened only for an update that **triggers** (qualifies to wake the Agent), so a 30 fps non-triggering sensor entry costs no span; every update still propagates context (below) |
| `wica.agent.reaction` | the context captured with the reaction's **parent trigger** (rule below); the other coalesced triggers are span **links** | `wica.reaction_id`, `wica.trigger_count`, `wica.outcome`, `wica.text_length`, `wica.command_count`, `wica.noop` |
| `wica.agent.model` | the reaction | GenAI semantic conventions: `gen_ai.operation.name = chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |
| `wica.agent.sink` | the reaction | — (its duration is the point) |
| `wica.agent.command` | the reaction; **current inside the Command's task** from dispatch to terminal write | `wica.command.name`, `wica.call_id`, `wica.command.state` |

`noop` opens no span; it is an attribute of the reaction.

**Propagation is what makes the ends join.** `contextvars` are the carrier:

- `World.update()` captures `contextvars.copy_context()` under its lock and hands it to `loop.call_soon_threadsafe(..., context=ctx)` for the trigger emission and for each listener (a sync listener is run through `ctx.run` on the executor). The Agent's trigger shim `create_task`s inside that emission, and an asyncio task inherits the current context — so `_handle_trigger` runs in the writer's context, and the Agent keeps that context with the entry in the window batch.
- A Command task is created inside its `wica.agent.command` span, so **any span an application or a third-party library opens inside the Command body nests under it with no WICA-specific API** — a TTS engine's synthesis and playback spans land under `say`. The Command's terminal `world.update` happens inside that span too, so the completion trigger carries it and the **follow-up reaction descends from the Command** (its parent is the completion's `wica.world.update` span, itself the command span's child): one trace runs recognizer → input write → reaction → `say` → synthesis → playback → follow-up reaction → `noop`.

**Which trigger is the reaction's parent.** An OpenTelemetry span has exactly one parent, and a coalesced reaction has several triggers. The rule: **the earliest trigger in the batch that is not a Command completion; if every trigger is a completion, the earliest one.** The other triggers become links. Rationale: the earliest external input is the one that has waited longest, so the root-to-response distance in the trace is the *worst-case* user-perceived latency — the honest number, and the one the recognizer's span makes end-to-end. Parenting to the last trigger (the loop's `representative` for logging) would hide the coalescing tax; parenting to nothing and linking everything would break the trace into pieces that viewers do not join. When a user input joins a window a Command completion opened, the input still wins, so its trace is not buried under the previous reaction's; the completion becomes a link.

### Always on

There is no enable/disable API. The cost per reaction is a handful of `datetime.now()` calls, one frozen dataclass, and no-op span objects; against a model call it is unmeasurable. Consumers opt in the way they do for every other `Event`: by subscribing, or by installing an SDK. This keeps one rule for all instrumentation ("observation only, always emitted") instead of a second code path whose disabled branch nobody tests.

### Metrics: per-reaction measures in core

`instrumentation.py` also holds the definitions, as pure functions over records, so the Harness, a panel and a script compute the same numbers:

| Measure | From |
|---|---|
| `reaction_latency` | earliest external `TriggerTrace.written_at` → `ReactionTrace.ended_at`, for reactions with `command_call_ids` or `text_length > 0` |
| `coalescing_wait` | `window_opened_at` → `window_closed_at` |
| `hop` | per trigger, `written_at` → `arrived_at` (the cross-thread hop; a sanity number) |
| `render_time` | `window_closed_at` → `prompt_ready_at` |
| `model_latency` | `model_started_at` → `model_ended_at` |
| `sink_time` | `sink_duration` |
| `busy_time` | `window_closed_at` → `ended_at` — the interval during which a new input is dropped |
| `command_duration` | `CommandTrace.started_at` → `ended_at` |
| `reactions_per_input` | reactions between two reactions whose parent trigger is external (the re-trigger chain's cost) |
| `dropped` | count of `on_trigger_dropped` |

Distributions (min, median, p95) and thresholds are the Harness's ([wica-tester.md](wica-tester.md)); core gives per-reaction samples only.

### Consumers

- **Transcript** ([gradio-contrib.md](gradio-contrib.md)): the reaction group opens *pending* (a spinner while the model thinks) on `on_prompt` and, on `on_agent_reaction_ended`, loses the spinner and gains a duration — the same `metadata.duration` its Command items already use. `reaction_id` matches its `reaction N` numbering. `on_agent_command_ended` can replace its per-`call_id` World listener for the terminal edit, but need not. Dropped inputs stay hidden (decided: the transcript shows what the Agent reacted to). These are changes to that spec, made by the implementation plan (status `Updated` while the code lags).
- **WicaTester Harness** ([wica-tester.md](wica-tester.md)): the Probe records `reaction_ended`, `trigger_dropped` and `command_ended` events from the three new Events; the framework's stamps replace the Harness's arrival stamps for every latency measure, and its metric computation delegates to core's definitions. The Harness keeps stamping its *own* actions (`set`, `wait`, expectations), pairing them with reactions, and building distributions and the report.
- **Exporters** — an application that wants traces installs `opentelemetry-sdk` and an exporter (console, OTLP → Jaeger / Grafana Tempo / Langfuse) and configures a `TracerProvider` before `start()`; WICA needs nothing. How the conversation demo opts in (an env var selecting a console or OTLP exporter) is an example concern, documented with the demo when built.
- **A metrics panel** — a fourth Gradio contrib component (current state idle / coalescing / thinking / speaking, rolling last / median / p95 of the measures above) is the natural next consumer; its shape is an open question below.

## Open questions

Deferrals; none blocks the design above.

1. **The metrics panel.** Presenter + panel like the other three (`MetricsLog(wica)` + `metrics_panel`), or a strip inside the transcript? Wait for the records to exist and look at real numbers first.
2. **`reaction_id` on the existing Events.** `on_prompt`, `on_command`, `on_text` could carry the reaction id so a consumer correlates without relying on loop-thread ordering. Additive; do it when a consumer needs it rather than widen four payloads speculatively.
3. **OpenTelemetry Metrics API.** Histograms of the measures above could be recorded through the Metrics API as well as derived from spans. The reaction record and the span layer cover the need; revisit if a backend wants native metrics.
4. **An origin timestamp on `update()`.** A recognizer that is *not* OpenTelemetry-instrumented could still report when speech ended by passing an explicit origin time to `world.update`, which the record would carry as the true start of the reaction. A plain-timestamp fallback to the span layer; deferred until an input source needs it.
5. **Span for non-triggering updates.** Only triggering updates open a `wica.world.update` span, to keep high-rate sensors quiet. If a trace ever needs to show a passive entry's write, a per-entry opt-in at `register()` is the likely shape.
