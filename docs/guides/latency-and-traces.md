# Latency and traces

Every reaction leaves a record, and every phase of it a span. Subscribe to the records for the numbers (how long after an input the agent acted, how much of it was the model), or export the spans to see one interaction end to end next to your own components' spans. Instrumentation is always on and costs nothing to ignore. The design is [specs/instrumentation.md](../../specs/instrumentation.md).

## A reaction and its phases

A **reaction** is what one trigger (or a coalesced burst) causes: the window opens on the first trigger, closes, the prompt renders, the model is called, its text goes to the sink and its Commands are dispatched; then the reaction ends. The record is a `ReactionTrace`, emitted on `wica.on_agent_reaction_ended`:

```python
from wica import ReactionTrace
from wica.instrumentation import reaction_latency


def on_reaction(trace: ReactionTrace) -> None:
    print(
        f"reaction {trace.reaction_id}: {trace.outcome},"
        f" latency={reaction_latency(trace)}, model={trace.model_latency},"
        f" busy={trace.busy_time:.3f}s, commands={list(trace.command_call_ids)},"
        f" tokens={trace.usage.input_tokens if trace.usage else None}"
    )


wica.on_agent_reaction_ended.subscribe(on_reaction)
```

| Field or measure | What it is |
|---|---|
| `triggers` | The `TriggerTrace`s that joined the window: `key`, `version_id`, `written_at` (the `update()` call), `arrived_at` (the loop), `is_command_completion` |
| `window_opened_at`, `window_closed_at`, `prompt_ready_at`, `model_started_at`, `model_ended_at`, `ended_at` | The phase stamps, wall clock |
| `outcome` | `ok`, `empty` (no text and no Command), `model_error`, `model_timeout` (the call passed `model_timeout`), `cancelled`; `error` carries the text |
| `text_length`, `sink_duration`, `command_call_ids`, `noop` | What the step produced |
| `usage` | `TokenUsage(input_tokens, output_tokens, cache_read_tokens)` as the provider reported it |
| `reaction_latency(trace)` | Earliest external trigger's `written_at` → `ended_at`: how long the person waited for the agent to act. `None` when the reaction only followed a Command completion |
| `trace.model_latency` | `model_started_at` → `model_ended_at`: the model's share |
| `trace.coalescing_wait`, `trace.render_time` | The window, the rendering |
| `trace.busy_time` | `window_closed_at` → `ended_at`: the interval during which a new input is dropped |
| `reactions_per_input(reactions)` | How many reactions each external input cost (the completion chain) |

Two more Events complete the picture: **`on_agent_trigger_dropped`** (a `WorldEntry` whose update arrived while a step was in flight; the count is the harness's `dropped_inputs`) and **`on_agent_command_ended`** (a `CommandTrace`: `call_id`, `name`, `reaction_id`, `started_at`, `ended_at`, `state`, `duration`). The reaction panel shows one reaction's numbers on demand ([gradio-panels.md](gradio-panels.md)); the `TestRun` harness pools them into distributions ([testing.md](testing.md)).

## Exporting traces

WICA opens OpenTelemetry spans through `opentelemetry-api` only and never imports an SDK, so nothing changes if you do not configure one. To view traces in a backend (Jaeger, Grafana Tempo, Langfuse, or the console), install `opentelemetry-sdk` and set a `TracerProvider` **before** `Wica.init`:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

wica = Wica.init(config)
```

The span tree per reaction: `wica.agent.reaction`, a child of the `wica.world.update` span of the earliest external trigger that woke it (the other coalesced triggers are links), with `wica.agent.model`, `wica.agent.sink` and one `wica.agent.command` per Command under it. Context propagates through `world.update()` and into Command tasks, so **a span your own code opens inside a Command** (a TTS engine's synthesis, a robot move) nests under that Command's span with no WICA-specific API, and a speech recognizer's span around its `world.update()` call joins the same trace. The conversation demo wires exactly this behind `WICA_DEMO_TRACES=console` ([`app.py`](../../examples/conversation_demo/app.py)).

## Reading the numbers

- **Reaction latency** is what the person feels: input written to action taken. The model's share is `model_latency`; the rest is the coalescing window (200 ms by default, `Wica.init(config, coalesce_window=…)`), rendering, dispatch and the sink.
- **Gemini thinks by default**, which can add seconds to every reaction: cap it through `model_kwargs` ([../reference/configuration.md](../reference/configuration.md#providers)).
- **Busy time** is the window of vulnerability: an input arriving inside it is dropped. A long sink (speaking inline in the sink) lengthens it; an output Command does not, since the step ends at dispatch ([commands.md](commands.md#the-output-command)).
- **Reactions per input** above one is the completion chain: each Command completion wakes a step; `triggers_on_completion=False` on a Command trims it.
