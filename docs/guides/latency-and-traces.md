# Latency and traces

Every reaction leaves a record, and every phase of it a span. Subscribe to the records for the numbers (how long after an input the agent acted, how much of it was the model), or export the spans to see one interaction end to end next to your own components' spans. Instrumentation is always on and costs nothing to ignore. The design is [specs/instrumentation.md](../../specs/instrumentation.md).

## A reaction and its phases

A **reaction** is what one trigger (or a coalesced burst) causes: the window opens on the first trigger, closes, the prompt renders, the model is called, what it says is handed to the output Command as it is written and its Commands are dispatched; then the reaction ends. The record is a `ReactionTrace`, emitted on `wica.on_agent_reaction_ended`:

```python
from wica import ReactionTrace
from wica.instrumentation import reaction_latency


def on_reaction(trace: ReactionTrace) -> None:
    print(
        f"reaction {trace.reaction_id}: {trace.outcome},"
        f" latency={reaction_latency(trace)}, model={trace.model_latency},"
        f" first_output={trace.time_to_first_output},"
        f" busy={trace.busy_time:.3f}s, commands={list(trace.command_call_ids)},"
        f" tokens={trace.usage.input_tokens if trace.usage else None}"
    )


wica.on_agent_reaction_ended.subscribe(on_reaction)
```

| Field or measure | What it is |
|---|---|
| `triggers` | The `TriggerTrace`s that joined the window: `key`, `version_id`, `written_at` (the `update()` call), `arrived_at` (the loop), `is_command_completion` |
| `window_opened_at`, `window_closed_at`, `prompt_ready_at`, `model_started_at`, `first_content_at`, `first_output_at`, `model_ended_at`, `ended_at` | The phase stamps, wall clock. `first_content_at` is the first streamed chunk carrying text or a Command call, `None` when none arrived. `first_output_at` is when the reaction's first output opened, its first words handed to the output Command; `None` for a reaction that said nothing |
| `outcome` | `ok` (a `<noop/>` included), `empty` (no text and no Command), `model_error`, `model_timeout` (the call passed `model_timeout`), `rejected` (the [response guards](../reference/api.md#response-guards) still rejected the response after the last retry), `cancelled`; `error` carries the text |
| `attempts`, `rejections`, `guard_duration` | Model calls made (more than one after a rejected response), each rejection's feedback, the seconds spent in the guards |
| `text_length`, `command_call_ids`, `noop` | What the step produced: the length of the response's text as written, tags included; the `call_id` of each Command it issued, each output counting as one; whether the model declined to react (`<noop/>` or an empty response) |
| `usage` | `TokenUsage(input_tokens, output_tokens, cache_read_tokens)` as the provider reported it, summed over the attempts |
| `reaction_latency(trace)` | Earliest external trigger's `written_at` → `first_output_at` when the reaction spoke, else → `ended_at` when it dispatched a Command: how long the person waited for the agent to speak or act. `None` when the reaction did neither, or only followed a Command completion |
| `trace.model_latency` | `model_started_at` → `model_ended_at`: the model's share, every attempt and the guards between them included |
| `trace.time_to_first_content`, `trace.generation_time` | `model_latency` in two: the wait before the answer began to arrive, then the answer arriving. Text arrives progressively; a Command call's arguments arrive progressively on some providers and all at the end on others |
| `trace.time_to_first_output` | `model_started_at` → `first_output_at`: how long the model call took to give the output Command its first words. It follows `time_to_first_content` by the tag's opening and any private text before it; for text with no tag, delivered whole at the end of the response, it equals `model_latency`. `None` when the reaction opened no output |
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

The span tree per reaction: `wica.agent.reaction`, a child of the `wica.world.update` span of the earliest external trigger that woke it (the other coalesced triggers are links), with `wica.agent.model` and one `wica.agent.command` per Command, an output included, under it; the model span carries a `wica.first_content` event and, when the reaction speaks, a `wica.first_output` one. Context propagates through `world.update()` and into Command tasks, so **a span your own code opens inside a Command** (a TTS engine's synthesis, a robot move) nests under that Command's span with no WICA-specific API, and a speech recognizer's span around its `world.update()` call joins the same trace. The conversation demo wires exactly this behind `WICA_DEMO_TRACES=console` ([`app.py`](../../examples/conversation_demo/app.py)).

## Reading the numbers

- **Reaction latency** is what the person feels: input written to the first words handed to your output Command, or to the Commands dispatched when the reaction does not speak. It no longer waits for the end of the response. What your voice does after it has the first words (synthesis, playback) is yours to measure, with a span inside the output Command. The rest of the latency is the coalescing window (200 ms by default, `Wica.init(config, coalesce_window=…)`), rendering, and the model up to its first output: `time_to_first_output`.
- **A streaming output Command starts earlier.** Annotated `OutputStream`, it has the first words at `first_output_at`; annotated `str`, it is called when the tag closes ([commands.md](commands.md#the-whole-text-or-the-words-as-they-are-written)). A response with no tag is delivered at the end of the model call either way.
- **Gemini thinks by default**, which can add seconds to every reaction: cap it through `model_kwargs` ([../reference/configuration.md](../reference/configuration.md#providers)).
- **Busy time** is the window of vulnerability: an input arriving inside it is dropped. It ends with the model's response, not with the speech: an output runs as its own task, so a long utterance does not lengthen it ([commands.md](commands.md#make-your-agent-speak)).
- **Reactions per input** above one is the completion chain: each Command completion wakes a step, and so does the end of an utterance that carried a `<next>` note; `triggers_on_completion=False` on a Command and `max_next_steps` trim it.
