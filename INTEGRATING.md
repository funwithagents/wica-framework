# Integrating WICA

**This file is for an agent (or developer) building an application that depends on WICA.** It is the consumer's entry point: the mental model, the public API you import, the canonical wiring recipes, the limits to design around, and where to read deeper. It deliberately excludes how to *develop WICA itself* — that's [AGENTS.md](AGENTS.md) (specs, plans, status discipline). Human readers evaluating the framework may prefer the prose [README.md](README.md).

> WICA ships **no model provider**. Install the extra for the one you use:
> ```bash
> uv add "wica[anthropic] @ git+https://github.com/funwithagents/wica-framework"
> # or wica[openai], or wica[huggingface-hub]
> ```

## Mental model (read this first)

You talk to **one object — `Wica`** — the single entry point. `Wica.init(config)` stands up the whole system from a config: it owns one event loop and a **World** + **Agent** pair that both run on it, and surfaces everything you need behind one surface. State lives in the World — a registry of named, typed entries that renders to the LLM prompt. You don't thread a conversation; you maintain state, and every reasoning step renders the relevant World into the prompt (a *view of state*, not a transcript).

“Multimodal” spans both sides differently: Inputs serialize perception into provider-neutral `Content`, while Commands produce speech, movement, display changes, API effects, and other physical or digital outputs. The conversational sink is one complete text response per reasoning step; it is not the full output surface.

- **Wica**: the entry point. `Wica.init(config)` builds and wires everything; `wica.set_output_sink(...)` / `wica.set_output_command(...)` wire the output channel afterwards (so the objects they belong to can be built against the Wica first — build, then wire, then start); `wica.start()` / `wica.stop()` run a restartable shared lifecycle, and `wica.close()` performs terminal teardown. `wica.world` is the World; `wica.register_command(...)` adds Commands; its `Event`s surface what the loop is doing.
- **W — World** (`wica.world`): you `register()` a key with a type + a `serialize_fn` (value → [`Content`](specs/content.md)), then `update()` it as data changes. Values must be deep-copyable: the World copies on ingress and on outward getters/callbacks so only `update()` can change versioned state. `update()` is callable from any thread, but only **while the system is running** (between `start()` and `stop()`); `register`/`get` work any time.
- **I — Inputs**: not a class — a *role* a World entry plays when an external producer feeds it. A registered entry marked `triggers_llm_call=True` wakes the Agent when updated. The producer can run on any thread.
- **C — Commands**: the agent's unit of action. Register a plain function (or an off-the-shelf LangChain tool wrapped as `Command(tool)`) with `wica.register_command`; the Agent binds it as a native model tool. Each runs as a cancellable `asyncio` task, tracked as a World entry.
- **A — Agent** (`wica.agent`): the reasoning loop. Built from config, runs on the shared loop, builds the prompt from World state, runs inference, dispatches Commands, and delivers one complete text response per step to your output sink. Rarely touched directly — the everyday paths are surfaced on `Wica`.

## Public API — what you import

Everything below is re-exported from the top-level `wica` package ([`src/wica/__init__.py`](src/wica/__init__.py)):

| Import | Kind | Use |
|---|---|---|
| `Wica` | class | The single entry point (`init`/`start`/`stop`/`register_command`, `wica.world`, the instrumentation Events) |
| `World` | class | The state registry (`register`/`update`/`get`/`unregister`, `start`/`stop`/`is_running`); reached as `wica.world` |
| `TextPart`, `ImagePart` | dataclass | Multimodal content parts; `Content` is a `list` of them |
| `Content`, `ContentPart` | type alias | What a `serialize_fn` returns |
| `Agent` | class | The reasoning loop; usually owned by `Wica`, but constructible directly (see the direct seam below) |
| `Command` | class | Definition wrapper for a callable or an off-the-shelf LangChain tool; use it to override callable metadata or wrap an existing tool |
| `CommandIssued` | dataclass | Payload of `wica.on_agent_command` (`name`, `args`, `call_id` — the `agent:command:<call_id>` entry's key suffix; the Event fires once that entry is registered, so a handler may `add_listener` on it) |
| `CommandExecution` | dataclass | Value of an `agent:command:<call_id>` World entry (`name`, `args`, `state`, `result`, `error`); `isinstance` against it to recognize a Command execution when reading the World or writing a `display_entry` hook (recipe 4) |
| `Event` | class | The pub/sub primitive the instrumentation signals use (`subscribe`/`unsubscribe`) |
| `WicaConfig`, `AgentConfig` | dataclass | Plain configuration objects; construct directly or parse strictly from a dictionary, a JSON string, or a JSON file |
| `ConfigError`, `MissingEnvError` | exception | Invalid parsed config or a reference that cannot be resolved when the Agent is built |
| `WorldEntry`, `WorldEntryConfig`, `WorldEntryVersion` | dataclass | Entry introspection (rarely needed directly) |

Instrumentation is a set of `Event`s you `.subscribe(...)` on — multi-consumer, so panels, a logger, and a metrics sink can all watch the same signal:

| Event | Payload | Fires |
|---|---|---|
| `wica.on_world_trigger` | `WorldEntry` | **raw** — every qualifying World update (pre-coalescing), including triggers the busy loop later drops |
| `wica.on_agent_trigger` | `WorldEntry` | **filtered** — once per trigger a run-to-completion step actually observed |
| `wica.on_agent_prompt` | `list[BaseMessage]` | the exact rendered messages before each model call |
| `wica.on_agent_command` | `CommandIssued` | each Command the model issues, at dispatch |
| `wica.on_agent_text` | `str` | the step's complete free text, right before the output sink receives it (observe the free-text channel without taking the sink slot) |

Signatures you'll actually call:

```python
WicaConfig(agent=AgentConfig(...))
WicaConfig.from_dict(data, *, base_dir=None)
WicaConfig.from_json(text, *, base_dir=None)
WicaConfig.from_json_file(path)
Wica.init(config: WicaConfig, *, coalesce_window=0.2, loop=None) -> Wica
wica.world            # the World (below); wica.agent — the Agent (rarely needed)
wica.set_output_sink(async_fn | None)        # receives the model's free text each step
wica.set_output_command(fn_or_command | None)  # the user-facing output Command (free text → private)
wica.register_command(fn_or_command)   # plain callable, or Command(...) for metadata/tool wrapping
wica.start(); wica.stop(); wica.start()   # stop is a reversible pause
wica.close()                              # terminal; releases an owned event loop
wica.on_world_trigger / on_agent_trigger / on_agent_prompt / on_agent_command / on_agent_text   # .subscribe(handler)

world.register(key: str, type: type[T], *,
               serialize_fn: Callable[[T | None, T | None], Content],  # (value, previous) -> Content
               archival_serialize_fn=None,      # lighter render for older turns; defaults to serialize_fn
               include_in_prompt=True,           # False → internal/shared state the model never sees
               triggers_llm_call=False,          # True → an update wakes the Agent (this is an Input)
               trigger_condition_fn=None,        # (old, new) -> bool: gate which updates trigger
               ttl=None,                         # timedelta → auto-expire the entry to None
               bypass_coalescing=False)          # True → fire immediately, skip the batch window
world.update(key, value)     # from any thread — but only while the system is running
world.get(key)               # defensive copy of current value (works any time)
world.unregister(key)

# The direct seam (advanced / tests): construct an Agent yourself instead of via Wica.
Agent(config: AgentConfig, *, world: World, loop, coalesce_window=0.2, model=None)
agent.set_output_sink(...); agent.set_output_command(...)   # same setters as on Wica
```

## Recipes

### 1. Feed a perception (an Input)

```python
from wica import TextPart

wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, prev: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,          # updating this wakes the Agent
)
# From anywhere — a UI callback, a sensor loop, a hardware interrupt (while the system is running):
wica.world.update("speech_input", "hello robot")
```

### 2. Give the agent an action (a Command)

```python
import asyncio

async def dance() -> str:
    """Perform a fun little dance. Takes about 10 seconds."""   # docstring → the tool description
    await asyncio.sleep(10)
    return "Finished the dance."

wica.register_command(dance)
```

A plain callable needs no WICA-specific hooks. Wrap an off-the-shelf LangChain tool as
`Command(existing_tool)` before registration; use `Command(fn, name=..., description=...)` to
override a callable's inferred metadata. Each execution is tracked as a World entry
`agent:command:<call_id>` that renders `running` while in flight and terminal (`result`/`error`)
once done, so a later step can *see* an action still running. The Agent auto-registers a native
`cancel_command(call_id)` so the model can abort its own in-flight Commands.

> **Prefer `async` for anything cancellable.** Cancellation (`cancel_command`, or `stop()`) cancels the `asyncio` task: an `async` Command unwinds cleanly at its next `await`. A plain **sync** function works too — it's offloaded to a thread — but Python can't kill a running thread, so on cancel the entry flips to `cancelled` immediately while the thread runs the function to completion in the background, result discarded. Make a long-running sync Command **cooperative** (poll a `threading.Event`) if it needs to actually stop. See [specs/commands.md](specs/commands.md) ("Cancellation reaches the task, not always the work").

### 3. Wire and run the system

```python
from wica import AgentConfig, TextPart, Wica, WicaConfig

async def speak(text: str) -> None:     # the output sink: async, takes the model's text
    print("robot says:", text)

config = WicaConfig(
    agent=AgentConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        api_key_env="WICA_ANTHROPIC_API_KEY",
        system_prompt="You are a friendly social robot.",
    )
)
wica = Wica.init(config)                             # owns the loop + World + Agent
wica.set_output_sink(speak)                          # wire the output after init, before start

# Register entries and Commands against the owned World, then start.
wica.world.register(
    "speech_input", str,
    serialize_fn=lambda value, prev: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,
)
wica.register_command(dance)
wica.on_agent_prompt.subscribe(lambda messages: ...)   # optional instrumentation

wica.start()
wica.world.update("speech_input", "hello!")   # Agent wakes, reasons, calls speak()/Commands
# ... keep the process alive while the system runs on its own loop ...
wica.close()
```

`wica.world` / `wica.agent` are **borrowed references** — valid for the life of the `Wica`. While stopped, reactive mutation through `world.update()` raises `RuntimeError("World is not running")`; schema operations and reads remain available, and `start()` resumes the same objects with their registrations, values, subscriptions, Commands, and Agent history intact. `close()` is terminal. To reset state rather than resume it, close this Wica and initialize a new one.

### 4. Add a Gradio UI (the three observability panels)

Install `wica[gradio]` and import from `wica.contrib.gradio` (never re-exported from `wica`). The
package ships three panels — the live World-state table, the prompt history, and the conversation
transcript — each as a **presenter** built over your `Wica` plus a **panel** function you call
inside your own `gr.Blocks`. The whole flow is **build → UI → wire → start → launch → close**:

```python
import gradio as gr

from wica import AgentConfig, CommandExecution, TextPart, Wica, WicaConfig, WorldEntry
from wica.contrib.gradio import (
    EntryDisplay, PromptLog, TranscriptLog,
    conversation_panel, prompt_panel, world_state_panel,
)

# 1. Build the Wica and register your World entries.
config = WicaConfig(agent=AgentConfig(
    provider="anthropic", model="claude-sonnet-5",
    api_key_env="WICA_ANTHROPIC_API_KEY", system_prompt="You are a friendly social robot.",
))
wica = Wica.init(config)
wica.world.register(
    "speech_input", str,
    serialize_fn=lambda value, prev: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,
)

# 2. Build the presenters over the Wica (they subscribe to its Events in their constructor),
#    then the page. One optional hook says how your entries read to a person; return None for
#    the generic default (`⚡ key = value`, `🦾 name(args)` for a Command execution).
def display_entry(entry: WorldEntry) -> EntryDisplay | None:
    value = entry.current.value
    if entry.key == "speech_input":
        return EntryDisplay(f'🗣️ "{value}"')
    if isinstance(value, CommandExecution) and value.name == "say":
        return EntryDisplay("🗣️ say", str(value.args.get("text", "")))
    return None

transcript = TranscriptLog(wica, display_entry)   # pass the same hook to both presenters
prompts = PromptLog(wica, display_entry)          # so transcript and prompt labels agree

with gr.Blocks() as page:
    with gr.Row():
        with gr.Column():
            conversation_panel(transcript)          # panels go inside the Blocks context
            msg = gr.Textbox(placeholder="Say something…", show_label=False)
        with gr.Column():
            world_state_panel(wica.world)
            prompt_panel(prompts)
    # Input widgets only touch the World; the panels refresh on their own timers.
    msg.submit(lambda text: wica.world.update("speech_input", text) or "", inputs=msg, outputs=msg)

# 3. Wire: the Commands (the voice as the output Command), then hand the transcript's sink back.
async def say(text: str) -> str:
    """Speak out loud to the person in front of you."""
    ...                                            # your TTS / robot voice
    return "Said it."

async def dance() -> str:
    """Perform a fun little dance."""
    ...
    return "Finished the dance."

wica.set_output_sink(transcript.output_sink)     # the model's free text → the 💭 item
wica.set_output_command(say)                     # the user-facing channel → the 🗣️ item
wica.register_command(dance)

# 4. Start the system, then serve the page; tear down when the server returns.
wica.start()
try:
    page.launch()
finally:
    wica.close()
```

Panel signatures (each owns the `gr.Timer` that keeps it live and returns its component(s)):

```python
world_state_panel(world, *, refresh_s=0.2) -> gr.HTML
prompt_panel(prompts, *, refresh_s=0.2, lines=16) -> tuple[gr.Dropdown, gr.Textbox]
conversation_panel(transcript, *, refresh_s=0.2, height=420) -> gr.Chatbot
TranscriptLog(wica, display_entry=None); PromptLog(wica, display_entry=None)   # wica=None → renders empty, subscribes nothing
```

**The ordering rules behind that sequence** — what is fixed and what is free:

- **Presenters after `Wica.init`.** `TranscriptLog`/`PromptLog` subscribe to the Wica's Events in their constructor, so the Wica must exist first. They are plain objects: build them before or outside the `gr.Blocks` context. Panels, by contrast, create Gradio components and **must** be called inside a `gr.Blocks` context.
- **Wire before `start()`.** `set_output_command` recomposes the system prompt to name the Command; setting it after the first step rebuilds the message and drops the cached prefix. `set_output_sink` and `register_command` also belong before `start()`. The order *among* those three calls does not matter, nor does the order between registering World entries and building presenters.
- **The sink's meaning depends on the output Command.** With an output Command set (as above), the free text the transcript's sink receives is the model's **private reasoning**, shown as the `💭 output sink` item, and the `🗣️` Command item is what the person hears. Without an output Command, that same sink text *is* the utterance. The transcript renders both cases identically; only the reading changes.
- **Input widgets update the World, which works only while running.** `world.update()` raises while stopped, so start the Wica before `launch()`; a callback that fires after `stop()` fails the same way.
- **Tear down from the Gradio thread.** `launch()` blocks until the server exits; call `wica.close()` after it returns. Never call `stop()`/`close()` from a sink, a Command, or an Event subscriber — those run on the Wica's loop thread and the call raises `RuntimeError`.
- **One hook for both presenters.** `display_entry` is optional, but pass the same one to `TranscriptLog` and `PromptLog` so the transcript's input items and the prompt history's labels read alike.

The layout is yours: the panels are building blocks, not a page. Nor is the inline shape above
the required one — a larger app can let its layout function build the presenters next to their
panels and return them (the presenter is the UI's read model, so it is natural UI state), then
wire the returned transcript's sink in its main. The shipped conversation demo does exactly that:
[`app_ui.py`](examples/conversation_demo/app_ui.py) owns the presenters and hands them back on a
small handle, and [`app.py`](examples/conversation_demo/app.py) wires and starts — the same
sequence with a Speaking panel, sensor buttons and an explore-only fallback when no key is set.
[specs/gradio-contrib.md](specs/gradio-contrib.md) carries the design rationale.

### Latency and traces

Every reaction ends with `wica.on_agent_reaction_ended` firing a `ReactionTrace` — subscribe to it
for the per-reaction numbers: `reaction_latency(trace)` (input written → reaction ended, `None`
when the reaction only re-triggered from a Command completion), `trace.model_latency`, and
`trace.busy_time` (how long the Agent was unavailable for the next trigger). `wica.instrumentation`
also holds `reactions_per_input` (the re-trigger chain's cost) and the raw measures each stamp
enables — see [specs/instrumentation.md](specs/instrumentation.md) ("Metrics").

For a trace you can view in a backend (Jaeger, Grafana Tempo, Langfuse, or just the console),
install `opentelemetry-sdk` and configure a `TracerProvider` **before** `Wica.init` — WICA itself
depends only on `opentelemetry-api` and never imports the SDK, so nothing changes if you don't:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

wica = Wica.init(config)
```

A span opened inside a Command's body (an application or third-party library calling
`tracer.start_as_current_span(...)`) automatically nests under that Command's `wica.agent.command`
span, with no WICA-specific API — the pattern a TTS engine's synthesis/playback spans, or a speech
recognizer's own span around `world.update()`, use to join one end-to-end trace. See
[specs/instrumentation.md](specs/instrumentation.md) ("Layer 2") for the full span tree. The
conversation demo wires exactly this behind `WICA_DEMO_TRACES=console`
([`app.py`](examples/conversation_demo/app.py)).

## Configuration

`Wica.init()` consumes a `WicaConfig`, regardless of where its values originate. Construct the
plain dataclasses directly when Python owns the settings, call `WicaConfig.from_dict()` for a
mapping supplied by a larger application, `WicaConfig.from_json()` for a JSON string (same optional
`base_dir` as `from_dict()`), or `WicaConfig.from_json_file()` for a dedicated file. The loaders are
strict: missing required keys, unknown keys (typos), invalid combinations, and
wrong types fail with `ConfigError`.

For a larger application configuration, extract the WICA-shaped subsection:

```python
config = WicaConfig.from_dict(
    app_settings["wica"],
    base_dir=app_settings_path.parent,
)
```

`base_dir` is only needed to give a relative `system_prompt_file` the same stable origin it would
have in a dedicated file. Without it, `from_dict()` keeps a relative path verbatim. Direct
dataclass construction does not run the loaders' strict runtime validation, so callers using that
path are responsible for valid types and field combinations.

The config dataclasses are frozen and enforce their structural invariants on construction (exactly
one of `system_prompt`/`system_prompt_file`, at most one of `api_key`/`api_key_env`) — so direct
construction fails with the same `ConfigError` the loaders raise; derive a variant with
`dataclasses.replace(...)` rather than assigning.

The dictionary/JSON representation has this shape:

```json
{
  "agent": {
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "api_key_env": "WICA_ANTHROPIC_API_KEY",
    "system_prompt_file": "prompts/robot.md",
    "model_kwargs": { "temperature": 0.7 }
  }
}
```

| Key | Required | Notes |
|---|---|---|
| `agent.provider` | yes | `anthropic` \| `openai` \| `huggingface-hub` \| `fake` (deterministic test double — see "Testing flows deterministically") |
| `agent.model` | yes | Model id (or Hub `repo_id` for `huggingface-hub`) |
| `agent.system_prompt` / `system_prompt_file` | exactly one | Inline, or a path located according to the construction method described above |
| `agent.api_key` / `api_key_env` | at most one | Literal key, or an env var read at **Agent build** (`Wica.init`). Neither → provider's standard env var. Prefer `api_key_env` so the config carries no secret and is safe to commit |
| `agent.model_kwargs` | no | Forwarded to the provider (e.g. `temperature`) |
| `agent.hf_provider` | no | Only for `huggingface-hub`: the Hub backend (`auto`/`fireworks-ai`/…). Default `auto` |

For `from_json_file()`, a relative `system_prompt_file` is located relative to the JSON file. No
loader reads the prompt file or resolves `api_key_env`; those operations happen when `Wica.init()`
builds the Agent. A caller that degrades on `MissingEnvError` or an unreadable prompt therefore
wraps `Wica.init()`, not config creation. Code-only wiring (`output_sink`, `output_command`,
`coalesce_window`, and an optional event loop) also belongs in `Wica.init()`. Selecting a provider
whose extra is not installed fails there with a clear `ImportError`.

**Logging is your application's concern, not WICA's.** WICA is a library: it emits records under the `wica.*` loggers and installs only a `NullHandler` — it never sets a level or adds handlers. Configure logging in your app (`logging.basicConfig(...)` and `logging.getLogger("wica").setLevel(...)`); raise the `wica` level to `DEBUG` for a full World+Agent lifecycle trace.

## Testing flows deterministically

Your own e2e tests can drive the whole loop against a **scripted** model instead of a live LLM — no key, no network, fully deterministic — by selecting `provider: "fake"`. The model replays `model_kwargs.script` step by step: each reasoning step consumes the next entry, so you assert the *exact* Commands and utterances a real provider could never pin. Once the script is spent it returns `model_kwargs.default` (so an extra step never crashes a test); `delay_s` (default 200 ms) simulates latency and is worth setting to `0` in a test.

```python
from wica import Wica, WicaConfig

config = WicaConfig.from_dict({          # or a committed *.config.json, same as production
    "agent": {
        "provider": "fake",
        "model": "scripted",
        "system_prompt": "You are a test double.",
        "model_kwargs": {
            "delay_s": 0,
            "script": [
                {"tool_calls": [{"name": "walk_to", "args": {"place": "kitchen"}}]},
                {"text": "On my way."},
            ],
        },
    },
})
wica = Wica.init(config)
wica.set_output_sink(my_sink)
wica.world.register("prompt", str, serialize_fn=..., triggers_llm_call=True)
wica.register_command(walk_to)
wica.start()
# ... update a triggering World entry, then assert the command entry + sink output ...
wica.close()
```

Each scripted `tool_calls[].name` is validated against your registered Commands (a typo raises, rather than silently emitting an unknown call). For a script that can't live in JSON (a reactive/programmatic double), construct the Agent directly with the raw-model override: `from wica.fake_model import FakeChatModel` → `Agent(AgentConfig(provider="fake", model="…", system_prompt="…"), world=world, loop=loop, model=FakeChatModel(...))`. Full design in [specs/fake-provider.md](specs/fake-provider.md).

## v1 limits to design around

- **One reasoning call at a time.** A trigger arriving while a step is in flight is **dropped, not queued**. Concurrency/interruption (`cancel_reaction`, barge-in) is designed but deferred — see [specs/agent.md](specs/agent.md).
- **Trigger coalescing.** A burst of triggers within `coalesce_window` (default 200 ms) is batched into a single step. Set an entry's `bypass_coalescing=True` for something that must act at once (a stop button); `coalesce_window=0` disables batching.
- **Lifecycle is restartable.** `wica.start()`/`wica.stop()` may repeat on the same instance and preserve its state. Call `wica.close()` for terminal teardown; a closed Wica cannot restart.
- **The conversational sink is complete text.** `output_sink(text)` is `async` and receives the model's assistant text per step; broader output modalities are Commands. Streaming sink output is deferred.
- **The World is a snapshot, not a log.** It holds current + previous per entry; conversation history lives in the Agent (as re-renderable snapshots). Heavy multimodal data (an image) renders inline on the turn it arrives and light thereafter.
- **Command names are unique, and `noop`/`cancel_command` are reserved.** Registering a duplicate or a reserved name raises `ValueError` (the output Command's name is checked at construction too).
- **World keys must be safe to embed.** A key is a non-empty string with no whitespace and none of `" < > &` (it goes verbatim into the `<entry key="…">` envelope); `register()` raises `ValueError` otherwise. The `agent:` prefix is framework-owned by convention.
- **Don't stop Wica from inside its own loop.** When Wica owns the loop, calling `wica.stop()`/`wica.close()` from a sink, Command, or Event subscriber raises `RuntimeError` (a thread can't join itself); hand the call to another thread.
- **Distributed via git, not PyPI**, and ships with no provider bundled (install an extra).

## Where to read deeper

Route to the spec that governs what you're touching (each carries the full rationale and open questions):

| Need to… | Read |
|---|---|
| Understand the entry point (`Wica`), the shared loop + lifecycle, the surfaced Events | [specs/wica.md](specs/wica.md) |
| Model multimodal content (`TextPart`/`ImagePart`, add a part type) | [specs/content.md](specs/content.md) |
| Understand entry versioning, `ttl`, `include_in_prompt`, lifecycle, rendering | [specs/world.md](specs/world.md) |
| Design an Input (the entry-as-Input pattern, triggering) | [specs/inputs.md](specs/inputs.md) |
| Write Commands (lifecycle, cancellation, `cancel_command`) | [specs/commands.md](specs/commands.md) |
| Understand the reasoning loop, history rendering, coalescing, instrumentation Events | [specs/agent.md](specs/agent.md) |
| Author config / add a provider | [specs/config.md](specs/config.md) |
| Write deterministic e2e tests with the scripted `fake` provider | [specs/fake-provider.md](specs/fake-provider.md) |
| Show the World, the prompts and the transcript in a Gradio app (`wica[gradio]`) | Recipe 4 above, then [specs/gradio-contrib.md](specs/gradio-contrib.md) for the design |
| See it all wired in a runnable app | [specs/conversation-demo.md](specs/conversation-demo.md) → [`examples/conversation_demo/app.py`](examples/conversation_demo/app.py) |

Spec index with statuses: [specs/_index.md](specs/_index.md).
