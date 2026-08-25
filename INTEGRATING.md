# Integrating WICA

**This file is for an agent (or developer) building an application that depends on WICA.** It is the consumer's entry point: the mental model, the public API you import, the canonical wiring recipes, the limits to design around, and where to read deeper. It deliberately excludes how to *develop WICA itself* — that's [AGENTS.md](AGENTS.md) (specs, plans, status discipline). Human readers evaluating the framework may prefer the prose [README.md](README.md).

> WICA ships **no model provider**. Install the extra for the one you use:
> ```bash
> uv add "wica[anthropic] @ git+https://github.com/<owner>/wica-framework"
> # or wica[openai], or wica[huggingface-hub]
> ```

## Mental model (read this first)

You talk to **one object — `Wica`** — the single entry point. `Wica.init(config)` stands up the whole system from a config: it owns one event loop and a **World** + **Agent** pair that both run on it, and surfaces everything you need behind one surface. State lives in the World — a registry of named, typed entries that renders to the LLM prompt. You don't thread a conversation; you maintain state, and every reasoning step renders the relevant World into the prompt (a *view of state*, not a transcript).

“Multimodal” spans both sides differently: Inputs serialize perception into provider-neutral `Content`, while Commands produce speech, movement, display changes, API effects, and other physical or digital outputs. The conversational sink is one complete text response per reasoning step; it is not the full output surface.

- **Wica**: the entry point. `Wica.init(config, *, output_sink=…)` builds and wires everything; `wica.start()` / `wica.stop()` run the shared lifecycle. `wica.world` is the World; `wica.register_command(...)` adds Commands; four `Event`s surface what the loop is doing.
- **W — World** (`wica.world`): you `register()` a key with a type + a `serialize_fn` (value → [`Content`](specs/content.md)), then `update()` it as data changes. Values must be deep-copyable: the World copies on ingress and on outward getters/callbacks so only `update()` can change versioned state. `update()` is callable from any thread, but only **while the system is running** (between `start()` and `stop()`); `register`/`get` work any time.
- **I — Inputs**: not a class — a *role* a World entry plays when an external producer feeds it. A registered entry marked `triggers_llm_call=True` wakes the Agent when updated. The producer can run on any thread.
- **C — Commands**: the agent's unit of action. Register a plain function (or an off-the-shelf LangChain tool) with `wica.register_command`; the Agent binds it as a native model tool. Each runs as a cancellable `asyncio` task, tracked as a World entry.
- **A — Agent** (`wica.agent`): the reasoning loop. Built from config, runs on the shared loop, builds the prompt from World state, runs inference, dispatches Commands, and delivers one complete text response per step to your output sink. Rarely touched directly — the everyday paths are surfaced on `Wica`.

## Public API — what you import

Everything below is re-exported from the top-level `wica` package ([`src/wica/__init__.py`](src/wica/__init__.py)):

| Import | Kind | Use |
|---|---|---|
| `Wica` | class | The single entry point (`init`/`start`/`stop`/`register_command`, `wica.world`, four Events) |
| `World` | class | The state registry (`register`/`update`/`get`/`unregister`, `start`/`stop`/`is_running`); reached as `wica.world` |
| `TextPart`, `ImagePart` | dataclass | Multimodal content parts; `Content` is a `list` of them |
| `Content`, `ContentPart` | type alias | What a `serialize_fn` returns |
| `Agent` | class | The reasoning loop; usually owned by `Wica`, but constructible directly (see the direct seam below) |
| `CommandIssued` | dataclass | Payload of `wica.on_agent_command` (`name`, `args`) |
| `Event` | class | The pub/sub primitive the four instrumentation signals use (`subscribe`/`unsubscribe`) |
| `WicaConfig`, `AgentConfig` | dataclass | Config, loaded strictly from JSON |
| `apply_logging` | fn | Set the `wica` logger level from `config.logging` (Wica.init already calls it) |
| `ConfigError`, `MissingEnvError` | exception | Raised on invalid config / unset env var (at `Wica.init`, i.e. build) |
| `WorldEntry`, `WorldEntryConfig`, `WorldEntryVersion` | dataclass | Entry introspection (rarely needed directly) |

Instrumentation is four `Event`s you `.subscribe(...)` on — multi-consumer, so panels, a logger, and a metrics sink can all watch the same signal:

| Event | Payload | Fires |
|---|---|---|
| `wica.on_world_trigger` | `WorldEntry` | **raw** — every qualifying World update (pre-coalescing), including triggers the busy loop later drops |
| `wica.on_agent_trigger` | `WorldEntry` | **filtered** — once per trigger a run-to-completion step actually observed |
| `wica.on_agent_prompt` | `list[BaseMessage]` | the exact rendered messages before each model call |
| `wica.on_agent_command` | `CommandIssued` | each Command the model issues, at dispatch |

Signatures you'll actually call:

```python
Wica.init(config: WicaConfig, *, output_sink=None, coalesce_window=0.2, loop=None) -> Wica
wica.world            # the World (below); wica.agent — the Agent (rarely needed)
wica.register_command(fn, *, name=None, description=None)   # fn: plain callable or LangChain BaseTool
wica.start(); wica.stop()          # once per instance — to "reset", discard and init a new one
wica.on_world_trigger / on_agent_trigger / on_agent_prompt / on_agent_command   # .subscribe(handler)

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
Agent(config: AgentConfig, *, world: World, loop, coalesce_window=0.2, output_sink=None, model=None)
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

A Command needs no WICA-specific hooks — off-the-shelf LangChain tools register unmodified. Its execution is tracked as a World entry `agent:command:<call_id>` that renders `running` while in flight and terminal (`result`/`error`) once done, so a later step can *see* an action still running. The Agent auto-registers a native `cancel_command(call_id)` so the model can abort its own in-flight Commands.

> **Prefer `async` for anything cancellable.** Cancellation (`cancel_command`, or `stop()`) cancels the `asyncio` task: an `async` Command unwinds cleanly at its next `await`. A plain **sync** function works too — it's offloaded to a thread — but Python can't kill a running thread, so on cancel the entry flips to `cancelled` immediately while the thread runs the function to completion in the background, result discarded. Make a long-running sync Command **cooperative** (poll a `threading.Event`) if it needs to actually stop. See [specs/commands.md](specs/commands.md) ("Cancellation reaches the task, not always the work").

### 3. Wire and run the system

```python
from wica import TextPart, WicaConfig, Wica

async def speak(text: str) -> None:     # the output sink: async, takes the model's text
    print("robot says:", text)

config = WicaConfig.from_json("agent.config.json")   # provider/model/key/persona from a file
wica = Wica.init(config, output_sink=speak)          # owns the loop + World + Agent; applies logging

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
wica.stop()
```

`wica.world` / `wica.agent` are **borrowed references** — valid for the life of the `Wica`. After `wica.stop()`, reactive mutation through `world.update()` raises `RuntimeError("World is not running")`; schema operations and reads remain available for inspection, but the stopped objects should be discarded rather than reused. To "reset", initialize a new `Wica`.

## Config schema

An Agent is stood up from one JSON file, so switching provider or editing the persona is a file edit, not a code change. Loading is strict — missing required keys, unknown keys (typos), and wrong types all fail loudly at load time.

```json
{
  "logging": "INFO",
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
| `agent.system_prompt` / `system_prompt_file` | exactly one | Inline, or a path resolved relative to the config file |
| `agent.api_key` / `api_key_env` | at most one | Literal key, or an env var read at **Agent build** (`Wica.init`). Neither → provider's standard env var. Prefer `api_key_env` so the config carries no secret and is safe to commit |
| `agent.model_kwargs` | no | Forwarded to the provider (e.g. `temperature`) |
| `agent.hf_provider` | no | Only for `huggingface-hub`: the Hub backend (`auto`/`fireworks-ai`/…). Default `auto` |
| `logging` | no | `wica` logger level. Default `INFO` |

Loading is a two-call composition — `WicaConfig.from_json` → `Wica.init` — keeping code-only wiring (output sink, `coalesce_window`, an optional pre-existing event loop) in `Wica.init`'s keyword arguments, where JSON can't reach. `Wica.init` applies logging itself, and it's where a referenced-but-unset `api_key_env` raises `MissingEnvError` (so a caller that degrades — e.g. to explore-only — wraps `Wica.init`, not `from_json`). Selecting a provider whose extra isn't installed fails at runtime with a clear `ImportError`.

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
wica = Wica.init(config, output_sink=my_sink)
wica.world.register("prompt", str, serialize_fn=..., triggers_llm_call=True)
wica.register_command(walk_to)
wica.start()
# ... update a triggering World entry, then assert the command entry + sink output ...
wica.stop()
```

Each scripted `tool_calls[].name` is validated against your registered Commands (a typo raises, rather than silently emitting an unknown call). For a script that can't live in JSON (a reactive/programmatic double), construct the Agent directly with the raw-model override: `from wica.fake_model import FakeChatModel` → `Agent(AgentConfig(provider="fake", model="…", system_prompt="…"), world=world, loop=loop, model=FakeChatModel(...))`. Full design in [specs/fake-provider.md](specs/fake-provider.md).

## v1 limits to design around

- **One reasoning call at a time.** A trigger arriving while a step is in flight is **dropped, not queued**. Concurrency/interruption (`cancel_reaction`, barge-in) is designed but deferred — see [specs/agent.md](specs/agent.md).
- **Trigger coalescing.** A burst of triggers within `coalesce_window` (default 200 ms) is batched into a single step. Set an entry's `bypass_coalescing=True` for something that must act at once (a stop button); `coalesce_window=0` disables batching.
- **Lifecycle is once per instance.** `wica.start()`/`wica.stop()` run once; the owned loop thread can't be restarted. To reset, discard the `Wica` and `init` a new one.
- **The conversational sink is complete text.** `output_sink(text)` is `async` and receives the model's assistant text per step; broader output modalities are Commands. Streaming sink output is deferred.
- **The World is a snapshot, not a log.** It holds current + previous per entry; conversation history lives in the Agent (as re-renderable snapshots). Heavy multimodal data (an image) renders inline on the turn it arrives and light thereafter.
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
| See it all wired in a runnable app | [specs/conversation-demo.md](specs/conversation-demo.md) → [`examples/conversation_demo.py`](examples/conversation_demo.py) |

Spec index with statuses: [specs/_index.md](specs/_index.md).
