# Integrating WICA

**This file is for an agent (or developer) building an application that depends on WICA.** It is the consumer's entry point: the mental model, the public API you import, the canonical wiring recipes, the limits to design around, and where to read deeper. It deliberately excludes how to *develop WICA itself* — that's [AGENTS.md](AGENTS.md) (specs, plans, status discipline). Human readers evaluating the framework may prefer the prose [README.md](README.md).

> WICA ships **no model provider**. Install the extra for the one you use:
> ```bash
> uv add "wica[anthropic] @ git+https://github.com/<owner>/wica-framework"
> # or wica[openai], or wica[huggingface-hub]
> ```

## Mental model (read this first)

State lives in **one World** — a registry of named, typed entries that renders to the LLM prompt. You don't thread a conversation; you maintain state, and every reasoning step renders the relevant World into the prompt (a *view of state*, not a transcript).

- **W — World**: `get_world()` singleton. You `register()` a key with a type + a `serialize_fn` (value → [`Content`](specs/content.md)), then `update()` it as data changes. Sync and thread-safe.
- **I — Inputs**: not a class — a *role* a World entry plays when an external producer feeds it. A registered entry marked `triggers_llm_call=True` wakes the Agent when updated. The producer can run on any thread.
- **C — Commands**: the agent's unit of action. Register a plain function (or an off-the-shelf LangChain tool) with `agent.register_command`; the Agent binds it as a native model tool. Each runs as a cancellable `asyncio` task, tracked as a World entry.
- **A — Agent**: the reasoning loop. Owns one event loop (daemon thread by default), builds the prompt from World state, runs inference, dispatches Commands, streams text to your output sink.

## Public API — what you import

Everything below is re-exported from the top-level `wica` package ([`src/wica/__init__.py`](src/wica/__init__.py)):

| Import | Kind | Use |
|---|---|---|
| `get_world` | fn → `World` | Get the singleton World |
| `World` | class | The state registry (`register`/`update`/`get`/`unregister`) |
| `TextPart`, `ImagePart` | dataclass | Multimodal content parts; `Content` is a `list` of them |
| `Content`, `ContentPart` | type alias | What a `serialize_fn` returns |
| `Agent` | class | The reasoning loop (`register_command`/`start`/`stop`); also at `wica.agent.Agent` |
| `WicaConfig`, `AgentConfig` | dataclass | Config, loaded strictly from JSON |
| `apply_logging` | fn | Set the `wica` logger level from `config.logging` |
| `ConfigError`, `MissingEnvError` | exception | Raised on invalid config / unset env var |
| `WorldEntry`, `WorldEntryConfig`, `WorldEntryVersion` | dataclass | Entry introspection (rarely needed directly) |

Signatures you'll actually call:

```python
world.register(key: str, type: type[T], *,
               serialize_fn: Callable[[T | None, T | None], Content],  # (value, previous) -> Content
               archival_serialize_fn=None,      # lighter render for older turns; defaults to serialize_fn
               include_in_prompt=True,           # False → internal/shared state the model never sees
               triggers_llm_call=False,          # True → an update wakes the Agent (this is an Input)
               trigger_condition_fn=None,        # (old, new) -> bool: gate which updates trigger
               ttl=None,                         # timedelta → auto-expire the entry to None
               bypass_coalescing=False)          # True → fire immediately, skip the batch window
world.update(key, value)     # from any thread
world.get(key)               # current value
world.unregister(key)

Agent(model, *, system_prompt, world=None, loop=None, coalesce_window=0.2,
      output_sink=None, on_prompt=None, on_trigger=None, on_command=None)
Agent.from_config(config: AgentConfig, **kwargs) -> Agent   # kwargs = the code-only wiring above
agent.register_command(fn, *, name=None, description=None)  # fn: plain callable or LangChain BaseTool
agent.start(); agent.stop()
```

## Recipes

### 1. Feed a perception (an Input)

```python
from wica import get_world, TextPart

world = get_world()
world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, prev: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,          # updating this wakes the Agent
)
# From anywhere — a UI callback, a sensor loop, a hardware interrupt:
world.update("speech_input", "hello robot")
```

### 2. Give the agent an action (a Command)

```python
import asyncio

async def dance() -> str:
    """Perform a fun little dance. Takes about 10 seconds."""   # docstring → the tool description
    await asyncio.sleep(10)
    return "Finished the dance."

agent.register_command(dance)
```

A Command needs no WICA-specific hooks — off-the-shelf LangChain tools register unmodified. Its execution is tracked as a World entry `agent:command:<call_id>` that renders `running` while in flight and terminal (`result`/`error`) once done, so a later step can *see* an action still running. The Agent auto-registers a native `cancel_command(call_id)` so the model can abort its own in-flight Commands.

> **Prefer `async` for anything cancellable.** Cancellation (`cancel_command`, or `stop()`) cancels the `asyncio` task: an `async` Command unwinds cleanly at its next `await`. A plain **sync** function works too — it's offloaded to a thread — but Python can't kill a running thread, so on cancel the entry flips to `cancelled` immediately while the thread runs the function to completion in the background, result discarded. Make a long-running sync Command **cooperative** (poll a `threading.Event`) if it needs to actually stop. See [specs/commands.md](specs/commands.md) ("Cancellation reaches the task, not always the work").

### 3. Wire and run the Agent

```python
import asyncio
from wica import get_world, TextPart, WicaConfig, apply_logging
from wica.agent import Agent

world = get_world()
# ... register Inputs as in recipe 1 ...

async def speak(text: str) -> None:     # the output sink: async, takes the model's text
    print("robot says:", text)

config = WicaConfig.from_json("agent.config.json")   # provider/model/key/persona from a file
apply_logging(config.logging)
agent = Agent.from_config(config.agent, output_sink=speak)   # code-only bits go as kwargs
agent.register_command(dance)
agent.start()

world.update("speech_input", "hello!")   # Agent wakes, reasons, calls speak()/Commands
# ... keep the process alive while the agent runs on its own loop ...
agent.stop()
```

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
| `agent.provider` | yes | `anthropic` \| `openai` \| `huggingface-hub` |
| `agent.model` | yes | Model id (or Hub `repo_id` for `huggingface-hub`) |
| `agent.system_prompt` / `system_prompt_file` | exactly one | Inline, or a path resolved relative to the config file |
| `agent.api_key` / `api_key_env` | at most one | Literal key, or an env var to read at load time. Neither → provider's standard env var. Prefer `api_key_env` so the config carries no secret and is safe to commit |
| `agent.model_kwargs` | no | Forwarded to the provider (e.g. `temperature`) |
| `agent.hf_provider` | no | Only for `huggingface-hub`: the Hub backend (`auto`/`fireworks-ai`/…). Default `auto` |
| `logging` | no | `wica` logger level. Default `INFO` |

Loading is a two-call composition — `WicaConfig.from_json` → `Agent.from_config` — keeping code-only wiring (World instance, event loop, output sink, instrumentation hooks) in `**kwargs`, where JSON can't reach. Selecting a provider whose extra isn't installed fails at runtime with a clear `ImportError`.

## v1 limits to design around

- **One reasoning call at a time.** A trigger arriving while a step is in flight is **dropped, not queued**. Concurrency/interruption (`cancel_reaction`, barge-in) is designed but deferred — see [specs/agent.md](specs/agent.md).
- **Trigger coalescing.** A burst of triggers within `coalesce_window` (default 200 ms) is batched into a single step. Set an entry's `bypass_coalescing=True` for something that must act at once (a stop button); `coalesce_window=0` disables batching.
- **Output is text via one sink.** `output_sink(text)` is `async` and receives the model's assistant text per step. Streaming output is deferred.
- **The World is a snapshot, not a log.** It holds current + previous per entry; conversation history lives in the Agent (as re-renderable snapshots). Heavy multimodal data (an image) renders inline on the turn it arrives and light thereafter.
- **Distributed via git, not PyPI**, and ships with no provider bundled (install an extra).

## Where to read deeper

Route to the spec that governs what you're touching (each carries the full rationale and open questions):

| Need to… | Read |
|---|---|
| Model multimodal content (`TextPart`/`ImagePart`, add a part type) | [specs/content.md](specs/content.md) |
| Understand entry versioning, `ttl`, `include_in_prompt`, rendering | [specs/world.md](specs/world.md) |
| Design an Input (the entry-as-Input pattern, triggering) | [specs/inputs.md](specs/inputs.md) |
| Write Commands (lifecycle, cancellation, `cancel_command`) | [specs/commands.md](specs/commands.md) |
| Understand the reasoning loop, history rendering, coalescing, hooks | [specs/agent.md](specs/agent.md) |
| Author config / add a provider | [specs/config.md](specs/config.md) |
| See it all wired in a runnable app | [specs/conversation-demo.md](specs/conversation-demo.md) → [`examples/conversation_demo.py`](examples/conversation_demo.py) |

Spec index with statuses: [specs/_index.md](specs/_index.md).
