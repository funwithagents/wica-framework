# WICA

**WICA is an agentic framework for agents that take multimodal inputs and produce multimodal outputs.**
Its state is centered on a single **World** — a store of the current interaction context that serializes to an LLM-facing prompt. Perceptions flow *in* as **Inputs**, the agent acts *out* through **Commands**, and an **Agent** reasoning loop sits in the middle, observing the World and deciding what to do.

Here, multimodal output includes Command-mediated action in the physical or digital world: speech, movement, a display change, an API action, and so on. WICA represents those capabilities as tools and serializes their state/results back into the World for reasoning. The v1 conversational output sink is intentionally narrower—it receives one complete text response per step—while Commands carry the broader output modalities.

The name is the model:

| Letter | Pillar | What it is |
|---|---|---|
| **W** | [World](specs/world.md) | The store of current state/context — a registry of typed entries that renders to a prompt |
| **I** | [Inputs](specs/inputs.md) | External multimodal data entering the World (speech, a camera frame, a sensor event) |
| **C** | [Commands](specs/commands.md) | The agent's unit of action on the World — async, cancellable |
| **A** | [Agent](specs/agent.md) | The reasoning loop that observes the World and issues Commands |

## Why WICA

- **State-first, not chat-first.** Instead of threading a conversation, you maintain a **World** of named entries — what the agent heard, who's nearby, how it feels, what it's doing. Every reasoning step renders the relevant World state into the prompt. The prompt is a *view of state*, not a transcript you append to.
- **Multimodal in, multimodal out.** Inputs serialize through the neutral [`Content`](specs/content.md) model (`TextPart`, `ImagePart`, more to come), while output modalities are Commands acting on the physical or digital world and reporting their state/results back through the World. A camera Input can be inline while fresh and a light text description afterwards; Commands can speak, move, display, or call external systems without making those effects `Content` return values.
- **Actions are async and cancellable.** Commands run as `asyncio` tasks on the Agent's event loop, so a long-running action (walk to the kitchen, do a 10-second dance) can be **cancelled** — by the framework, or by the model itself issuing `cancel_command(call_id)`.
- **Reactive by construction.** Marking an Input `triggers_llm_call=True` is all it takes to wake the Agent when a new perception arrives. Bursts of perceptions are coalesced into a single step.
- **Provider-agnostic.** Switching between Anthropic, OpenAI, or Hugging Face Hub is a config edit, not a code change. LangChain is used only as a low-level primitive (model + tool schemas), quarantined to the Agent's I/O boundary — everything upstream stays SDK-free.

## How it works

### The entry point — `Wica`

You talk to one object. `Wica.init(config)` stands up the whole system from a config — it owns a single event loop and a **World** + **Agent** pair that both run on it — and surfaces everything you need: `wica.world` for state, `wica.register_command(...)` for actions, restartable `wica.start()`/`wica.stop()` cycles, terminal `wica.close()`, and four `Event`s to observe the loop.

### The World — state that becomes a prompt

The World (`wica.world`) is a registry. You `register()` a key with a declared type and a `serialize_fn` that turns its value into `Content`, then `update()` it as data changes:

```python
from wica import TextPart

# Register an entry: how it's typed, serialized, and whether it wakes the agent.
wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, prev: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,   # a new value wakes the Agent
)

# Later, from anywhere (a UI callback, a sensor thread, a hardware interrupt), while running:
wica.world.update("speech_input", "hello robot")
```

Each entry is versioned (a small incrementing `id`), timestamped, and rendered inside an XML-style `<entry key="..." id="...">…</entry>` block the model can reference precisely. Values are deep-copied on update and when exposed through getters/callbacks, so versioned state cannot be changed silently without another `update()`. Entries can be marked `include_in_prompt=False` to act as pure internal/shared state that never reaches the model, and given a `ttl` to auto-expire. The World holds a *snapshot* (current + previous), not an event-sourced log — history lives in the Agent.

### Inputs — perception coming in

An **Input** isn't a class; it's a *role a World entry plays* when an external producer feeds it.
A user utterance, a "closest person detected" event, a camera frame — each is just a `register()`ed entry that some producer `update()`s. `update()` is callable from **any thread** (a Gradio callback, a sensor poll loop, a hardware interrupt) — it hops to the shared loop internally — so an Input producer never needs to know about the loop the World and Agent share.

```python
wica.world.register("closest_user", str, serialize_fn=_render_user, triggers_llm_call=True)
wica.world.update("closest_user", "alice")   # someone stepped up
wica.world.update("closest_user", None)      # ...and walked away
```

### Commands — the agent acting out

A **Command** is the agent's unit of action on the World. You register plain functions (sync or async) as Commands; the Agent binds them as the model's native tool calls:

```python
async def dance() -> str:
    """Perform a fun little dance. Takes about 10 seconds."""
    await asyncio.sleep(10)
    return "Finished the dance."

wica.register_command(dance)
```

Off-the-shelf LangChain tools work unmodified — a Command needs no WICA-specific hooks. Every Command runs as a cancellable `asyncio` task and its execution is tracked as a World entry `agent:command:<call_id>` that renders `running` while in flight and terminal (`result`/`error`) once done — so a later reasoning step can *see* an action still running and decide to cancel it. The Agent auto-registers a native `cancel_command(call_id)` so the model can abort its own in-flight Commands.

### The Agent — the reasoning loop

The Agent is triggered by the World, builds a prompt from World state, runs inference against a configured provider, dispatches Commands, and delivers each complete text response to a pluggable sink. It runs on the **single event loop** `Wica` owns (a daemon thread by default) — shared with the World — so Command cancellation is race-free while `update()` stays callable from any thread. It keeps the conversation **history** as re-renderable World snapshots — so the newest observation renders rich (an image inline) and older ones render light, keeping the deep prompt prefix byte-stable and cacheable.

Perceptions that arrive in a burst are **coalesced** into a single step (a ~200 ms leading-edge window, configurable via `coalesce_window`); an entry can set `bypass_coalescing=True` to act immediately (a stop button, a barge-in utterance).

> **v1 scope.** The Agent currently runs **one reasoning call at a time**: a trigger arriving while a step is in flight is dropped, not queued. The full concurrency/interruption model (`cancel_reaction`, streaming output, barge-in) is designed and deferred — see [specs/agent.md](specs/agent.md).

## Quick start

WICA is distributed via git (not PyPI), and ships with **no** model provider — you install the one you use as an extra:

```bash
# In a consuming project:
uv add "wica[anthropic] @ git+https://github.com/funwithagents/wica-framework"
# or wica[openai], or wica[huggingface-hub]
```

A minimal agent starts from a `WicaConfig`. For configuration owned by Python code, construct it directly:

```python
from wica import AgentConfig, TextPart, Wica, WicaConfig

async def speak(text: str) -> None:
    print("robot says:", text)

config = WicaConfig(
    agent=AgentConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        api_key_env="WICA_ANTHROPIC_API_KEY",
        system_prompt="You are a friendly social robot.",
        model_kwargs={"temperature": 0.7},
    )
)
wica = Wica.init(config, output_sink=speak)   # owns the loop + World + Agent

wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda v, prev: [TextPart(f'The person said: "{v}"')],
    triggers_llm_call=True,
)
wica.start()

# Feed a perception; the Agent wakes, reasons, and replies.
wica.world.update("speech_input", "hello!")

# ... keep the process alive while the system runs on its own loop ...
wica.close()  # terminal teardown; use stop() instead when you intend to start again
```

## Configuration

`Wica.init()` accepts a `WicaConfig`. The configuration object is the stable API; construct it directly, parse it from a dictionary owned by a larger application, or load it from a dedicated JSON file.

### Construct directly in Python

Direct construction is the simplest choice when configuration belongs in application code:

```python
from wica import AgentConfig, WicaConfig

config = WicaConfig(
    agent=AgentConfig(
        provider="openai",
        model="gpt-4o",
        system_prompt="You are a concise assistant.",
        # No key field: let the provider use its standard environment variable.
    )
)
```

These are plain dataclasses. Direct construction benefits from static type checking, but does not run the strict runtime validation performed by the loaders below; the caller is responsible for providing valid field combinations.

### Parse part of an application configuration

`from_dict()` is useful when WICA settings are nested inside a larger JSON document, settings object, or configuration system. Pass the WICA-shaped subsection, not the unrelated application keys:

```python
import json
from pathlib import Path

from wica import WicaConfig

settings_path = Path("app.config.json")
settings = json.loads(settings_path.read_text(encoding="utf-8"))
config = WicaConfig.from_dict(
    settings["wica"],
    base_dir=settings_path.parent,
)
```

`base_dir` is optional. When the dictionary contains a relative `system_prompt_file`, it locates that path relative to `base_dir`; without one, a relative path is kept as written and is read relative to the process working directory when the Agent is built.

### Load a dedicated JSON file

`from_json()` reads the same nested shape from a standalone file:

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

```python
from wica import WicaConfig

config = WicaConfig.from_json("agent.config.json")
```

For this path, a relative `system_prompt_file` is automatically located relative to the JSON file's directory, so a config-plus-prompts folder remains relocatable.

All three forms produce the same type and use the same framework entry point:

```python
wica = Wica.init(config, output_sink=speak, coalesce_window=0.2)
```

- **API key** — give a literal `api_key`, or an `api_key_env` naming the env var to read at **Agent build** (`Wica.init`), at most one. With neither, the provider's standard env var is used. An env-referenced config carries no secret and is safe to commit; a literal-key config should be git-ignored.
- **System prompt** — give exactly one of inline `system_prompt` or `system_prompt_file`. Relative file-path handling depends on the construction method as described above.
- **Strict parsing** — `from_dict()` and `from_json()` reject missing required keys, unknown keys (including typos), invalid field combinations, and wrong types with `ConfigError`.

Creating a config is side-effect-light: the loaders validate references but do not read an API-key environment variable or the prompt file. Those are resolved when `Wica.init()` builds the Agent; that is where a referenced-but-unset `api_key_env` raises `MissingEnvError`, or an unreadable prompt file raises `ConfigError`. Runtime-only wiring (`output_sink`, `output_command`, `coalesce_window`, and an optional event loop) remains in `Wica.init()` because it consists of callables and runtime objects rather than configuration data. WICA does not configure logging; the embedding application owns its handlers and levels.

### Providers

Core `wica` bundles no provider; install the extra for the one you use. Selecting a provider whose extra isn't installed fails at runtime with a clear `ImportError`.

| `provider` | Install extra | Backed by |
|---|---|---|
| `anthropic` | `wica[anthropic]` | `langchain-anthropic` |
| `openai` | `wica[openai]` | `langchain-openai` |
| `huggingface-hub` | `wica[huggingface-hub]` | `langchain-huggingface` (serverless Inference Providers) |
| `fake` | none | built-in deterministic test double; no network or key |

## The conversation demo

The repo ships a runnable example — a browser UI to talk to a simulated social robot and *watch the framework work*: a live view of the World state, the exact prompt sent to the model each step, and sensor-input buttons. It makes the four pillars tangible without writing code.

```bash
export WICA_ANTHROPIC_API_KEY=sk-...
uv run --group demo python examples/conversation_demo.py
```

Without a key set it still opens and is explorable — it just can't run the robot's reasoning.
See [specs/conversation-demo.md](specs/conversation-demo.md).

## Project layout

| Path | What's there |
|---|---|
| `src/wica/` | The library — facade, World, Content, config, Agent/Commands, Events, and the scripted fake model; see the complete module map in [`AGENTS.md`](AGENTS.md) |
| `specs/` | Pre-implementation design docs, one per concept — start at [specs/_index.md](specs/_index.md) |
| `plans/` | Implementation plans turning specs into buildable steps — [plans/_index.md](plans/_index.md) |
| `tests/` | Fast, deterministic, no-network tests (the default `pytest` run) |
| `tests-e2e/` | Opt-in full-loop tests: deterministic fake flows plus live provider cases |
| `examples/` | Runnable examples (the Gradio conversation demo) |

The design is documented spec-first: each spec in [`specs/`](specs/_index.md) carries a status (`Draft`/`Stable`/…) and declares the code and tests it governs. Read the specs for the full rationale behind every decision above.

**Integrating WICA into your own app?** [INTEGRATING.md](INTEGRATING.md) is the consumer-facing entry point — the public API surface, the wiring recipes, the v1 limits, and a spec-routing table — written to be read by a coding agent.

## Development

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --dev          # install deps + dev tooling
uv run ruff check .    # lint
uv run pyright         # type check
uv run pytest          # fast test tier (no network)
```

The deterministic full-loop flows are also opt-in because `tests-e2e/` is outside the default test path:

```bash
uv run pytest tests-e2e -k fake
```

The live set calls real providers and needs a provider API key:

```bash
uv run pytest tests-e2e            # each provider whose key is set runs; the rest skip
uv run pytest tests-e2e -k openai  # one provider
```
