# API reference

The public surface of WICA: what you import, the entry point and what it owns, the World API, `Command`, the Events, the errors, and the contracts that hold across them. The normative design, with its rationale, is in the specs each section links.

On this page: [Imports](#imports) · [Construction](#construction) · [The `Wica` surface](#the-wica-surface) · [The World](#the-world) · [`Command`](#command) · [Events](#events) · [Errors](#errors) · [Contracts](#contracts).

## Imports

From `wica`:

| Name | What it is |
|---|---|
| `Wica` | The entry point: owns the loop, a `World` and an `Agent`; `init`, `start`/`stop`/`close`, `register_command`, `set_output_sink`/`set_output_command`, the Events |
| `World`, `WorldEntry`, `WorldEntryConfig`, `WorldEntryVersion` | The state registry (reached as `wica.world`), an entry snapshot (`key`, `type`, `current`, `previous`), its registration options, one version (`id`, `value`, `timestamp`) |
| `TextPart`, `ImagePart`, `Content`, `ContentPart` | Provider-neutral multimodal content: `Content` is a `list` of parts, what a `serialize_fn` returns |
| `Command` | A Command definition over a callable or an off-the-shelf LangChain tool, for a name or description override, `triggers_on_completion`, or tool wrapping |
| `CommandIssued`, `CommandExecution` | The payload of `on_agent_command` (`name`, `args`, `call_id`); the value of an `agent:command:<call_id>` entry (`name`, `args`, `state`, `result`, `error`) |
| `Event` | The pub/sub primitive the Events are: `subscribe(handler)`, `unsubscribe(handler)` |
| `WicaConfig`, `AgentConfig` | The configuration ([configuration.md](configuration.md)) |
| `ReactionTrace`, `TriggerTrace`, `CommandTrace`, `TokenUsage` | The instrumentation records ([../guides/latency-and-traces.md](../guides/latency-and-traces.md)) |
| `ConfigError`, `MissingEnvError` | The errors a caller catches ([Errors](#errors)) |
| `Agent` | The reasoning loop, owned by `Wica` and reached as `wica.agent`; constructible directly for a programmatic model double ([../guides/testing.md](../guides/testing.md)) |

Three subpackages are imported explicitly, never re-exported from `wica`: **`wica.contrib.gradio`** (the browser panels, behind the `gradio` extra: [../guides/gradio-panels.md](../guides/gradio-panels.md)), **`wica.testing`** (the Probe, the `TestRun` harness, the report, the scripted tester, scenario files and checks: [../guides/testing.md](../guides/testing.md)) and **`wica.benchmark`** (the matrix, `Target`, the runner, the summary, the CLI: [../guides/benchmarking.md](../guides/benchmarking.md)). `wica.fake_model.FakeChatModel` is the scripted model behind `provider: "fake"`, for the direct seam. `wica.instrumentation` holds the measures over the records (`reaction_latency`, `reactions_per_input`).

## Construction

```
Wica.init(config: WicaConfig, *, coalesce_window: float = 0.2, loop: asyncio.AbstractEventLoop | None = None) -> Wica
```

The one constructor. It builds a `World` and an `Agent` from the config and wires them. With no `loop`, the Wica creates one and runs it in a daemon thread from `start()`; with one, it adopts it and leaves running it to you. `coalesce_window` is the batch window for trigger bursts, in seconds ([../guides/world-and-inputs.md](../guides/world-and-inputs.md#bursts-are-coalesced)); `0` disables it. Resolution happens here: the key variable is read, the prompt file is read, the provider's package is imported, so `MissingEnvError`, `ConfigError` and `ImportError` surface at `init`, not at config creation. `init` takes no application callables: the sink and the output Command are wired after it.

## The `Wica` surface

```
wica.world                      # the World (below); wica.agent — the Agent, rarely needed
wica.register_command(fn_or_command)              # a plain callable, or Command(...) for options
wica.set_output_sink(async_fn | None)             # receives the model's complete free text each step
wica.set_output_command(fn_or_command | None)     # the user-facing output channel (the free text turns private)
wica.start(); wica.stop(); wica.start()           # restartable: stop() is a pause
wica.close()                                      # terminal: releases an owned loop; no restart
wica.is_running                                   # True between start() and stop()
wica.on_<event>.subscribe(handler)                # the eight Events (below)
```

- **`register_command(fn)`**: a sync or async function (its name and docstring become the tool's name and description), or a `Command`. Names are unique and `noop`/`cancel_command` are reserved: a duplicate or reserved name raises `ValueError`. Register before `start()`. [../guides/commands.md](../guides/commands.md).
- **`set_output_sink(sink)`**: an `async def sink(text: str) -> None`, called once per reasoning step with the model's complete free text, when there is any. `None` clears it. Without an output Command, this text is the reply; with one, it is the model's private reasoning.
- **`set_output_command(fn)`**: the Command the model must call to address the user (speech, a chat bubble). It is registered like any Command and also recomposes the system prompt, so set it before `start()`; a bare callable gets `triggers_on_completion=True`. `None` clears it.
- **Lifecycle**: `start()` starts the World then the Agent (and the loop thread when owned); `stop()` cancels in-flight Commands and stops both, keeping registrations, values, subscriptions and history; `close()` stops if needed and releases the loop. `start()` on a running Wica is a no-op; `start()` after `close()` raises `RuntimeError`.
- **`wica.world` and `wica.agent` are borrowed references**, valid for the life of the Wica.

[specs/wica.md](../../specs/wica.md).

## The World

```
world.register(key: str, type: type[T], *,
               serialize_fn: Callable[[T | None, T | None], Content],   # (value, previous) -> Content
               archival_serialize_fn=None,    # a lighter render for older observations; defaults to serialize_fn
               include_in_prompt=True,        # False: internal state the model never sees
               triggers_llm_call=False,       # True: an update wakes the Agent (the entry is an Input)
               trigger_condition_fn=None,     # (old, new) -> bool: which updates trigger
               ttl=None,                      # timedelta: the value resets to None after it
               bypass_coalescing=False)       # True: an update closes the batch window at once
world.update(key, value)        # from any thread, while running; TypeError on a wrong type
world.get(key) -> Any           # a defensive copy of the current value
world.get_entry(key) -> WorldEntry
world.get_config(key) -> WorldEntryConfig
world.is_registered(key) -> bool;  world.keys() -> list[str]
world.unregister(key)
world.add_listener(key, callback); world.remove_listener(key, callback)   # sync or async, on the loop
world.is_running
```

- **Keys** are non-empty strings with no whitespace and none of `" < > &`: they go verbatim into the prompt's `<entry key="…" id="…">` envelope. `register()` raises `ValueError` otherwise. The `agent:` prefix is framework-owned (`agent:command:<call_id>` for Command executions).
- **Values are deep-copied** on `update()` and on every read and callback, so only `update()` changes versioned state. Each update advances the entry's `id` by one and stamps it; the World keeps the current and the previous version, not a log.
- **Listeners** receive a `WorldEntry` snapshot on the loop (a sync callback runs on the loop's thread pool, an async one as a task); an exception in one is logged, never propagated to `update()`'s caller.
- **Rendering**: `world.render_entry(entry)` and `world.render_full_prompt()` produce the `Content` the model reads; a `serialize_fn` that returns `[]` omits the entry from that render.

[../guides/world-and-inputs.md](../guides/world-and-inputs.md); [specs/world.md](../../specs/world.md).

## `Command`

```
Command(fn: Callable | BaseTool, *, name: str | None = None, description: str | None = None, triggers_on_completion: bool = True)
```

Wraps a callable (name from `__name__`, description from the docstring, both overridable) or an existing LangChain `BaseTool` (its own name and description; overriding them raises `ValueError`). `triggers_on_completion=False` makes the Command's terminal completion wake no reasoning step. [../guides/commands.md](../guides/commands.md); [specs/commands.md](../../specs/commands.md).

## Events

Each is an `Event[T]` you `.subscribe(handler)` on, multi-consumer, emitted on the loop thread; a raising handler is isolated from the others. The same objects are reachable on `wica.world` and `wica.agent`.

| Event | Payload | Fires |
|---|---|---|
| `wica.on_world_trigger` | `WorldEntry` | Raw: every qualifying World update, before coalescing, including the triggers a busy Agent later drops |
| `wica.on_agent_trigger` | `WorldEntry` | Filtered: once per trigger a reasoning step observed (several per step when a burst was coalesced) |
| `wica.on_agent_trigger_dropped` | `WorldEntry` | A trigger that arrived while a step was in flight and was dropped |
| `wica.on_agent_prompt` | `list[BaseMessage]` | The exact rendered messages, before each model call |
| `wica.on_agent_command` | `CommandIssued` | Each Command the model issues, at dispatch, `noop` included; the `agent:command:<call_id>` entry exists when it fires |
| `wica.on_agent_command_ended` | `CommandTrace` | A Command's terminal state (`complete`, `failed`, `cancelled`), with its timing |
| `wica.on_agent_text` | `str` | The step's complete free text, right before the output sink receives it |
| `wica.on_agent_reaction_ended` | `ReactionTrace` | Every reaction's record, at its end: the triggers, the phase stamps, the outcome, the Commands, the token usage |

[../guides/latency-and-traces.md](../guides/latency-and-traces.md); [specs/instrumentation.md](../../specs/instrumentation.md).

## Errors

| Error | Raised by | When |
|---|---|---|
| `ConfigError` | the loaders, `AgentConfig`, `Wica.init` | A missing, unknown or mistyped key, an invalid field combination; an unreadable prompt file; a `history_budget` with no known context window |
| `MissingEnvError` | `Wica.init` | `api_key_env` names a variable that is unset (a `ConfigError` subclass) |
| `ImportError` | `Wica.init` | The selected provider's extra is not installed |
| `ValueError` | `register_command`, `set_output_command`, `world.register`, `Command` | A duplicate or reserved Command name; an invalid World key or an already registered one; a name or description override on a `BaseTool` |
| `TypeError` | `world.update` | A value that is not of the entry's declared type (`None` is always allowed) |
| `RuntimeError` | `world.update`, `start`, `stop`, `close` | `update()` while not running; `start()` after `close()`, or on an injected loop that is not running; `stop()`/`close()` called from the Wica's own loop thread |

A Command that raises does not fail the step: its entry turns `failed` with the error text, and the model sees it. A sink that raises is logged and the step's Commands still run.

## Contracts

- **One loop, many threads.** Everything WICA runs (the Agent's steps, the Commands, the listeners, the Event handlers, the sink) runs on the one event loop `Wica` owns or adopted. `update()` may be called from any thread and hops to it. Never call `stop()`/`close()` from code running on that loop.
- **Setup before start.** `register`, `register_command`, `set_output_sink`, `set_output_command` and the subscriptions need no running system; `update()` does. The order among the wiring calls does not matter.
- **Restart keeps, close releases.** `stop()`/`start()` preserves registrations, values, subscriptions, Commands and history. `close()` is terminal; a reset is a new `Wica.init`.
- **One reasoning step at a time.** A trigger arriving while a step is in flight is dropped, not queued (`on_agent_trigger_dropped`). Commands are not so limited: all the calls of one response dispatch at once and run concurrently, and a later step can observe and cancel one still running. [../guides/limits.md](../guides/limits.md).
- **Cancellation reaches the task.** `cancel_command`, `stop()` and `close()` cancel the asyncio task running a Command: an `async` Command unwinds at its next `await`; a sync one is offloaded to a thread, whose work cannot be interrupted, so its entry reads `cancelled` while the function runs to completion in the background.
