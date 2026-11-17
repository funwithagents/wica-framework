# API reference

The public surface of WICA: what you import, the entry point and what it owns, the World API, `Command`, the output Command, the Events, the errors, and the contracts that hold across them. The normative design, with its rationale, is in the specs each section links.

On this page: [Imports](#imports) · [Construction](#construction) · [The `Wica` surface](#the-wica-surface) · [The World](#the-world) · [`Command`](#command) · [The output Command](#the-output-command) · [Response guards](#response-guards) · [Events](#events) · [Errors](#errors) · [Contracts](#contracts) · [Migrating from the output sink](#migrating-from-the-output-sink).

## Imports

From `wica`:

| Name | What it is |
|---|---|
| `Wica` | The entry point: owns the loop, a `World` and an `Agent`; `init`, `start`/`stop`/`close`, `register_command`, `set_output_command`, the Events |
| `World`, `WorldEntry`, `WorldEntryConfig`, `WorldEntryVersion` | The state registry (reached as `wica.world`), an entry snapshot (`key`, `type`, `current`, `previous`), its registration options, one version (`id`, `value`, `timestamp`) |
| `TextPart`, `ImagePart`, `Content`, `ContentPart` | Provider-neutral multimodal content: `Content` is a `list` of parts, what a `serialize_fn` returns |
| `Command` | A Command definition over a callable or an off-the-shelf LangChain tool, for a name or description override, `triggers_on_completion`, or tool wrapping |
| `OutputStream` | What an output Command receives when its first parameter is annotated with it: the text of one output as the model writes it ([The output Command](#the-output-command)) |
| `CommandIssued`, `CommandExecution` | The payload of `on_agent_command` (`name`, `args`, `call_id`); the value of an `agent:command:<call_id>` entry (`name`, `args`, `state`, `result`, `error`) |
| `Event` | The pub/sub primitive the Events are: `subscribe(handler)`, `unsubscribe(handler)` |
| `WicaConfig`, `AgentConfig` | The configuration ([configuration.md](configuration.md)) |
| `ReactionTrace`, `TriggerTrace`, `CommandTrace`, `TokenUsage` | The instrumentation records ([../guides/latency-and-traces.md](../guides/latency-and-traces.md)) |
| `ConfigError`, `MissingEnvError` | The errors a caller catches ([Errors](#errors)) |
| `Agent` | The reasoning loop, owned by `Wica` and reached as `wica.agent`; constructible directly for a programmatic model double ([../guides/testing.md](../guides/testing.md)) |

Three subpackages are imported explicitly, never re-exported from `wica`: **`wica.contrib.gradio`** (the browser panels, behind the `gradio` extra: [../guides/gradio-panels.md](../guides/gradio-panels.md)), **`wica.testing`** (the Probe, the `TestRun` harness, the report, the scripted tester, scenario files and checks: [../guides/testing.md](../guides/testing.md)) and **`wica.benchmark`** (the matrix, `Target`, the runner, the summary, the CLI: [../guides/benchmarking.md](../guides/benchmarking.md)). `wica.fake_model.FakeChatModel` is the scripted model behind `provider: "fake"`, for the direct seam. `wica.instrumentation` holds the measures over the records (`reaction_latency`, `reactions_per_input`). `wica.guards` holds the response-guard types ([Response guards](#response-guards)); writing your own guard is experimental. `wica.text_output` holds the parser of the model's text; an application needs only `OutputStream` from it, which `wica` re-exports.

## Construction

```
Wica.init(config: WicaConfig, *, coalesce_window: float = 0.2, loop: asyncio.AbstractEventLoop | None = None) -> Wica
```

The one constructor. It builds a `World` and an `Agent` from the config and wires them. With no `loop`, the Wica creates one and runs it in a daemon thread from `start()`; with one, it adopts it and leaves running it to you. `coalesce_window` is the batch window for trigger bursts, in seconds ([../guides/world-and-inputs.md](../guides/world-and-inputs.md#bursts-are-coalesced)); `0` disables it. Resolution happens here: the key variable is read, the prompt file is read, the provider's package is imported, so `MissingEnvError`, `ConfigError` and `ImportError` surface at `init`, not at config creation. `init` takes no application callables: the output Command is wired after it.

## The `Wica` surface

```
wica.world                      # the World (below); wica.agent — the Agent, rarely needed
wica.register_command(fn_or_command)              # a plain callable, or Command(...) for options
wica.set_output_command(fn_or_command | None)     # the one output channel: what the model says to its user
wica.start(); wica.stop(); wica.start()           # restartable: stop() is a pause
wica.close()                                      # terminal: releases an owned loop; no restart
wica.is_running                                   # True between start() and stop()
wica.remove_response_guard(name)                  # drop a default response guard (below)
wica.response_guards                              # name -> guard, in the order they run
wica.on_<event>.subscribe(handler)                # the nine Events (below)
```

- **`register_command(fn)`**: a sync or async function (its name and docstring become the tool's name and description), or a `Command`. Names are unique and `noop`, `next` and `cancel_command` are reserved: a duplicate or reserved name raises `ValueError`. Register before `start()`. [../guides/commands.md](../guides/commands.md).
- **`set_output_command(fn)`**: the one function that receives what the model says to its user (speech, a chat bubble). The model speaks by writing a tag named after it in its response text; the function is not a tool. Setting it recomposes the system prompt, so set it before `start()`. `None` clears it: no tag is taught and all the model's text is private. The contract is [The output Command](#the-output-command).
- **Lifecycle**: `start()` starts the World then the Agent (and the loop thread when owned); `stop()` cancels in-flight Commands and stops both, keeping registrations, values, subscriptions and history; `close()` stops if needed and releases the loop. `start()` on a running Wica is a no-op; `start()` after `close()` raises `RuntimeError`.
- **Response guards** check every model response before it is used, and send a wrong one back to the model: [Response guards](#response-guards).
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

## The output Command

```
wica.set_output_command(fn_or_command | None)

async def say(text: OutputStream, emotion: Literal["happy", "sad", "neutral"] = "neutral") -> str | None
def say(text: str, emotion: str = "neutral") -> str | None        # sync or async

wica.agent.output_command_name        # its name, None when none is set
wica.agent.output_wakes(call_id)      # whether that output's end wakes a step; None when the call is not an output
```

The output Command is the application's one output channel. The model issues it by writing a tag in its response text, named after the Command: `<say emotion="happy">Hello there!</say>`. Only the tag's body is delivered; text outside a tag is private ([Events](#events), `on_agent_private_text`). The task-oriented page is [../guides/commands.md](../guides/commands.md#make-your-agent-speak).

**The signature** is read by `set_output_command`, which raises `ValueError` when it does not fit:

| Part | Rule |
|---|---|
| First parameter | Receives the text. Annotated `OutputStream`: the function is called at the output's first words and receives them as they are written; it must be `async def`. Annotated `str`, or not annotated: the function is called once the tag closes, with the whole text; it may be sync or async |
| Every other parameter | A tag attribute: annotated `str` or a `Literal` of strings, with a string default. An attribute the model omits, misspells or gives a value outside the `Literal` falls back to the default; the text is still delivered. `*args` and `**kwargs` are refused |
| Name | The tag's name: a letter, then letters, digits, `_` or `-`. Not `noop`, `next` or `cancel_command`, and not the name of a registered Command |
| Docstring | Taught to the model in the system prompt, with each attribute and its allowed values |
| Return value | The execution's `result`, stringified; `None` reads as `Done.` |
| Definition | A callable, or `Command(fn, name=…, description=…)`. A `Command` wrapping a LangChain `BaseTool` is refused. `triggers_on_completion` is not read |

**`OutputStream`** is the text of one output as the model writes it:

| Member | What it is |
|---|---|
| `async for piece in stream` | The text pieces, in order; the iteration ends when the tag's body ends |
| `stream.text` | Everything received so far |
| `await stream.read()` | Waits for the end of the body and returns it whole |
| `stream.heard` | `None` by default. A function cancelled part-way may set it to the text it actually delivered: the model's history then keeps that instead of the whole body |

**The three tags** WICA reads in the model's text:

| Tag | Meaning |
|---|---|
| `<NAME attr="value">…</NAME>` | An output: `NAME` is the output Command's name, the body is delivered, the attributes are its other parameters |
| `<noop/>` | Do nothing. It counts only as the whole response; an empty response counts the same. The reaction's record has `noop=True` and no Command Event fires |
| `<next>…</next>` | A private note, written next to an output, asking to be called again once that output ends. `max_next_steps` caps how many steps in a row a note may wake ([configuration.md](configuration.md#the-agent-block)) |

The parsing is lenient: a response with text and no output tag is delivered whole, as one output, when the response ends; a tag left open is closed by the end of the response; a `<noop/>` beside an output is ignored; any other `<…>` is ordinary text. With no output Command set, only `<noop/>` is read.

**Each output is a Command execution.** It gets an `agent:command:<call_id>` entry whose `CommandExecution` has the output Command's `name` and, as `args`, the attributes plus `text`. It differs from a Command issued as a tool call in these ways:

- **It is not a tool.** It is not bound to the model and is not in `wica.agent.commands`.
- **It starts before the response is complete.** The entry reads `running` from the output's first words; `on_agent_command` fires once the body is complete, with the whole `text`.
- **Several tags are several outputs**, delivered one after the other in the order written.
- **One voice.** A response that starts an output cancels every output still running or waiting from an earlier response. Other Commands are left alone. `cancel_command` cancels an output like any Command.
- **Its end wakes a step only through a `<next>` note.** A cancelled output wakes nothing.
- **A cancelled output corrects the history.** Its body is cut to `stream.heard` when the function set it, and an output of the same response that never started is removed.

[specs/agent.md](../../specs/agent.md), "Output: tags in the model's text"; [specs/commands.md](../../specs/commands.md), "The output Command".

## Response guards

Before a model response is used (its Commands dispatched, its text recorded), WICA checks it with its **response guards**. A guard that finds a mistake rejects the response: the model is called again, over the same conversation plus a correction, up to `max_retries` times (default `1`, [configuration.md](configuration.md#the-agent-block)). Only the accepted response enters the history, so later prompts never show the mistake. Out of retries, the reaction ends with the `rejected` outcome: nothing is done, and an ERROR log line carries the reason.

**A response that has started an output is not guarded.** An output is delivered as the response streams, so by the end of the response part of it has been heard and cannot be taken back: the response is accepted as it is and no guard runs. An unknown or invalid Command call beside the output is dispatched and ends as a `failed` entry the model observes. The guards therefore cover the responses that only act or do nothing.

Three guards are on by default:

| Guard | Rejects a response that |
|---|---|
| `incomplete_response` | was cut at the output cap (`max_tokens` and the like) in the middle of a Command call; the model is asked again with a length to stay under. Without this guard a cut call is dropped, or dispatched with the arguments that survived |
| `unknown_command` | calls a Command that is not registered |
| `invalid_args` | calls a Command with arguments that do not fit its signature |

Each retry is a full model call the person waits for, and the Agent drops new inputs meanwhile, like during any step. To turn a guard off:

```python
wica.remove_response_guard("invalid_args")
```

`max_retries: 0` keeps the guards but never retries: a rejected response ends the reaction at once, its other Commands included.

**Your own guards are experimental.** `wica.add_response_guard(guard, name=None)` adds one: a function that receives a `wica.guards.ModelResponse` (`text`, `commands`, `bound`, `malformed`, `truncated`, `output_tokens`) and returns `None` to accept or a `wica.guards.Retry(feedback, show_response=True)` to reject; `show_response=False` retries with a reminder instead of showing the model its response. The shape may change without notice while we learn how guards behave, so do not build on it yet. A sync guard runs on the event loop and holds everything while it runs; an async one (an LLM judging the answer) holds the reaction, so inputs are dropped for as long as it takes. [specs/agent.md](../../specs/agent.md), "Response guards".

## Events

Each is an `Event[T]` you `.subscribe(handler)` on, multi-consumer, emitted on the loop thread; a raising handler is isolated from the others. The same objects are reachable on `wica.world` and `wica.agent`.

| Event | Payload | Fires |
|---|---|---|
| `wica.on_world_trigger` | `WorldEntry` | Raw: every qualifying World update, before coalescing, including the triggers a busy Agent later drops |
| `wica.on_agent_trigger` | `WorldEntry` | Filtered: once per trigger a reasoning step observed (several per step when a burst was coalesced) |
| `wica.on_agent_trigger_dropped` | `WorldEntry` | A trigger that arrived while a step was in flight and was dropped |
| `wica.on_agent_prompt` | `list[BaseMessage]` | The exact rendered messages, before each reaction's first model call |
| `wica.on_agent_response_rejected` | `ResponseRejection` | A response the [response guards](#response-guards) rejected: `attempt`, `response`, `feedback`, `show_response`, `will_retry` |
| `wica.on_agent_command` | `CommandIssued` | Each Command the model issues, at dispatch; the `agent:command:<call_id>` entry exists when it fires. An output is reported the same way, once per tag, when its body is complete: `args` holds its attributes and its whole `text`. A `<noop/>` response fires nothing |
| `wica.on_agent_command_ended` | `CommandTrace` | A Command's terminal state (`complete`, `failed`, `cancelled`), with its timing |
| `wica.on_agent_private_text` | `str` | The text the model wrote outside its output tags in a step, when there is any, once the response is complete. Nobody hears it. With no output Command, all the model's text. The `<noop/>` and `<next>` tags are not part of it |
| `wica.on_agent_reaction_ended` | `ReactionTrace` | Every reaction's record, at its end: the triggers, the phase stamps, the outcome, the Commands, the token usage |

[../guides/latency-and-traces.md](../guides/latency-and-traces.md); [specs/instrumentation.md](../../specs/instrumentation.md).

## Errors

| Error | Raised by | When |
|---|---|---|
| `ConfigError` | the loaders, `AgentConfig`, `Wica.init` | A missing, unknown or mistyped key, an invalid field combination; an unreadable prompt file; a `history_budget` with no known context window |
| `MissingEnvError` | `Wica.init` | `api_key_env` names a variable that is unset (a `ConfigError` subclass) |
| `ImportError` | `Wica.init` | The selected provider's extra is not installed |
| `ValueError` | `register_command`, `set_output_command`, `remove_response_guard`, `add_response_guard`, `world.register`, `Command` | A duplicate or reserved Command name; an output Command whose name or signature does not fit [the contract](#the-output-command); an unknown or duplicate guard name; an invalid World key or an already registered one; a name or description override on a `BaseTool` |
| `TypeError` | `world.update` | A value that is not of the entry's declared type (`None` is always allowed) |
| `RuntimeError` | `world.update`, `start`, `stop`, `close` | `update()` while not running; `start()` after `close()`, or on an injected loop that is not running; `stop()`/`close()` called from the Wica's own loop thread |

A Command that raises does not fail the step: its entry turns `failed` with the error text, and the model sees it. The same holds for an output Command.

## Contracts

- **One loop, many threads.** Everything WICA runs (the Agent's steps, the Commands, the output Command, the listeners, the Event handlers) runs on the one event loop `Wica` owns or adopted. `update()` may be called from any thread and hops to it. Never call `stop()`/`close()` from code running on that loop.
- **Setup before start.** `register`, `register_command`, `set_output_command` and the subscriptions need no running system; `update()` does. The order among the wiring calls does not matter.
- **Restart keeps, close releases.** `stop()`/`start()` preserves registrations, values, subscriptions, Commands and history. `close()` is terminal; a reset is a new `Wica.init`.
- **One reasoning step at a time.** A trigger arriving while a step is in flight is dropped, not queued (`on_agent_trigger_dropped`). Commands are not so limited: all the calls of one response dispatch at once and run concurrently, and a later step can observe and cancel one still running. [../guides/limits.md](../guides/limits.md).
- **Cancellation reaches the task.** `cancel_command`, `stop()` and `close()` cancel the asyncio task running a Command: an `async` Command unwinds at its next `await`; a sync one is offloaded to a thread, whose work cannot be interrupted, so its entry reads `cancelled` while the function runs to completion in the background.

## Migrating from the output sink

The output sink and the forced tool call are removed, with no alias. One output path replaces them: the output Command, issued as a tag in the model's text.

| Before | Now |
|---|---|
| `wica.set_output_sink(fn)` | `wica.set_output_command(fn)`; the function's first parameter receives the text ([The output Command](#the-output-command)) |
| An output Command `say(text: str)` called as a tool | The same function, issued as `<say>…</say>` in the text. It is no longer in `wica.agent.commands` |
| `agent.force_tool_call` in the config | Removed: the loaders reject it as an unknown key. Nothing is forced; speaking does not use `tool_choice` |
| The `noop` tool | `<noop/>` as the whole response; read `ReactionTrace.noop`. No `on_agent_command` fires for it |
| A step after every utterance (`triggers_on_completion` on the output Command) | The model's `<next>` note, capped by `agent.max_next_steps` |
| `wica.on_agent_text` | `wica.on_agent_private_text`: the text outside the output tags |
| The `output_command_required` guard | Removed: text with no tag is delivered whole |
| `ReactionTrace.sink_duration` | Removed. `first_output_at` and `time_to_first_output` are new; `reaction_latency` ends when the first output opens |
| `TestRun.expect_text`, the scenario expectation `{"text": …}` | `expect_private_text`, `{"private_text": …}`; `expect_output` matches the output's `text` |
| A scripted model speaking with `{"tool_calls": [{"name": "say", …}]}`, silent with a `noop` call | `{"text": "<say>Hello</say>"}`, `{"text": "<noop/>"}` |
| `TranscriptLog.output_sink` | Removed: the transcript subscribes to the Events by itself |
