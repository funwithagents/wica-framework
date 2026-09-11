---
code:
  - src/wica/agent.py
  - src/wica/command.py
tests:
  - tests/test_agent.py
  - tests/test_command.py
---

# Commands

**Status:** Implemented

## Purpose

A **Command** is WICA's unit of agent action on the World — the **C** in WICA (World / Inputs / Commands / Agents). When an Agent decides to *do* something (add two numbers, walk to the kitchen, cancel an in-flight call, speak), it issues a Command; the Command runs, and its execution is tracked as World state.

A Command is realized **under the hood as a LangChain tool** — the provider-agnostic schema/binding primitive the model actually emits a call against. But "tool" is an implementation detail at the Agent's I/O boundary, exactly like the model SDK and provider message blocks:

- **LangChain is "a primitive, not a framework"** ([agent.md](agent.md)) — confined to the model-facing edge: the Agent, the `Command` wrapper, and the prompt Event.
- **`Content` decouples WICA from provider blocks** ([content.md](content.md)) — the SDK shape lives only in the Agent's adapter.
- A **tool is the primitive a Command is exposed and invoked through** — WICA code and specs speak *Commands*; the word *tool* survives only where LangChain's `@tool`/`bind_tools` machinery is literally in play.

This keeps the framework's vocabulary aligned with its name: agents act on the World via Commands, and swapping or adding a model SDK never changes what a Command *is*.

## Commands and tools

- **Off-the-shelf LangChain tools work unmodified.** Registering an existing tool *is* how you commonly define a Command — the Agent must not require a tool to carry any WICA-specific hook (a describe/serialize function, a mode annotation, …). Anything WICA needs beyond `name`/`args`/`result` is optional and lives at the *registration/wrapper* layer, never inside the tool ([agent.md](agent.md), "Commands"). That wrapper layer is now a concrete object — the **`Command`** below.
- **Registering a Command builds its backing tool.** The Agent exposes `register_command(fn)` (over LangChain's `@tool`/`bind_tools`); the resulting tool is what gets bound to the model.
- **Not every Command is backed by an off-the-shelf tool.** Command is the umbrella for *all* agent actions on the World. Some are plain registered tools (`add`, `walk_to`); others are WICA-native control actions the Agent implements directly — e.g. `cancel_command(call_id)` and the always-registered `noop` (both below), the deferred `cancel_reaction(reaction_id)` (which would cancel an in-flight reasoning reaction — see [agent.md](agent.md), "Future improvements"), and the optional **output Command** (below), the generalized realization of the once-deferred `speak(text)`. All are Commands; the tool is just the most common backing.

## The `Command` object

WICA is *named* for Commands (the **C** in WICA), yet until now it had no `Command` *type* — only the runtime companions `CommandExecution`, `CommandRecord`, and `CommandIssued` (all in [agent.py](../src/wica/agent.py)). `Command` is the missing member: the **definition-time** object, the concrete home for the "registration/wrapper layer" the tenet above names, and the seam that keeps LangChain out of the framework's command-definition surface.

- **`Command` wraps a callable or an off-the-shelf tool and holds the backing `BaseTool` internally.** Its constructor accepts **either** a plain `Callable[..., Any]` **or** a LangChain `BaseTool`:
  - `Command(fn, *, name=None, description=None)` — builds the backing tool from the function (name from `__name__`, description from the docstring; explicit `name`/`description` override). An undocumented function with no `description` fails loudly, the same `tool()` requirement WICA already inherits.
  - `Command(existing_tool)` — wraps an off-the-shelf `BaseTool` unmodified. "Works unmodified" still holds: you *wrap*, you don't *modify* — it's just explicit now.
- **`Command` is the one place `BaseTool` appears at the definition layer.** LangChain construction (`@tool`/binding of a `BaseTool`) lives inside `Command`; the Agent unwraps a `Command` to its tool only where it already speaks LangChain — `bind_tools` and the model-call adapter. This keeps the command-*definition* surface LangChain-free even though `Command` itself imports LangChain to build the tool (see [agent.md](agent.md), "LangChain as a primitive, not a framework"): before, `register_command` accepted a raw `BaseTool`, leaking LangChain into how you *define* commands; now the command-definition surface speaks `Command | Callable` only.
- **`register_command` takes exactly one argument: `fn: Callable[..., Any] | Command`.** A bare callable is auto-wrapped into a `Command` internally (the everyday convenience — name and description come from `__name__`/docstring); a `Command` is used directly. There are **no** `name`/`description` keyword arguments on `register_command` any longer — overriding either is expressed by constructing a `Command(fn, name=…, description=…)` and passing that. In practice every application/test call site is already the bare `register_command(fn)` form; the only sites that overrode name/description were WICA's own control Commands, which now build a `Command` (see `cancel_command`, `noop`).
- **`Command` is where deferred per-command options will live.** The sync/async `mode` (see "Sync vs. async") and a custom result-rendering hook ([Open questions](#open-questions) #5) are natural `Command` fields when built — additive, and none required today.
- **Names are unique and two are reserved.** `noop` and `cancel_command` belong to WICA. `Agent.register_command()` raises `ValueError` for either name and for any name already registered. The optional output Command's name is checked at `Agent` construction (a reserved name raises `ValueError` there, so `Wica.init` fails fast) and is then reserved too. WICA's own Commands are attached by a private idempotent path used by `Agent.start()`, so a restart re-attaches them without ever replacing an application Command.

### `cancel_command` — the model aborting its own in-flight Command

A running Command is genuinely current state (its `agent:command:<call_id>` entry renders `running`), so the model can decide to abort one it previously issued. `cancel_command(call_id)` is a WICA-native control Command the Agent **auto-registers** (no app wiring, like the deferred `cancel_reaction`) — as a `Command(self._cancel_command_action, name=…, description=…)`, now that `register_command` carries no `name`/`description` kwargs — and implements directly over the same loop-thread cancellation path the framework already uses (`Agent.cancel_command` — `stop()` cancels every Agent-owned task through the same loop-thread mechanism; see [agent.md](agent.md), "Concurrency and long-running Commands in v1"). It targets a running **output Command** (below) exactly like any other, which is what makes barge-in — cancelling in-flight speech — possible.

- **No new identifier.** The target is named by its `call_id`, which is **already visible** to the model: the World renders every command entry inside an `<entry key="agent:command:<call_id>" …>` envelope ([world.md](world.md)), so the `call_id` is on screen with no extra rendering. The Command reads it straight from there — the entry body is left unchanged. The handler is lenient: it accepts either the bare `call_id` or the full `agent:command:<call_id>` key (it strips the prefix), so a verbatim copy of the envelope key also works.
- **id-guarded no-op.** Cancelling a Command that has already finished, is not running, or never existed is a harmless no-op (the running-task lookup misses) — same guard as `Agent.cancel_command` and the World's TTL `expected_id` pattern. The Command's `result` reports which happened (`cancelling <call_id>` vs. a "not a running command" message), so the model gets feedback through the normal completion-observation path.
- **Uniform lifecycle — no special-casing.** `cancel_command` is dispatched, tracked, and observed exactly like any other Command: it gets its own `agent:command:<call_id>` entry that goes terminal and re-triggers. So a single cancel produces **two** triggers — the cancel Command's own completion and the target going `cancelled` — which the single-in-flight loop handles by its normal drop-and-leave-in-place rule (one is processed, the other's terminal entry waits to be observed; see [agent.md](agent.md), "Future improvements"). No new machinery.

`cancel_command` cancels a *Command*; the deferred `cancel_reaction` cancels an in-flight *reaction* — siblings, not the same Command. Only `cancel_command` is buildable in v1, where concurrent LLM calls don't exist yet but concurrent Commands do (a step dispatches async Commands and goes idle; a later step observes them still `running`).

### The output Command — deliberate, cancellable user-facing output

The output Command is the generalized realization of the once-deferred `speak(text)` ([agent.md](agent.md), "Future improvements"): instead of a fixed `speak`, the **application registers whatever Command is its user-facing output channel** — TTS, a chat bubble, a robot's mouth — and the Agent treats it as *the* output channel. It is **optional**: when none is set, the Agent keeps its v1 free-text-as-output behavior unchanged (see [agent.md](agent.md), "Output").

Why a Command rather than reusing the `output_sink`? The point is **not** "get text to TTS" — an async `output_sink` could already call TTS. It is the two things only the Command lifecycle provides:

- **Cancellable, barge-in-able output.** An output Command runs as a tracked async task with an `agent:command:<call_id>` entry, so `cancel_command` (above) can interrupt in-flight speech when a new input arrives. An `output_sink` is an opaque inline `await` with no handle — the model can neither *observe* nor *cancel* it.
- **Observable output.** Because the running entry is `include_in_prompt=True`, a concurrent step *sees* "speaking … (running)" and can decide to cancel it or wait. The model gains explicit, per-step control over **what** it says and **whether** it says anything — its free text becomes private reasoning (see [agent.md](agent.md), "Output").

The output Command runs through the **same uniform lifecycle** as any other Command below — `running` → terminal `agent:command:<call_id>` entry — with exactly one tunable difference: **whether its terminal completion re-triggers a step.** A module-level constant `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION` (default **`True`**) gates the `triggers_llm_call` its entry is registered with:

- **`True` (default):** speaking re-triggers, so the Agent gets a follow-up step and can chain a second utterance, self-correct, or **`noop`** (below) to stop. Cost: **≥2 LLM calls per utterance** (the utterance, then the "anything else?" step).
- **`False`:** speaking never wakes a fresh step (speak → idle). Saves the extra inference but forbids self-continuation without a new external input. The terminal entry is then observed and retired by the next input-driven step, exactly like any completion whose trigger was dropped (see "Command execution as a World entry").

Shipping default is `True`; the constant exists so flipping the trade is a one-line change, not a refactor. The choice is recorded, not hidden, precisely because the cost/behavior trade is real. `include_in_prompt` stays `True` regardless, so barge-in works either way.

### `noop` — the model choosing not to act

LLMs are unreliable at returning *genuinely* empty output, so "reply with empty text to do nothing" is a fragile terminator. Instead the Agent **auto-registers a zero-argument WICA-native `noop` Command** (always, independent of any output Command — "the agent may choose not to react to an observation" is generally useful) that the model calls to signal *no reaction*.

`noop` is explained to the model in two complementary places: the **WICA runtime primer** appended to the system prompt (which establishes that declining to react is allowed at all — see [agent.md](agent.md), "System prompt composition") and its own tool **description** (which states *when* to call it). A live e2e test verifies models actually reach for it (see [testing.md](testing.md)).

`noop` is the **one Command that is deliberately not an action on the World**, so it breaks the uniform lifecycle on purpose:

- It creates **no** `agent:command:<call_id>` entry, spawns **no** task, and **never triggers** — it exists precisely to *stop* a cycle (were it to re-trigger, `noop` → completion → `noop` would loop forever). Its non-triggering is **not** tunable (contrast the output Command's flag).
- It **is** still a command the model issued, so the Agent's `on_command` instrumentation Event fires for it (`CommandIssued("noop", {})`) like for every other issued command — observability, not action. A consumer wanting only World actions filters `noop` (and the output Command) out by name; one wanting to *show* that the agent declined (as the conversation demo does) renders it. See [agent.md](agent.md), "Instrumentation".
- It **is** recorded in history so later context shows the agent *chose* not to react — as a **dedicated `NoReactionRecord`** (`call_id` only), *not* a `CommandRecord`: `noop` has no `name`/`args` worth recording and never yields an Observation outcome (see [agent.md](agent.md), "History record shape"). To keep the provider message stream valid (a native `tool_call` needs a matching `tool_result`), it renders as the model's native `noop` tool call paired with a **plain acknowledgement** `tool_result` — *not* the entry-pointer ack the real Commands use (there is no entry to point at); the render path selects that ack by record type. Because there is always an assistant turn (the `noop` call) between observations, it also sidesteps the consecutive-observation rendering wrinkle a truly empty response would create.

`noop` is the natural terminator for the output Command's re-trigger chain (when `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION` is `True`), and independently the way any agent declines to react.

## Command execution as a World entry (event-driven)

Command execution is **stored in the World** — an in-flight Command is genuinely *current state* and fits the World's snapshot model ([world.md](world.md)). The lifecycle maps directly onto existing World primitives, with **no new machinery**:

- Register the command-execution entry (`agent:command:<call_id>`) with `triggers_llm_call=True` **and** `trigger_condition_fn=lambda old, new: new.is_terminal()`. Start/progress updates land in the entry (visible to any reader) but **fail the condition → no trigger**; only **complete/failed/cancelled** triggers a new Agent step.
- The `<call_id>` in the entry key is **WICA-owned**: the Agent uses the provider's tool-call id when it satisfies the World key grammar ([world.md](world.md)) and no entry with that key exists, and otherwise generates one (`uuid4().hex`). The provider's id is kept separately on the history record (`CommandRecord.tool_call_id`) for reconstructing the native `tool_call`/`tool_result` pair; the two are equal in the common case. The model only ever sees `call_id`.
- Mark it `include_in_prompt=True` so a *concurrent* step renders "navigate → running (task 7)" and can decide to cancel it before it finishes.
- On completion, the model continues via the **reconstruct path**: at the next step the Agent re-renders the earlier Command from its durable, provider-neutral `CommandRecord` (name/args/`call_id`) as the model's **native `tool_call`**, paired with a **`tool_result`** — but that `tool_result` carries only a **fixed acknowledgement pointing at the command-execution entry** (`agent:command:<call_id>`), **never the outcome**. The provider pins a `tool_result` immediately after its `tool_use` (a `tool_use` can't be left dangling across turns), so it can only ever sit at the Command's *dispatch* site, not where the Command actually finished — and while the Command is still in flight, a `tool_result` present at all would read as "the call returned." So the **status and outcome are delivered by the command-execution entry itself, rendered as an observation** — `running` while in flight, terminal (`result`/`error`) at the step where the completion is observed — landing the outcome at the causally-correct point in history. Consequently the `agent:command:<call_id>` entry **is** rendered into the observation like any other entry (this drops the earlier skip-by-prefix rule). The native pair is reconstructed *only at render time*, inside the Agent's message adapter — the same place LangChain/provider blocks already live — never stored in history; the Agent renders the entry body with its own command serializer, passed via `render_entry`'s `serialize_fn` override, so the World stays command-agnostic and a **retired** entry still re-renders from its history snapshot (see [agent.md](agent.md), "Rendering to messages"). *(Rendering the Command as a `"Calling walk_to…"` assistant **text** line is deliberately avoided: a model shown that prose imitates it, emitting command descriptions as free text instead of real tool calls. The native `tool_call` keeps that fixed, and the running/terminal status rides an observation `<entry>` — a user-role block — so there is nothing for the model to imitate as its own output.)*

This is the **default execution model for all Commands**: uniform, event-driven, needs no per-command sync/async declaration, and keeps the "full World resend" cheap because fresh/archival rendering + prompt caching means only the completion entry (and any inputs that arrived meanwhile) miss cache.

### CommandExecution value model

The value stored in the entry is an immutable `CommandExecution` snapshot:

| Field | Description |
|---|---|
| `name` | The Command's name (the backing tool's name) |
| `args` | The arguments the model issued the Command with |
| `state` | `running` → `complete` \| `failed` \| `cancelled`. `is_terminal()` is true for anything but `running` — this is the predicate `trigger_condition_fn` checks so only a finished Command starts a new step |
| `result` | Stringified result on `complete`, else `None` |
| `error` | Error message on `failed`, else `None` |

## Generic call description (no command cooperation)

The Agent always has a Command's **name**, its **args** (from the call), and its **result** (stringified). So it can always describe an execution — `add(a=1, b=2) → 3` — with zero cooperation from the backing tool. A future per-registration rendering override could provide a nicer result summary, but no such hook exists in v1 and no Command cooperation is required.

Consequence for history ([agent.md](agent.md), "History record shape"): the durable record of a Command's *action* is the **neutral `CommandRecord`** (name/args/`call_id`); its *outcome* is the command-execution World-entry snapshot (`CommandExecution`) captured in a later Observation record — **not** raw provider `tool_use`/`tool_result` messages. At render time the Agent reconstructs the native `tool_call` from the `CommandRecord` (paired with the fixed-ack `tool_result`), and the outcome renders from the World-entry snapshot via the generic description above; both live only in the Agent's message adapter and are then discarded, never stored. History stays snapshots + neutral records.

## Sync vs. async (deferred optimization)

Both sync and async Commands are two inferences (a second pass is always needed to use a result), so async carries **no extra-inference penalty**. The only real difference: a fast Command *could* be handled inline (append `tool_result` to the frozen step array, native thread preserved) instead of ending the step and re-observing the whole World. This is a **latency optimization, not a behavioral change**, and is **deferred** — the uniform event-driven model above ships first.

When added, the sync/async decision is made **without annotating the Command**, via either (or both):

- **(a) Auto by latency (preferred):** `await` the Command inline with a short timeout (~1–2s). Returns in time → handle inline (native pair, sync). Exceeds it → detach: convert to the history-text line, register the running command-execution entry, end the step, resume on completion (async). The Command's *actual speed* decides, per invocation — no declaration.
- **(b) Optional registration flag:** `register_command(fn, mode="auto" | "sync" | "async")`, default `auto`. A one-word wrapper flag (not a tool change) to pin known cases (`walk_to` always async, `add` always sync).

## Cancellation reaches the task, not always the work

`cancel_command` (and the framework's own `Agent.cancel_command`, or the blanket cancellation `stop()` performs) cancels the **asyncio task** running the Command — it lands a `CancelledError` at the awaiting coroutine's next suspension point, flips the command-execution entry to `cancelled`, and re-triggers. That is prompt and reliable for the *bookkeeping*. Whether it stops the actual *work* depends on how the backing function is written:

- **Native `async` Command (has real `await` points).** Genuinely cancellable: `CancelledError` lands at an `await`, the coroutine unwinds, `finally` blocks run. This is the case the whole cancellation guarantee is built on.
- **Sync function.** WICA runs it — LangChain's `ainvoke` offloads a sync tool to a thread-pool executor, so it works and doesn't block the loop. But **Python cannot forcibly kill a thread** (no safe `Thread.kill`; the `PyThreadState_SetAsyncExc` hack can't interrupt a blocking C call — precisely the slow case you'd want to cancel). So on cancel the *awaiting task* unwinds and the entry reads `cancelled` immediately, while the **worker thread keeps running the sync function to completion in the background, its result discarded**. The cancellation is honest for the World, not for the CPU.
- **Cooperative sync function** (periodically checks a `threading.Event`/flag and returns early) is the way to make a sync Command actually stoppable without going async.

**Guidance:** prefer `async` for any Command that can run long enough to be worth cancelling; if it must be sync and long-running, make it cooperative. The only way to *forcibly* kill uncooperative blocking work is to run it in a **subprocess** (`terminate()`/`SIGKILL`), which trades away cheap in-process World access for serialized args/results — a deliberate escape hatch, not the default. This mirrors the same thread-can't-be-cancelled reasoning that bans `init_chat_model`'s local `huggingface` pipeline in [agent.md](agent.md) ("Provider-agnostic model, from config").

## Open questions

1. **Failure rendering.** How a *failed* Command renders in its command-execution entry (currently `Called explode() → failed: boom`) — whether that's rich enough, and how it reads alongside the fixed-ack `tool_result` that the reconstruct path pairs with the reconstructed `tool_call`.
2. **Mixed-speed parallel Commands (deferred sync path).** When one turn emits **multiple** (parallel) Commands of mixed speed, whether to detach all vs. return results for the fast ones inline plus acks for the slow — and the auto-latency timeout value.
3. **Stale-`id` handling.** What happens when a Command references a World entry `id` that no longer matches the entry's current `id` ([world.md](world.md) open question #1 flags this as "likely primarily a Commands-spec concern"). Since `id` advances on *every* `update()`, a stale reference is detectable after any change; the response policy (reject, re-observe, proceed) is unspecified.
4. **Generic command-activity listening.** A listener can't subscribe to "any Command" without knowing every `call_id` in advance, since `add_listener` needs a concrete key. One possible fix ([agent.md](agent.md) open question #5) is a persistent broadcast entry (`agent:command_activity`, `include_in_prompt=False`, `triggers_llm_call=False`) that the Agent overwrites on every status change. Not yet built.
5. **Custom result rendering.** Whether registration should accept an optional nicer-summary hook, and what its shape should be, is unspecified.
