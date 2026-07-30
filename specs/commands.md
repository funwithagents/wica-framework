# Commands

**Status:** Draft

## Purpose

A **Command** is WICA's unit of agent action on the World — the **C** in WICA (World / Inputs / Commands / Agents). When an Agent decides to *do* something (add two numbers, walk to the kitchen, cancel an in-flight call, speak), it issues a Command; the Command runs, and its execution is tracked as World state.

A Command is realized **under the hood as a LangChain tool** — the provider-agnostic schema/binding primitive the model actually emits a call against. But "tool" is an implementation detail at the Agent's I/O boundary, exactly like the model SDK and provider message blocks:

- **LangChain is "a primitive, not a framework"** ([agent.md](agent.md)) — quarantined to the Agent.
- **`Content` decouples WICA from provider blocks** ([content.md](content.md)) — the SDK shape lives only in the Agent's adapter.
- A **tool is the primitive a Command is exposed and invoked through** — WICA code and specs speak *Commands*; the word *tool* survives only where LangChain's `@tool`/`bind_tools` machinery is literally in play.

This keeps the framework's vocabulary aligned with its name: agents act on the World via Commands, and swapping or adding a model SDK never changes what a Command *is*.

## Commands and tools

- **Off-the-shelf LangChain tools work unmodified.** Registering an existing tool *is* how you commonly define a Command — the Agent must not require a tool to carry any WICA-specific hook (a describe/serialize function, a mode annotation, …). Anything WICA needs beyond `name`/`args`/`result` is optional and lives at the *registration/wrapper* layer, never inside the tool ([agent.md](agent.md), "Commands").
- **Registering a Command builds its backing tool.** The Agent exposes `register_command(fn)` (over LangChain's `@tool`/`bind_tools`); the resulting tool is what gets bound to the model.
- **Not every Command is backed by an off-the-shelf tool.** Command is the umbrella for *all* agent actions on the World. Some are plain registered tools (`add`, `walk_to`); others are WICA-native control actions the Agent implements directly — e.g. `cancel_command(call_id)` (below), the deferred `cancel_activity(task_id)` (which cancels an in-flight *LLM call* — see [agent.md](agent.md), "Concurrency & interruption"), and the deferred `speak(text)` output Command. All are Commands; the tool is just the most common backing.

### `cancel_command` — the model aborting its own in-flight Command

A running Command is genuinely current state (its `agent:command:<call_id>` entry renders `running`), so the model can decide to abort one it previously issued. `cancel_command(call_id)` is a WICA-native control Command the Agent **auto-registers** (no app wiring, like the deferred `cancel_activity`) and implements directly over the same loop-thread cancellation path the framework already uses (`Agent.cancel_command`, run from `stop()` — see [agent.md](agent.md), "Concurrency & interruption").

- **No new identifier.** The target is named by its `call_id`, which is **already visible** to the model: the World renders every command entry inside an `<entry key="agent:command:<call_id>" …>` envelope ([world.md](world.md)), so the `call_id` is on screen with no extra rendering. The Command reads it straight from there — the entry body is left unchanged. The handler is lenient: it accepts either the bare `call_id` or the full `agent:command:<call_id>` key (it strips the prefix), so a verbatim copy of the envelope key also works.
- **id-guarded no-op.** Cancelling a Command that has already finished, is not running, or never existed is a harmless no-op (the running-task lookup misses) — same guard as `Agent.cancel_command` and the World's TTL `expected_id` pattern. The Command's `result` reports which happened (`cancelling <call_id>` vs. a "not a running command" message), so the model gets feedback through the normal completion-observation path.
- **Uniform lifecycle — no special-casing.** `cancel_command` is dispatched, tracked, and observed exactly like any other Command: it gets its own `agent:command:<call_id>` entry that goes terminal and re-triggers. So a single cancel produces **two** triggers — the cancel Command's own completion and the target going `cancelled` — which the single-in-flight loop handles by its normal drop-and-leave-in-place rule (one is processed, the other's terminal entry waits to be observed; see [agent.md](agent.md), "Future improvements"). No new machinery.

`cancel_command` cancels a *Command*; the deferred `cancel_activity` cancels an in-flight *LLM call* — siblings, not the same Command. Only `cancel_command` is buildable in v1, where concurrent LLM calls don't exist yet but concurrent Commands do (a step dispatches async Commands and goes idle; a later step observes them still `running`).

## Command execution as a World entry (event-driven)

Command execution is **stored in the World** — an in-flight Command is genuinely *current state* and fits the World's snapshot model ([world.md](world.md)). The lifecycle maps directly onto existing World primitives, with **no new machinery**:

- Register the command-execution entry (`agent:command:<call_id>`) with `triggers_llm_call=True` **and** `trigger_condition_fn=lambda old, new: new.is_terminal()`. Start/progress updates land in the entry (visible to any reader) but **fail the condition → no trigger**; only **complete/failed/cancelled** triggers a new Agent step.
- Mark it `include_in_prompt=True` so a *concurrent* step renders "navigate → running (task 7)" and can decide to cancel it before it finishes.
- On completion, the model continues via the **reconstruct path**: at the next step the Agent re-renders the earlier Command from its durable, provider-neutral `CommandRecord` (name/args/`call_id`) as the model's **native `tool_call`**, paired with a **`tool_result`** — but that `tool_result` carries only a **fixed acknowledgement pointing at the command-execution entry** (`agent:command:<call_id>`), **never the outcome**. The provider pins a `tool_result` immediately after its `tool_use` (a `tool_use` can't be left dangling across turns), so it can only ever sit at the Command's *dispatch* site, not where the Command actually finished — and while the Command is still in flight, a `tool_result` present at all would read as "the call returned." So the **status and outcome are delivered by the command-execution entry itself, rendered as an observation** — `running` while in flight, terminal (`result`/`error`) at the step where the completion is observed — landing the outcome at the causally-correct point in history. Consequently the `agent:command:<call_id>` entry **is** rendered into the observation like any other entry (this drops the earlier skip-by-prefix rule). The native pair is reconstructed *only at render time*, inside the Agent's message adapter — the same place LangChain/provider blocks already live — never stored in history; the Agent renders the entry body with its own command serializer, passed via `render_entry`'s `serialize_fn` override, so the World stays command-agnostic and a **retired** entry still re-renders from its history snapshot (see [agent.md](agent.md), "Rendering to messages"). *(Earlier design rendered the Command as a `"Calling walk_to…"` assistant **text** line; that backfired — the model imitated the prose and emitted command descriptions as free text instead of real tool calls. The native `tool_call` keeps that fixed; the running/terminal status now rides an observation `<entry>` — a user-role block — so there is again nothing for the model to imitate as its own output.)*

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

The Agent always has a Command's **name**, its **args** (from the call), and its **result** (stringified). So it can always describe an execution — `add(a=1, b=2) → 3` — with zero cooperation from the backing tool. A per-registration override (`render_result`) is *optional sugar* for a nicer result summary, never required.

Consequence for history ([agent.md](agent.md), "History record shape"): the durable record of a Command's *action* is the **neutral `CommandRecord`** (name/args/`call_id`); its *outcome* is the command-execution World-entry snapshot (`CommandExecution`) captured in a later Observation record — **not** raw provider `tool_use`/`tool_result` messages. At render time the Agent reconstructs the native `tool_call` from the `CommandRecord` (paired with the fixed-ack `tool_result`), and the outcome renders from the World-entry snapshot via the generic description above; both live only in the Agent's message adapter and are then discarded, never stored. History stays snapshots + neutral records.

## Sync vs. async (deferred optimization)

Both sync and async Commands are two inferences (a second pass is always needed to use a result), so async carries **no extra-inference penalty**. The only real difference: a fast Command *could* be handled inline (append `tool_result` to the frozen step array, native thread preserved) instead of ending the step and re-observing the whole World. This is a **latency optimization, not a behavioral change**, and is **deferred** — the uniform event-driven model above ships first.

When added, the sync/async decision is made **without annotating the Command**, via either (or both):

- **(a) Auto by latency (preferred):** `await` the Command inline with a short timeout (~1–2s). Returns in time → handle inline (native pair, sync). Exceeds it → detach: convert to the history-text line, register the running command-execution entry, end the step, resume on completion (async). The Command's *actual speed* decides, per invocation — no declaration.
- **(b) Optional registration flag:** `register_command(fn, mode="auto" | "sync" | "async")`, default `auto`. A one-word wrapper flag (not a tool change) to pin known cases (`walk_to` always async, `add` always sync).

## Open questions

1. **Failure rendering.** How a *failed* Command renders in its command-execution entry (currently `Called explode() → failed: boom`) — whether that's rich enough, and how it reads alongside the fixed-ack `tool_result` that the reconstruct path pairs with the reconstructed `tool_call`.
2. **Mixed-speed parallel Commands (deferred sync path).** When one turn emits **multiple** (parallel) Commands of mixed speed, whether to detach all vs. return results for the fast ones inline plus acks for the slow — and the auto-latency timeout value.
3. **Stale-`id` handling.** What happens when a Command references a World entry `id` that no longer matches the entry's current `id` (world.md open question #2 flags this as "likely primarily a Commands-spec concern"). Since `id` advances on *every* `update()`, a stale reference is detectable after any change; the response policy (reject, re-observe, proceed) is unspecified.
4. **Generic command-activity listening.** A listener can't subscribe to "any Command" without knowing every `call_id` in advance, since `add_listener` needs a concrete key. Sketched fix (agent.md open question #12): a persistent broadcast entry (`agent:command_activity`, `include_in_prompt=False`, `triggers_llm_call=False`) the Agent plain-overwrites on every status change. Not yet built.
5. **`render_result` override.** The optional per-registration nicer-summary hook is named but unspecified in shape.
