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
- **Not every Command is backed by an off-the-shelf tool.** Command is the umbrella for *all* agent actions on the World. Some are plain registered tools (`add`, `walk_to`); others are WICA-native control actions the Agent implements directly — e.g. `cancel_activity(task_id)` (already a "model-issued command" in [agent.md](agent.md), "Concurrency & interruption") and the deferred `speak(text)` output Command. All are Commands; the tool is just the most common backing.

## Command execution as a World entry (event-driven)

Command execution is **stored in the World** — an in-flight Command is genuinely *current state* and fits the World's snapshot model ([world.md](world.md)). The lifecycle maps directly onto existing World primitives, with **no new machinery**:

- Register the command-execution entry (`agent:command:<call_id>`) with `triggers_llm_call=True` **and** `trigger_condition_fn=lambda old, new: new.is_terminal()`. Start/progress updates land in the entry (visible to any reader) but **fail the condition → no trigger**; only **complete/failed/cancelled** triggers a new Agent step.
- Mark it `include_in_prompt=True` so a *concurrent* step renders "navigate → running (task 7)" and can decide to cancel it before it finishes.
- On completion, the model continues via the **reconstruct path**: it sees its own `"Calling walk_to…"` line in history *plus* the result in World state, and picks up from there. (This is **not** the native `tool_use`/`tool_result` mechanism — that's dropped on this path by design; the history-text line is what keeps it coherent.)

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

The Agent always has a Command's **name**, its **args** (from the call), and its **result** (stringified). So it can always render a plain-text description of an execution — `Called add(a=1, b=2) → 3` — with zero cooperation from the backing tool. This generic default is what lands in history; a per-registration override (`render_result`) is *optional sugar* for a nicer summary, never required.

Consequence for history ([agent.md](agent.md), "History record shape"): the durable record of a Command is this **description**, not the raw `tool_use`/`tool_result` messages. The native pair, if used at all, is a within-step transport detail, discarded when the step ends; history is snapshots + descriptions.

## Sync vs. async (deferred optimization)

Both sync and async Commands are two inferences (a second pass is always needed to use a result), so async carries **no extra-inference penalty**. The only real difference: a fast Command *could* be handled inline (append `tool_result` to the frozen step array, native thread preserved) instead of ending the step and re-observing the whole World. This is a **latency optimization, not a behavioral change**, and is **deferred** — the uniform event-driven model above ships first.

When added, the sync/async decision is made **without annotating the Command**, via either (or both):

- **(a) Auto by latency (preferred):** `await` the Command inline with a short timeout (~1–2s). Returns in time → handle inline (native pair, sync). Exceeds it → detach: convert to the history-text line, register the running command-execution entry, end the step, resume on completion (async). The Command's *actual speed* decides, per invocation — no declaration.
- **(b) Optional registration flag:** `register_command(fn, mode="auto" | "sync" | "async")`, default `auto`. A one-word wrapper flag (not a tool change) to pin known cases (`walk_to` always async, `add` always sync).

## Open questions

1. **Failure rendering.** How a *failed* Command renders in both the running entry and the history description (currently `Called explode() → failed: boom`) — whether that's rich enough, and how it interacts with the reconstruct path's coherence.
2. **Mixed-speed parallel Commands (deferred sync path).** When one turn emits **multiple** (parallel) Commands of mixed speed, whether to detach all vs. return results for the fast ones inline plus acks for the slow — and the auto-latency timeout value.
3. **Stale-`id` handling.** What happens when a Command references a World entry `id` that no longer matches the entry's current `id` (world.md open question #3 flags this as "likely primarily a Commands-spec concern"). Since `id` advances on *every* `update()`, a stale reference is detectable after any change; the response policy (reject, re-observe, proceed) is unspecified.
4. **Generic command-activity listening.** A listener can't subscribe to "any Command" without knowing every `call_id` in advance, since `add_listener` needs a concrete key. Sketched fix (agent.md open question #12): a persistent broadcast entry (`agent:command_activity`, `include_in_prompt=False`, `triggers_llm_call=False`) the Agent plain-overwrites on every status change. Not yet built.
5. **`render_result` override.** The optional per-registration nicer-summary hook is named but unspecified in shape.
