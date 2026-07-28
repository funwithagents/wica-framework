# Agent

**Status:** Draft

## Purpose

The Agent is the reasoning loop that observes the World and acts on it. It is triggered by the World (via `set_trigger_handler`), builds an LLM prompt from World state, runs inference against a configurable provider, executes **Commands** (see [commands.md](commands.md)), and streams output to a pluggable sink. It owns the conversation **history**; the World owns current **state**.

This spec is an early draft: the sections under "Settled" reflect decisions already made in design discussion; "Open questions" holds everything still being brainstormed — several are load-bearing and not yet resolved.

## Settled

### LangChain as a primitive, not a framework

The Agent uses LangChain for two things only: the **provider-agnostic chat model** abstraction and **tool schemas / binding** (the primitive WICA **Commands** are built on — see [commands.md](commands.md)). It does **not** use a prebuilt agent loop (LangGraph's `create_react_agent` or similar) — WICA already owns state (World) and history, and the requirements below (cancellation, storing Command execution in the World, streaming to a custom sink, a pluggable output Command) want full control of the loop. So the Agent hand-writes its control loop over LangChain's model + tool primitives.

LangChain is quarantined to the Agent's I/O boundary. Everything upstream (World, `Content`, Inputs, Commands) stays SDK-agnostic. The Agent owns the single adapter that converts WICA `Content` (see [content.md](content.md)) into provider message blocks and back.

### Provider-agnostic model, from config

The model is selected from configuration (provider + model name + params), via LangChain's `init_chat_model`-style construction. Switching providers is a config change, not a code change. Backed by a small settings object (pydantic-style), source TBD (env / file).

### Configurable system prompt

The Agent takes a system prompt from config, so the agent/robot's persona and behavior are customizable per deployment without code changes.

### The Agent owns the event loop; the World stays sync

The World is sync and thread-based (see [world.md](world.md), open question #4): `update()` and its callbacks can be driven from any thread (sensor/hardware/bookkeeping producers). The Agent needs an asyncio event loop for cancellable async Commands. Resolution:

- The **Agent owns exactly one event loop**. Its constructor takes `loop: AbstractEventLoop | None = None`; if `None`, it creates one and runs it in a daemon thread. Injecting aids testing and lets the app own the loop; the default keeps simple cases one-liner-easy.
- The World's (sync) trigger handler is a thin bridge: `asyncio.run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)`. It runs on the World's pool thread, schedules onto the loop, and returns immediately.
- **All task manipulation happens on the loop thread.** The bridge never touches tasks directly; `_handle_trigger` (running on the loop) is where step start / cancel / barge-in live, so cancel-and-restart is race-free. This is the whole reason for wanting the loop.

The World is **not** given the loop (rejected alternative in world.md open question #4).

### History: re-renderable World snapshots, owned by the Agent

History is owned by the Agent, not the World (the World is snapshot-only — `current` + `previous` — and structurally can't hold a growing log).

- **History is an ordered list of WICA's own wrapper objects** over World snapshots (`WorldEntry`/`WorldEntryVersion` copies, which the World already hands out immutably — cheap, safe to keep). The wrapper is ours (not raw LangChain messages) so we retain maximum data and convert to LangChain messages only at call time.
- **Messages are re-rendered from these objects on every LLM call**, not stored as frozen messages. Storing objects (not baked messages) is precisely what enables position-aware rendering (below).
- **Each trigger's snapshot is the full `include_in_prompt` World state, not just the entry that fired.** History holds one bundle per trigger — every currently-included `WorldEntry` (via the World's `get_prompt_entries()`), not only the one whose update caused the trigger — so entries that are `include_in_prompt=True` but never themselves `triggers_llm_call` (passive context) still reach the model. Resolves world.md open question #2 in favor of "full."

### History record shape

History is an ordered list of three record kinds:

| Record | Fields | Notes |
|---|---|---|
| Observation | `entries: list[WorldEntry]` | The full `include_in_prompt` bundle at trigger time (`get_prompt_entries()`), not just the entry that fired (see above) |
| Assistant text | `text: str` | The model's own free-text output for a step — the utterance, under free-text-as-speech (see Open question #2) |
| Command | `call_id: str`, `name: str`, `args: dict[str, Any]` | Written at dispatch time, before the result is known — no `result` field here; the result surfaces later via a subsequent Observation record once the Command's World entry goes terminal, not by mutating this record (code: `CommandRecord`) |

All three are immutable once created; history is append-only — no record is ever edited after creation, including the Command record once its result is known.

**Rendering to messages (resolves Open question #9).** A step's records render as: its Observation → a user message of `<entry>` blocks; then that step's Assistant-text + Command records → a **single** assistant message whose text is the utterance and whose `tool_calls` are the Commands rendered as the model's **native tool calls** (reconstructed from the neutral `CommandRecord` — see [commands.md](commands.md)); then one **`tool_result`** per Command carrying its stringified outcome (a placeholder while still running). Commands are deliberately **not** rendered as `"Calling foo…"` assistant *text* — the model imitated that prose and emitted descriptions as free text instead of real tool calls. Because a Command's outcome rides on its `tool_result`, **any** `agent:command:<call_id>` World entry is skipped when rendering the Observation, so the result isn't duplicated (and a retired entry needn't be re-serialized). The skip is by key prefix, not by state — a completed entry is what matters in the v1 single-in-flight loop (a command is always terminal by the time it re-observes), but skipping every `agent:command:*` key is also what's correct once concurrency lands: an in-flight command already renders as a native `tool_call` awaiting its `tool_result`, so it too must not double as an `<entry>` block.

### Fresh vs. archival rendering

At each call the Agent renders the snapshot list → messages, applying a **freshness policy**: the newest observation is rendered **fresh** (rich, e.g. an image inline as `ImagePart`); older snapshots are rendered **archival** (light, e.g. a short text description). The mechanism lives in the World (`serialize_fn` vs. `archival_serialize_fn`, selected by `render_entry(entry, archival=...)` — see world.md); the **policy** (which entry counts as fresh) lives in the Agent.

Payoffs:
- Heavy multimodal data (images) is sent inline for exactly one turn, then downgraded to text — big token/cost savings for a multimodal robot.
- **Caching still holds.** Each snapshot flips fresh → archival exactly once, always at the tail (the previously-newest one); everything deeper is a pure function of an immutable snapshot, so it renders byte-identically every turn and the deep prefix stays cached. Only the last entry or two re-encode each turn — the same cost a normal growing conversation pays.
- Invariant to preserve: **archival rendering must be a pure function of that snapshot alone** — no "superseded by a later view" awareness, no reordering — or the deep prefix stops being cacheable.

Consequence: `render_full_prompt()` is **not** the Agent's path. The Agent has its own snapshot-history renderer (`snapshots -> list[BaseMessage]`) that is position-aware and multimodal. World rendering primitives (`render_entry`, the `Content` model) are the building blocks it composes; `render_full_prompt` remains a text/debug/full-current-state convenience.

### Commands

The Agent's acting mechanism is **Commands** — WICA's unit of agent action on the World. Their concept, tool backing, execution-as-World-entry lifecycle, generic call description, and the deferred sync/async optimization are all specced in [commands.md](commands.md); this section covers only how the Agent *loop* uses them.

- Commands are registered easily — the Agent exposes `register_command(fn)` over LangChain's `@tool` / `bind_tools`, building the backing tool the model is bound to.
- **Commands are used as-implemented.** The Agent must not require a backing tool to provide any WICA-specific hook (a describe/serialize function, a mode annotation, …). Off-the-shelf LangChain tools work unmodified. Anything WICA needs beyond `name`/`args`/`result` is optional and lives at the *registration/wrapper* layer, never inside the tool.
- Command execution is **async** and **cancellable** (asyncio tasks on the Agent's loop), tracked as a `agent:command:<call_id>` World entry whose value is a `CommandExecution` (see [commands.md](commands.md), "Command execution as a World entry"). A terminal execution triggers the next step. The durable history record is the neutral `CommandRecord` (name/args/`call_id`) plus its result; the model's native `tool_call`/`tool_result` pair is **reconstructed from that neutral record at render time** (see "History → Rendering to messages"), not stored.
- v1 always takes the uniform event-driven path for every Command; the inline fast-path for quick Commands is deferred (see [commands.md](commands.md), "Sync vs. async").

### Concurrency & interruption

Multiple LLM calls may be **in flight at once** — a new trigger starts a new call even while an earlier one is still thinking or speaking. All calls run as coroutines on the Agent's single event loop (cooperative interleaving at `await` points, not OS-thread parallelism); the World→loop bridge (`run_coroutine_threadsafe`) is what lets a thread-dispatched World trigger start one.

**Mutual awareness via World state.** The Agent publishes each in-flight call's live activity as a World entry — e.g. `agent:activity[<task_id>] = {state: thinking | speaking, utterance, cause, …}` — with `include_in_prompt=True` and **`triggers_llm_call=False`** (otherwise activity updates would self-trigger the agent forever). So a newly-started call renders "a call is currently speaking X (spawned by input #42)" and can reason about it. This extends "store execution in the World" (see Commands, and [commands.md](commands.md)) up to the LLM calls themselves.

**Interruption is binary and model-decided.** A new call has exactly two options:
- **cancel the running call and proceed itself**, or
- **do nothing** — yield entirely, producing no output or action.

There is no "defer and speak later." This binary is what keeps action/speech serialized *without* a separate rule: a call that proceeds has necessarily cancelled the other, and a call that yields does nothing — so two calls are never speaking/acting at once **by construction** (this is why "does non-speech action also serialize?" needs no separate answer — it falls out of the binary).

Cancellation is itself a **Command** (`cancel_activity(task_id)`, a WICA-native control Command not backed by an off-the-shelf tool — see [commands.md](commands.md)), run on the loop thread and **id-guarded** so a cancel targeting an already-finished call is a no-op (same pattern as the World's TTL `expected_id` guard).

**Live progress is written to the World by the acting function itself.** The output sink is not a passive consumer: the TTS/output function **updates a World entry as it speaks** (e.g. at sentence boundaries — "currently speaking: …" / "spoken so far: …"), `triggers_llm_call=False`. That live progress is what gives a concurrent call something current to judge against. Generalizes to any long action updating its own progress entry.

**Known limitation — accepted for now: races.** Concurrency is unbounded (as many concurrent calls as arise) and triggers originate on World dispatch threads, so genuine races exist: two concurrently-thinking calls can both decide to "cancel the other and proceed," cancel-cycling or briefly double-speaking. Knowingly accepted at this stage. Mitigating factor: the single loop reduces it to cooperative interleaving, and the World's `RLock` keeps state access safe — races are logical/ordering, not memory corruption. Future tightening (deferred, see open question below): bound concurrency; make "cancel current owner + take the output channel" an atomic critical section between `await`s on the loop; and/or a single-owner output sink as a hard backstop against double-speak.

### Instrumentation: observability hooks

The Agent takes a small family of optional callbacks, all **instrumentation, not control flow**: none can alter what the loop does (return values are ignored), and each is invoked through a single guard that catches and logs any exception so a misbehaving hook can never abort a step. They exist because the interesting moments of a step (what was sent, what fired it, what it decided to do) are otherwise internal to `_run_step`.

| Hook | Signature | Fired |
|---|---|---|
| `on_trigger` | `Callable[[WorldEntry], None]` | at the start of a step, with the World entry whose update caused it — **only for steps that actually run** (a dropped single-in-flight trigger fires nothing). Includes command-completion re-triggers; a consumer that only cares about external inputs filters out `agent:command:*` keys itself |
| `on_prompt` | `Callable[[list[BaseMessage]], None]` | with the exact rendered messages, immediately before each model `ainvoke`. Fresh/archival rendering keeps the deep prefix byte-stable, so a hook logging every prompt sees the same cacheable prefix the provider does |
| `on_command` | `Callable[[str, dict[str, Any]], None]` | with each Command's `(name, args)` at dispatch time, as the model issues it |

The first consumer is the conversation demo (see [conversation-demo.md](conversation-demo.md)): `on_prompt` feeds its prompt panel, `on_trigger` shows inputs on the conversation's input side, and `on_command` shows the robot's actions on the assistant side. They are equally plain debugging aids.

Complementing the hooks, the World and Agent emit **lifecycle logs** under the standard-library `wica.*` loggers, so the whole loop is traceable without wiring any hook: **DEBUG** for per-event detail (registrations, every World update and whether it triggered a call, trigger receipt, step start/complete, LLM output, command dispatch/start/end), **INFO** for coarse lifecycle (agent start/stop, dropped triggers), and **WARNING** for command failures. This is debug tracing, not the project's eventual error-handling story — fire-and-forget listener/trigger exceptions in the World are still swallowed (see [world.md](world.md)).

## Open questions

These are unresolved and several are central. Do not treat the "Settled" split above as covering them.

1. **Command execution model — mostly settled (see "Commands" above and [commands.md](commands.md)); residual details** now tracked in commands.md's open questions: the command-execution entry's failure rendering, and — for the deferred sync path — the auto-latency timeout value and how a single turn emitting **multiple** (parallel) Commands of mixed speed is handled.

2. **Output: explicit `speak` Command vs. free-text-as-speech. (Q5, undecided — but streaming works either way.)** Settled parts: the output sink is pluggable, and it **publishes live progress to the World** as it speaks (see Concurrency & interruption). Streaming to TTS is possible in both models — a Command's arguments stream incrementally too, just as partial JSON. The remaining choice: **(a)** free-text content *is* the utterance (streams trivially by concatenating text deltas; thinking/speaking separation comes from extended-thinking blocks; Commands are for actions only) vs. **(b)** an explicit `speak(text)` Command (explicit control over exactly what's spoken, at the cost of partial-JSON extraction to stream the `text` arg — keep its schema a single string to make that easy). Decide by whether explicit control over spoken-vs-internal text is needed, or "visible text = speech" suffices.

3. **Concurrency races (deferred tightening).** The concurrency/interruption *model* is settled (see Concurrency & interruption): unbounded concurrent calls, binary model-decided interruption (cancel-and-proceed or do-nothing), mutual awareness via `agent:activity` World entries. What's deferred is hardening against the accepted races: whether to **bound** concurrency, make **take-over atomic** (cancel current owner + acquire the output channel in one critical section between `await`s), and/or enforce a **single-owner output sink** as a hard backstop against double-speak. Revisit once the basic loop runs.

4. **Full vs. incremental prompt — resolved.** Committed strategy: re-render from an Agent-owned history of snapshots each call (see Fresh vs. archival above; not a from-scratch `render_full_prompt()` every time), and each snapshot captured per trigger is the *full* `include_in_prompt` World state via `get_prompt_entries()`, not only the entry that fired — see "History" above. World open question #2 updated to match.

5. **Where the freshness/renderer policy is configured.** The World provides fresh/archival serialization per entry; the Agent decides *when* archival applies and how snapshots group into messages. Exact shape of that Agent-side renderer/registry is TBD.

6. **Step termination.** When is a step "done"? No more Commands issued? Output produced? A max-iterations guard? Undefined.

7. **Error handling.** The World swallows callback exceptions (fire-and-forget, a known World gap). The Agent can't inherit that — Command failures need to surface into World state / history so the next run can react ("navigation failed: path blocked"). Needs a real story (ties to the project's pending logging story).

8. **Context-window growth / truncation.** Append-only snapshot history needs an eventual truncation/summarization strategy.

9. **History wrapper object shape — resolved, see "History record shape" above.** The three record kinds and their fields are settled (Observation, Assistant text, Command), and so is how they render into messages: a step's Assistant-text + Command records collapse into one assistant message (text + native `tool_calls`) followed by `tool_result`s — see "Rendering to messages" above.

10. **Single agent for now.** One World trigger handler ⇒ one Agent, matching the World singleton. No multi-agent partitioning yet.

11. **`set_trigger_handler` shape.** May fold into a broader Agent registration API once Commands/Agents are further specced (noted in the World implementation plan).

12. **Generic command-activity listening.** The "one World entry per Command" design (see [commands.md](commands.md), "Command execution as a World entry") is race-free — each call fully owns its own key, so concurrent calls never contend on the same `update()` — but it means a listener can't subscribe to "any Command" in general without already knowing every `call_id` in advance, since `add_listener` needs a concrete key. Sketched fix, not yet decided or built: keep the per-call entries as the race-free source of truth, and additionally register one persistent broadcast entry (e.g. `agent:command_activity`, `include_in_prompt=False`, `triggers_llm_call=False`) that the Agent plain-overwrites (never merges) on every command status change, so a listener can subscribe once and be notified of all command activity without needing `call_id`s ahead of time. A plain overwrite carries no read-modify-write race, unlike a shared list would. Revisit once a concrete consumer needs this.

## Future improvements (deferred out of v1)

The v1 implementation ships a reduced slice on purpose. These are already directionally decided — not open questions — but not yet built, so they don't get lost once v1 ships:

- **Full concurrency & interruption model.** `agent:activity` World entries, model-issued `cancel_activity(task_id)`, and genuinely concurrent in-flight calls (see "Concurrency & interruption" above). v1 ships single-in-flight instead: a new trigger arriving while a call is in flight is dropped and logged, not queued or coalesced — no step is started for it and no history record is created for it. **One exception on cleanup:** a dropped trigger that is a *command completion* still retires its `agent:command:<call_id>` World entry (the Agent doesn't run a reasoning step to react to it, but the entry must not leak — otherwise a completed Command lingers in the World and in every subsequent prompt indefinitely; its effect already lives in whatever other World entries the Command wrote). More generally, the Agent retires **every** terminal command entry it observes on each step, not only the one that fired, so a completion dropped mid-step is still cleaned up on the next step that runs. Revisit once the basic loop runs — this is Open question #3's tightening, plus the concurrency model itself.
- **Sync/async Command optimization.** The inline fast-path for quick Commands (auto-by-latency or a registration flag — see [commands.md](commands.md), "Sync vs. async"). v1 always takes the async/event-driven path for every Command, per that spec's "default execution model" framing.
- **Streaming output + live progress.** Token-level streaming to the output sink, and the sink writing live "currently speaking: …" progress to a World entry as it goes (this only matters once concurrent calls exist, so a competing call has something current to judge against). v1's sink receives one complete string per step.
- **`speak()` Command.** v1 uses free-text-as-speech (Open question #2, resolved *for v1* in this direction). An explicit `speak(text)` Command remains a live alternative if free-visible-text-is-speech turns out too coarse — e.g. needing internal reasoning text that isn't spoken.
- **`AgentConfig` from env/file.** v1 constructs `AgentConfig` directly in code; loading it from a config file or environment variables is unbuilt.
- **History truncation / summarization.** `self._history` is append-only and unbounded in v1 (Open question #8).
- **Agent-level (non-Command) error handling.** A model call itself raising isn't yet given a World-state/history story — only Command failures are (they land in a terminal `CommandExecution` with the error). Ties into the project's broader logging story (Open question #7).
- **Multi-agent partitioning.** Still one `Agent` per one World (Open question #10).
