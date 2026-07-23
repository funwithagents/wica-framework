# Agent

**Status:** Draft

## Purpose

The Agent is the reasoning loop that observes the World and acts on it. It is triggered by the World (via `set_trigger_handler`), builds an LLM prompt from World state, runs inference against a configurable provider, executes tools, and streams output to a pluggable sink. It owns the conversation **history**; the World owns current **state**.

This spec is an early draft: the sections under "Settled" reflect decisions already made in design discussion; "Open questions" holds everything still being brainstormed — several are load-bearing and not yet resolved.

## Settled

### LangChain as a primitive, not a framework

The Agent uses LangChain for two things only: the **provider-agnostic chat model** abstraction and **tool schemas / binding**. It does **not** use a prebuilt agent loop (LangGraph's `create_react_agent` or similar) — WICA already owns state (World) and history, and the requirements below (cancellation, storing tool execution in the World, streaming to a custom sink, a pluggable output tool) want full control of the loop. So the Agent hand-writes its control loop over LangChain's model + tool primitives.

LangChain is quarantined to the Agent's I/O boundary. Everything upstream (World, `Content`, Inputs, Commands) stays SDK-agnostic. The Agent owns the single adapter that converts WICA `Content` (see [content.md](content.md)) into provider message blocks and back.

### Provider-agnostic model, from config

The model is selected from configuration (provider + model name + params), via LangChain's `init_chat_model`-style construction. Switching providers is a config change, not a code change. Backed by a small settings object (pydantic-style), source TBD (env / file).

### Configurable system prompt

The Agent takes a system prompt from config, so the agent/robot's persona and behavior are customizable per deployment without code changes.

### The Agent owns the event loop; the World stays sync

The World is sync and thread-based (see [world.md](world.md), open question #4): `update()` and its callbacks can be driven from any thread (sensor/hardware/bookkeeping producers). The Agent needs an asyncio event loop for cancellable async tools. Resolution:

- The **Agent owns exactly one event loop**. Its constructor takes `loop: AbstractEventLoop | None = None`; if `None`, it creates one and runs it in a daemon thread. Injecting aids testing and lets the app own the loop; the default keeps simple cases one-liner-easy.
- The World's (sync) trigger handler is a thin bridge: `asyncio.run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)`. It runs on the World's pool thread, schedules onto the loop, and returns immediately.
- **All task manipulation happens on the loop thread.** The bridge never touches tasks directly; `_handle_trigger` (running on the loop) is where step start / cancel / barge-in live, so cancel-and-restart is race-free. This is the whole reason for wanting the loop.

The World is **not** given the loop (rejected alternative in world.md open question #4).

### History: re-renderable World snapshots, owned by the Agent

History is owned by the Agent, not the World (the World is snapshot-only — `current` + `previous` — and structurally can't hold a growing log).

- **History is an ordered list of WICA's own wrapper objects** over World snapshots (`WorldEntry`/`WorldEntryVersion` copies, which the World already hands out immutably — cheap, safe to keep). The wrapper is ours (not raw LangChain messages) so we retain maximum data and convert to LangChain messages only at call time.
- **Messages are re-rendered from these objects on every LLM call**, not stored as frozen messages. Storing objects (not baked messages) is precisely what enables position-aware rendering (below).

### Fresh vs. archival rendering

At each call the Agent renders the snapshot list → messages, applying a **freshness policy**: the newest observation is rendered **fresh** (rich, e.g. an image inline as `ImagePart`); older snapshots are rendered **archival** (light, e.g. a short text description). The mechanism lives in the World (`serialize_fn` vs. `archival_serialize_fn`, selected by `render_entry(entry, archival=...)` — see world.md); the **policy** (which entry counts as fresh) lives in the Agent.

Payoffs:
- Heavy multimodal data (images) is sent inline for exactly one turn, then downgraded to text — big token/cost savings for a multimodal robot.
- **Caching still holds.** Each snapshot flips fresh → archival exactly once, always at the tail (the previously-newest one); everything deeper is a pure function of an immutable snapshot, so it renders byte-identically every turn and the deep prefix stays cached. Only the last entry or two re-encode each turn — the same cost a normal growing conversation pays.
- Invariant to preserve: **archival rendering must be a pure function of that snapshot alone** — no "superseded by a later view" awareness, no reordering — or the deep prefix stops being cacheable.

Consequence: `render_full_prompt()` is **not** the Agent's path. The Agent has its own snapshot-history renderer (`snapshots -> list[BaseMessage]`) that is position-aware and multimodal. World rendering primitives (`render_entry`, the `Content` model) are the building blocks it composes; `render_full_prompt` remains a text/debug/full-current-state convenience.

### Tools

- Tools are registered easily (LangChain `@tool` / `bind_tools`) — the Agent exposes a simple register API over that.
- Tool execution is **async** and **cancellable** (asyncio tasks on the Agent's loop).
- **Tools are used as-implemented.** The Agent must not require a tool to provide any WICA-specific hook (a describe/serialize function, a mode annotation, …). Off-the-shelf LangChain tools work unmodified. Anything WICA needs beyond `name`/`args`/`result` is optional and lives at the *registration/wrapper* layer, never inside the tool.

#### Generic call description (no tool cooperation)

The Agent always has the tool's **name**, its **args** (from the tool call), and its **result** (stringified). So it can always render a plain-text description of a call — `Called get_calendar(date=today) → 3 meetings` — with zero tool cooperation. This generic default is what lands in history; a per-registration override (`render_result`) is *optional sugar* for a nicer summary, never required.

Consequence for history: the durable record of a tool call is this **description**, not the raw `tool_use`/`tool_result` messages. The native pair, if used at all, is a within-step transport detail (below), discarded when the step ends; history is snapshots + descriptions.

#### Tool lifecycle as a World entry (event-driven)

Tool execution is **stored in the World** — an in-flight tool is genuinely *current state* and fits the World's snapshot model. The lifecycle maps directly onto existing World primitives, with **no new machinery**:

- Register the tool-status entry with `triggers_llm_call=True` **and** `trigger_condition_fn=lambda old, new: new.is_terminal()`. Start/progress updates land in the entry (visible to any reader) but **fail the condition → no trigger**; only **complete/failure** triggers a new Agent step.
- Mark it `include_in_prompt=True` so a *concurrent* step renders "navigate → running (task 7)" and can decide to cancel it before it finishes.
- On completion, the model continues via the **reconstruct path**: it sees its own `"calling tool xxx…"` line in history *plus* the result in world state, and picks up from there. (This is not the native `tool_use`/`tool_result` mechanism — that's dropped on this path by design; the history-text line is what keeps it coherent.)

This is the **default execution model for all tools**: uniform, event-driven, needs no per-tool sync/async declaration, and keeps the "full world resend" cheap because fresh/archival + caching means only the completion entry (and any inputs that arrived meanwhile) miss cache.

#### Sync vs. async (deferred optimization)

Both sync and async are two inferences (a second pass is always needed to use a result), so async carries **no extra-inference penalty**. The only real difference: a fast tool *could* be handled inline (append `tool_result` to the frozen step array, native thread preserved) instead of ending the step and re-observing the whole world. This is a **latency optimization, not a behavioral change**, and is **deferred** — the uniform event-driven model above ships first.

When added, the sync/async decision is made **without annotating the tool**, via either (or both):
- **(a) Auto by latency (preferred):** `await` the tool inline with a short timeout (~1–2s). Returns in time → handle inline (native pair, sync). Exceeds it → detach: convert to the history-text line, register the running-tool world entry, end the step, resume on completion (async). The tool's *actual speed* decides, per invocation — no declaration.
- **(b) Optional registration flag:** `register_tool(fn, mode="auto" | "sync" | "async")`, default `auto`. A one-word wrapper flag (not a tool change) to pin known cases ("walk_to" always async, "add" always sync).

### Concurrency & interruption

Multiple LLM calls may be **in flight at once** — a new trigger starts a new call even while an earlier one is still thinking or speaking. All calls run as coroutines on the Agent's single event loop (cooperative interleaving at `await` points, not OS-thread parallelism); the World→loop bridge (`run_coroutine_threadsafe`) is what lets a thread-dispatched World trigger start one.

**Mutual awareness via World state.** The Agent publishes each in-flight call's live activity as a World entry — e.g. `agent:activity[<task_id>] = {state: thinking | speaking, utterance, cause, …}` — with `include_in_prompt=True` and **`triggers_llm_call=False`** (otherwise activity updates would self-trigger the agent forever). So a newly-started call renders "a call is currently speaking X (spawned by input #42)" and can reason about it. This extends "store execution in the World" (see Tools) up to the LLM calls themselves.

**Interruption is binary and model-decided.** A new call has exactly two options:
- **cancel the running call and proceed itself**, or
- **do nothing** — yield entirely, producing no output or action.

There is no "defer and speak later." This binary is what keeps action/speech serialized *without* a separate rule: a call that proceeds has necessarily cancelled the other, and a call that yields does nothing — so two calls are never speaking/acting at once **by construction** (this is why "does non-speech action also serialize?" needs no separate answer — it falls out of the binary).

Cancellation is a **model-issued command** (`cancel_activity(task_id)`), run on the loop thread and **id-guarded** so a cancel targeting an already-finished call is a no-op (same pattern as the World's TTL `expected_id` guard).

**Live progress is written to the World by the acting function itself.** The output sink is not a passive consumer: the TTS/output function **updates a World entry as it speaks** (e.g. at sentence boundaries — "currently speaking: …" / "spoken so far: …"), `triggers_llm_call=False`. That live progress is what gives a concurrent call something current to judge against. Generalizes to any long action updating its own progress entry.

**Known limitation — accepted for now: races.** Concurrency is unbounded (as many concurrent calls as arise) and triggers originate on World dispatch threads, so genuine races exist: two concurrently-thinking calls can both decide to "cancel the other and proceed," cancel-cycling or briefly double-speaking. Knowingly accepted at this stage. Mitigating factor: the single loop reduces it to cooperative interleaving, and the World's `RLock` keeps state access safe — races are logical/ordering, not memory corruption. Future tightening (deferred, see open question below): bound concurrency; make "cancel current owner + take the output channel" an atomic critical section between `await`s on the loop; and/or a single-owner output sink as a hard backstop against double-speak.

## Open questions

These are unresolved and several are central. Do not treat the "Settled" split above as covering them.

1. **Tool execution model — mostly settled (see "Tools" above); residual details:** the tool-status entry's value model and terminal-state predicate (`running` / `complete` / `failed`, what `is_terminal()` checks); how a *failure* renders in both the running entry and the history description; and, for the deferred sync path, the auto-latency timeout value and how a single turn emitting **multiple** (parallel) tool calls of mixed speed is handled (detach all vs. return results for the fast ones + acks for the slow).

2. **Output: explicit `speak` tool vs. free-text-as-speech. (Q5, undecided — but streaming works either way.)** Settled parts: the output sink is pluggable, and it **publishes live progress to the World** as it speaks (see Concurrency & interruption). Streaming to TTS is possible in both models — a tool call's arguments stream incrementally too, just as partial JSON. The remaining choice: **(a)** free-text content *is* the utterance (streams trivially by concatenating text deltas; thinking/speaking separation comes from extended-thinking blocks; tools are for actions only) vs. **(b)** an explicit `speak(text)` tool (explicit control over exactly what's spoken, at the cost of partial-JSON extraction to stream the `text` arg — keep its schema a single string to make that easy). Decide by whether explicit control over spoken-vs-internal text is needed, or "visible text = speech" suffices.

3. **Concurrency races (deferred tightening).** The concurrency/interruption *model* is settled (see Concurrency & interruption): unbounded concurrent calls, binary model-decided interruption (cancel-and-proceed or do-nothing), mutual awareness via `agent:activity` World entries. What's deferred is hardening against the accepted races: whether to **bound** concurrency, make **take-over atomic** (cancel current owner + acquire the output channel in one critical section between `await`s), and/or enforce a **single-owner output sink** as a hard backstop against double-speak. Revisit once the basic loop runs.

4. **Full vs. incremental prompt.** Effectively leaning to "re-render from snapshots each call" (see Fresh vs. archival above), which supersedes the earlier full-`render_full_prompt`-each-time idea. Confirm this is the committed strategy and update World open question #2 accordingly once locked.

5. **Where the freshness/renderer policy is configured.** The World provides fresh/archival serialization per entry; the Agent decides *when* archival applies and how snapshots group into messages. Exact shape of that Agent-side renderer/registry is TBD.

6. **Step termination.** When is a step "done"? No more tool calls? Output produced? A max-iterations guard? Undefined.

7. **Error handling.** The World swallows callback exceptions (fire-and-forget, a known World gap). The Agent can't inherit that — tool failures need to surface into World state / history so the next run can react ("navigation failed: path blocked"). Needs a real story (ties to the project's pending logging story).

8. **Context-window growth / truncation.** Append-only snapshot history needs an eventual truncation/summarization strategy.

9. **History wrapper object shape.** The concrete fields of the Agent's own history wrapper are still undefined, but the kinds of record are now clearer: (a) **World snapshots** (observations), (b) the Agent's own **assistant text** turns, and (c) **tool-call descriptions** (the generic `Called xxx(args) → result` text, per "Tools" above — not raw `tool_use`/`tool_result`). Tool activity is no longer a gap: it's captured both as the history description and, for anything run via the World-entry lifecycle, as a snapshot. Remaining: the exact record union/fields and how assistant text turns interleave with snapshots at render time.

10. **Single agent for now.** One World trigger handler ⇒ one Agent, matching the World singleton. No multi-agent partitioning yet.

11. **`set_trigger_handler` shape.** May fold into a broader Agent registration API once Commands/Agents are further specced (noted in the World implementation plan).
