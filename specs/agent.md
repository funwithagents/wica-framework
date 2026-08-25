---
code:
  - src/wica/agent.py
tests:
  - tests/test_agent.py
---

# Agent

**Status:** Implemented

## Purpose

The Agent is the reasoning loop that observes the World and acts on it. It is triggered by the World (by subscribing to the World's `on_trigger` Event — see "The shared event loop"), builds an LLM prompt from World state, runs inference against a configurable provider, executes **Commands** (see [commands.md](commands.md)), and streams output to a pluggable sink. It owns the conversation **history**; the World owns current **state**.

The v1 design is settled and shipped: the sections under "Settled" describe what the Agent does today (single-in-flight loop, trigger coalescing, Command execution as World state, config-driven provider construction). "Open questions" now holds genuine **deferrals** — post-v1 directions (full concurrency/interruption, streaming, error-handling and truncation stories) tracked so they aren't lost — not unresolved blockers to the current implementation.

## Settled

### LangChain as a primitive, not a framework

The Agent uses LangChain for two things only: the **provider-agnostic chat model** abstraction and **tool schemas / binding** (the primitive WICA **Commands** are built on — see [commands.md](commands.md)). It does **not** use a prebuilt agent loop (LangGraph's `create_react_agent` or similar) — WICA already owns state (World) and history, and the requirements below (cancellation, storing Command execution in the World, streaming to a custom sink, a pluggable output Command) want full control of the loop. So the Agent hand-writes its control loop over LangChain's model + tool primitives.

LangChain is quarantined to the Agent's I/O boundary. Everything upstream (World, `Content`, Inputs, Commands) stays SDK-agnostic. The Agent owns the single adapter that converts WICA `Content` (see [content.md](content.md)) into provider message blocks and back.

### Provider-agnostic model, from config

The model is selected from configuration (provider + model name + params), via LangChain's `init_chat_model`-style construction. Switching providers is a config change, not a code change. Backed by a small settings object (`AgentConfig`), loaded from a JSON framework config — see [config.md](config.md).

**`AgentConfig` is unresolved data; construction is where its references resolve.** A loaded `AgentConfig` mirrors the JSON and holds `api_key`/`api_key_env` and `system_prompt`/`system_prompt_file` verbatim — no env or file read has happened. **`Agent.__init__` takes the `AgentConfig` directly** and performs those reads at build: it calls `resolve_system_prompt(config)` (prompt-file read) and `build_chat_model(config)` (which calls `resolve_api_key(config)` — env-var read, `MissingEnvError` if unset — then constructs the LangChain model). Both resolvers are pure functions owned by [config.md](config.md); the Agent is just the consumer that invokes them — so an unset key env or unreadable prompt surfaces at `Agent(config)`, not at config load. Config-driven construction *is* the constructor: one build path, which the `Wica` facade drives as `Agent(config.agent, world=…, …)` (see [wica.md](wica.md)). A deterministic model for tests is selected through config too, via the `fake` provider (see [fake-provider.md](fake-provider.md)).

**One additive seam: an optional `model=` override.** `Agent.__init__` accepts an optional `model: BaseChatModel | None` — when a bespoke model is passed, it's used verbatim instead of calling `build_chat_model` (the system prompt is still config-expressed). This is the raw-model injection seam: for a model no config can express, and what the Agent's own unit tests use to drive the loop over a fully-scripted double the scripted `fake` provider can't express (blocking model calls, arbitrary tool-call sequences). `Wica.init` deliberately does **not** surface it — the facade stays config-only; the override lives at the `Agent` layer for direct construction.

**Most providers construct via `init_chat_model`; `huggingface-hub` is the exception.** `anthropic`, `openai`, and other LangChain-supported providers are built by `init_chat_model(model, model_provider=provider, …)`. `huggingface-hub` — the Hugging Face Hub's serverless Inference Providers — takes a dedicated branch in `build_chat_model` that constructs `ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id=model, provider=<hf_provider>, huggingfacehub_api_token=<resolved api_key>, …))` (config surface — the `hf_provider` field, the extras — in [config.md](config.md), "Providers"). Because this branch builds the model directly instead of going through `init_chat_model`, it also owns the kwarg mapping: the key that `resolve_api_key(config)` returns (from `api_key`/`api_key_env`) is forwarded as `huggingfacehub_api_token`, not the generic `api_key` the `init_chat_model` providers receive. It must **not** use `init_chat_model`'s builtin `huggingface` provider, which builds a *local `transformers` pipeline*: that path is a blocking, in-process generation call, and running it under `ainvoke` offloads it to a thread pool the Agent's `task.cancel()` cannot stop — so a cancelled or barged-in step would leave a zombie generation burning compute, violating the cancellation guarantee this loop is built on (see "The shared event loop"). WICA uses the hosted endpoint only. (Even the hosted path's cancellation quality depends on `langchain-huggingface` implementing a truly-async `_agenerate`; that's an implementation verification item, not a design change.)

This is still one adapter at the Agent's I/O boundary, not a leak upstream: the `huggingface-hub` branch is just one more provider-construction detail inside `build_chat_model`. Everything above the Agent stays SDK-agnostic regardless of provider.

### Configurable system prompt

The Agent takes a system prompt from config, so the agent/robot's persona and behavior are customizable per deployment without code changes. The config may carry it inline (`system_prompt`) or as a file reference (`system_prompt_file`); `Agent.__init__` resolves the two to a single string via `resolve_system_prompt(config)` at build (see [config.md](config.md), "System prompt"), so the loop never sees the file indirection — internally it works with a plain resolved `system_prompt: str`.

### The shared event loop

The Agent's Commands are cancellable async tasks, so the Agent runs on an asyncio event **loop**. That loop is **owned and run by the `Wica` facade** (in one daemon thread) and **injected into both the Agent and the World** — one loop for the whole system (see [wica.md](wica.md), "Lifecycle"; [world.md](world.md), "The shared event loop"). The Agent's constructor takes the `loop`; it does not create or own a thread of its own, and `stop()` does not tear the loop down (that's `Wica`'s job).

- **All task manipulation happens on the loop thread.** Step start / cancel / barge-in, the coalescing window, and every Command task live on this one loop, so cancel-and-restart is race-free without locks — the whole reason the loop exists.
- **The Agent is just a subscriber to the World's trigger — no thread-to-loop bridge.** The Agent subscribes to the World's `on_trigger` Event (see [world.md](world.md), "The trigger — the `on_trigger` Event"), attaching a one-line **sync shim** that schedules its async work: `world.on_trigger.subscribe(lambda e: self._loop.create_task(self._handle_trigger(e)))`. The World emits `on_trigger` on the loop thread (`call_soon_threadsafe`), so the shim runs there and `create_task` is safe — no `run_coroutine_threadsafe`. The shim is the whole bridge, and other observers can subscribe to the same raw signal alongside the Agent.
- **The World is handed the same loop**, rather than owning a `ThreadPoolExecutor` bridged to a separate Agent loop — this is what makes the system symmetric and single-owner, and it resolves [world.md](world.md) open question #3. `update()` is still callable from any thread; the World hops to the loop via `call_soon_threadsafe` internally.

### History: re-renderable World snapshots, owned by the Agent

History is owned by the Agent, not the World (the World is snapshot-only — `current` + `previous` — and structurally can't hold a growing log).

- **History is an ordered list of WICA's own wrapper objects** over World snapshots (`WorldEntry`/`WorldEntryVersion` copies, which the World already hands out immutably — cheap, safe to keep). The wrapper is ours (not raw LangChain messages) so we retain maximum data and convert to LangChain messages only at call time.
- **Messages are re-rendered from these objects on every LLM call**, not stored as frozen messages. Storing objects (not baked messages) is precisely what enables position-aware rendering (below).
- **Each trigger's snapshot is the full `include_in_prompt` World state, not just the entry that fired.** History holds one bundle per trigger — every currently-included `WorldEntry` (via the World's `get_prompt_entries()`), not only the one whose update caused the trigger — so entries that are `include_in_prompt=True` but never themselves `triggers_llm_call` (passive context) still reach the model. Resolves world.md open question #1 in favor of "full."

### History record shape

History is an ordered list of three record kinds:

| Record | Fields | Notes |
|---|---|---|
| Observation | `entries: list[WorldEntry]` | The full `include_in_prompt` bundle at trigger time (`get_prompt_entries()`), not just the entry that fired (see above) |
| Assistant text | `text: str` | The model's own free-text output for a step — the utterance, under free-text-as-speech (see Open question #2) |
| Command | `call_id: str`, `name: str`, `args: dict[str, Any]` | Written at dispatch time, before the result is known — no `result` field here; the result surfaces later via a subsequent Observation record once the Command's World entry goes terminal, not by mutating this record (code: `CommandRecord`) |

All three are immutable once created; history is append-only — no record is ever edited after creation, including the Command record once its result is known.

**Rendering to messages (resolves Open question #9).** A step's records render as: its Observation → a user message of `<entry>` blocks; then that step's Assistant-text + Command records → a **single** assistant message whose text is the utterance and whose `tool_calls` are the Commands rendered as the model's **native tool calls** (reconstructed from the neutral `CommandRecord` — see [commands.md](commands.md)); then one **`tool_result`** per Command carrying a **fixed acknowledgement** that points at the Command's World entry (`agent:command:<call_id>`) — **not** its outcome. Commands are deliberately **not** rendered as `"Calling foo…"` assistant *text* — a model shown its own actions as prose imitates it, emitting descriptions as free text instead of issuing real tool calls.

A `tool_result` is pinned immediately after its `tool_use` (a `tool_use` can't be left dangling across turns), so it can only sit at the Command's *dispatch* site — not where the Command actually finished. Putting the outcome there would misplace it, and while a Command is still in flight a `tool_result` present at all reads as "the call returned." So the outcome is carried **not** by the `tool_result` but by the Command's World entry, rendered into the Observation like any other entry: `running` while in flight, terminal (`result`/`error`) at the step where the completion is observed — the causally-correct point. The `agent:command:<call_id>` entry is therefore **never skipped**. Because the World must not know what a `CommandExecution` is, the Agent renders these entries with its own command serializer, passed via `render_entry`'s `serialize_fn` override — which also lets a **retired** command entry re-render from its history snapshot after the World has unregistered it.

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
- Command execution is **async** and **cancellable** (asyncio tasks on the Agent's loop), tracked as a `agent:command:<call_id>` World entry whose value is a `CommandExecution` (see [commands.md](commands.md), "Command execution as a World entry"). A terminal execution triggers the next step. The durable history record of the *action* is the neutral `CommandRecord` (name/args/`call_id`); the *outcome* is the command-execution entry's own `CommandExecution` snapshot, captured in a later Observation. At render time the Agent reconstructs the native `tool_call` from the `CommandRecord`, pairs it with a **fixed-ack** `tool_result`, and lets the outcome render from the World-entry snapshot (see "History → Rendering to messages"); none of the provider-native blocks are stored.
- v1 always takes the uniform event-driven path for every Command; the inline fast-path for quick Commands is deferred (see [commands.md](commands.md), "Sync vs. async").
- The Agent **auto-registers one WICA-native control Command, `cancel_command(call_id)`** (see [commands.md](commands.md), "`cancel_command`"), so the model can abort a Command it previously issued that is still `running`. It targets the command by its `call_id` — already on screen in the `<entry key="agent:command:<call_id>" …>` envelope, so no extra rendering — and reuses the Agent's existing loop-thread cancellation path (the same one `stop()` drives via `Agent.cancel_command`); an already-finished/unknown target is a harmless no-op. This is the v1-buildable sibling of the deferred `cancel_reaction(reaction_id)`, which cancels an in-flight *reaction* rather than a Command (see "Concurrency & interruption").
- **Tool-calling support is not verified.** Because Commands *are* native tool calls, they only work against a provider/model that actually supports tool calling — universal on `openai`, but model-and-backend-dependent on `huggingface-hub`. WICA does **no** capability probe: a model that can't do tool calls either hard-rejects (a provider error surfaces from `ainvoke` — the pre-existing agent-level error gap, Open question #7) or silently emits no Commands. v1 relies on these runtime signals rather than pre-flight detection.

### Trigger coalescing

Perceptions often arrive in bursts — several Inputs updating within a few milliseconds (a speech event plus a proximity change, or a camera frame plus its detection). Firing one LLM call per trigger would waste calls, and under v1's single-in-flight loop it's worse than wasteful: the first trigger's step snapshots the World *before* the rest of the burst has landed, and the later triggers are then **dropped** as "busy" — their changes lost until some unrelated later trigger happens to fire. So the Agent coalesces a burst into one step.

- **A fixed leading-edge window.** The first trigger (when the Agent isn't already busy) opens a **coalescing window** — an `asyncio` timer on the loop thread, default **200 ms**, configurable via the Agent's `coalesce_window` (seconds). Triggers arriving while the window is open **join** it but do **not** extend it — so latency is bounded by the window and a fast stream can never starve the step (rejecting classic reset-on-each-trigger debounce for exactly that reason). When the window closes, **exactly one** step runs; because a step already snapshots the full `include_in_prompt` World (`get_prompt_entries()` — see "History"), every entry that changed during the window is observed together, for free.
- **`coalesce_window=0` disables it** — each trigger fires immediately, exactly the pre-coalescing behavior. This isn't a separate code path to reason about: a zero window is simply a window that closes at once.
- **Per-entry bypass.** An entry registered `bypass_coalescing=True` (see [world.md](world.md)) skips the wait: an update to it closes the current window **early**, running the step now and carrying along any triggers already collected in the window. For a stop/panic button or a barge-in utterance, the ~200 ms tax isn't worth paying. The Agent reads the flag off the `WorldEntry` snapshot the trigger handler is handed (it's carried there like `type` — see [world.md](world.md)).
- **Coalescing only batches the arrival burst *before* a step starts.** A trigger — bypass or not — arriving **while a step is already in flight** is still **dropped**, exactly as v1 does today (see "Concurrency & interruption" and "Future improvements"). Bypass skips the *window*, not the *in-flight drop*. Collecting triggers that arrive *during* a running step and coalescing them into a follow-up step is the concurrency/queue question deferred below; the two compose but are separate.
- **All window manipulation happens on the loop thread.** The window timer is opened, joined, flushed, and cancelled only on the single shared event loop — the same invariant that makes cancellation race-free (see "The shared event loop"). The World schedules `_handle_trigger` onto that loop when a trigger fires; the window logic lives inside it.
- **`agent.on_trigger` fires once per collected trigger.** Even though a coalesced burst runs a single step, the Agent's `on_trigger` instrumentation Event fires **once for each trigger** that joined the window, so a consumer like the demo still shows every processed input on the conversation's input side (see Instrumentation). The other per-step artifacts — the Observation record, `on_prompt`, and the model call — happen **once** for the coalesced step. (This is the *filtered* trigger view — the World's `on_trigger`, which the Agent subscribes to, is the raw one that also fires for triggers dropped while busy.)

### Concurrency & interruption

Multiple LLM calls may be **in flight at once** — a new trigger starts a new call even while an earlier one is still thinking or speaking. All calls run as coroutines on the single shared event loop (cooperative interleaving at `await` points, not OS-thread parallelism); the World scheduling the Agent's async `_handle_trigger` onto that loop (`create_task`) is what lets a thread-dispatched World trigger start one.

**Mutual awareness via World state.** The Agent publishes each in-flight reaction's live state as a World entry — e.g. `agent:reaction[<reaction_id>] = {state: thinking | speaking, utterance, cause, …}` — with `include_in_prompt=True` and **`triggers_llm_call=False`** (otherwise reaction updates would self-trigger the agent forever). So a newly-started reaction renders "a reaction is currently speaking X (spawned by input #42)" and can reason about it. This extends "store execution in the World" (see Commands, and [commands.md](commands.md)) up to the LLM calls themselves.

**Interruption is binary and model-decided.** A new call has exactly two options:
- **cancel the running call and proceed itself**, or
- **do nothing** — yield entirely, producing no output or action.

There is no "defer and speak later." This binary is what keeps action/speech serialized *without* a separate rule: a call that proceeds has necessarily cancelled the other, and a call that yields does nothing — so two calls are never speaking/acting at once **by construction** (this is why "does non-speech action also serialize?" needs no separate answer — it falls out of the binary).

Cancellation is itself a **Command** (`cancel_reaction(reaction_id)`, a WICA-native control Command not backed by an off-the-shelf tool — see [commands.md](commands.md)), run on the loop thread and **id-guarded** so a cancel targeting an already-finished call is a no-op (same pattern as the World's TTL `expected_id` guard).

**Live progress is written to the World by the acting function itself.** The output sink is not a passive consumer: the TTS/output function **updates a World entry as it speaks** (e.g. at sentence boundaries — "currently speaking: …" / "spoken so far: …"), `triggers_llm_call=False`. That live progress is what gives a concurrent call something current to judge against. Generalizes to any long action updating its own progress entry.

**Known limitation — accepted for now: races.** Concurrency is unbounded (as many concurrent calls as arise) and triggers originate on World dispatch threads, so genuine races exist: two concurrently-thinking calls can both decide to "cancel the other and proceed," cancel-cycling or briefly double-speaking. Knowingly accepted at this stage. Mitigating factor: the single loop reduces it to cooperative interleaving, and the World's `RLock` keeps state access safe — races are logical/ordering, not memory corruption. Future tightening (deferred, see open question below): bound concurrency; make "cancel current owner + take the output channel" an atomic critical section between `await`s on the loop; and/or a single-owner output sink as a hard backstop against double-speak.

### Instrumentation: observability Events

The Agent exposes a small family of **`Event`s** (see [events.md](events.md)), all **instrumentation, not control flow**: subscribing changes nothing about what the loop does, and `Event.emit`'s per-subscriber isolation catches and logs a raising subscriber so it can neither abort a step nor starve sibling subscribers. They exist because the interesting moments of a step (what was sent, what fired it, what it decided to do) are otherwise internal to `_run_step`. The Agent **emits on them internally** at the right points — there is no separate hook-callback parameter and no `_fire_hook` guard; the Event *is* the seam, and any number of consumers subscribe.

| Event | Type | Emitted |
|---|---|---|
| `on_trigger` | `Event[WorldEntry]` | **once per trigger that a run-to-completion step observes**, with the World entry whose update caused it. A burst of *N* triggers coalesced into one step (see "Trigger coalescing") emits *N* times but runs a single step; a dropped single-in-flight trigger emits nothing. Includes command-completion re-triggers; a consumer that only cares about external inputs filters out `agent:command:*` keys itself. This is the **filtered** trigger view — distinct from the World's raw `on_trigger` the Agent subscribes to (see "The shared event loop") |
| `on_prompt` | `Event[list[BaseMessage]]` | with the exact rendered messages, immediately before each model `ainvoke`. Fresh/archival rendering keeps the deep prefix byte-stable, so a subscriber logging every prompt sees the same cacheable prefix the provider does |
| `on_command` | `Event[CommandIssued]` | with each Command the model issues, at dispatch time, as a `CommandIssued(name, args)` value (a small frozen dataclass carrying what a bare `(name, args)` tuple would, but named and evolvable) |

The first consumer is the conversation demo (see [conversation-demo.md](conversation-demo.md)): `on_prompt` feeds its prompt panel, `on_trigger` shows processed inputs on the conversation's input side, and `on_command` shows the robot's actions on the assistant side. They are equally plain debugging aids.

**The `Wica` facade surfaces these Events directly** (see [wica.md](wica.md)) — `wica.on_agent_trigger`, `wica.on_agent_prompt`, `wica.on_agent_command` *are* the Agent's Events, plus `wica.on_world_trigger` for the World's raw one. No adapters: because each is already an `Event`, `Wica` just re-exposes the object, and multiple consumers (demo panels, a logger, a metrics sink) subscribe to the same signal. `on_prompt` carrying `list[BaseMessage]` (a LangChain type) is natural here — the Agent *is* the framework's LangChain I/O boundary — and it does not leak into the `Event` primitive, which stays a pure project-agnostic leaf ([events.md](events.md)).

Complementing the hooks, the World and Agent emit **lifecycle logs** under the standard-library `wica.*` loggers, so the whole loop is traceable without wiring any hook: **DEBUG** for per-event detail (registrations, every World update and whether it triggered a call, trigger receipt, step start/complete, LLM output, command dispatch/start/end), **INFO** for coarse lifecycle (agent start/stop, dropped triggers), and **WARNING** for command failures. This is debug tracing, not the project's eventual error-handling story — fire-and-forget listener/trigger exceptions in the World are still swallowed (see [world.md](world.md)).

## Open questions

These are deferrals beyond v1, not blockers to what's built — the "Settled" split above covers the current implementation. Several point at the bigger post-v1 concurrency/interruption and error-handling work.

1. **Command execution model — mostly settled (see "Commands" above and [commands.md](commands.md)); residual details** now tracked in commands.md's open questions: the command-execution entry's failure rendering, and — for the deferred sync path — the auto-latency timeout value and how a single turn emitting **multiple** (parallel) Commands of mixed speed is handled.

2. **Output: explicit `speak` Command vs. free-text-as-speech. (Q5, undecided — but streaming works either way.)** Settled parts: the output sink is pluggable, and it **publishes live progress to the World** as it speaks (see Concurrency & interruption). Streaming to TTS is possible in both models — a Command's arguments stream incrementally too, just as partial JSON. The remaining choice: **(a)** free-text content *is* the utterance (streams trivially by concatenating text deltas; thinking/speaking separation comes from extended-thinking blocks; Commands are for actions only) vs. **(b)** an explicit `speak(text)` Command (explicit control over exactly what's spoken, at the cost of partial-JSON extraction to stream the `text` arg — keep its schema a single string to make that easy). Decide by whether explicit control over spoken-vs-internal text is needed, or "visible text = speech" suffices.

3. **Concurrency races (deferred tightening).** The concurrency/interruption *model* is settled (see Concurrency & interruption): unbounded concurrent calls, binary model-decided interruption (cancel-and-proceed or do-nothing), mutual awareness via `agent:reaction` World entries. What's deferred is hardening against the accepted races: whether to **bound** concurrency, make **take-over atomic** (cancel current owner + acquire the output channel in one critical section between `await`s), and/or enforce a **single-owner output sink** as a hard backstop against double-speak. Revisit once the basic loop runs.

4. **Full vs. incremental prompt — resolved.** Committed strategy: re-render from an Agent-owned history of snapshots each call (see Fresh vs. archival above; not a from-scratch `render_full_prompt()` every time), and each snapshot captured per trigger is the *full* `include_in_prompt` World state via `get_prompt_entries()`, not only the entry that fired — see "History" above. World open question #1 updated to match.

5. **Where the freshness/renderer policy is configured.** The World provides fresh/archival serialization per entry; the Agent decides *when* archival applies and how snapshots group into messages. Exact shape of that Agent-side renderer/registry is TBD.

6. **Step termination.** When is a step "done"? No more Commands issued? Output produced? A max-iterations guard? Undefined.

7. **Error handling.** The World swallows callback exceptions (fire-and-forget, a known World gap). The Agent can't inherit that — Command failures need to surface into World state / history so the next run can react ("navigation failed: path blocked"). Needs a real story (ties to the project's pending logging story).

8. **Context-window growth / truncation.** Append-only snapshot history needs an eventual truncation/summarization strategy.

9. **History wrapper object shape — resolved, see "History record shape" above.** The three record kinds and their fields are settled (Observation, Assistant text, Command), and so is how they render into messages: a step's Assistant-text + Command records collapse into one assistant message (text + native `tool_calls`) followed by **fixed-ack** `tool_result`s — the outcome rides the Command's World-entry observation, not the `tool_result` — see "Rendering to messages" above.

10. **Single agent for now.** One World trigger handler ⇒ one Agent, matching the World singleton. No multi-agent partitioning yet.

11. **World trigger wiring — resolved.** The World exposes its trigger as an `on_trigger: Event[WorldEntry]` and the Agent is simply a subscriber (via an async-scheduling shim) — see "The shared event loop" and [world.md](world.md). A single subscriber still matches the one-agent-per-World assumption (#10), but the Event admits additional observers with no API change.

12. **Generic command-activity listening.** The "one World entry per Command" design (see [commands.md](commands.md), "Command execution as a World entry") is race-free — each call fully owns its own key, so concurrent calls never contend on the same `update()` — but it means a listener can't subscribe to "any Command" in general without already knowing every `call_id` in advance, since `add_listener` needs a concrete key. Sketched fix, not yet decided or built: keep the per-call entries as the race-free source of truth, and additionally register one persistent broadcast entry (e.g. `agent:command_activity`, `include_in_prompt=False`, `triggers_llm_call=False`) that the Agent plain-overwrites (never merges) on every command status change, so a listener can subscribe once and be notified of all command activity without needing `call_id`s ahead of time. A plain overwrite carries no read-modify-write race, unlike a shared list would. Revisit once a concrete consumer needs this.

## Future improvements (deferred out of v1)

The v1 implementation ships a reduced slice on purpose. These are already directionally decided — not open questions — but not yet built, so they don't get lost once v1 ships:

- **Full concurrency & interruption model.** `agent:reaction` World entries, model-issued `cancel_reaction(reaction_id)`, and genuinely concurrent in-flight calls (see "Concurrency & interruption" above). v1 ships single-in-flight instead: a new trigger arriving *while a call is in flight* is dropped and logged, not queued or coalesced — no step is started for it and no history record is created for it. (This is specifically the *in-flight* case; the pre-step **coalescing window** — see "Trigger coalescing" — does batch a burst that arrives *before* a step starts, and ships in v1.) **A dropped *command completion* is not reacted to, but its `agent:command:<call_id>` entry is left in place** — *not* eagerly retired. It stays part of current World state and is **rendered into history and only then retired** by the next step that observes it (the Agent renders **and** retires *every* terminal command entry it observes, not only the one that fired). This preserves a load-bearing invariant: **every completed Command is always present either in history (once a step has observed its entry) or in the current World state (while its completion is still pending observation) — a completed Command is never silently lost.** If the agent goes idle forever right after a dropped completion, that terminal entry simply persists as current state until some later step looks at it; that's the invariant working, not a leak. Revisit once the basic loop runs — this is Open question #3's tightening, plus the concurrency model itself.
- **Sync/async Command optimization.** The inline fast-path for quick Commands (auto-by-latency or a registration flag — see [commands.md](commands.md), "Sync vs. async"). v1 always takes the async/event-driven path for every Command, per that spec's "default execution model" framing.
- **Streaming output + live progress.** Token-level streaming to the output sink, and the sink writing live "currently speaking: …" progress to a World entry as it goes (this only matters once concurrent calls exist, so a competing call has something current to judge against). v1's sink receives one complete string per step.
- **`speak()` Command.** v1 uses free-text-as-speech (Open question #2, resolved *for v1* in this direction). An explicit `speak(text)` Command remains a live alternative if free-visible-text-is-speech turns out too coarse — e.g. needing internal reasoning text that isn't spoken.
- **History truncation / summarization.** `self._history` is append-only and unbounded in v1 (Open question #8).
- **Agent-level (non-Command) error handling.** A model call itself raising isn't yet given a World-state/history story — only Command failures are (they land in a terminal `CommandExecution` with the error). Ties into the project's broader logging story (Open question #7).
- **Multi-agent partitioning.** Still one `Agent` per one World (Open question #10).
