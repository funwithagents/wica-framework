# Gradio conversation demo (first runnable example)

**Status:** In progress

## Goal

Implement the conversation demo specced in [specs/conversation-demo.md](../specs/conversation-demo.md): the first runnable
example that lets someone *see* WICA work end-to-end — type "speech" at an agent, watch it reason
over the World, issue Commands, and reply, with the live World state and the exact prompt sent to
the model visible alongside. It doubles as a manual test harness for the framework
(World → Inputs → Commands → Agent all exercised through the public API).

This plan covers the build; the product/UX (surfaces, robot persona, sensor inputs, robot actions)
lives in [specs/conversation-demo.md](../specs/conversation-demo.md) and is not restated here.

Non-goal: exercising v1's deferred features (concurrency/interruption, streaming). v1 Agent is
single-in-flight and hands the sink one complete string per step; the demo lives within that.

## Decisions (from brainstorming)

- **LLM backend:** real provider via `AgentConfig`, provider/model/key from env
  (`WICA_PROVIDER` default `anthropic`, `WICA_MODEL` default `claude-sonnet-5`,
  key via a WICA-namespaced var `WICA_<PROVIDER>_API_KEY`, e.g. `WICA_ANTHROPIC_API_KEY`, routed
  into the provider's standard env var internally so it can't collide with an ambient key). If the key is missing the
  app still **launches** and shows a clear "set your key" banner instead of crashing — nothing
  that talks to the model runs until a key is present.
- **Prompt visibility:** add a small optional `on_prompt` debug hook to the `Agent` (a real,
  reusable library change — Agent spec is Draft), fired just before `ainvoke`. The demo renders
  the captured messages into a read-only panel.
- **World panel:** raw structured values only (key, id, type, value, updated-at) — not the
  rendered prompt Content.
- **Packaging:** new top-level `examples/` dir; `gradio` in a new `demo` uv dependency group
  (not core deps). Run with `uv run --group demo python examples/conversation_demo.py`.

## Library change: `Agent.on_prompt` hook

Add `on_prompt: Callable[[list[BaseMessage]], None] | None = None` to `Agent.__init__` (and pass
through `from_config`'s `**kwargs`). In `_run_step`, immediately before
`await self._bound_model.ainvoke(messages)`, call `self._on_prompt(messages)` if set. Guard it so
a raising hook can't break the step (log and continue) — the hook is debug instrumentation, not
control flow.

- **Spec:** update [specs/agent.md](../specs/agent.md) — add a short bullet under "Settled" (or a
  small "Instrumentation" note) documenting the hook as an optional observability seam. Agent spec
  is **Draft**, so no `Stable → Updated` dance; just keep it accurate.
- **Tests:** add to `tests/test_agent.py` a test asserting the hook fires once per step with the
  same messages the model receives (drive via the existing fake-model test setup), and that a
  hook raising doesn't abort the step.

## The demo (`examples/conversation_demo.py`)

### World entries the demo registers at startup

| Key | Type | in_prompt | triggers_llm | Role |
|---|---|---|---|---|
| `speech_input` | `str` | yes | **yes** | User's spoken utterance. Send box → `update()`. serialize: `User said: "…"` |
| `closest_user` | `str` | yes | **yes** | Nearest detected user id, or `None` when gone. Buttons → `update()`. serialize: `Closest user: <id>` / `No user nearby` when `None` |
| `emotion` | `str` | yes | no | Set by the `set_emotion` Command. serialize: `Current emotion: <e>` |
| `tracked_users` | `list` | yes | no | Ids currently being tracked. Managed by the tracking Commands. serialize: `Tracking users: [...]` / `Not tracking anyone` |
| `active_tracked_user` | `str` | yes | no | The one tracked user in focus (for `switch`). |

`triggers_llm_call=False` on Command-written entries (`emotion`, tracking) so the agent's own
actions don't re-trigger it into a loop — matches the `agent:activity` reasoning in agent.md.
`None`-valued entries still render (that's how "no user nearby" / "not tracking" reach the model).

### Commands registered on the Agent (the fake robot actions)

- `dance() -> str` — **`await asyncio.sleep(10)`** to simulate a robot dancing for ~10s, then
  returns a short confirmation (`"Did a little dance."`). No World mutation of its own, but this is
  the demo's best showcase of the async Command lifecycle: the `agent:command:<id>` entry sits in
  `running` (visible live in the World panel) for 10s, then its `running → complete` transition
  triggers a follow-up agent step. A concurrent speech input during those 10s is *dropped* by v1's
  single-in-flight loop (expected — worth noting in the UI so it doesn't look like a bug).
- `set_emotion(emotion: str) -> str` — `update("emotion", emotion)`.
- `start_user_tracking(user_id: str) -> str` — add id to `tracked_users` (idempotent).
- `stop_user_tracking(user_id: str) -> str` — remove id; clear `active_tracked_user` if it was it.
- `switch_user_tracking(user_id: str) -> str` — set `active_tracked_user` (adding to
  `tracked_users` if not already tracked).

Tracking Commands read-modify-write the `tracked_users` list via `get`/`update`. Single-in-flight
v1 + all mutations on the agent loop thread means no contention in practice.

### Inputs surfaced as buttons

- **Closest user detected**: a small id textbox + "Detect" button → `update("closest_user", id)`.
- **Clear closest user**: button → `update("closest_user", None)`.

(These are the `triggers_llm_call=True` sensor-style inputs; the send box is the speech input.)

### Gradio UI (Blocks — needs side panels, so not `ChatInterface`)

- **Conversation** (`gr.Chatbot`) + speech input textbox + Send.
- **World state** panel — raw values, one row per demo-registered key. `gr.Dataframe` or JSON.
- **Prompt** panel — read-only textbox showing the last messages captured via `on_prompt`,
  flattened to readable text (role + `Content.to_string()`-style block dump).
- **Input buttons** — the closest-user controls above.

### Wiring UI ↔ Agent (thread bridge, no asyncio in Gradio)

The Agent owns its own loop in a daemon thread; the World is thread-safe. So Gradio stays sync:

- `output_sink` (async, runs on the agent loop) pushes each reply string onto a thread-safe
  `queue.Queue`. A `gr.Timer(~0.4s)` drains it into the Chatbot.
- `on_prompt` stores the latest messages in a shared holder (lock-guarded); the same timer
  refreshes the Prompt panel.
- The same timer re-reads the demo's known World keys via `world.get_entry(key)` and repaints the
  World panel. (No "list all entries" API exists; the demo knows its own keys, which is enough.)
- Sending speech / pressing input buttons just calls `world.update(...)` synchronously from the
  Gradio handler; the trigger fires the agent on its loop thread.

### Missing-key handling

On startup, detect whether the provider key is set. If not, render a banner and disable Send /
input buttons (or let them no-op with a notice) so the app is explorable but never calls a model
without credentials.

## Project-map / index bookkeeping

- Add the `examples/` row to the **Top-level layout** table in [AGENTS.md](../AGENTS.md) (new root
  dir — the project-map discipline requires it; `tests/test_project_map.py` enforces root-dir
  coverage).
- Add this plan's row to [plans/_index.md](_index.md); flip to **Done** once verified.

## Verification

- ✅ `uv run ruff check .`, `uv run pyright`, `uv run pytest` all pass (53 tests, incl. two new
  `on_prompt` tests). `examples/` is linted by ruff but kept out of pyright's `include` (it depends
  on `gradio`, a non-core dep) — the reusable `on_prompt` hook is what's unit-tested, not the UI.
- ✅ Headless smoke test (no key): module imports, the World bridge (`on_send`/`on_detect`/`tick`)
  updates state correctly, and `build_ui()` builds the Blocks — the no-key path stays explorable.
- ⏳ **Remaining — live run.** `uv run --group demo python examples/conversation_demo.py` with a
  real `WICA_ANTHROPIC_API_KEY` — send speech, confirm a reply, a Command firing (watch the 10s
  `dance` sit `running` in the World panel), and the Prompt panel updating. Not run here: no
  provider key is available in this environment. This is the last step before flipping to **Done**.

Once the live run is confirmed, set this plan **Done** and promote
[specs/conversation-demo.md](../specs/conversation-demo.md) from **Draft** to
**Stable** (its status rule: Stable requires a Done plan).
