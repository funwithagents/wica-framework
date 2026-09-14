# Conversation demo: generic transcript with live Command state, and a Speaking panel

**Status:** Done

Implements the [specs/conversation-demo.md](../specs/conversation-demo.md) update of 2026-09-14
("What the user sees" 1.–2., "A reusable transcript"): the conversation surface becomes a
**generic, reusable transcript** whose only application-specific knowledge enters through one
`display_entry(entry)` hook; every Command item shows its **execution state live** (in progress →
complete/failed/cancelled, with result/error and duration); and the simulated TTS's word-by-word
progress moves out of the transcript into a dedicated **Speaking panel** under it.

No framework change: the transcript follows a Command through the seam added by plan
[202609131341](202609131341_tester-prerequisites.md) — `on_agent_command` fires with a `call_id`
after the `agent:command:<call_id>` entry is registered, so the handler can `add_listener` on it.
`wica.contrib`/[gradio-contrib.md](../specs/gradio-contrib.md) are deliberately untouched (the
transcript is refactored *inside the demo*; lifting it is a later decision).

## Goal

1. **`examples/conversation_demo/transcript.py` — `TranscriptLog`**, the generic presenter:
   - `EntryDisplay(label, detail=None)` and the hook type `DisplayEntry = Callable[[WorldEntry],
     EntryDisplay | None]`; `None` falls back to the generic default (`⚡ key = value`; for a
     `CommandExecution` value `🦾 name` / `name(args)`, `🗣️ name` when it is the output Command).
   - Handlers for the four framework signals: `on_trigger(entry)` (input side, via the hook; a
     Command-completion re-trigger is not shown but labels the step), `on_prompt(messages)` (prompt
     history + opens the reaction group), `on_command(issued)` (creates the Command item in issue
     order under the current reaction, `status: "pending"`; `🚫 noop` for `noop`; attaches an
     **async** World listener on `agent:command:<call_id>` once attached to a `Wica`),
     `on_command_update(entry)` (the listener: re-renders the item through the hook, and on a
     terminal state removes `status` — not `"done"`, which would make Gradio auto-collapse the
     item — and adds a ✅/❌/⏹ suffix, result/error, and `duration`), and
     `output_sink(text)` (💭 item).
   - `attach(wica)`: binds the World (for listeners), reads `wica.agent.output_command_name`,
     subscribes the three Events. The application still passes `log.output_sink` to `Wica.init`.
   - `snapshot()` / `prompt_text()` as before, with the transcript signature covering content
     **and** title/status so in-place state edits reach the UI.
2. **`app_state.py` — `DemoState`** becomes the demo-specific composition: a `TranscriptLog` built
   with the demo's hook, plus `Speaking`/`SpeakingSlot` — the current `say`'s text, words spoken
   so far, and state (`speaking`/`complete`/`cancelled`) behind a lock; `read()` for the UI.
3. **`app.py`**: `display_entry` next to the `serialize_fn`s (speech, closest user, `say`);
   `say()` drives the speaking slot only (marks `cancelled` on `CancelledError` and re-raises,
   `complete` after the last word) and no longer touches the transcript; `build_app` attaches the
   log.
4. **`app_ui.py`**: a **Speaking** `gr.HTML` panel under the chatbot (spoken words vs. remaining,
   `spoken/total`, state badge), fed by the conversation tick; intro text updated.
5. **Tests**: `tests/test_conversation_demo.py` covers the default and custom hook, the Command
   item lifecycle (issued → running → complete / failed / cancelled, noop, duration/status), the
   signature change, the speaking slot, and `say()` cancellation; `tests-e2e/test_example_flow.py`
   asserts the `say` item's title/detail/status and the speaking slot, plus a **barge-in** flow (a
   second input while `say` runs → scripted `cancel_command` → the `say` item and the slot read
   cancelled).
6. Spec [conversation-demo.md](../specs/conversation-demo.md) back to `Implemented` (file + index).

## Steps

1. [x] Spec edit (status `Updated`, index row).
2. [x] `transcript.py`: `EntryDisplay`, `DisplayEntry`, `TranscriptLog`, prompt rendering helpers
   moved from `app_state.py`.
3. [x] `app_state.py`: `Speaking` / `SpeakingSlot`, `DemoState` composition.
4. [x] `app.py`: `display_entry`, `say()` over the slot, `build_app` wiring.
5. [x] `app_ui.py`: Speaking panel + tick output, signature-driven chatbot push, intro text.
6. [x] Tests (unit + e2e), then `ruff check`, `ruff format`, `pyright`, `pytest`,
   `pytest tests-e2e -k "fake or example"`.
7. [x] Spec → `Implemented`; plan → `Done`; both indexes.

## Verification

`uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`,
`uv run pytest tests-e2e -k "fake or example"` all pass.
