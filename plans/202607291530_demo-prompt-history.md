# Demo prompt history dropdown

**Status:** Done

Implements the updated Prompt surface in [specs/conversation-demo.md](../specs/conversation-demo.md):
the demo keeps every reasoning step's prompt (not just the last), exposes them through a dropdown
labelled by time + trigger, and auto-snaps the view to the newest when a new step runs.

## Motivation

Today [examples/conversation_demo.py](../examples/conversation_demo.py) keeps a single
`_latest_prompt` string that `on_prompt` overwrites each step, so the user can only ever see the
most recent prompt. Watching how successive World states turn into successive prompts — the whole
teaching point of the panel — is impossible once a second step runs.

## Design

- **History store.** Replace the `_latest_prompt` string with `_prompts: list[dict[str, str]]`,
  each `{"label": ..., "text": ...}`, guarded by the existing `_state_lock`. Never mutate an entry
  in place (append-only) so `tick`/`on_select_prompt` can read `text` outside the lock.
- **Labelling.** `on_trigger` and `on_prompt` fire back-to-back per step on the agent loop
  ([agent.py:274-277](../src/wica/agent.py#L274-L277)), so `on_trigger` stashes the current trigger
  description into a module global `_last_trigger_label`, and `on_prompt` reads it to build
  `"HH:MM:SS — <trigger>"`. `on_trigger` stashes for **every** step (including command-completion
  re-triggers, which don't reach the events queue) so the label is never stale. `_describe_trigger`
  gains an `agent:command:` branch (`⚡ command finished: <name>`) so those steps read cleanly.
- **Snap-to-newest.** A module global `_last_shown_count` tracks how many prompts the UI has shown.
  `tick` compares it to `len(_prompts)`: on growth it returns `gr.update(choices=…, value=newest)`
  for the dropdown and `gr.update(value=<newest text>)` for the textbox; otherwise it returns bare
  `gr.update()` for both, leaving the user's current selection/scroll untouched between steps.
- **Manual browse.** A `gr.Dropdown` (choices are `(label, index)` pairs, value = index) sits above
  the read-only `prompt_view` textbox. Its `.change` handler `on_select_prompt(index)` returns the
  selected prompt's text. `tick` wiring above means a new step overrides any manual selection.

## Steps

1. Swap `_latest_prompt` → `_prompts` list + `_last_trigger_label` / `_last_shown_count` globals.
2. Update `on_trigger` (stash label, keep the command-key early-return for the events queue) and
   `_describe_trigger` (command branch).
3. Rewrite `on_prompt` to append `{label, text}` with a timestamp.
4. Add `on_select_prompt`; extend `tick`'s return to the dropdown + textbox updates.
5. Add the `gr.Dropdown` to the layout, wire `.change` and the timer's extra outputs.

## Verification

- `uv run ruff check .`, `uv run pyright`, `uv run pytest` all pass (demo has no automated tests;
  the Agent `on_prompt` hook tests in `tests/test_agent.py` are unaffected).
- Manual: run the demo, send several utterances / sensor events, confirm each step's prompt appears
  in the dropdown labelled by time+trigger, the view snaps to the newest on each new step, and older
  prompts are browsable while the robot is idle.
