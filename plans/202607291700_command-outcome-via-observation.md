# Command outcome delivered via its World-entry observation

**Status:** Done

Implements the reworked command rendering in [specs/commands.md](../specs/commands.md)
("Command execution as a World entry") and [specs/agent.md](../specs/agent.md)
("History → Rendering to messages"): a Command's `tool_result` becomes a **fixed ack** and its
**status/outcome is delivered by the `agent:command:<call_id>` World entry rendered as an
observation**, at the causally-correct point in history.

## Motivation

Bug (see [specs/_todo.md](../specs/_todo.md), first line): *"if dance then say hello during dance,
does not seem to know dance is not finished."* Trace:

- A long-running Command (`dance`) is dispatched as a background task and the step returns, so
  `_busy` clears while the command is still running. A new input (`say hello`) therefore starts a
  fresh step *while `dance` is in flight*.
- In that step, [_render_messages](../src/wica/agent.py#L368) (a) **skips** the running
  `agent:command:*` entry (skip-by-prefix) and (b) renders the earlier `dance` `tool_call` paired
  with a `tool_result` of `"(in progress)"`. A `tool_result` being present reads to the model as
  *"the call returned"*, and `"(in progress)"` looks like its return value — so the model concludes
  `dance` finished and says hello as if it were done.

Root cause: the outcome was riding the `tool_result`, which the provider pins to the Command's
*dispatch* site and which cannot express "still running." The fix moves the outcome onto the
Command's World entry (observed at the point it happens) and reduces the `tool_result` to a fixed
pointer at that entry.

## Design

Mirrors the settled spec decisions:

1. **`World.render_entry` gains an optional `serialize_fn` override.** When provided, it is used for
   the body and the config lookup is skipped entirely, so a caller can render an entry whose key has
   since been **unregistered**. The `<entry …>…Updated…</entry>` envelope stays owned by the World
   (single source of truth); only the body function is swapped. Signature:
   `render_entry(entry, *, archival=False, serialize_fn=None)`.

2. **The `tool_result` becomes a fixed ack.** A module helper `_command_ack(call_id)` returns a
   constant pointer, e.g. `"Dispatched. Live status and result appear in the World state as entry
   agent:command:<call_id>."` `flush_assistant` emits this for every reconstructed `tool_call`,
   regardless of the command's state. It no longer reads any per-call result.

3. **Command entries are rendered into the observation, never skipped.** In `_render_messages`, the
   observation loop drops the `startswith("agent:command:")` skip. Command entries render through
   `render_entry(world_entry, archival=archival, serialize_fn=_serialize_command_execution)` — the
   Agent's own command serializer — so the World stays command-agnostic and a retired entry still
   re-renders from its history snapshot. Normal entries keep going through the registered path
   (`render_entry(world_entry, archival=archival)`), unchanged. The command-key test reuses a
   module constant `_COMMAND_KEY_PREFIX = "agent:command:"` (also used by `_dispatch_command`).

4. **`_command_results` is removed** — the dict, its `__init__` initialization, and the three
   assignments in `_run_command`. The outcome now flows via the World snapshot captured in the
   Observation record, so nothing reads it anymore.

5. **Running serialization reads clearly as in-progress.** Keep `_serialize_command_execution`; its
   `running` branch (`"Calling <call>…"`) already conveys in-flight — optionally tighten to make
   the running state explicit (e.g. `"Calling <call>… (running)"`). Terminal branches already
   render `Called <call> → <result>` / `→ failed: <error>` / `→ cancelled`.

6. **Dropped completions stay in current context until observed (invariant: no completed Command is
   ever lost).** [_handle_trigger](../src/wica/agent.py#L252) still drops and logs the trigger, but
   **no longer** eagerly retires the entry — the `_cleanup_command_entry` call is removed from the
   dropped-completion branch (and `_is_terminal_command` is now used only by `_append_observation`).
   The terminal entry stays as current World state and is **rendered into history and then retired**
   by the next step that observes it (`_append_observation` already retires every terminal command
   entry it observes; command entries are now also rendered on the way out). So a completed Command
   is always present either in history (observed) or in current World state (pending observation); if
   the agent goes idle forever after a dropped completion, the entry simply persists as current
   state. Documented in agent.md "Future improvements".

No change to `CommandExecution`, the trigger/cleanup machinery, `CommandRecord`, or the
snapshot-history model.

## Steps

1. **world.py** — add the `serialize_fn` override to `render_entry` (skip config lookup when
   provided; keep the envelope). 
2. **agent.py** — add `_COMMAND_KEY_PREFIX` and `_command_ack`; use the prefix constant in
   `_dispatch_command`.
3. **agent.py** — `_render_messages`: `flush_assistant` emits `_command_ack(...)` for each call;
   observation loop renders command entries via the `serialize_fn` override instead of skipping.
4. **agent.py** — remove `_command_results` (field init + `_run_command` assignments).
5. Adjust `_serialize_command_execution` running-branch wording if tightening (step D5).

## Tests

- **Update** [tests/test_agent.py](../tests/test_agent.py):
  - `test_...tool_result...` (the `add` test): `tool_result` content is now the ack, not `"3"`;
    assert the ack, and assert the completed command renders in the observation
    (`Called add(a=1, b=2) → 3`).
  - `test_parallel_tool_calls_independent_keys_and_mixed_status_line`: both `tool_result`s are acks;
    assert the observation contains `fast_tool`'s result and `slow_tool`'s **running** line; keep the
    `fast` retired / `slow` still-running World assertions.
  - `test_past_commands_render_as_native_tool_calls_not_prose`: confirm still passes (native
    `tool_calls`, no `"Calling …"` prose in assistant text).
  - `test_dropped_command_completion_is_still_cleaned_up`: **reframe** — the dropped completion's
    entry now **persists** as current World state (no longer eagerly retired). Assert it is still
    registered after the busy step, then that a subsequent step **observes it, renders its outcome
    into that step's prompt, and retires it** (in history-or-current-context at all times).
- **New** regression test (the reported bug): dispatch a still-running command, fire a second input
  trigger to start a concurrent step, and assert that step's prompt renders the running command as
  an observation `<entry>` (in-progress text) and its `tool_result` is the ack — i.e. the model is
  told the command is *not* finished.
- **New** test: after a command completes and its entry is retired, a later step's prompt still
  renders the completed command entry at its historical position (no `KeyError` from the
  unregistered config) — exercises the `render_entry` override on a retired entry.
- **New/adjust** [tests/test_world.py](../tests/test_world.py): `render_entry` with a `serialize_fn`
  override uses it (and works after `unregister`).
- Check [tests-e2e/test_agent.py](../tests-e2e/test_agent.py) for any assertion on `tool_result`
  content and update to the ack + observation model.

## Verification

`uv run ruff check .`, `uv run pyright`, `uv run pytest` — all green before marking Done.
