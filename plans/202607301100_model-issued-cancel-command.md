# Model-issued `cancel_command`

**Status:** Done

Implements the [commands.md](../specs/commands.md) "`cancel_command`" section (and the matching bullet in [agent.md](../specs/agent.md), "Commands"): expose the Agent's existing loop-thread command cancellation to the **model** as a WICA-native, auto-registered Command, so the LLM can abort a Command it previously issued that is still `running`.

## Background

The Agent already has a Python-facing cancel path — `Agent.cancel_command(call_id)` → `_cancel_command_on_loop`, which cancels the asyncio task; the task's `_run_command` catches `CancelledError`, marks the `agent:command:<call_id>` entry `cancelled`, and that terminal update re-triggers a step. It's used by `stop()` and tests but is **not** bound to the model. This plan promotes it to a model-issued Command.

Even under v1's single-in-flight loop, this is useful: a step can dispatch several async Commands and then go idle; a later trigger (a new input, or a completion) starts a step where the model observes the others still `running` and can decide to abort one.

## Design decisions (settled with the user)

- **No new identifier / no render change.** The target is named by its `call_id`, already visible in the World's `<entry key="agent:command:<call_id>" …>` envelope. The command body render (`_serialize_command_execution`) is left untouched.
- **Lenient argument.** The handler accepts the bare `call_id` *or* the full `agent:command:<call_id>` key (strips the prefix), so a verbatim copy of the envelope key works.
- **Auto-registered by the Agent**, not app-wired — it needs Agent internals. Always present (WICA-native control action).
- **Async handler** so it executes on the Agent's loop thread (via `command.ainvoke`), keeping `_running_tasks` access race-free. A sync tool would run in a thread-pool off the loop.
- **Scope:** `cancel_command` only. The deferred `cancel_reaction` (cancelling in-flight *LLM calls*) is a separate topic, unbuildable until the concurrency model lands.

## Steps

1. **`src/wica/agent.py` — constants.** Add `_CANCEL_COMMAND_NAME = "cancel_command"` and a `_CANCEL_COMMAND_DESCRIPTION` (tells the model to pass the `call_id` — the part after `agent:command:` in the running command's entry key — and that cancelling a finished/unknown command is a no-op).

2. **`_cancel_command_on_loop` returns a bool.** Change it to return whether it actually cancelled a running task (`False` when the task is missing or already `done()`), so the tool result can report the outcome. `stop()`'s call ignores the return; `cancel_command()`'s Python API is unaffected.

3. **`_cancel_command_action(self, call_id: str) -> str` (async method).** Strip an optional `agent:command:` prefix, call `_cancel_command_on_loop`, and return `"cancelling <call_id>"` or a `"<call_id> is not a running command …"` message. Give it a docstring.

4. **Auto-register in `__init__`.** After the command-state fields are initialized, call `self.register_command(self._cancel_command_action, name=_CANCEL_COMMAND_NAME, description=_CANCEL_COMMAND_DESCRIPTION)` so it's bound to the model on construction (before any user commands; `register_command` re-binds on every registration, so order is irrelevant to binding).

## Tests (`tests/test_agent.py`)

- **Model aborts a running Command end-to-end.** Register a `block_forever` user Command. Script the FakeChatModel: step 1 dispatches `block_forever`; a second input trigger drives step 2 where the model issues `cancel_command(call_id=<target>)`; assert the target entry reaches `state == "cancelled"` and a later prompt's observation renders `Called block_forever() → cancelled`, and the run reaches a final text output.
- **Lenient on the full key.** `cancel_command(call_id="agent:command:<id>")` cancels the same target as the bare id.
- **No-op on unknown/finished id.** Issuing `cancel_command` for an id with no running task returns the "not a running command" result and raises nothing; the loop continues.

## Verification

`uv run ruff check .`, `uv run pyright`, `uv run pytest` all green. Then flip this plan's status (here and in [plans/_index.md](_index.md)) to `Done`.
