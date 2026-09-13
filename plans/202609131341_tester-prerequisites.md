# WicaTester prerequisites: World/Agent introspection and `call_id` on `CommandIssued`

**Status:** Done

The three small framework gaps the WicaTester study ([specs/_wica-tester-study.md](../specs/_wica-tester-study.md),
§2 G1–G3, decision §8.7) needs closed before a test harness can wire itself to a `Wica` under test
from the public surface alone. Each is an additive public read or payload field; no behavior of the
loop changes except the *moment* `on_command` fires for a dispatched Command.

## Goal

1. **G1 — World schema enumeration.** `World.keys()` returns every registered key (registration
   order); `World.get_config(key)` returns a copy of the key's `WorldEntryConfig` (`KeyError` if
   unregistered). Both unguarded reads, like `is_registered`.
2. **G2 — Agent command enumeration.** `Agent.commands` is a read-only `Mapping[str, Command]`
   of the currently bound set; `Agent.output_command_name` is `str | None`.
3. **G3 — `call_id` on `CommandIssued`, emitted after registration.** `CommandIssued` gains a
   third field `call_id: str`. For a dispatched Command, `on_command` fires after
   `world.register(agent:command:<call_id>)` and before the initial `running` update, so a
   subscriber can `add_listener` on the key inside the handler and see `running` → terminal.
   For `noop`, `call_id` is the `NoReactionRecord`'s id (no entry exists).
4. Specs [world.md](../specs/world.md), [agent.md](../specs/agent.md), [wica.md](../specs/wica.md)
   describe the above (already edited, `Updated`) and return to `Implemented`; INTEGRATING.md's
   API table matches.

Out of scope: the `wica.testing` subpackage itself (next plan), a broadcast
`agent:command_activity` entry ([commands.md](../specs/commands.md) OQ#4 stays open), exposing
`Agent._history`.

## Steps

1. **World** ([src/wica/world.py](../src/wica/world.py)): add `keys()` and `get_config()` next to
   `is_registered`. Tests in `tests/test_world.py`: `keys()` reflects register/unregister and
   order; `get_config()` returns the declared type/flags and a copy (mutating the returned config
   does not change what `get_prompt_snapshot()` reports); `KeyError` on an unknown key.
2. **Agent** ([src/wica/agent.py](../src/wica/agent.py)): `commands` property
   (`types.MappingProxyType(self._commands)`), `output_command_name` property. Tests in
   `tests/test_agent.py`: after `register_command` + `start()` the mapping holds the application
   Command, `noop`, `cancel_command`, and the output Command; the mapping rejects item assignment;
   `output_command_name` is `None` / the name.
3. **`CommandIssued.call_id`**: add the field; in `_run_step` emit `CommandIssued(noop, {}, call_id)`;
   in `_dispatch_command` move the emit to after `self._world.register(...)` and before the
   `running` update, passing `call_id`. Update the existing equality assertions in
   `tests/test_agent.py`, `tests/test_wica.py`, `tests/test_conversation_demo.py`. New test: an
   `on_command` subscriber calls `world.add_listener(f"agent:command:{c.call_id}", …)` inside the
   handler and the listener receives the `running` version then the `complete` one; for `noop`
   the `call_id` has no registered entry.
4. Flip `world.md`/`agent.md`/`wica.md` and their index rows back to `Implemented`; this plan and
   its index row to `Done`.

Verification after each code step: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
`uv run pytest`; finally `uv run pytest tests-e2e -k "fake or example"`.
