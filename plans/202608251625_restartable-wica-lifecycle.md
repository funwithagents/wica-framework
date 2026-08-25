# Restartable Wica lifecycle

**Status:** Done

Implements the restartable lifecycle update in [specs/wica.md](../specs/wica.md), with the matching
World and Agent lifecycle behavior in [specs/world.md](../specs/world.md) and
[specs/agent.md](../specs/agent.md).

## Goal

Support repeated `wica.start()` → `wica.stop()` cycles on one `Wica` while preserving the owned
World, Agent, registrations, instrumentation Events, and Agent history. An owned asyncio loop stays
open across reversible stops; every start creates a fresh daemon thread for that loop. A new
terminal `wica.close()` stops the system and closes an owned loop when the instance will not be used
again.

## Steps

1. Change the facade lifecycle state from one-shot `_stopped` to explicit running/closed states.
   Make `start()` and `stop()` idempotent, create a fresh owned thread per running cycle, retain the
   shared loop across `stop()`, and add idempotent terminal `close()`.
2. Make Agent stop cancel every trigger, reasoning, and Command task, draining cancellation before
   an owned loop is stopped so pending work cannot be frozen into a later cycle.
3. Make World restart restore TTL behavior for retained non-`None` values, and make its lifecycle
   transition atomic with concurrent updates.
4. Add functional tests for two complete cycles, idempotent lifecycle calls, terminal close, task
   cancellation, and TTL behavior across a pause.
5. Update consumer documentation and final teardown call sites to use `close()` where no restart is
   intended.

## Verification

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest`
- `uv run pytest tests-e2e -k fake`

## Status bookkeeping

- `wica.md`, `world.md`, and `agent.md`: Implemented → Updated while implementation is in
  progress, then back to Implemented after verification.
- This plan: In progress → Done after verification.
