---
code:
  - src/wica/events.py
tests:
  - tests/test_events.py
---

# Event (pub/sub primitive)

**Status:** Implemented

## Purpose

A tiny, **generic, project-agnostic** publish/subscribe primitive — `Event[T]` in `events.py`. Its whole job is to let one producer *publish* a value and any number of consumers *subscribe* to receive it, decoupled from each other. It knows **nothing** about whatever it is embedded in: no framework types, no domain concepts, only the standard library. That deliberate ignorance is the point — it's the kind of leaf you lift into another project unchanged.

Boundaries: this spec owns only the primitive. Who *uses* it and what flows through it are entirely up to a consumer — `Event[T]` is the mechanism; a consumer decides the payloads and the wiring. Nothing in this spec references any particular consumer, and nothing here would change if the file were copied into an unrelated project.

## Decided

### The type: `Event[T]`

A generic signal carrying a single value of type `T`:

| Method | Role |
|---|---|
| `subscribe(handler: Callable[[T], None])` | Register `handler` to be called on every future `emit`. |
| `unsubscribe(handler)` | Remove a previously subscribed `handler` (raises `ValueError` if it was never subscribed). |
| `emit(value: T)` | Call every subscribed handler with `value`, **in subscription order**. |

- **Generic over the payload.** `Event[T]` is parameterized so the payload type is checked end-to-end (pyright): `Event[str]` delivers `str`s, `Event[None]` is a pure "it happened" signal. A producer that owns several signals gives each its own concrete `Event[...]`.
- **Single value per `emit`.** One positional payload, not `*args` — so `T` names exactly what a handler receives. A signal that needs to convey several things carries them as one composite value (a tuple / dataclass), which keeps the generic parameter honest. `Event[None]` covers the no-payload case.

### Semantics

- **Synchronous, inline dispatch.** `emit` calls each handler directly, in the order they subscribed, and returns once they've all run. There is no queue, no scheduling, no thread of its own — the *caller's* thread runs the handlers. Crossing threads, if a consumer needs it, is the consumer's concern, not this primitive's.
- **Snapshot iteration.** `emit` iterates a copy of the handler list, so a handler may `subscribe`/`unsubscribe` (itself or another) *during* dispatch without corrupting the in-progress round: the set that fires this round is fixed when `emit` begins; changes take effect next round.
- **Stateless beyond its handlers.** An `Event` holds only its handler list — it carries a value to subscribers and stores nothing. It does not replay the last value to a late subscriber; a subscriber sees only `emit`s after it subscribed.
- **No error isolation (deliberate, for now).** A handler that raises propagates out of `emit` (and later handlers in that round don't run). Handlers are expected to be small and non-throwing; swallowing/aggregating handler errors is an [open question](#open-questions), added only if a real consumer needs it.

### Why its own module

Keeping `Event[T]` in a standalone `events.py` with **zero project imports** is what makes it a pure dependency leaf: a producer can own its `Event[...]` outputs while depending only on the primitive, nothing from the surrounding project. It also makes the primitive trivially reusable — nothing here would change if it were copied into an unrelated project.

## Open questions

- **Error isolation.** Whether `emit` should isolate a throwing handler (catch-and-continue, optionally collecting errors) rather than propagating. Deferred until a consumer actually needs one bad handler not to break the others.
- **Async handlers.** Only sync `Callable[[T], None]` handlers today. If a consumer ever needs to `await` in a handler, an async variant (or scheduling onto a loop) would be a separate, additive design.
