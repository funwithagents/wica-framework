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
- **Subscriber isolation.** A handler that raises is **caught and logged** (via a module logger, `logging.getLogger(__name__)`) and dispatch **continues** to the remaining handlers — one bad subscriber can neither abort the `emit` nor starve its siblings. This is what makes `Event` safe to use as a multi-subscriber fan-out point: the World's trigger and the Agent's instrumentation are `Event`s (surfaced by [wica.md](wica.md)) where several independent consumers subscribe to the same signal, and a bug in one must not silence the rest (matching the Agent's "instrumentation never breaks the loop" guarantee). `logging` is the only dependency this needs — `events.py` stays a pure standard-library leaf with zero project imports, so it's copyable into another project unchanged. A consumer that wants to *know* a handler failed reads the log (aggregating/collecting handler errors is a possible future addition, not built).

### Why its own module

Keeping `Event[T]` in a standalone `events.py` with **zero project imports** is what makes it a pure dependency leaf: a producer can own its `Event[...]` outputs while depending only on the primitive, nothing from the surrounding project. It also makes the primitive trivially reusable — nothing here would change if it were copied into an unrelated project.

## Open questions

- **Error aggregation.** `emit` isolates a throwing handler (catch-and-log-and-continue — see "Semantics"). What it does **not** do is *collect* handler errors and hand them back to the `emit` caller (e.g. an aggregated exception once all handlers have run). Deferred until a consumer needs the failures programmatically rather than just logged.
- **Async handlers.** Only sync `Callable[[T], None]` handlers today. If a consumer ever needs to `await` in a handler, an async variant (or scheduling onto a loop) would be a separate, additive design.
