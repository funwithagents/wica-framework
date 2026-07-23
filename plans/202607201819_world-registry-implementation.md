# World registry implementation

**Status:** Done

Implements the settled behavior in `specs/world.md` ("Core concepts"). The three spec "Open questions" are deferred (see end).

## Scope

- `src/wica/world.py` — `WorldEntryConfig`, `WorldEntryVersion`, `WorldEntry`, `World`, plus module-level `get_world()` / `reset_world()`
- `src/wica/__init__.py` — export `World`, `WorldEntry`, `WorldEntryVersion`, `WorldEntryConfig`, `get_world`
- `tests/conftest.py` — autouse fixture resetting the singleton between tests
- `tests/test_world.py` — functional tests

## Data model

Three plain dataclasses (`WorldEntryVersion`/`WorldEntry` frozen; none generic — see "Type safety"):

```python
@dataclass
class WorldEntryConfig:
    type: type
    serialize_fn: Callable[[Any, Any], str]  # (value, previous_value) -> str
    include_in_prompt: bool = True
    triggers_llm_call: bool = False
    trigger_condition_fn: Callable[[Any, Any], bool] | None = None
    ttl: timedelta | None = None

@dataclass(frozen=True)
class WorldEntryVersion:
    id: int
    value: Any
    timestamp: datetime  # tz-aware, UTC

@dataclass(frozen=True)
class WorldEntry:
    key: str
    type: type          # copied from config at register() time; self-describing for callbacks
    current: WorldEntryVersion
    previous: WorldEntryVersion | None
```

Config (static schema) and live state are kept as two separate objects so the snapshot handed to callbacks (`WorldEntry`) carries only plain data, never the callables in `WorldEntryConfig`.

## `World` API

```python
class World:
    def register[T](
        self,
        key: str,
        type: type[T],
        *,
        serialize_fn: Callable[[T | None, T | None], str],
        include_in_prompt: bool = True,
        triggers_llm_call: bool = False,
        trigger_condition_fn: Callable[[T | None, T | None], bool] | None = None,
        ttl: timedelta | None = None,
    ) -> None: ...

    def unregister(self, key: str) -> None: ...
    def update(self, key: str, value: Any) -> None: ...
    def get(self, key: str) -> Any: ...
    def get_entry(self, key: str) -> WorldEntry: ...
    def add_listener(self, key: str, callback: Callable[[WorldEntry], None]) -> None: ...
    def remove_listener(self, key: str, callback: Callable[[WorldEntry], None]) -> None: ...
    def set_trigger_handler(self, handler: Callable[[WorldEntry], None] | None) -> None: ...
    def render_entry(self, entry: WorldEntry) -> str: ...
    def render_full_prompt(self) -> str: ...
```

Internal state: `_configs: dict[str, WorldEntryConfig]`, `_entries: dict[str, WorldEntry]`, `_id_counters: dict[str, int]` (**never cleared** — so re-`register()` after `unregister()` continues the `id` sequence instead of restarting at 1), `_listeners: dict[str, list[Callable]]`, `_timers: dict[str, threading.Timer]`, `_trigger_handler`, `_lock: threading.RLock`, `_executor: concurrent.futures.ThreadPoolExecutor`.

Key behaviors not already spelled out in the spec:

- **Type safety** — only `register()` is generic. `type: type[T]` binds `T` exactly from the class argument, so a mismatched `serialize_fn`/`trigger_condition_fn` is caught statically at the call site. Every other method takes a plain `str`/`Any`; correctness there is enforced by the runtime `isinstance` check in `update()`, not the type checker. (See "Revision history" for why the API isn't generic end-to-end.)
- **Sync, not async** — every method is a plain (non-`async`) dict mutation. The only I/O-bound thing is the trigger handler eventually making an LLM call; that stays the handler's concern, so `World` never forces `async` coloring on trivial callers like `get()`.
- **Concurrency / dispatch** — the `RLock` guards the registry/counters/timers (a TTL timer fires on a background thread). It's released *before* any callback dispatch. Listener callbacks and the trigger handler are fire-and-forget: each is `_executor.submit(...)`ed as its own task so a slow one (e.g. a real LLM call) neither blocks `update()`'s caller nor holds up the others. Callbacks receive the exact `WorldEntry` instance `update()` just built — no defensive copy, since every `update()` swaps a brand-new frozen `WorldEntry`/`WorldEntryVersion` into `_entries[key]` rather than mutating in place, so a later `update()` can't alter an already-dispatched snapshot.
- **TTL** — `threading.Timer` scheduled off `config.ttl`, (re)started on each `update()` that sets a non-`None` value, cancelled on `unregister()` and on every subsequent `update()`. Expiry runs the same `update(key, None)` codepath via an internal flag that suppresses the trigger handler (per spec) but still notifies listeners and still advances `id`. The timer is armed with the version `id` it corresponds to (`_ttl_expire(key, expected_id)`); under the lock, the reset no-ops unless the entry's current `id` still equals `expected_id`. This closes a race where a timer firing just as a fresh `update()` lands would otherwise clobber the new value and cancel its timer.
- **Singleton** — module-level `get_world()` lazily creates/caches one `World`. `reset_world()` is **test-only** (cancels timers, shuts the executor down, clears the cached instance); imported directly from `wica.world` by `conftest.py`, not re-exported from the package.

## Rendering

`render_entry(entry)` looks up `_configs[entry.key]` (raising `KeyError` if the key has since been unregistered — the snapshot can outlive its registration), calls `serialize_fn(entry.current.value, entry.previous.value if entry.previous else None)`, and ignores `include_in_prompt`:

```
<entry key="{entry.key}" id="{entry.current.id}">
{serialize_fn(value, previous_value)}
Updated: {entry.current.timestamp.isoformat()}
</entry>
```

`render_full_prompt()` selects `_entries` whose config has `include_in_prompt=True`, calls `render_entry(entry)` for each in ascending-`timestamp` order, and joins with newlines.

## Implementation steps

1. Add the three dataclasses.
2. Add `World.__init__` (the internal state above) and the `get_world()` / `reset_world()` pair.
3. `register` (build config + initial `WorldEntry` with `id=1`, `value=None`, `previous=None`; `ValueError` if already registered), `unregister` (drop both dicts + listeners + timer), `get`, `get_entry` (each `KeyError` if unregistered).
4. `update` (validate type; under lock, advance `id`, build new frozen version/entry, swap in, (re)arm TTL timer; release lock; dispatch listeners + gated trigger handler). Factor a private `_update(key, value, *, ttl_reset, expected_id=None)` so the TTL path reuses it while suppressing the trigger handler and guarding on the version `id` (see the TTL bullet above).
5. TTL timer wiring + `_cancel_all_timers()` (used by `reset_world()`).
6. `render_entry` then `render_full_prompt` on top of it.
7. Exports in `__init__.py`.
8. `tests/conftest.py` autouse fixture calling `reset_world()` before/after each test.

## Tests (`tests/test_world.py`)

All functional — drive the public API, assert observable behavior. Callback assertions use a `threading.Event` set inside a test-double callback + `event.wait(timeout=...)`, since dispatch is fire-and-forget.

- `register` then `get` → `None`; `update` changes `get`; `update` with wrong type raises `TypeError` and leaves the prior value/id intact; `update(key, None)` always allowed.
- `id`: `register` alone renders `id="1"`; every `update` (real value *or* clear) advances it; re-`register` after `unregister` continues past the last `id` (no restart at 1). Re-`register` with a *different* type then `get` just returns the new value (no staleness error — there's no handle to go stale).
- `unregister` removes the entry (later `get`/`update` → `KeyError`); re-`register` without `unregister` → `ValueError`.
- `get_entry` returns a `WorldEntry` matching `get` and the registered `key`/`type`; `KeyError` when unregistered.
- Listeners: fire with the updated `WorldEntry`, not for unrelated keys; `remove_listener` stops them.
- Trigger handler: invoked when `triggers_llm_call` and the (optional) `trigger_condition_fn` passes; gated off when the condition returns `False`; the `lambda old, new: old != new` + repeated-value case confirms no-op suppression; never invoked by `unregister` or on a `triggers_llm_call=False` entry.
- Non-blocking: a slow callback blocked on an `Event` proves `update()` already returned (new value visible via `get()`) before the callback is released.
- Snapshot safety: after `update(k, "first")` then `update(k, "second")`, a listener that recorded its argument still sees `"first"`.
- `serialize_fn` receives the right `previous_value` across `A → B → None` transitions (checked via `render_entry`).
- TTL: short `ttl` clears the value on its own (advancing `id`, without firing the trigger handler even when `triggers_llm_call=True`); an `update` before expiry postpones the reset; a stale expiry for a superseded version `id` (invoked directly, since it can't be triggered publicly) no-ops instead of clobbering the fresh value.
- Rendering: `render_entry` matches the exact tag format; renders `include_in_prompt=False` entries; raises `KeyError` on an unregistered snapshot. `render_full_prompt` omits excluded entries, orders by timestamp, and each block equals `render_entry(get_entry(key))`.
- Singleton: two `get_world()` calls return the same object.

## Revision history

The API went through three pivots during review; captured here so the current shape's rationale is legible without cluttering the design above.

1. **Typed key handle, then removed.** An interim design had `register()` return a generic `WorldKey[T]` handle carried by every method, for end-to-end static type checking. Two problems sank it: (a) pyright doesn't pin a shared `TypeVar` from one argument and strictly check another — it solves for *any* type satisfying both jointly, silently widening away the mismatch (a frozen dataclass is also inferred *covariant*, compounding this; a classic invariant `TypeVar` was needed to make the check fire at all); (b) once that complexity was weighed against a benefit that only helps when a type checker runs, the scope was cut to generic-`register()`-only. At that point the handle carried nothing past `register()` except a `.type` that could go *stale* across `unregister()`+re-`register()`, needing its own runtime guard — a risk the handle itself introduced. Dropping the handle for a plain `str` key removed that whole failure mode. **Net:** plain-`str` keys everywhere, `register()` the sole generic method, runtime `isinstance` the real safety net.
2. **`id` simplified.** An interim design let `id` be `int | None` and *persist* across a clear (to distinguish "cleared" from "replaced" for a future Commands consumer). Dropped for: `id` is always a plain `int` starting at `1` from `register()`, advancing on every `update()` including clears. Simpler; the Commands use case doesn't exist yet.
3. **`render_entry` takes a `WorldEntry`, not a key.** So it renders exactly the snapshot the caller holds (from a callback or `get_entry`) rather than re-reading live state that a concurrent `update()` may have moved on.

## Deferred / open follow-ups

- Spec open questions — concurrent trigger coalescing (currently: trigger immediately), full-vs-incremental serialization (`render_entry` is the building block; strategy is an Agent-loop decision), stale-`id` handling (likely a Commands-spec concern).
- Type-checking tool choice — a `specs/project.md` question, independent of this work.
- `set_trigger_handler` shape — revisit once Commands/Agents specs exist (may fold into a broader Agent registration API).
