# World registry implementation

Implements `specs/world.md` (Status: Draft, but the settled behavior described throughout "Core concepts" is implementation-ready; the three "Open questions" are either resolved for now or explicitly deferred — see "Deferred / not in scope" below).

## Scope

- `src/wica/world.py` — `WorldEntryConfig`, `WorldEntryVersion`, `WorldEntry`, and `World`
- `src/wica/__init__.py` — export `World`, `WorldEntry`, `WorldEntryVersion`, `WorldEntryConfig`
- `tests/test_world.py` — functional tests mirroring `src/wica/world.py` per `specs/project.md`

## Design decisions made while planning (flagging for review)

These aren't nailed down by `world.md` itself; picking a concrete, reasonable choice so the plan is actionable. Called out here so they're easy to challenge during review rather than buried in code.

1. **LLM-trigger hookup** (resolved via user input): `World.set_trigger_handler(handler)` sets a single `Callable[[WorldEntry], None] | None`, default `None` (no-op). When `triggers_llm_call` is set and `trigger_condition_fn` (if any) passes, `update()` **dispatches** the handler call rather than invoking it inline — see decision 13 (fire-and-forget dispatch). This keeps `World` decoupled from an actual Agent loop, which isn't specced yet — the future Agents implementation just calls `set_trigger_handler` on startup — and means a slow handler (a real LLM call) never makes `update()`'s caller wait.
2. **`trigger_condition_fn` signature**: `Callable[[old_value, new_value], bool]`. Gives it enough information to gate on a transition (e.g. "only trigger if the value increased"), not just the new value. Stays a plain, synchronously-*evaluated* predicate — it's the gate deciding *whether* to dispatch, not something itself dispatched. See decision 24: there's deliberately no automatic "skip if unchanged" behavior built on top of this — that's exactly what this predicate is for, if wanted.
3. **Listener callback signature**: `Callable[[WorldEntry], None]` — passed the full updated entry (not just the raw value), since key/id/timestamp are often needed alongside the value. Like the trigger handler, dispatched rather than invoked inline (decision 13).
4. **`register()` does not take an initial `value`.** Every entry starts as `None` (consistent with `None` always being a valid value regardless of declared `type`) — the caller sets the real value with a follow-up `update(key, value)`. Corrected from an earlier draft of this plan that had `register()` accept an initial value; per user direction, register-then-set-to-`None` is the intended flow, not register-with-a-value.
5. **Re-registering an already-registered key without `unregister()` first raises `ValueError`.** Only `register()` after `unregister()` is allowed and continues the `id` sequence (per spec). This makes the id-continuity guarantee meaningful — the counter dict persists across unregister, and re-register is the only path back to a key.
6. **`unregister(key)` also drops any listeners registered for that key.** They'd otherwise be orphaned (never firing again, silently). Not stated in spec, but "removes an entry entirely" reads as including its subscriptions.
7. **`type` validation is `isinstance`-based**, so `type` must be a plain class or tuple of classes. Subscripted generics (`list[int]`, `dict[str, Any]`, etc.) aren't supported — `isinstance` doesn't accept them. Out of scope for this draft; flagged as a limitation, not silently handled.
8. **Singleton implementation**: module-level `get_world()` accessor (lazily creates and caches a module-global `World` instance on first call), not a classic `__new__`-overridden singleton class. More idiomatic Python (cf. `logging.getLogger()`), and avoids the gotcha where `__init__` re-runs on every direct `World()` call even when `__new__` returns the cached instance. `World()` itself stays a plain, directly-constructible class — calling it directly instead of `get_world()` is discouraged by convention, not structurally prevented. Adds a **test-only** `reset_world()` module function to clear the cached instance between tests — required for pytest isolation, not part of the public spec'd API. A `conftest.py` fixture (autouse, `tests/`) calls it before/after each test.
9. **TTL enforcement**: `threading.Timer`, restarted on every `update()` (including TTL-driven resets, which don't recreate a timer since the value is now `None`) and cancelled on `unregister()`. A `threading.RLock` guards the registry/counters/timers since the timer fires on a background thread; it's released before dispatching listener/trigger-handler callbacks (decision 13) so building/queuing that dispatch never happens while holding the lock. TTL-driven resets go through the same `update()` codepath with an internal flag that suppresses the trigger-handler call (per spec) but still notifies listeners (spec doesn't say otherwise, and a listener arguably does want to know the value went away).
10. **`register()` and `update()` return `None`** (per user direction) — no `WorldEntry` handed back to the caller. A caller that needs the current value/id uses `get(key)` / `render_entry(key)` / `render_full_prompt()` (or, for tests, an internal accessor); `register`/`update` are pure side-effecting calls.
11. **`id` is `int | None`, not always `int`, and — revised per user input — persists across a clear** rather than resetting to `None`: `id` is `None` **only** in the true "never had any value yet" state, right after `register()` and before the first non-`None` `update()`. From then on, `id` only ever changes (increments) when `update()` sets a **non-`None`** value — an explicit `update(key, None)` or a TTL-driven reset carries the *same* `id` forward into the new "current" version rather than nulling it. This matters for Commands (spec open question 3): something that references `(key, id)` needs to be able to validate against that version even after the entry's since been cleared, not only while it still holds a value — `value is None` is the sole signal for "empty," `id is None` means "never had anything." The per-key counter still persists across `unregister()`/re-`register()` regardless, so the first non-`None` value given to a key after re-registering still gets a fresh integer one past the last real `id` that key ever had, rather than restarting at 1. *(This plan originally had `id` reset to `None` on every clear, symmetric with `register()`'s initial state; revised after discussing a concrete Commands use case that needs a stale-but-real `id` to check against even on a cleared entry.)*
12. **`World`'s public API is synchronous, not `async`.** Considered making it fully `async` (to match an eventual async Agent/LLM loop) but reverted: none of `register`/`unregister`/`update`/`get`/listeners/`render_full_prompt` do any I/O of their own — they're dict mutations. Forcing `async def` on all of them would mean every caller, including a trivial `get()`, has to be `async` too ("async coloring"), for no benefit to those methods themselves. The one genuinely I/O-bound thing is the trigger handler eventually making an LLM call — that's handled by keeping the handler itself a plain callable and letting it (or the future Agent loop) bridge into async on its own terms, rather than `World` imposing that on everything.
13. **`update()` never blocks on listener/trigger-handler callbacks** (resolved via user input): both are fire-and-forget — dispatched to run in the background, with `update()` returning as soon as the entry's own state is mutated. A shared `concurrent.futures.ThreadPoolExecutor` on the `World` instance runs them (bounded pool, vs. an unbounded `threading.Thread` per call under high update frequency); the trigger handler and each listener are submitted as separate tasks so slow ones don't hold up others.
14. **Callbacks receive the exact `WorldEntry` instance `update()` just built — no defensive copy needed.** Since `WorldEntry` is replaced wholesale in `_entries[key]` on every `update()` rather than mutated in place, and `WorldEntryVersion` is frozen (decision 21), the object handed to a callback can never be changed out from under it by a later `update()` — that later call constructs an entirely new `WorldEntry`/`WorldEntryVersion` pair and swaps it into `_entries[key]`, leaving the one already dispatched to callbacks untouched. (An earlier draft of this plan built an explicit shallow copy for this purpose; the wholesale-replacement approach in decision 21 makes that unnecessary.)
15. **Exceptions raised inside a dispatched callback are not surfaced to `update()`'s caller** — a consequence of fire-and-forget: nothing calls `.result()` on the submitted `Future`, so `ThreadPoolExecutor` just holds the exception. Acceptable for this draft (no logging infra exists yet in the project to report it properly) but worth flagging as a real gap: a broken listener currently fails silently. Revisit once the project has a logging story.
16. **`render_entry(key)` omits the `id` attribute entirely when `id` is `None`** (consequence of decision 11), rather than printing `id="None"` — the tag is just `<entry key="...">`. Applies only to a never-updated entry now (a real `id` persists through subsequent clears per the revised decision 11); `render_full_prompt()` inherits this since it's built on `render_entry` (decision 20).
17. **Registration metadata and live state are two parallel per-key dictionaries** (resolved via user input), not one merged object: `_configs: dict[str, WorldEntryConfig]` holds `type`/`include_in_prompt`/`triggers_llm_call`/`trigger_condition_fn`/`serialize_fn`/`ttl` — set once at `register()`, unchanged until `unregister()` (no "reconfigure" API). `_entries: dict[str, WorldEntry]` holds the live, versioned state (`key`/`type`/`current`/`previous` — see decision 21) — mutated by every `update()`. `register()` creates one of each; `unregister()` removes both. `update()` reads the config (for validation/trigger gating) but only ever touches the entry.
18. **The object handed to listener/trigger-handler callbacks is a `WorldEntry` (decision 14), never anything from `WorldEntryConfig`.** Keeps the callback payload plain data (`key`/`type`/`current`/`previous`, and their nested `WorldEntryVersion` fields) with no callables in it — those stay in `WorldEntryConfig` and are never touched by the dispatch path.
19. **`type` is duplicated onto `WorldEntry`, not just `WorldEntryConfig`** (resolved via user input): copied from the config at `register()` time and never touched again. This makes a `WorldEntry` — and in particular the object handed to listener/trigger callbacks (decision 18) — self-describing about the value's declared type without needing a `_configs` lookup. Unlike `current`/`previous` (decision 21), `type` doesn't change across updates, so it lives once on `WorldEntry` rather than being duplicated onto every `WorldEntryVersion`.
20. **`render_entry(key)` is a new public method** (resolved via user input), factored out of `render_full_prompt()`: it renders one key's `WorldEntry` into the same XML-style block, and `render_full_prompt()` becomes "filter `_configs` by `include_in_prompt`, call `render_entry(key)` for each surviving key in timestamp order, join." Motivated by the earlier discussion of whether the trigger handler should receive the fully-rendered World: it shouldn't (`World` staying agnostic to that still-open spec question — open question 2), but a handler that *wants* to render just the entry that fired now has a direct way to do it, without needing `World` to hand it a pre-rendered string. `render_entry(key)` **ignores `include_in_prompt`** — it renders any registered key regardless, since an explicit per-key request bypasses the aggregate-prompt filter; only `render_full_prompt()` applies that filter, by choosing which keys to call `render_entry` for.
21. **`WorldEntryVersion` is a new, separate, frozen sub-object holding `id`/`value`/`timestamp`** (resolved via user input): `WorldEntry` no longer stores those flat — it holds `current: WorldEntryVersion` and `previous: WorldEntryVersion | None`. `register()` creates `current = WorldEntryVersion(id=None, value=None, timestamp=now())` and `previous = None`. Every `update()` builds a brand-new `WorldEntryVersion` for the new state, shifts the old `current` into `previous`, and constructs a brand-new `WorldEntry` (also effectively frozen — replaced wholesale, never mutated in place) to swap into `_entries[key]`. `previous` is `None` only in the window between `register()` and the first `update()`. Motivated by wanting `serialize_fn` to see the prior value (decision 23) without the World keeping a full history — just one step back, which stays consistent with "snapshot, not event log" (spec).
22. **`serialize_fn` signature widens to `(value, previous_value)`** (resolved via user input): lets it phrase things differently depending on the transition — e.g. distinguishing "just cleared" from "never set," or rendering a diff-like description — especially around `None`. `render_entry(key)` calls it as `config.serialize_fn(entry.current.value, entry.previous.value if entry.previous is not None else None)`. Collapsing "no prior version at all" (`entry.previous is None`) and "prior version's value happened to be `None`" into the same `previous_value=None` argument is an accepted simplification: `register()` always starts a key at value `None` anyway (decision 4), so in practice there's nothing meaningfully different between those two cases from `serialize_fn`'s point of view.
23. **No built-in equality-based trigger suppression** (documentation-only, resolved via user input): `update()` never compares old vs. new value before evaluating `trigger_condition_fn` — no automatic "skip if the value didn't change" behavior. Reasoning (same as decision 2): equality is ambiguous or expensive for arbitrary types (e.g. images), and some entries legitimately want to re-trigger on an unchanged value (e.g. a heartbeat). Documented directly on `WorldEntryConfig.trigger_condition_fn` in `specs/world.md` as the recommended pattern (`trigger_condition_fn=lambda old, new: old != new`) rather than baked into `update()` itself.

## Deferred / not in scope (per spec's own open questions)

- **Concurrent trigger coalescing** — spec says "Currently: always trigger immediately," so `update()` dispatches the trigger handler individually each time (fire-and-forget, per decision 13), rather than batching multiple triggers into one call; no batching logic to build.
- **Incremental serialization** — `render_entry(key)` (decision 20) gives a per-entry building block a future incremental path could use, but no actual incremental/partial *prompt-assembly* strategy (deciding what an LLM call should actually resend) is implemented — that's an Agent-loop-level decision, left open.
- **Stale-`id` handling** — no `get_by_id(key, id)` or staleness detection; `id` now persisting across a clear (decision 11) makes this kind of check *possible* in principle, but the actual mismatch-handling behavior spec says this likely belongs to the (unwritten) Commands spec.

## `WorldEntryConfig`

Static registration metadata (decision 17) — set once at `register()`, never mutated until `unregister()`:

```python
@dataclass
class WorldEntryConfig:
    type: type | tuple[type, ...]
    serialize_fn: Callable[[Any, Any], str]  # (value, previous_value) -> str — decision 22
    include_in_prompt: bool = True
    triggers_llm_call: bool = False
    trigger_condition_fn: Callable[[Any, Any], bool] | None = None
    ttl: timedelta | None = None
```

## `WorldEntryVersion`

A single versioned snapshot of an entry's value (decision 21) — frozen (immutable once created); `update()` builds a new one rather than mutating an existing one in place:

```python
@dataclass(frozen=True)
class WorldEntryVersion:
    id: int | None
    value: Any
    timestamp: datetime  # tz-aware, UTC
```

## `WorldEntry`

Live state for a registered key (decision 21) — also effectively frozen: `update()` constructs a brand-new `WorldEntry` and swaps it into `_entries[key]` rather than mutating fields on the existing one:

```python
@dataclass(frozen=True)
class WorldEntry:
    key: str
    type: type | tuple[type, ...]
    current: WorldEntryVersion
    previous: WorldEntryVersion | None
```

`type` is copied from `WorldEntryConfig` at `register()` time and never touched again (decision 19) — duplicated so a `WorldEntry` is self-describing, but only once (not per version, unlike `current`/`previous`).

## `World`

```python
class World:
    def register(
        self,
        key: str,
        type: type | tuple[type, ...],
        *,
        serialize_fn: Callable[[Any, Any], str],
        include_in_prompt: bool = True,
        triggers_llm_call: bool = False,
        trigger_condition_fn: Callable[[Any, Any], bool] | None = None,
        ttl: timedelta | None = None,
    ) -> None: ...

    def unregister(self, key: str) -> None: ...
    def update(self, key: str, value: Any) -> None: ...
    def get(self, key: str) -> Any: ...
    def add_listener(self, key: str, callback: Callable[[WorldEntry], None]) -> None: ...
    def remove_listener(self, key: str, callback: Callable[[WorldEntry], None]) -> None: ...
    def set_trigger_handler(self, handler: Callable[[WorldEntry], None] | None) -> None: ...
    def render_entry(self, key: str) -> str: ...
    def render_full_prompt(self) -> str: ...
```

Internal state: `_configs: dict[str, WorldEntryConfig]`, `_entries: dict[str, WorldEntry]` (decision 17 — two parallel per-key dicts; each `WorldEntry` nests its `current`/`previous` `WorldEntryVersion`, decision 21), `_id_counters: dict[str, int]` (never cleared), `_listeners: dict[str, list[Callable]]`, `_timers: dict[str, threading.Timer]`, `_trigger_handler`, `_lock: threading.RLock`, `_executor: concurrent.futures.ThreadPoolExecutor` (fire-and-forget dispatch of listener/trigger-handler callbacks, per decision 13).

### Module-level accessor

```python
_world: World | None = None

def get_world() -> World:
    global _world
    if _world is None:
        _world = World()
    return _world

def reset_world() -> None:  # test-only
    global _world
    if _world is not None:
        _world._cancel_all_timers()
        _world._executor.shutdown(wait=True)
    _world = None
```

### `render_entry(key)` / `render_full_prompt()` format

`render_entry(key)` looks up `_entries[key]` (raising `KeyError` if unregistered) and `_configs[key].serialize_fn`, ignoring `_configs[key].include_in_prompt` (decision 20). It calls `serialize_fn(entry.current.value, entry.previous.value if entry.previous is not None else None)` (decision 22):

```
<entry key="{key}" id="{id}">
{serialize_fn(value, previous_value)}
Updated: {timestamp.isoformat()}
</entry>
```

...except when `id` is `None` (decision 16 — only possible for a never-updated entry, per revised decision 11), where the attribute is dropped rather than printed as `id="None"`:

```
<entry key="{key}">
{serialize_fn(value, previous_value)}
Updated: {timestamp.isoformat()}
</entry>
```

Note the `Updated:` line sits *inside* `<entry>...</entry>`, after `serialize_fn`'s output — not as an attribute (keeps the "informational, not referenceable" distinction from `key`/`id`), but structurally part of the entry's own block rather than a free-floating line after the closing tag.

`render_full_prompt()` filters `_configs` down to keys with `include_in_prompt=True`, calls `render_entry(key)` for each in ascending-timestamp order, and joins the blocks with a newline between them.

## Implementation steps

1. Add `WorldEntryConfig`, `WorldEntryVersion`, and `WorldEntry` dataclasses to `src/wica/world.py` (decisions 17, 21) — `WorldEntryVersion` and `WorldEntry` both frozen.
2. Add `World` (plain class, no singleton machinery), with a `threading.RLock` and a `concurrent.futures.ThreadPoolExecutor`, plus the module-level `get_world()`/`reset_world()` accessor pair.
3. Implement `register`: builds and stores a `WorldEntryConfig` in `_configs[key]` and an initial `WorldEntry(key, type=config.type, current=WorldEntryVersion(id=None, value=None, timestamp=now()), previous=None)` in `_entries[key]`; raises `ValueError` if the key is already registered. Implement `unregister`: removes both, cancels any timer, drops listeners. Implement `get`: returns `_entries[key].current.value`.
4. Implement `update()`: look up `_configs[key]` (raise `KeyError` if unregistered), validate `value`'s type against `config.type` (`None` always allowed). Under the lock: read `old_entry = _entries[key]`; compute the new `id` — `_id_counters[key] + 1` (and store it back) if `value is not None`, else `old_entry.current.id` unchanged (decision 11); build `new_version = WorldEntryVersion(id=..., value=value, timestamp=now())`; build `new_entry = WorldEntry(key=key, type=old_entry.type, current=new_version, previous=old_entry.current)`; assign `_entries[key] = new_entry`. Release the lock, then `_executor.submit(...)` the listener callbacks and (if `config.triggers_llm_call` and `config.trigger_condition_fn` gating passes) the trigger handler, passing `new_entry` directly (decision 14 — no copy needed).
5. Implement TTL: `threading.Timer` scheduling off `config.ttl` in `register`/`update`, cancellation in `unregister`/on every new `update`, internal TTL-driven reset path (calls the same `update()` value-`None` logic, so `id` persists per decision 11) that skips the trigger handler, and `_cancel_all_timers()` used by `reset_world()`.
6. Implement `render_entry(key)` (raises `KeyError` if unregistered, ignores `include_in_prompt` per decision 20, calls `serialize_fn(value, previous_value)` per decision 22), then `render_full_prompt()` on top of it: filter `_configs` to `include_in_prompt=True` keys, call `render_entry(key)` for each in ascending-timestamp order, join.
7. Export `get_world`, `World`, `WorldEntry`, `WorldEntryVersion`, `WorldEntryConfig` from `src/wica/__init__.py` (`reset_world` stays test-only, imported directly from `wica.world` in `conftest.py`, not re-exported publicly).
8. Add `tests/conftest.py` with an autouse fixture calling `reset_world()` before/after each test (shuts down the previous instance's executor so no background threads leak between tests).

## Tests (`tests/test_world.py`)

All functional — driving the public API, asserting observable behavior, not internals. Since listener/trigger-handler dispatch is now fire-and-forget (background thread pool), tests that assert on a callback having run use a `threading.Event` set inside the test double and `event.wait(timeout=...)` rather than asserting immediately after `update()` returns:

- `register` + `get` returns `None` — an entry starts with no value until the first `update`.
- `update` changes what `get` returns.
- `update` with a mismatched type raises `TypeError` and leaves the previous value/id untouched (no partial mutation).
- `update(key, None)` always succeeds regardless of declared `type`.
- `register` alone (no `update` yet) renders with no `id` attribute at all (value is `None`, decision 16); the first non-`None` `update` gives it `id="1"`, and successive non-`None` updates increment from there.
- `update(key, None)` after a real value **keeps** the same `id` (revised decision 11 — no longer resets to "no version"), while clearing the value; a subsequent non-`None` update advances to a new integer rather than reusing an old one.
- `unregister` removes the entry: subsequent `get`/`update` raise `KeyError`.
- Re-`register`ing the same key without `unregister` raises `ValueError`.
- Re-`register`ing a key *after* `unregister` and giving it a non-`None` value: the first real `id` is one past the last real `id` that key ever had, not a restart at 1.
- `add_listener` callback eventually fires with the updated `WorldEntry` on `update` (awaited via `Event`); unrelated keys' updates don't fire it.
- `remove_listener` stops further notifications.
- `set_trigger_handler` + `triggers_llm_call=True`, no `trigger_condition_fn`: handler eventually invoked on `update`.
- `trigger_condition_fn` gates the handler: not invoked when it returns `False`, invoked (with correct old/new values) when `True`. Include a case with `trigger_condition_fn=lambda old, new: old != new` and an `update()` call that repeats the current value, confirming the handler is *not* invoked — demonstrating the documented no-op-suppression pattern (decision 23), since `update()` itself never suppresses this.
- `unregister` never invokes the trigger handler, even for an entry with `triggers_llm_call=True`.
- `update` on a `triggers_llm_call=False` entry never invokes the handler.
- `update()` returns without waiting on a slow trigger handler/listener: a test double blocks on an `Event` the test controls, and the test asserts `update()` already returned (and the entry's new value is visible via `get()`) before it sets that `Event` to release the callback — proves the dispatch is truly non-blocking, not just "fast in practice."
- Snapshot safety: a listener records the `WorldEntry` it received (and its `.current.value`); immediately after `update(key, "first")` returns (before waiting for the listener), the test calls `update(key, "second")`. After waiting for the listener via `Event`, it asserts the listener's recorded entry still shows `"first"` — proves the object handed to callbacks is unaffected by a later `update()` (decision 14), since that `update()` built an entirely separate `WorldEntry`/`WorldEntryVersion` rather than mutating the one already dispatched.
- `serialize_fn` receives the correct `previous_value`: register a key, `update` it to `"A"` (serializer sees `(value="A", previous_value=None)`), `update` it to `"B"` (`(value="B", previous_value="A")`), then `update(key, None)` (`(value=None, previous_value="B")`) — verified via `render_entry(key)`'s content matching what a test `serialize_fn` recorded/produced for each transition.
- TTL: entry with a short `ttl` resets its value to `None` on its own (poll with timeout), **keeping its `id`** (revised decision 11), without invoking the trigger handler even when `triggers_llm_call=True`.
- TTL restart: an `update` before expiry postpones the reset — value is still non-`None` at the original expiry time, and only clears a full `ttl` after the *last* update.
- `render_entry(key)` output matches the spec's format exactly for a known entry: `key`/`id` as opening-tag attributes, `serialize_fn` content, then the `Updated:` ISO-timestamp line, all before the closing `</entry>` tag.
- `render_entry(key)` on an unregistered key raises `KeyError`.
- `render_entry(key)` renders an `include_in_prompt=False` entry anyway (decision 20) — the filter is `render_full_prompt()`'s job, not `render_entry`'s.
- `render_full_prompt()` omits `include_in_prompt=False` entries (while `render_entry(key)` would still render that same key directly).
- `render_full_prompt()` orders entries by timestamp — updating an older entry moves its block to the end.
- `render_full_prompt()`'s block for a given key is identical to calling `render_entry(key)` directly for it.
- Singleton: two `get_world()` calls return the same object, and a mutation via one reference is visible via the other.

## Open follow-ups (not blocking this plan)

- Type checking tool choice (mypy/pyright/none) — `specs/project.md` open question, independent of this work.
- Once Commands/Agents specs exist, revisit whether `set_trigger_handler` is the right long-term shape or should become part of a broader Agent registration API.
