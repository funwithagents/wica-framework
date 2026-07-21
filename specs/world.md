# World

**Status:** Draft

## Purpose

The World stores all data about the current interaction context — arbitrary registered objects — for the rest of the system to read and react to. It serves two distinct roles:

1. **Shared state.** Any registered entry can be read (`get`) and reacted to (`add_listener`) by other modules, independent of the LLM entirely — e.g. bookkeeping, coordination state, or data one part of the system needs to hand to another.
2. **LLM-facing context.** Entries marked `include_in_prompt=True` also get rendered into a semantic text string (`render_full_prompt()`) suitable for inclusion in an LLM prompt.

These roles are independent: an entry can be stored purely for other modules to consume — via listeners reacting to its updates — without ever being serialized into a prompt (`include_in_prompt=False`). `include_in_prompt` is the switch between the two: it doesn't control whether an entry is *stored* or *reactive*, only whether it's *shown to the model*.

The World holds current state only — a snapshot, not an event-sourced log of the Commands that produced it — and stores values directly, including multimodal data (e.g. images, audio), rather than by reference to an external blob store.

## Core concepts

### Registration and WorldEntryConfig

A key must be registered before any data can be stored under it — `register()` is how a key comes into existence in the World, and that call is where its static behavior/schema is defined: `type`, whether it reaches the LLM, whether it triggers a call, how it serializes, its TTL. This is captured in a `WorldEntryConfig`, set once at `register()` and unchanged until `unregister()` (there's no "reconfigure" API — changing behavior means unregistering and re-registering):

| Field | Description |
|---|---|
| `type` | Declared type of the stored value. `update()` validates new values against it and raises `TypeError` on mismatch (see `WorldEntryVersion.value` below for the `None` exemption) |
| `include_in_prompt` | Whether this entry is included when rendering the World prompt |
| `triggers_llm_call` | Whether updating this entry's value should trigger a new LLM call |
| `trigger_condition_fn` (optional) | Extra condition function evaluated on update; must pass, in addition to `triggers_llm_call`, for the update to actually trigger the LLM call. There's no automatic "skip if unchanged" behavior — `update()` never compares old vs. new value itself (equality is ambiguous or expensive for arbitrary types, e.g. images, and some entries legitimately want to re-trigger on an unchanged value, e.g. a heartbeat). If a no-op update shouldn't trigger, encode that here: `trigger_condition_fn=lambda old, new: old != new` |
| `serialize_fn` | Function that converts a value into a semantic string for the rendered prompt: `serialize_fn(value, previous_value)` (values only — does not receive the timestamp). Receiving `previous_value` lets it phrase things differently depending on the transition (e.g. distinguishing "just cleared" from "never set"), especially around `None` |
| `ttl` (optional) | Duration since the last update after which the value automatically resets to `None` via `update(key, None)`, if not refreshed before then. Enforced actively via a timer, not lazily on read; the timer restarts on every update. A TTL-driven reset behaves like `update(key, None)` for the value, but — unlike an explicit `update(key, None)` call — never triggers an LLM call even if `triggers_llm_call=True` |

### WorldEntryVersion

A single versioned snapshot of an entry's value — immutable once created; `update()` builds a new one rather than mutating an existing one in place:

| Field | Description |
|---|---|
| `id` | Identifier for this specific *version* of the entry's value; regenerated (incremented) whenever `update()` sets a **non-`None`** value — and, once assigned, persists across later clears (see below). `None` only in the true "never had a value yet" state, right after `register()` and before the first `update()`. Included in the serialized output alongside `key` so the LLM (or a Command) can reference a specific entry/version — a mismatched `id` signals the data has changed since it was last seen |
| `value` | The actual stored object (any type, including multimodal data — e.g. images, audio). `None` is always a valid value here regardless of the entry's declared `type` — exempt from the type-check that otherwise raises on mismatch. This is also how a value gets cleared; there's no separate `delete` |
| `timestamp` | Set when this version is created; used to order entries when rendering and stamped, in absolute form, onto the rendered entry |

`id` is a small per-key incrementing integer (`1, 2, 3, ...`) rather than a UUID/hash: any reference to an entry always pairs `id` with `key`, so global uniqueness isn't needed, and small integers are far easier for a model to reproduce exactly than a hex string.

Crucially, **clearing a value does not reset its `id`**: an explicit `update(key, None)` or a TTL-driven reset produces a new "current" version with the *same* `id` as before (just `value=None`) — only a non-`None` update ever advances the counter. This matters for Commands: something that references `(key, id)` needs to be able to validate against that version even after the entry has since been cleared, not only while it still holds a value. `value is None` is the sole signal for "empty"; `id is None` is reserved for "never had any value at all."

The counter for each key lives in a dictionary separate from the live entries and is never cleared — so after `unregister()`/re-`register()`, the first non-`None` value given back to that key still gets a fresh integer one past the last real `id` it ever had, rather than restarting at 1 (which would let a stale old reference collide with a reused `id`).

### WorldEntry

The live state for a registered key — created by `register()` and updated by every `update()`:

| Field | Description |
|---|---|
| `key` | Stable identifier for this entry's slot in the World (dictionary key); unchanged across updates |
| `type` | Declared type of the stored value, copied from the `WorldEntryConfig` set at `register()` — carried on `WorldEntry` too so a snapshot of it (e.g. handed to a listener/trigger-handler callback) is self-describing without a separate config lookup. Unlike `current`/`previous` below, this never changes across updates, so it isn't duplicated per version |
| `current` | The entry's current `WorldEntryVersion` |
| `previous` | The `WorldEntryVersion` immediately before `current`, or `None` in the brief window between `register()` and the first `update()`. Exists so `serialize_fn` can react to the transition (e.g. rendering a cleared value differently from a never-set one) without the World keeping a full history — just one step back, consistent with the World being snapshot-based rather than event-sourced |

`register()` creates `current = WorldEntryVersion(id=None, value=None, timestamp=<register time>)` and `previous = None`. Every `update()` shifts the old `current` into `previous` and builds a fresh `current`.

### World API

The World is a **singleton** — one instance exists per process/run, and (for now) there's a single global World rather than any per-agent partitioning/nesting. It's accessed via a module-level `get_world()` accessor (lazily creating the shared instance on first call) rather than a classic `__new__`-overridden singleton class — the more idiomatic Python shape (cf. `logging.getLogger()`), and it avoids the gotcha where `__init__` re-runs on every direct `World()` call even when `__new__` returns the cached instance. Calling `World()` directly is discouraged by convention, not structurally prevented.

Internally, the World holds two parallel per-key stores: registration metadata (`WorldEntryConfig`) and live state (`WorldEntry`). Keeping them apart means, e.g., the snapshot handed to listener/trigger callbacks (below) only ever needs to copy the small `WorldEntry`, never the callables living in `WorldEntryConfig`.

The World exposes the following API:

- `register(...)` — registers a new entry: stores its `WorldEntryConfig` and creates its initial `WorldEntry` (current version `id=None`, `value=None`); can happen at any time, including mid-run
- `unregister(key)` — removes an entry entirely (both its `WorldEntryConfig` and its `WorldEntry` are gone); never triggers an LLM call, even if the entry had `triggers_llm_call=True`
- `update(key, value)` — shifts the entry's current version into `previous` and builds a new current version (new timestamp; new `id` if `value` is non-`None`, otherwise the same `id` as before — see `WorldEntryVersion` above), evaluates trigger conditions, and (currently) triggers an LLM call whenever `triggers_llm_call` is set and the condition passes. Clearing a value (without unregistering) is just `update(key, None)` — there's no separate `delete`. Dispatches the trigger call and any listener callbacks in the background rather than waiting on them, so a slow callback (e.g. the trigger handler making an actual LLM call) never blocks the caller. The `WorldEntry` these callbacks receive is a snapshot taken at dispatch time, not a reference to the live, further-mutable entry — otherwise a fast follow-up `update()` on the same key could race with a still-running callback reading stale in-place fields. This snapshot is cheap and safe to take because `WorldEntryVersion` objects are immutable once created — only the outer `WorldEntry`'s `current`/`previous` references ever change
- `get(key)` — returns the current value stored for a key
- `add_listener(key, callback)` / `remove_listener(key, callback)` — subscribe/unsubscribe a callback invoked whenever the entry at `key` is updated, dispatched in the background alongside the trigger handler (see `update()` above) rather than synchronously within `update()`
- `render_entry(key)` — serializes a single entry into the XML-style block described below, *regardless* of that entry's `include_in_prompt` — an explicit per-key request bypasses the aggregate-prompt filter, since the caller (e.g. a trigger handler wanting just the entry that fired, rather than the whole World) is asking for this key specifically
- `render_full_prompt()` — serializes all `include_in_prompt` entries into a single semantic text string, ordered by timestamp, by calling `render_entry(key)` for each and concatenating the blocks

`render_full_prompt()` is named to leave room for an alternative, incremental rendering path — see open question below; the name should not be assumed final until that's resolved. `render_entry(key)` is a first building block for that path: a trigger handler can call it directly on the entry that fired instead of re-rendering the whole World, without `World` itself having to pick which strategy is "right."

### Rendered entry format

Each entry — whether from `render_entry(key)` or as part of `render_full_prompt()` — is wrapped in an XML-style tag, chosen because Claude tracks tag boundaries and attribute values reliably — both reading them and echoing them back precisely, which matters since the model may need to reference a `key`/`id` exactly (e.g. in a Command):

```
<entry key="user_profile" id="a1b2c3d4">
Jane is logged in as an administrator, currently editing the billing page.
Updated: 2026-07-20T14:32:10Z
</entry>
```

- **Prefix** (opening tag attributes): `key` and `id` — structured and unambiguous, cheap for the model to copy verbatim. When the entry's `id` is `None` (no value has ever been set — the *only* time `id` is `None`; a subsequent clear keeps the last real `id`, see `WorldEntryVersion` above), the `id` attribute is omitted entirely rather than printed as the literal string `"None"` — there's no version to reference, so the tag is just `<entry key="...">`.
- **Content**: the untouched output of `serialize_fn(value, previous_value)`.
- **Trailing line** (inside the tag, after the content): a plain `Updated: <ISO 8601 timestamp>` line, deliberately not an attribute — visually separating stable identity (`key`/`id`, referenceable) from freshness (informational only), while staying structurally inside `<entry>...</entry>` so its association with this entry (and not a neighboring one) is unambiguous rather than relying on line ordering. The timestamp is always **absolute**, never relative phrasing ("updated 5s ago"): once text is sent to the LLM it's frozen, so a pre-rendered relative time would silently go stale as soon as that message sits in history (or a cached prompt prefix) longer than the phrasing implies. If recency framing is wanted, it's the model's job at inference time, informed by a small "current time" marker injected fresh outside the cached/stable part of the prompt on each call — not stored on the entry or persisted into history.

`render_full_prompt()` concatenates these blocks, one per included entry (each produced by `render_entry(key)`), in timestamp order.

## Open questions

1. **Concurrent trigger coalescing** — when multiple `triggers_llm_call` entries update within the same agent step, should each trigger its own LLM call immediately, or should updates be batched into a single LLM call right before the next turn? *(Currently: always trigger immediately.)*
2. **Full vs. incremental serialization** — should each LLM call resend the fully re-serialized World (`render_full_prompt()`), or only the data that triggered the call (`render_entry(key)`), layered on top of existing conversation history? Both will be tried empirically — `render_entry(key)` now exists as the building block for the incremental path, but which strategy an Agent loop actually uses per trigger is still undecided.
3. **Stale-`id` handling** — what happens when something (e.g. a Command) references an `id` that no longer matches the entry's current `id`? The World now guarantees `id` survives a clear specifically so this kind of check remains possible even against an emptied entry, but the actual mismatch-handling behavior is likely primarily a Commands-spec concern — the World may need to expose something like `get_by_id(key, id)` to support detecting this.
