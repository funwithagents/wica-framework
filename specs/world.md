# World

**Status:** Draft

## Purpose

The World is WICA's store of current context/state. It holds arbitrary registered objects — including multimodal data — and can be rendered into a semantic text string suitable for inclusion in an LLM prompt.

## Core concepts

### WorldEntry

Each piece of state registered in the World is wrapped in a `WorldEntry` object with the following fields:

| Field | Description |
|---|---|
| `key` | Stable identifier for this entry's slot in the World (dictionary key); unchanged across updates |
| `id` | Identifier for this specific *version* of the entry's value; regenerated on every `update()`. Included in the serialized output alongside `key` so the LLM can reference a specific entry/version (e.g. from a Command) — a mismatched `id` signals the data has changed since it was last seen |
| `type` | Declared type of the stored value, used for validation on register/update |
| `value` | The actual stored object (any type, including multimodal data — e.g. images, audio) |
| `include_in_prompt` | Whether this entry is included when rendering the World prompt |
| `triggers_llm_call` | Whether updating this entry's value should trigger a new LLM call |
| `trigger_condition_fn` (optional) | Extra condition function evaluated on update; must pass, in addition to `triggers_llm_call`, for the update to actually trigger the LLM call |
| `serialize_fn` | Function that converts `value` into a semantic string for the rendered prompt (value only — does not receive the timestamp) |
| `timestamp` | Updated whenever `value` changes; used to order entries when rendering and stamped, in absolute form, onto the rendered entry |
| `ttl` (optional) | Duration since the last update after which the value automatically resets to `None` via `update(key, None)`, if not refreshed before then. Enforced actively via a timer, not lazily on read; the timer restarts on every update |

### World

The World is a **singleton** — one instance exists per process/run.

The World exposes the following API:

- `register(...)` — registers a new entry; can happen at any time, including mid-run
- `unregister(key)` — removes an entry entirely (registration and value both gone)
- `update(key, value)` — updates an entry's value and timestamp, evaluates trigger conditions, and (currently) triggers an LLM call whenever `triggers_llm_call` is set and the condition passes. Clearing a value (without unregistering) is just `update(key, None)` — there's no separate `delete`
- `get(key)` — returns the current value stored for a key
- `add_listener(key, callback)` / `remove_listener(key, callback)` — subscribe/unsubscribe a callback invoked whenever the entry at `key` is updated
- `render_full_prompt()` — serializes all `include_in_prompt` entries into a single semantic text string, ordered by timestamp

`render_full_prompt()` is named to leave room for an alternative, incremental rendering path — see open question below; the name should not be assumed final until that's resolved.

### Rendered entry format

Each included entry is wrapped in an XML-style tag, chosen because Claude tracks tag boundaries and attribute values reliably — both reading them and echoing them back precisely, which matters since the model may need to reference a `key`/`id` exactly (e.g. in a Command):

```
<entry key="user_profile" id="a1b2c3d4">
Jane is logged in as an administrator, currently editing the billing page.
</entry>
Updated: 2026-07-20T14:32:10Z
```

- **Prefix** (opening tag attributes): `key` and `id` — structured and unambiguous, cheap for the model to copy verbatim.
- **Content**: the untouched output of `serialize_fn`.
- **Suffix**: a plain `Updated: <ISO 8601 timestamp>` line after the closing tag, deliberately not an attribute — visually separating stable identity (`key`/`id`, referenceable) from freshness (informational only).

`render_full_prompt()` concatenates these blocks, one per included entry, in timestamp order.

## Decided

- State is snapshot-based: the World holds current state only, not an event-sourced log of Commands
- Multimodal objects are stored directly in the World (not by reference to an external blob store)
- Serialization target is a semantic string, meant to be easily interpreted by an LLM
- Single global World for now (no per-agent partitioning/nesting)
- The World is a singleton
- `type` is used for typing/validation purposes (checked on register/update)
- Registration/unregistration can happen dynamically, mid-run
- Entries are ordered by `timestamp` when rendered
- `update(key, value)` raises if `value`'s type doesn't match the type declared at `register`
- `unregister(key)` does not trigger an LLM call, even if the entry had `triggers_llm_call=True`
- Each rendered entry carries its `timestamp` in **absolute** form (e.g. an ISO 8601 string), appended by the World's render step — not by `serialize_fn`. Relative/recency phrasing ("updated 5s ago") is never baked into entry serialization: once text is sent to the LLM it's frozen, so a pre-rendered relative time silently goes stale as soon as that message sits in history (or a cached prefix) for longer than the phrasing implies. If recency framing is wanted, it's the model's job at inference time, informed by a small "current time" marker injected fresh outside the cached/stable part of the prompt on each call — not stored on the entry or persisted into history.
- Each rendered entry also carries its `key` and `id` (see `WorldEntry` above), so the LLM can refer back to a specific entry/version — e.g. from a Command
- Rendered entries use the XML-style wrapper described in "Rendered entry format" above: `key`/`id` as opening-tag attributes, `serialize_fn` output as content, `Updated: <ISO 8601>` as a trailing suffix line
- `None` is always a valid value for `update(key, None)`, regardless of the entry's declared `type` — exempt from the type-check that otherwise raises on mismatch. This is also how a value gets cleared; there's no separate `delete`
- `id` is a per-key incrementing integer (`1, 2, 3, ...`), not a UUID/hash — any reference to an entry always pairs `id` with `key`, so global uniqueness isn't needed, and small integers are far easier for the LLM to reproduce exactly than a hex string
- Per-key `id` counters live in a dictionary separate from the live `WorldEntry` objects, and are never cleared — so re-registering a previously unregistered key continues its `id` sequence rather than restarting at 1, avoiding a stale reference colliding with a reused `id`
- `ttl` resets are enforced actively (a timer firing at expiry), not by lazily checking on next read; the timer restarts on every `update()`. A TTL-driven reset resets the value to `None` but does **not** trigger an LLM call, even if `triggers_llm_call=True` — unlike an explicit `update(key, None)` call, which follows the normal trigger logic like any other update

## Open questions

1. **Concurrent trigger coalescing** — when multiple `triggers_llm_call` entries update within the same agent step, should each trigger its own LLM call immediately, or should updates be batched into a single LLM call right before the next turn? *(Currently: always trigger immediately.)*
2. **Full vs. incremental serialization** — should each LLM call resend the fully re-serialized World (`render_full_prompt()`), or only the data that triggered the call, layered on top of existing conversation history? Both will be tried empirically; the outcome may require a second, complementary render function (and could affect naming of the first).
3. **Stale-`id` handling** — what happens when something (e.g. a Command) references an `id` that no longer matches the entry's current `id`? Likely belongs primarily to the Commands spec, but the World may need to expose something like `get_by_id(key, id)` to support detecting this.
