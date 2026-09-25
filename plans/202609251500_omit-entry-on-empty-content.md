# Omit an entry from the prompt when its serializer returns `[]`

**Status:** Done

Let a World entry opt out of the rendered prompt **per render**, by having its `serialize_fn` (or `archival_serialize_fn`) return an empty `Content`. Today an empty body still produces a hollow `<entry key=… id=…>` envelope with a blank line and an `Updated:` stamp, so there is no way for an entry to hide itself while it has nothing worth showing (e.g. `None` before the first perception): the only switch is `include_in_prompt`, fixed at `register()`. Implements the `Updated` parts of [world.md](../specs/world.md) ("Omitting an entry"), [inputs.md](../specs/inputs.md) (`serialize_fn` row) and [agent.md](../specs/agent.md) ("Rendering to messages", omitted entries). Flip all three back to `Implemented` when this plan is `Done`.

Design decisions:

- **`[]` is the signal, not `None`.** `Content` is `list[ContentPart]`, and `[]` is already a valid `Content`, so the serializer signature does not change and no `Optional` leaks into every registration. An empty body was never useful (a hollow envelope tells the model nothing), so giving it a meaning breaks nothing anyone relied on.
- **The World decides at `render_entry`, not in the aggregate.** `render_entry` returns `[]` when the selected serializer does — no envelope at all — so *every* consumer (the World's own `render_full_prompt`, the Agent's history renderer, any caller passing an explicit `serialize_fn`) omits the entry with no extra logic. The envelope is only ever emitted around a non-empty body.
- **Fresh and archival decide independently.** Each render consults one serializer; an entry can show while fresh and vanish from history, or the reverse.
- **Omission is rendering-only.** The entry is still stored, versioned, dispatches listeners and can still trigger a step. A trigger entry that renders to nothing still starts a step.
- **An all-omitted observation still renders as a user message**, holding one fixed placeholder text block. Providers reject empty user content and the first message after the system prompt must be a user turn, so the Agent cannot just skip it. When such an observation merges into a preceding observation's user message it contributes nothing, and a placeholder left by an earlier all-omitted observation is replaced once real blocks join it.
- **Command entries are unaffected**: the Agent's own command serializer never returns `[]`, so agent.md's "never skipped" rule for `agent:command:<call_id>` holds.

## Steps

### 1. Specs

- [world.md](../specs/world.md): `serialize_fn`/`archival_serialize_fn` rows, `render_entry`/`render_full_prompt` bullets, a new "Omitting an entry" subsection under "Rendered entry format"; status `Updated`.
- [inputs.md](../specs/inputs.md): `serialize_fn` row; status `Updated`.
- [agent.md](../specs/agent.md): "Rendering to messages" — omitted entries and the placeholder rule; status `Updated`.
- [specs/_index.md](../specs/_index.md) rows in sync.

### 2. World (`src/wica/world.py`)

- `render_entry`: after calling the selected serializer, return `[]` if the body is empty; otherwise wrap as before. `render_full_prompt` needs no change (it concatenates).

### 3. Agent (`src/wica/agent.py`)

- `_render_messages`: a module constant `_EMPTY_OBSERVATION` for the placeholder text. When an observation's blocks are empty and it does not merge into a preceding user message, append a user message holding the placeholder block. When merging, drop a leading placeholder from the previous message once real blocks exist, and re-add it only if the merged result is still empty.

### 4. Tests

- `tests/test_world.py`: `render_entry` returns `[]` for an empty fresh body, for an empty archival body (independently of the fresh one), and for an explicit `serialize_fn`; `render_full_prompt` leaves the entry out entirely (no key, no hollow envelope) and shows it again once its serializer has something to say.
- `tests/test_agent.py`: an omitted entry is absent from the observation's user message and appears once it has a value; an all-omitted observation renders as the placeholder user message, never an empty one, and two such consecutive observations merge into a single placeholder.

### 5. Verification

`uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`; then flip the three specs to `Implemented` and this plan to `Done`.
