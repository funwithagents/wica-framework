# World content-model update

Introduces the neutral `Content` model ([specs/content.md](../specs/content.md)) and switches World serialization/rendering from `str` to `Content`, plus adds the fresh/archival serialization split ([specs/world.md](../specs/world.md)). Builds on the shipped World registry (`src/wica/world.py`).

## Scope

- `src/wica/content.py` — **new** neutral module: `TextPart`, `ImagePart`, `ContentPart`, `Content`, each part with `to_string()`. No LangChain / provider imports.
- `src/wica/world.py` — `serialize_fn`/`archival_serialize_fn` return `Content`; `render_entry(entry, *, archival=False) -> Content`; `render_full_prompt() -> Content`.
- `src/wica/__init__.py` — export `TextPart`, `ImagePart`, `ContentPart`, `Content` (alongside the existing World exports).
- `tests/test_content.py` — **new**, functional tests for the parts + flattening.
- `tests/test_world.py` — update rendering assertions to the `Content` shape.

## `wica/content.py`

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TextPart:
    text: str
    def to_string(self) -> str:
        return self.text

@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str        # e.g. "image/png"
    def to_string(self) -> str:
        return f"[image {self.media_type}]"

ContentPart = TextPart | ImagePart     # open union; more parts later
Content = list[ContentPart]
```

- Frozen dataclasses, matching the World's immutable-snapshot discipline.
- `to_string()` is per-part (not a central `match`) so a future part type carries its own text fallback.
- No helper/normalization for text-only construction yet — callers write `[TextPart("...")]` (content.md open question #1).

## `wica/world.py` changes

**`WorldEntryConfig`:**

```python
@dataclass
class WorldEntryConfig:
    type: type
    serialize_fn: Callable[[Any, Any], Content]
    archival_serialize_fn: Callable[[Any, Any], Content] | None = None   # defaults to serialize_fn
    include_in_prompt: bool = True
    triggers_llm_call: bool = False
    trigger_condition_fn: Callable[[Any, Any], bool] | None = None
    ttl: timedelta | None = None
```

**`register[T]`:** add `archival_serialize_fn: Callable[[T | None, T | None], Content] | None = None` keyword; `serialize_fn` return type becomes `Content`. Store both on the config.

**`render_entry`:**

```python
def render_entry(self, entry: WorldEntry, *, archival: bool = False) -> Content:
    config = ...  # KeyError if unregistered, unchanged
    fn = config.serialize_fn
    if archival and config.archival_serialize_fn is not None:
        fn = config.archival_serialize_fn
    previous_value = entry.previous.value if entry.previous is not None else None
    body = fn(entry.current.value, previous_value)
    opening = TextPart(f'<entry key="{entry.key}" id="{entry.current.id}">\n')
    closing = TextPart(f"\nUpdated: {entry.current.timestamp.isoformat()}\n</entry>")
    return [opening, *body, closing]
```

- Note the wrapper is `TextPart`s bracketing the body parts; **no** merging of adjacent `TextPart`s here — that's the Agent adapter's concern.

**`render_full_prompt`:** select `include_in_prompt` entries, sort by timestamp, `render_entry(entry)` (fresh) each, and flatten the per-entry `Content` lists into one `Content`. (Previously joined strings with `"\n"`; now concatenate parts. To preserve the between-block newline that the string-join gave, the closing/opening `TextPart`s already carry their own `\n`; confirm the concatenated text matches the old block separation in a test.)

## Type safety

Only `register[T]` is generic, unchanged: `type: type[T]` binds `T`, so a `serialize_fn`/`archival_serialize_fn` whose parameters don't match the declared type is caught at the call site. Everything else stays `Any`-typed with the runtime `isinstance` check in `update()`.

## Tests

**`tests/test_content.py`** (functional):
- `TextPart("hi").to_string() == "hi"`.
- `ImagePart(b"...", "image/png").to_string() == "[image image/png]"`.
- Flattening a mixed `Content` (`[TextPart, ImagePart, TextPart]`) via `"".join(p.to_string() for p in content)` yields the expected text with the image placeholder in position.

**`tests/test_world.py`** (update existing rendering tests):
- `render_entry` returns a `Content` (list of parts): a text-only entry → `[TextPart(open), *serialize_body, TextPart(close)]`; flattening it reproduces the exact old `<entry ...>…</entry>` block (keeps the format contract).
- `archival=True` uses `archival_serialize_fn` when provided (assert the parts differ from fresh — e.g. fresh returns `[ImagePart(...)]`, archival returns `[TextPart("a photo")]`); falls back to `serialize_fn` when `archival_serialize_fn` is `None`.
- Multimodal entry: fresh `render_entry` places the `ImagePart` between the two wrapper `TextPart`s.
- `render_full_prompt` returns one `Content` for all `include_in_prompt` entries in timestamp order; flattening equals the concatenation of each `render_entry` flatten (and excluded entries are absent).
- `serialize_fn` still receives the right `previous_value` across `A → B → None` (now asserted through the returned parts).

## Out of scope / deferred

- Ergonomic `text()` / bare-`str` construction (content.md open question #1).
- `AudioPart` / `FilePart` and other part types (content.md open question #2).
- The `Content -> provider message blocks` adapter and adjacent-`TextPart` merging — Agent-side (content.md open question #3, [specs/agent.md](../specs/agent.md)).
- Fresh/archival *policy* (when an entry counts as archival) — Agent-side; the World only exposes the `archival` flag.
