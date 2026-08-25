---
code:
  - src/wica/content.py
tests:
  - tests/test_content.py
---

# Content

**Status:** Implemented

## Purpose

`Content` is WICA's neutral, provider-agnostic representation of a piece of multimodal material — text, an image, and (later) audio, files, etc. It is the common currency between the layers that produce material and the layer that talks to a model:

- The **World** returns `Content` from entry serialization (`serialize_fn` / `archival_serialize_fn`, `render_entry`, `render_full_prompt` — see [world.md](world.md)).
- The **Agent** converts `Content` into a concrete provider's message blocks at its I/O boundary — the *only* place a specific model SDK (LangChain) is imported (see [agent.md](agent.md)).
- **Inputs** use it to present multimodal perception. Commands report their state/results through
  World serialization, although the real-world effect of a Command (speech, movement, display,
  API action) is not itself required to be a `Content` return value.

The whole point is decoupling: everything that enters the model-facing prompt speaks `Content` and stays ignorant of LangChain or any model provider. Swapping the SDK, or supporting a second one, touches only the Agent's adapter — never the World, Inputs, Commands' World serialization, or a registrant's `serialize_fn`. WICA's broader “multimodal output” claim refers to Command-mediated action in the physical or digital world; the v1 output sink itself remains complete text.

`Content` lives in a neutral `wica/content` module that depends on **nothing** LLM- or provider-specific.

## Core concepts

### Parts

`Content` is an ordered list of typed *content parts*:

```python
Content = list[ContentPart]
ContentPart = TextPart | ImagePart      # open union — more parts expected later
```

The initial part types:

| Part | Fields | `to_string()` |
|---|---|---|
| `TextPart` | `text: str` | the text verbatim |
| `ImagePart` | `data: bytes`, `media_type: str` (e.g. `"image/png"`) | a short textual placeholder, e.g. `[image image/png]` |

`ContentPart` is an open union: audio, files, and other modalities are expected to be added, and adding one should not force changes on consumers that don't handle it (a text-flattening consumer just falls back to that part's `to_string()`).

Parts are immutable (frozen dataclasses), consistent with the World handing out immutable snapshots.

### `to_string()` — the text stand-in

Every part defines a `to_string() -> str` method returning its plain-text representation: the text itself for `TextPart`, a short placeholder for non-text parts (e.g. `[image image/png]`). This is deliberately per-part rather than a central `match` over the union, so a new part type carries its own text fallback with it.

Flattening a whole `Content` to text — for logging, debugging, or a text-only consumer that can't render parts — is just concatenating `part.to_string()` across the list. This is what keeps the World loggable and testable now that its rendering is multimodal rather than a plain string.

## Open questions

1. **Ergonomic construction.** For now, building text content is explicit — `[TextPart("...")]`. A convenience (a `text("...") -> Content` helper, or accepting a bare `str` and normalizing) is deferred until the verbosity actually bites.
2. **Part set growth.** Only `TextPart` and `ImagePart` exist initially. `AudioPart` (robot mic input), `FilePart`/document parts, and possibly a structured/JSON part will be added as concrete needs appear; each must ship with a sensible `to_string()`.
