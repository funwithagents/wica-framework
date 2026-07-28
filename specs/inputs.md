# Inputs

**Status:** Stable

## Purpose

An **Input** is external multimodal data entering the World — the **I** in WICA (World / Inputs / Commands / Agents). A user's typed utterance, a "closest person detected" sensor event, a camera frame, a microphone clip: whatever the outside world *pushes at* the agent so it has something to perceive and react to.

Unlike World, Content, Commands, and the Agent, **Inputs have no module and no type of their own.** An Input is not a class you instantiate — it's a **role a World entry plays**: a [World](world.md) entry that is *fed by an external producer* rather than by the agent's own actions. Everything an Input needs already exists in the World's `register()`/`update()` API ([world.md](world.md)) and the `Content` model ([content.md](content.md)); this spec describes the **pattern**, not new machinery.

Concretely, an Input is a World entry where:

- an **external producer** (UI, sensor, hardware driver, network callback — anything outside the Agent) calls `update(key, value)` when new data arrives, and
- the entry is usually registered with **`triggers_llm_call=True`**, so a fresh perception prompts the Agent to think.

That's the whole idea. "Input" is a way of *talking about* a subset of World entries by their role, so the four-pillar vocabulary stays complete — not a layer of code between the producer and the World.

## What makes an entry an Input

An Input is distinguished from other World entries by **who writes it** and **why**, not by any structural marker:

- **Inputs are written from outside the Agent** — the producer is a sensor, a UI event, a hardware callback. The Agent only ever *reads* an Input (it renders it into the prompt and reasons about it); it never produces one.
- **Agent-produced entries are not Inputs.** Command-execution entries (`agent:command:<call_id>` — see [commands.md](commands.md)), the Agent's own activity state (`agent:activity[...]` — see [agent.md](agent.md)), and entries the model sets via Commands (in the demo, `emotion` and `tracked_user`) are all written *by the Agent as a result of its own reasoning*. They're World state, but they're not perception coming in.
- **Purely internal/bookkeeping entries are not Inputs either** — e.g. coordination state one module hands another with `include_in_prompt=False` (see [world.md](world.md), "Shared state"). Those don't represent the outside world entering; they're internal plumbing.

So the same World primitive backs all three, and the World itself draws no line between them. "Input" is the design label for *external-perception* entries — useful for reasoning about the system (and for consumers like the demo's `on_trigger`, which shows Inputs on the conversation's input side and filters `agent:command:*` re-triggers out itself — see [agent.md](agent.md)).

## Defining an Input

An Input is defined exactly like any other World entry — with `register()` — and then driven with `update()` from its producer. Two Inputs from the conversation demo ([conversation-demo.md](conversation-demo.md)):

```python
world.register("speech_input", str, serialize_fn=_speech, triggers_llm_call=True)
world.register("closest_user", str, serialize_fn=_closest_user, triggers_llm_call=True)
```

and, when the outside world produces something:

```python
world.update("speech_input", text)   # the user said something
world.update("closest_user", uid)    # a person stepped in front of the robot
world.update("closest_user", None)   # ...and then walked away
```

The `register()` fields ([world.md](world.md), "Registration and WorldEntryConfig") carry all of an Input's behaviour; the ones that matter most for Inputs:

| Field | Why it matters for an Input |
|---|---|
| `serialize_fn` | Turns the raw perception into `Content` for the prompt — `[TextPart("The user said: …")]` for speech, an `ImagePart` for a camera frame, an `AudioPart` (once it exists — see [content.md](content.md)) for a mic clip. This is where a sensor reading becomes something the model can read |
| `triggers_llm_call` | Almost always `True` for an Input: a new perception is exactly the event that should wake the Agent. An Input registered `False` is one the Agent *sees but doesn't react to on arrival* (ambient context read on the next step that some other entry triggers) |
| `trigger_condition_fn` | Gates *which* perceptions are worth a step. `update()` never auto-skips unchanged values (see [world.md](world.md)), so an Input that should only fire on genuine change encodes it here: `trigger_condition_fn=lambda old, new: old != new`. Conversely, a heartbeat-style Input legitimately re-triggers on an unchanged value |
| `archival_serialize_fn` | Lets a heavy Input render rich while fresh and light once stale — a camera frame as an inline `ImagePart` on the turn it arrives, a one-line text description on later turns — so multimodal perception isn't re-sent every step (see [agent.md](agent.md), "Freshness policy") |
| `ttl` | Makes an Input **transient**: a perception that should decay to `None` if not refreshed (a "someone is speaking" flag, a proximity reading). The World resets it via `update(key, None)` after the TTL, which *doesn't* itself trigger a call (see [world.md](world.md)) |

Nothing here is Input-specific API — it's the ordinary World config, described from the Input's point of view.

## Multimodality

Inputs are the primary reason `Content` is multimodal. An Input's `serialize_fn` returns `Content` (an ordered list of `TextPart`/`ImagePart`/… parts — see [content.md](content.md)), so a camera Input yields an `ImagePart`, a microphone Input an audio part, a text Input a `TextPart`. Because everything upstream of the Agent speaks `Content`, an Input producer never touches LangChain or any provider SDK — the Agent's adapter is the only place a raw image becomes a provider image block (see [content.md](content.md), [agent.md](agent.md)).

## Producers and threads

An Input's producer can live anywhere and run on **any thread** — a Gradio callback, a sensor polling loop, a hardware interrupt handler. The World is sync and thread-based specifically so `update()` can be called from these producers without them knowing about the Agent's asyncio loop (see [world.md](world.md) open question #4 and [agent.md](agent.md), "Threading"). The Agent bridges to its own loop; the Input producer just calls `world.update(...)` and moves on.

## Relationship to the other pillars

- **World** ([world.md](world.md)) — the store an Input writes into; Inputs are World entries, full stop.
- **Content** ([content.md](content.md)) — what an Input serializes to; the neutral currency that keeps producers provider-agnostic.
- **Agent** ([agent.md](agent.md)) — the consumer; a `triggers_llm_call` Input update is what starts an Agent step, and the Agent renders the Input (fresh, then archival) into the prompt.
- **Commands** ([commands.md](commands.md)) — the mirror image: Commands are the Agent acting *out* on the World, Inputs are the world coming *in*. Both are just World entries with different producers (the Agent vs. the outside).

## Open questions

1. **Consume-on-trigger / transient perception.** A perception event is often meaningful *once* — it should prompt one step and then be gone, not linger in the World (and the prompt) as stale state. Today the closest tools are a short `ttl` (decays to `None` after a delay) or the producer manually clearing with `update(key, None)`. A first-class option — e.g. an entry that auto-resets right *after* it triggers an LLM call, and/or once the Agent has "taken it into account" — is sketched in `specs/_todo.md` ("specific TTL for destruction after LLM trigger", "add a parameter next to LLM for destruction on LLM trigger and/or taken into account") but not designed or built. Until then, transient Inputs use TTL or explicit clearing.
2. **Input provenance / metadata.** There is currently no structural marker that an entry *is* an Input (vs. agent-produced or bookkeeping state) — consumers that care (e.g. the demo's `on_trigger`) distinguish by convention/key-prefix filtering. Whether Inputs deserve an explicit tag (for routing, UI, or "which entries did the model take into account" — also flagged in `_todo.md`) is open; the four-pillar model treats "Input" as a role, not a type, and that has sufficed so far.
3. **Ergonomic Input construction.** Registering an Input is the same multi-argument `register()` call as any entry. If a small set of Input shapes recur (a text Input, an image Input), a thin helper (`register_text_input(key)`, …) could cut boilerplate — deferred until the verbosity actually bites, mirroring the same deferral for `Content` construction ([content.md](content.md) open question #1).
