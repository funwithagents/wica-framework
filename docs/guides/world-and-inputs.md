# The World and its Inputs

The World is the agent's state: a registry of named, typed entries that renders into the prompt at every reasoning step. An **Input** is not a class; it is the role an entry plays when an external producer feeds it, a speech recognizer, a camera, a sensor, a UI. This page is how to register an entry, how it reaches the model, and what wakes the Agent. The API in one place is [../reference/api.md](../reference/api.md#the-world); the design is [specs/world.md](../../specs/world.md) and [specs/inputs.md](../../specs/inputs.md).

## Registering an entry

```python
from wica import TextPart

wica.world.register(
    "closest_user",
    str,
    serialize_fn=lambda value, previous: (
        [TextPart(f"{value} is standing in front of you.")] if value else []
    ),
    triggers_llm_call=True,
)
```

- **`key`** is how the entry is named everywhere: in `update()`, in `get()`, and in the prompt, where the model sees `<entry key="closest_user" id="3">`. A key is a non-empty string with no whitespace and none of `" < > &`; `agent:` is the framework's prefix.
- **`type`** is checked at every `update()`: a value of another type raises `TypeError`. `None` is always allowed and means "nothing", which is also an entry's initial value.
- **`serialize_fn(value, previous)`** turns the value into the `Content` the model reads: a list of `TextPart` and `ImagePart`. It receives the previous value too, so a render can say what changed. Values arrive as defensive copies; the function must not depend on mutating them.
- **Registration needs no running system**, and may happen at any time, mid-run included. Register everything before `start()` in the usual flow.

Register one entry per fact the model should know: what the person said, who is in view, how the robot feels, what it is doing. The prompt is the rendering of the current World, in timestamp order, so a reasoning step sees every entry together.

## Updating it

```python
wica.world.update("closest_user", "Alice")   # someone stepped up
wica.world.update("closest_user", None)      # ...and walked away
```

`update()` is callable **from any thread** (a Gradio callback, a sensor poll loop, a hardware interrupt): it validates and deep-copies the value, advances the entry's version (`id` plus one, a new timestamp) under a lock, and hands the dispatch to the shared loop. It needs the system **running**: between `start()` and `stop()`; otherwise it raises `RuntimeError`. Reads (`get`, `get_entry`) work at any time and return copies.

The World keeps the current and the previous version of an entry, not a history: it is a snapshot of the present. The Agent keeps the history, as re-renderable snapshots of what it observed at each step.

## What the model sees

Each included entry renders inside an XML-style envelope the World owns, with the key and the version id the model can quote back, the serializer's parts, and an absolute timestamp:

```
<entry key="closest_user" id="3">
Alice is standing in front of you.
Updated: 2026-11-09T14:32:10Z
</entry>
```

- **`include_in_prompt=False`** makes an entry internal state the model never sees: shared data between Commands, a flag your listeners read.
- **An empty render omits the entry.** A `serialize_fn` that returns `[]` leaves no envelope, so the model never sees the key for that render; the entry is still stored, still versioned, and can still trigger. The example above omits `closest_user` while nobody is there.
- **`archival_serialize_fn`** is the lighter render the Agent uses for older observations in history, the current one rendering with `serialize_fn`. A camera frame can be the inline image while fresh and a line of text afterwards; the deep, cached part of the prompt stays small and byte-stable.
- **Multimodal content** is `ImagePart` next to `TextPart` in the same list ([specs/content.md](../../specs/content.md)).

## What wakes the Agent

- **`triggers_llm_call=True`** is what makes an entry an Input: every update that commits wakes the Agent, which observes the whole World (not just the entry that fired) and reasons. Entries left at the default never wake it and are context only.
- **`trigger_condition_fn(old, new) -> bool`** gates which updates trigger: `lambda old, new: new is not None` wakes the Agent when someone appears, not when they leave; `lambda old, new: old != new` ignores repeats. It is evaluated before the update commits; if it raises, nothing changes.
- **A Command's completion also wakes the Agent** (by default), so the model sees the outcome of what it did ([commands.md](commands.md)).
- **A trigger that arrives while a step is in flight is dropped**, not queued: v1 runs one reasoning step at a time. Its value is in the World for the next step to see, but nothing wakes the Agent for it. `wica.on_agent_trigger_dropped` tells you when it happens ([limits.md](limits.md)).

## Bursts are coalesced

Perceptions arrive in bursts: a speech event and a proximity change within a few milliseconds, a frame and its detection. The first trigger opens a **window** (200 ms by default, `Wica.init(config, coalesce_window=…)`); triggers arriving while it is open join it without extending it; when it closes, one step runs and observes them all. Latency is bounded by the window, and a fast stream can never starve the step.

- **`bypass_coalescing=True`** on an entry closes the window at once when that entry updates: a stop button, a barge-in utterance. It skips the window, not the in-flight drop.
- **`coalesce_window=0`** disables batching: every trigger fires immediately.

## Expiring values

`ttl=timedelta(seconds=5)` resets the entry to `None` five seconds after each update, with ordinary listener dispatch but no trigger: the Agent is not woken by an expiry. A "someone just spoke" entry that should not linger in the prompt, a detection that is stale after a few seconds. TTL time keeps elapsing while the system is stopped; an expired value is reset at the next `start()`.

## Listening to changes

Your own code can watch an entry too: `world.add_listener(key, callback)` calls `callback(entry)` on every update, with a `WorldEntry` snapshot (`entry.current.value`, `entry.previous`), on the shared loop; sync callbacks run on its thread pool, async ones as tasks, and an exception in one is logged, never raised to `update()`'s caller. This is how a UI mirrors the World, and how the transcript panel follows a Command's state.
