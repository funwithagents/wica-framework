# Consistency-audit fixes: reserved names, safe shutdown, snapshot rendering, namespaces, config invariants, transactional triggers

**Status:** Done

Implements the remaining items of the repository consistency audit recorded in
[specs/_fixes.md](../specs/_fixes.md) — the ones that need code, tests, or a design decision (the
wording-only findings were fixed on 2026-09-11, commit `21788f5`).

This plan is written to be executed step by step, in order, by an agent that has not read the
audit. Every step says **which file**, **what to change**, **what the result must look like**, and
**which test proves it**. Line numbers refer to the tree at commit `21788f5`; re-locate by the
quoted code if they have drifted. Read [AGENTS.md](../AGENTS.md) first for the verification and
status rules, then the five specs this plan edits: [commands.md](../specs/commands.md),
[wica.md](../specs/wica.md), [agent.md](../specs/agent.md), [world.md](../specs/world.md),
[config.md](../specs/config.md).

## Goal

Close six behavioral gaps. After this plan, each behavior below is written in its spec, implemented,
and pinned by a functional test, and `_fixes.md` has no implementation items left.

| # | Gap today | Decision |
|---|---|---|
| D1 | `Agent.register_command` silently overwrites a Command with the same name; the built-ins `noop`/`cancel_command` can be shadowed or shadow an app Command | Reserve the two names; duplicates raise `ValueError`; built-ins register through a private idempotent path |
| D2 | `Wica.stop()`/`close()` called from the owned loop thread (e.g. inside an output sink) tries to join its own thread and raises **after** tearing the Agent/World down; `Wica.start()` accepts an injected loop that is not running | Reject same-thread shutdown on an owned loop **before** any state change; reject a non-running injected loop at `start()` |
| D3 | History renders old observations with the World's **current** registration config, so `unregister()` breaks the next render and re-`register()` rewrites old prompts | Capture each entry's serializers at observation time and always render history from them |
| D4 | Any string is a World key (a `"` breaks the `<entry key="…">` envelope); provider tool-call ids are used verbatim as keys; rendering detects command entries by string prefix | Validate a key grammar at `register()`; use the provider id only when valid and free, else generate; drop the prefix check in rendering (D3 makes it unnecessary) |
| D5 | Direct `AgentConfig(...)` construction can set both/neither prompt field; configs are mutable | `__post_init__` invariants; `frozen=True` on both config dataclasses |
| D6 | `World._update` commits value/id/timer **before** running `trigger_condition_fn`, so a raising predicate leaves a half-applied update | Run the predicate first; if it raises, nothing changes and the error propagates |

## Ground rules for the implementer

- Work the steps **in order**; later steps assume earlier ones (D3 before D4's rendering change,
  D4's key validation before D4's call-id fallback).
- After **every** step run `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
  `uv run pytest`. Do not move on with a failure.
- Do not change public names other than those listed here. Do not add dependencies.
- Existing tests are the safety net: they must keep passing unchanged, except where a step says
  explicitly that a test is updated.
- Test style (from AGENTS.md): drive the public API, assert on observable results, never assert
  that a mock was called. Use the existing helpers named in each step.

## Step 0 — Bookkeeping: open the plan

1. In this file, change `**Status:** Todo` to `**Status:** In progress`; change the row in
   [plans/_index.md](_index.md) to `In progress`.
2. In each of `specs/commands.md`, `specs/wica.md`, `specs/agent.md`, `specs/world.md`,
   `specs/config.md`: change the `**Status:** Implemented` line to `**Status:** Updated`, and the
   matching row in [specs/_index.md](../specs/_index.md) to `Updated`.
3. Run `uv run pytest tests/test_project_map.py` (it must still pass; it does not check statuses,
   this is just a sanity check that the files are intact).

## Step 1 — Spec edits (design before code)

Make these edits now, so the code in later steps implements text that already exists. Keep the
prose style of the surrounding sections. Do not remove existing content unless told to.

### 1a. `specs/world.md`

- In "Registration and WorldEntryConfig", after the sentence about `register()` being generic,
  add a paragraph **"Key grammar"**: a key is a non-empty `str` with no whitespace and none of the
  four characters `"`, `<`, `>`, `&`; `register()` raises `ValueError` otherwise. Rationale: the
  key is interpolated verbatim into the `<entry key="…">` envelope and echoed back by the model, so
  it must be safe there without escaping. Keys beginning with `agent:` are used by the framework
  (`agent:command:<call_id>`); applications should avoid that prefix. This is a convention the
  World documents but does not enforce, so the World stays Agent-agnostic.
- In the "World API" list, add an entry after `get_prompt_entries`:
  `get_prompt_snapshot() -> list[tuple[WorldEntry, WorldEntryConfig]]` — same selection and order
  as `get_prompt_entries()`, but each entry is paired with a copy of its registration config, taken
  under one lock so the pair is consistent. Exists so the Agent can capture, at observation time,
  the serializers that governed each entry (see agent.md, "History"), and render that observation
  identically forever, regardless of later `unregister()`/`register()` calls.
- Add an entry `is_registered(key: str) -> bool` — unguarded read; `True` iff the key currently
  has a `WorldEntryConfig`.
- In the `update()` entry, replace "It evaluates the trigger condition over defensive copies and
  (currently) triggers an LLM call whenever `triggers_llm_call` is set and the condition passes."
  with: "It evaluates `trigger_condition_fn` **before committing** — over a deep copy of the old
  value and the new value, under the lock. If the predicate raises, the exception propagates to the
  `update()` caller and **nothing changes**: value, `id`, timestamp, TTL timer, listeners and
  trigger are all as before the call. If it returns `True` (or there is no predicate) and
  `triggers_llm_call` is set, the update triggers an LLM call."

### 1b. `specs/commands.md`

- In "The `Command` object", add a bullet **"Names are unique and two are reserved."**
  `noop` and `cancel_command` belong to WICA. `Agent.register_command()` raises `ValueError` for
  either name and for any name already registered. The optional output Command's name is checked
  at `Agent` construction (a reserved name raises `ValueError` there, so `Wica.init` fails fast)
  and is then reserved too. WICA's own Commands are attached by a private idempotent path used by
  `Agent.start()`, so a restart re-attaches them without ever replacing an application Command.
- In "Command execution as a World entry", after the first bullet, add: the `<call_id>` in the
  entry key is **WICA-owned**: the Agent uses the provider's tool-call id when it satisfies the
  World key grammar (world.md) and no entry with that key exists, and otherwise generates one
  (`uuid4().hex`). The provider's id is kept separately on the history record
  (`CommandRecord.tool_call_id`) for reconstructing the native `tool_call`/`tool_result` pair; the
  two are equal in the common case. The model only ever sees `call_id`.

### 1c. `specs/agent.md`

- In "History record shape", change the Observation row's Fields to
  `entries: list[ObservedEntry]` and its Notes to: each `ObservedEntry` is the `WorldEntry`
  snapshot **plus the `serialize_fn` and `archival_serialize_fn` that governed it at observation
  time** (captured via `World.get_prompt_snapshot()`); history renders through these captured
  serializers, never through the World's live registration, so an observation renders identically
  after the key is unregistered or re-registered.
- Same table: add `tool_call_id: str` to the Command row and the No-reaction row, noting it is the
  provider's id (used to reconstruct the native call) while `call_id` is the World-key suffix and
  `cancel_command` target (equal in the common case — see commands.md).
- In "Rendering to messages", replace the last sentence of the second paragraph (the one starting
  "Because the World must not know what a `CommandExecution` is…") with: "Command entries are
  registered with the Agent's own serializer, so the captured serializer already is that one — the
  renderer treats every observed entry uniformly and never inspects keys." Keep the surrounding
  paragraph.
- In "Commands", first bullet, append: "Command names are unique and `noop`/`cancel_command` are
  reserved — see commands.md, 'Names are unique and two are reserved'."

### 1d. `specs/wica.md`

- In "The event loop, restartable `start()`/`stop()`, and terminal `close()`", after the
  "Injected loop" paragraph, add **"Shutdown from the loop thread."** For an **owned** loop,
  `stop()` and `close()` called from that loop's own thread — from an output sink, a Command, or an
  Event subscriber — raise `RuntimeError` **before changing any state**; the system keeps running.
  A synchronous facade must join its thread, and there is no correct synchronous answer on that
  thread. To stop from inside the loop, hand the call to another thread (e.g.
  `threading.Thread(target=wica.stop).start()`). The injected-loop behavior described above (a
  same-thread `stop()` requests cancellation and returns) is unchanged.
- Same section: `start()` with an injected loop that is **not running** raises `RuntimeError`
  (in addition to the existing closed-loop check): World dispatch and the Agent's trigger shim need
  a running loop, so accepting a stopped one would queue callbacks forever.
- In "Open questions", add: **Async shutdown.** `astop()`/`aclose()` coroutines that a sink or
  subscriber could `await` to stop the system from the loop thread are a possible addition;
  deferred until a consumer needs it — today the rule is "reject on the owned loop thread".

### 1e. `specs/config.md`

- In "Plain dataclasses with dictionary and JSON loaders", replace "Callers may construct the
  dataclasses directly; that path relies on the caller and static type checking rather than running
  the loaders' validation." with: "Callers may construct the dataclasses directly. The **structural
  invariants** (exactly one of `system_prompt`/`system_prompt_file`, at most one of
  `api_key`/`api_key_env`) are enforced by `AgentConfig.__post_init__`, so direct construction fails
  with the same `ConfigError` the loaders raise; what the loaders add on top is the *shape*
  validation of external data (unknown keys, wrong types). Both dataclasses are `frozen=True`:
  derive a variant with `dataclasses.replace(...)` rather than assigning."
- In "Flow into the Agent", first bullet, replace "It does not run loader validation." with "It
  runs the invariant check but not the loaders' shape validation."

## Step 2 — D6: transactional trigger predicate — `src/wica/world.py`

**Where:** `World._update`, lines 224–317. The commit currently happens at lines 254–274
(`stored_value = copy.deepcopy(value)` … timer start) and the predicate runs afterwards at
lines 277–283.

**Change:** move the predicate evaluation to just after the `TypeError` check (line ~247) and
before `stored_value`'s commit. Concretely, inside the `with self._lock:` block, after the type
check:

```python
            stored_value = copy.deepcopy(value)
            # Evaluate the trigger predicate BEFORE committing, so a raising predicate leaves the
            # entry (value, id, timestamp, timer) untouched and the error reaches the caller.
            # See specs/world.md ("update()").
            triggers_configured = config.triggers_llm_call and not ttl_reset
            should_trigger = triggers_configured and (
                config.trigger_condition_fn is None
                or config.trigger_condition_fn(
                    copy.deepcopy(old_entry.current.value), copy.deepcopy(stored_value)
                )
            )
            new_id = self._id_counters[key] + 1
            ... (rest of the commit exactly as today)
```

and delete the later duplicate block (old lines 277–283). Keep `listeners = list(...)` where it is.
Nothing else in the method changes.

**Tests:** append to `tests/test_world.py` (fixtures `world`, `loop`; helpers `identity_serialize`,
`RecordingCallback`):

```python
def test_raising_trigger_condition_propagates_and_leaves_state_untouched(world: World):
    def boom(old, new):
        raise RuntimeError("predicate failed")

    world.register("k", str, serialize_fn=identity_serialize,
                   triggers_llm_call=True, trigger_condition_fn=boom)
    listener = RecordingCallback()
    world.add_listener("k", listener)
    trigger = RecordingCallback()
    world.on_trigger.subscribe(trigger)
    before = world.get_entry("k")

    with pytest.raises(RuntimeError, match="predicate failed"):
        world.update("k", "value")

    after = world.get_entry("k")
    assert after == before                 # same id, same (None) value, same timestamp
    assert not listener.event.wait(0.2)    # no listener dispatched
    assert not trigger.event.wait(0.2)     # no trigger emitted


def test_raising_trigger_condition_keeps_the_pending_ttl_timer(world: World):
    # A good update arms a TTL; a later raising update must not cancel or re-arm it.
    world.register("k", str, serialize_fn=identity_serialize, ttl=timedelta(seconds=0.3),
                   triggers_llm_call=True,
                   trigger_condition_fn=lambda old, new: (_ for _ in ()).throw(RuntimeError()) if new == "bad" else True)
    world.update("k", "good")
    id_after_good = world.get_entry("k").current.id
    with pytest.raises(RuntimeError):
        world.update("k", "bad")
    assert world.get("k") == "good"
    time.sleep(0.5)
    assert world.get("k") is None                           # the ORIGINAL timer fired
    assert world.get_entry("k").current.id == id_after_good + 1
```

(Write the second predicate as a small named function instead of the lambda trick if you prefer;
the behavior is what matters.)

## Step 3 — D4 part 1: key grammar and `is_registered` — `src/wica/world.py`

**Add** a module-level helper near the top (after `_describe_value`):

```python
_FORBIDDEN_KEY_CHARS = frozenset('"<>&')


def validate_key(key: str) -> None:
    """Raise ValueError unless `key` is safe to embed verbatim in an `<entry key="…">` envelope:
    non-empty, no whitespace, none of " < > &. See specs/world.md ("Key grammar")."""
    if not isinstance(key, str) or not key:
        raise ValueError("World key must be a non-empty string")
    if any(ch.isspace() for ch in key):
        raise ValueError(f"World key {key!r} must not contain whitespace")
    bad = sorted(set(key) & _FORBIDDEN_KEY_CHARS)
    if bad:
        raise ValueError(f"World key {key!r} must not contain {''.join(bad)!r}")
```

**Call it** as the first statement of `World.register()` (line 152, before `with self._lock:`):
`validate_key(key)`.

**Add** to `World` (next to `get_entry`):

```python
    def is_registered(self, key: str) -> bool:
        with self._lock:
            return key in self._configs
```

Export `validate_key` from `src/wica/__init__.py`? **No** — keep it module-level in `world.py`;
the Agent imports it from there. Do not add it to `__all__`.

**Tests:** append to `tests/test_world.py`:

```python
@pytest.mark.parametrize("bad", ["", "has space", "tab\tkey", 'q"uote', "a<b", "a>b", "a&b"])
def test_register_rejects_unsafe_keys(world: World, bad: str):
    with pytest.raises(ValueError):
        world.register(bad, str, serialize_fn=identity_serialize)
    assert not world.is_registered(bad)


@pytest.mark.parametrize("ok", ["speech_input", "agent:command:x", "a/b.c-d_e", "ünïcödé"])
def test_register_accepts_safe_keys(world: World, ok: str):
    world.register(ok, str, serialize_fn=identity_serialize)
    assert world.is_registered(ok)


def test_is_registered_reflects_register_and_unregister(world: World):
    assert not world.is_registered("k")
    world.register("k", str, serialize_fn=identity_serialize)
    assert world.is_registered("k")
    world.unregister("k")
    assert not world.is_registered("k")
```

Check that no existing test registers a key with a space or a quote (`grep -n 'register("' tests`);
if one does, rename that key.

## Step 4 — D3 part 1: `World.get_prompt_snapshot()` — `src/wica/world.py`

**Add** right after `get_prompt_entries` (line ~367):

```python
    def get_prompt_snapshot(self) -> list[tuple[WorldEntry, WorldEntryConfig]]:
        """Like get_prompt_entries(), but each entry is paired with a copy of its registration
        config, taken under one lock so the pair is consistent. The Agent captures the serializers
        from it at observation time so history renders identically regardless of later
        unregister/re-register. See specs/world.md."""
        with self._lock:
            pairs = [
                (entry, self._configs[key])
                for key, entry in self._entries.items()
                if self._configs[key].include_in_prompt
            ]
            pairs.sort(key=lambda pair: pair[0].current.timestamp)
            return [
                (copy.deepcopy(entry), dataclasses.replace(config)) for entry, config in pairs
            ]
```

Add `import dataclasses` at the top (keep the existing `from dataclasses import dataclass`).
`dataclasses.replace(config)` makes a shallow copy — the callables are shared, which is what we
want; do **not** deepcopy the config (deepcopying functions is pointless and deepcopying `type`
objects is fine but wasteful).

**Tests:** append to `tests/test_world.py`:

```python
def test_get_prompt_snapshot_pairs_each_entry_with_its_own_config(world: World):
    world.register("a", str, serialize_fn=identity_serialize)
    world.register("b", bytes, serialize_fn=image_serialize,
                   archival_serialize_fn=text_archival_serialize)
    world.register("hidden", str, serialize_fn=identity_serialize, include_in_prompt=False)
    snapshot = world.get_prompt_snapshot()
    assert [e.key for e, _ in snapshot] == [e.key for e in world.get_prompt_entries()]
    by_key = {e.key: c for e, c in snapshot}
    assert by_key["a"].serialize_fn is identity_serialize
    assert by_key["b"].archival_serialize_fn is text_archival_serialize
    assert "hidden" not in by_key


def test_get_prompt_snapshot_config_is_a_copy(world: World):
    world.register("a", str, serialize_fn=identity_serialize)
    (_, config), = world.get_prompt_snapshot()
    config.serialize_fn = lambda v, p: [TextPart("tampered")]
    world.update("a", "x")
    assert flatten(world.render_entry(world.get_entry("a"))) .count("tampered") == 0
```

## Step 5 — D3 part 2: history renders from captured serializers — `src/wica/agent.py`

**5a. Records** (lines 62–93). Add a new frozen dataclass before `ObservationRecord` and change
`ObservationRecord`:

```python
@dataclass(frozen=True)
class ObservedEntry:
    """One entry of an Observation: the WorldEntry snapshot plus the serializers that governed it
    when observed. History renders through these, never through the World's live registration, so
    an observation renders identically after the key is unregistered or re-registered. See
    specs/agent.md ("History record shape")."""

    entry: WorldEntry
    serialize_fn: Callable[[Any, Any], Content]
    archival_serialize_fn: Callable[[Any, Any], Content]


@dataclass(frozen=True)
class ObservationRecord:
    entries: list[ObservedEntry]
```

**5b. Capture** — `_append_observation` (lines 737–749) becomes:

```python
    def _append_observation(self) -> None:
        snapshot = self._world.get_prompt_snapshot()
        observed = [
            ObservedEntry(
                entry=entry,
                serialize_fn=config.serialize_fn,
                archival_serialize_fn=config.archival_serialize_fn or config.serialize_fn,
            )
            for entry, config in snapshot
        ]
        self._history.append(ObservationRecord(observed))
        # (keep the existing comment block about retirement)
        for item in observed:
            if self._is_terminal_command(item.entry):
                _logger.debug("retiring completed command entry %r", item.entry.key)
                self._cleanup_command_entry(item.entry.key)
```

**5c. Render** — in `_render_messages`, replace the loop body at lines 810–825 (the
`for world_entry in record.entries:` block with its `startswith` branch) with:

```python
                for item in record.entries:
                    serialize_fn = item.archival_serialize_fn if archival else item.serialize_fn
                    rendered = self._world.render_entry(item.entry, serialize_fn=serialize_fn)
                    blocks.extend(_content_to_message_blocks(rendered))
```

Note: `render_entry` with an explicit `serialize_fn` ignores its `archival` flag and never touches
the World's `_configs`, so this render cannot raise `KeyError` for an unregistered key. Do not pass
`archival=` any more. Remove the now-unused comment about "Command entries carry the outcome…"
or reword it to say every entry renders through its captured serializer.

**5d. Other readers of `record.entries`.** At `21788f5` the only sites outside `agent.py` are four
assertions in `tests/test_agent.py` (lines 187, 586, 1005, 1079) of the form
`[e.key for e in observation.entries]` — change each to `e.entry.key`. Re-check with
`grep -rn "\.entries" src tests tests-e2e examples`.

**Tests:** append to `tests/test_agent.py` (helpers: `make_agent`, `ProgrammableChatModel`,
`text_response`, `sequence`, `wait_until`, `identity_serialize`, fixtures `loop`, `world`, `sink`):

```python
def _step(model: ProgrammableChatModel, world: World, key: str, value: str, n_calls: int):
    """Trigger a step by updating `key` and wait until the model has been called n_calls times."""
    world.update(key, value)
    wait_until(lambda: len(model.calls) >= n_calls)


def _rendered_user_text(model: ProgrammableChatModel, call_index: int) -> str:
    return "".join(human_texts(m) for m in model.calls[call_index] if isinstance(m, HumanMessage))


def test_history_renders_an_unregistered_entry_from_its_captured_serializer(loop, world, sink):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink, coalesce_window=0)
    world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"mood={v}")])
    agent.start()
    world.update("mood", "happy")
    _step(model, world, "speech", "one", 1)
    assert "mood=happy" in _rendered_user_text(model, 0)

    world.unregister("mood")                      # would KeyError at render time before D3
    _step(model, world, "speech", "two", 2)
    assert "mood=happy" in _rendered_user_text(model, 1)   # observation 1 still renders as it did
    agent.stop()


def test_reregistering_a_key_does_not_rewrite_earlier_observations(loop, world, sink):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink, coalesce_window=0)
    world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"OLD:{v}")])
    agent.start()
    world.update("mood", "happy")
    _step(model, world, "speech", "one", 1)
    first_render = _rendered_user_text(model, 0)

    world.unregister("mood")
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"NEW:{v}")])
    world.update("mood", "calm")
    _step(model, world, "speech", "two", 2)
    second = model.calls[1]
    # The first observation is byte-identical to what step one sent (cache-stable prefix) …
    assert _rendered_user_text(model, 1).startswith(first_render.split("</entry>")[0])
    assert "OLD:happy" in _rendered_user_text(model, 1)
    # … and the new observation uses the new serializer.
    assert "NEW:calm" in _rendered_user_text(model, 1)
    assert "NEW:happy" not in _rendered_user_text(model, 1)
    agent.stop()


def test_application_entry_under_agent_command_prefix_uses_its_own_serializer(loop, world, sink):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink, coalesce_window=0)
    world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("agent:command:mine", str, serialize_fn=lambda v, p: [TextPart(f"MINE:{v}")])
    agent.start()
    world.update("agent:command:mine", "x")
    _step(model, world, "speech", "one", 1)
    assert "MINE:x" in _rendered_user_text(model, 0)     # not "(no command)" / not a crash
    agent.stop()
```

`human_texts` already exists in the module; `HumanMessage` is already imported there — check
with `grep -n "HumanMessage" tests/test_agent.py` and add the import if missing.

## Step 6 — D4 part 2: WICA-owned call ids — `src/wica/agent.py`

**6a. Records.** Add `tool_call_id: str` to `CommandRecord` and `NoReactionRecord`:

```python
@dataclass(frozen=True)
class CommandRecord:
    call_id: str        # World-key suffix (agent:command:<call_id>) and cancel_command target
    tool_call_id: str   # the provider's id — used to reconstruct the native tool_call/tool_result
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class NoReactionRecord:
    call_id: str
    tool_call_id: str
```

**6b. Choosing the key.** Add a method to `Agent`:

```python
    def _world_call_id(self, provider_id: str | None) -> str:
        """The `<call_id>` for a new agent:command:<call_id> entry: the provider's tool-call id
        when it is a valid World key and that key is free, else a generated one. See
        specs/commands.md ("Command execution as a World entry")."""
        if provider_id:
            try:
                validate_key(provider_id)
            except ValueError:
                pass
            else:
                if not self._world.is_registered(f"{_COMMAND_KEY_PREFIX}{provider_id}"):
                    return provider_id
        return uuid.uuid4().hex
```

Import `validate_key` from `wica.world` (extend the existing `from wica.world import World,
WorldEntry` line).

**6c. Use it** in `_run_step` (lines 645–661). Replace
`call_id = call["id"] or uuid.uuid4().hex` with:

```python
            tool_call_id = call["id"] or uuid.uuid4().hex
            call_id = self._world_call_id(tool_call_id)
```

and pass both into the records: `NoReactionRecord(call_id, tool_call_id)` and
`CommandRecord(call_id, tool_call_id, call["name"], copy.deepcopy(args))`. For a `noop` the
`call_id` is only recorded (no entry is created), so it is fine that it equals `tool_call_id`.

**6d. Render** in `_render_messages` (lines 829–844): the native `tool_calls[].id` and
`pending_acks` keys must use **`record.tool_call_id`**; `_command_ack(record.call_id)` keeps using
`call_id` (that is the entry the model is pointed at). Concretely:

```python
            elif isinstance(record, CommandRecord):
                pending_calls.append({"name": record.name, "args": copy.deepcopy(record.args),
                                      "id": record.tool_call_id})
                pending_acks[record.tool_call_id] = _command_ack(record.call_id)
            elif isinstance(record, NoReactionRecord):
                pending_calls.append({"name": _NOOP_COMMAND_NAME, "args": {}, "id": record.tool_call_id})
                pending_acks[record.tool_call_id] = _NOOP_ACK
```

**6e. Existing tests.** At `21788f5` no test constructs `CommandRecord`/`NoReactionRecord` directly
(verify with `grep -rn "CommandRecord(\|NoReactionRecord(" tests tests-e2e`), so no test edit is
needed for the new field. All existing key assertions such as `"agent:command:call1"` keep
passing because valid, free provider ids are used verbatim.

**Tests:** append to `tests/test_agent.py`:

```python
def test_invalid_provider_call_id_gets_a_safe_world_key_but_keeps_its_message_id(loop, world, sink):
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    bad_id = 'call "quoted"'
    model = ProgrammableChatModel(respond=sequence(
        tool_call_response([("add", {"a": 1, "b": 2}, bad_id)]), text_response("done")))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink, coalesce_window=0)
    agent.register_command(add)
    world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    agent.start()
    world.update("speech", "go")
    wait_until(lambda: len(model.calls) >= 2)          # completion re-triggered a second step
    keys = [k for k in world_keys_seen(model) if k.startswith("agent:command:")]
    assert keys and all('"' not in k for k in keys)
    ai = next(m for m in model.calls[1] if isinstance(m, AIMessage))
    assert ai.tool_calls[0]["id"] == bad_id          # the provider's id is what the model sees
    agent.stop()
```

Write `world_keys_seen(model)` as a small helper that regex-extracts `key="…"` from the rendered
user text of every call (`re.findall(r'<entry key="([^"]*)"', text)`), or simply assert on
`agent._history` records: `record.call_id != bad_id` and `record.tool_call_id == bad_id`. Either is
acceptable; asserting on the rendered prompt is closer to the public surface.

```python
def test_colliding_provider_call_ids_get_distinct_entries(loop, world, sink):
    release = threading.Event()

    async def slow() -> str:
        """Block until released."""
        await asyncio.get_running_loop().run_in_executor(None, release.wait)
        return "done"

    model = ProgrammableChatModel(respond=tool_call_response([("slow", {}, "same")]))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink, coalesce_window=0)
    agent.register_command(slow)
    world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    agent.start()
    world.update("speech", "one")
    wait_until(lambda: world.is_registered("agent:command:same"))
    world.update("speech", "two")                    # second step, same provider id, first still running
    wait_until(lambda: len(model.calls) >= 2)
    wait_until(lambda: len([k for k in agent._command_keys]) == 2)
    assert "agent:command:same" in agent._command_keys
    release.set()
    agent.stop()
```

(Reading `agent._command_keys` is an internal peek; it is acceptable here because the observable
alternative — two distinct running rows in the prompt — is what the following assertion on
`model.calls[1]` should check if you prefer: the rendered text contains two `slow()` running lines.)

## Step 7 — D1: reserved names and duplicate rejection — `src/wica/agent.py`

**7a. Constants.** After `_NOOP_COMMAND_NAME` add:

```python
_RESERVED_COMMAND_NAMES = frozenset({_CANCEL_COMMAND_NAME, _NOOP_COMMAND_NAME})
```

**7b. `__init__`** (lines 335–344): after computing `self._output_command_name`, add:

```python
        if self._output_command_name in _RESERVED_COMMAND_NAMES:
            raise ValueError(
                f"output_command may not be named {self._output_command_name!r}: "
                f"{sorted(_RESERVED_COMMAND_NAMES)} are reserved by WICA"
            )
```

**7c. `register_command`** (lines 404–414) becomes:

```python
    def register_command(self, fn: Callable[..., Any] | Command) -> None:
        """(keep the docstring; add:) Names are unique: registering a name already registered, a
        WICA-reserved name (`noop`, `cancel_command`), or the output Command's name raises
        ValueError. See specs/commands.md ("Names are unique and two are reserved")."""
        command = fn if isinstance(fn, Command) else Command(fn)
        if command.name in _RESERVED_COMMAND_NAMES:
            raise ValueError(f"command name {command.name!r} is reserved by WICA")
        if command.name == self._output_command_name:
            raise ValueError(f"command name {command.name!r} is the output Command's name")
        if command.name in self._commands:
            raise ValueError(f"command {command.name!r} is already registered")
        self._attach_command(command)

    def _attach_command(self, command: Command) -> None:
        """Store the Command and rebind the model to the full current tool set."""
        self._commands[command.name] = command
        self._bound_model = self.model.bind_tools([c.tool for c in self._commands.values()])
        _logger.debug("registered command %r", command.name)

    def _register_builtin(self, command: Command) -> None:
        """Idempotent attach for WICA-native Commands and the output Command (used by start(), which
        may run again after stop()). Their names are reserved from application use, so this can
        never replace an application Command."""
        if self._commands.get(command.name) is command:
            return
        self._attach_command(command)
```

**7d. `start()`** (lines 425–443): the three `self.register_command(...)` calls become
`self._register_builtin(...)`. Build the two built-in `Command` objects **once** in `__init__`
(e.g. `self._cancel_command = Command(self._cancel_command_action, name=…, description=…)` and
`self._noop_command = …`) so the identity check in `_register_builtin` holds across restarts;
`start()` then passes those attributes. Do not build a fresh `Command(...)` inside `start()`.

**7e. `wica.py`**: nothing to change; `Wica.register_command` delegates and the `ValueError`
propagates. Update its docstring to mention the uniqueness rule in one sentence.

**Tests:** append to `tests/test_agent.py`:

```python
def _dummy_agent(loop, world, sink, **kwargs) -> Agent:
    return make_agent(ProgrammableChatModel(respond=text_response("ok")),
                      world=world, loop=loop, output_sink=sink, **kwargs)


def _named(name: str) -> Command:
    def fn() -> str:
        """A test command."""
        return "x"
    return Command(fn, name=name, description="A test command.")


@pytest.mark.parametrize("name", ["noop", "cancel_command"])
def test_register_command_rejects_reserved_names(loop, world, sink, name):
    agent = _dummy_agent(loop, world, sink)
    with pytest.raises(ValueError, match="reserved"):
        agent.register_command(_named(name))


def test_register_command_rejects_duplicate_names(loop, world, sink):
    agent = _dummy_agent(loop, world, sink)
    agent.register_command(_named("wave"))
    with pytest.raises(ValueError, match="already registered"):
        agent.register_command(_named("wave"))


@pytest.mark.parametrize("name", ["noop", "cancel_command"])
def test_output_command_with_reserved_name_is_rejected_at_construction(loop, world, sink, name):
    with pytest.raises(ValueError, match="reserved"):
        _dummy_agent(loop, world, sink, output_command=_named(name))


def test_register_command_rejects_the_output_commands_name(loop, world, sink):
    agent = _dummy_agent(loop, world, sink, output_command=_named("say"))
    with pytest.raises(ValueError, match="output Command"):
        agent.register_command(_named("say"))


def test_restart_keeps_exactly_one_of_each_builtin(loop, world, sink):
    agent = _dummy_agent(loop, world, sink, output_command=_named("say"))
    agent.register_command(_named("wave"))
    agent.start(); agent.stop(); agent.start()
    assert sorted(agent._commands) == ["cancel_command", "noop", "say", "wave"]
    agent.stop()
```

and to `tests/test_wica.py` (fixture `wica_factory`, helper `fake_config`; add
`from wica import Command` to its imports — it is not imported there yet):

```python
def test_wica_register_command_rejects_duplicates_and_reserved_names(wica_factory):
    wica = wica_factory(fake_config())

    def wave() -> str:
        """Wave."""
        return "waved"

    wica.register_command(wave)
    with pytest.raises(ValueError):
        wica.register_command(wave)
    with pytest.raises(ValueError):
        wica.register_command(Command(wave, name="noop", description="Wave."))
```

Also update `tests-e2e/test_fake_flows.py` only if it registers a command twice (it should not).

## Step 8 — D2: safe shutdown — `src/wica/wica.py`

**8a. Same-thread guard.** Add a private helper and call it first thing in `stop()` (line 148) and
`close()` (line 162), before the lock is taken:

```python
    def _reject_if_on_owned_loop_thread(self, operation: str) -> None:
        if self._owns_loop and threading.current_thread() is self._loop_thread:
            raise RuntimeError(
                f"cannot {operation} Wica from its own event-loop thread (an output sink, Command, "
                "or Event subscriber); call it from another thread"
            )
```

`stop()`: `self._reject_if_on_owned_loop_thread("stop")` as the first line. `close()`:
`self._reject_if_on_owned_loop_thread("close")` as the first line. Because `close()` calls
`stop()` internally the second check is redundant but harmless and gives the right verb in the
message. `_loop_thread` is `None` when not running, so the guard is a no-op then — correct, since
there is nothing to join.

**8b. Non-running injected loop.** In `start()` (line 132, after the `is_closed()` check) add:

```python
            if not self._owns_loop and not self._loop.is_running():
                raise RuntimeError("injected event loop is not running")
```

**Tests:** append to `tests/test_wica.py`:

```python
def test_stop_from_an_output_sink_is_rejected_and_the_system_keeps_running(wica_factory):
    wica_ref: list[Wica] = []
    errors: list[BaseException] = []
    done = threading.Event()

    async def sink(text: str) -> None:
        try:
            wica_ref[0].stop()
        except BaseException as exc:       # noqa: BLE001 - we want the exact error
            errors.append(exc)
        done.set()

    wica = wica_factory(fake_config([{"text": "hi"}]), output_sink=sink, coalesce_window=0)
    wica_ref.append(wica)
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.start()
    wica.world.update("speech", "hello")
    assert done.wait(WAIT_TIMEOUT)
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "event-loop thread" in str(errors[0])
    assert wica.is_running and wica.world.is_running      # nothing was torn down
    wica.stop()                                           # from the test thread: clean
    assert not wica.is_running


def test_stop_from_an_event_subscriber_is_rejected(wica_factory):
    errors: list[BaseException] = []
    done = threading.Event()
    wica = wica_factory(fake_config([{"text": "hi"}]), coalesce_window=0)

    def on_prompt(messages) -> None:
        try:
            wica.close()
        except RuntimeError as exc:
            errors.append(exc)
        done.set()

    wica.on_agent_prompt.subscribe(on_prompt)
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.start()
    wica.world.update("speech", "hello")
    assert done.wait(WAIT_TIMEOUT)
    assert len(errors) == 1
    assert wica.is_running


def test_start_rejects_an_injected_loop_that_is_not_running():
    loop = asyncio.new_event_loop()
    try:
        wica = Wica.init(fake_config(), loop=loop)
        with pytest.raises(RuntimeError, match="not running"):
            wica.start()
        assert not wica.is_running
        wica.close()
    finally:
        loop.close()
```

`test_injected_loop_wica_can_restart_without_owning_the_loop` (existing, line 284) must still pass
unchanged — its loop is running.

## Step 9 — D5: config invariants and frozen configs — `src/wica/config.py`, `tests-e2e/support.py`

**9a.** Change `@dataclass` to `@dataclass(frozen=True)` on both `AgentConfig` (line 22) and
`WicaConfig` (line 46).

**9b.** Add to `AgentConfig`, after the fields:

```python
    def __post_init__(self) -> None:
        # The structural invariants the loaders enforce, applied to direct construction too, so
        # every AgentConfig instance is valid. See specs/config.md.
        has_inline = self.system_prompt is not None
        has_file = self.system_prompt_file is not None
        if has_inline and has_file:
            raise ConfigError("agent: specify at most one of 'system_prompt'/'system_prompt_file', not both")
        if not has_inline and not has_file:
            raise ConfigError("agent: specify one of 'system_prompt'/'system_prompt_file'")
        if self.api_key is not None and self.api_key_env is not None:
            raise ConfigError("agent: specify at most one of 'api_key'/'api_key_env', not both")
```

`ConfigError` is defined above the dataclass in the same module, so no ordering issue. The loaders
call `cls(**…)` after their own checks, so they pass through `__post_init__` without change.

**9c.** In `resolve_system_prompt`, keep the final `raise ConfigError(...)` but change its comment
to `# unreachable: __post_init__ guarantees exactly one is set`.

**9d.** `tests-e2e/support.py` line 57 (`config.agent.system_prompt = system_prompt`) becomes:

```python
    if system_prompt is not None:
        config = WicaConfig(
            agent=dataclasses.replace(config.agent, system_prompt=system_prompt, system_prompt_file=None)
        )
```

with `import dataclasses` added. Update the docstring sentence "(inline wins over
`system_prompt_file` — see specs/config.md)" to "(replacing the committed persona; the file field
is cleared so the exactly-one invariant holds)".

**9e.** `grep -rn "\.agent\.[a-z_]* = \|config\.[a-z_]* = " tests tests-e2e examples src` — there
must be no other assignment into a config; if one turns up, convert it to `dataclasses.replace`.

**Tests:** append to `tests/test_config.py`:

```python
def test_direct_construction_with_both_prompt_fields_raises():
    with pytest.raises(ConfigError):
        AgentConfig(provider="fake", model="m", system_prompt="a", system_prompt_file="b.md")


def test_direct_construction_with_no_prompt_raises():
    with pytest.raises(ConfigError):
        AgentConfig(provider="fake", model="m")


def test_direct_construction_with_both_key_fields_raises():
    with pytest.raises(ConfigError):
        AgentConfig(provider="fake", model="m", system_prompt="a", api_key="k", api_key_env="E")


def test_configs_are_frozen():
    cfg = WicaConfig(agent=AgentConfig(provider="fake", model="m", system_prompt="a"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent.system_prompt = "b"       # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent = cfg.agent               # type: ignore[misc]
    variant = dataclasses.replace(cfg.agent, system_prompt="b")
    assert variant.system_prompt == "b" and cfg.agent.system_prompt == "a"
```

Add `import dataclasses` to the test module. pyright will flag the assignments; the
`# type: ignore[misc]` comments are expected there.

## Step 10 — Docs drift guard (low priority, small)

Add to `tests/test_project_map.py`:

```python
_CONSUMER_DOCS = [_REPO_ROOT / "README.md", _REPO_ROOT / "INTEGRATING.md"]
_DOC_CALL = re.compile(r"\b(WicaConfig|Wica|wica|world|agent)\.([a-z_]+)\(")
_DOC_TARGETS = {"WicaConfig": WicaConfig, "Wica": Wica, "wica": Wica, "world": World, "agent": Agent}


def test_consumer_docs_only_call_methods_that_exist():
    missing: list[str] = []
    for doc in _CONSUMER_DOCS:
        text = doc.read_text(encoding="utf-8")
        fences = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
        for fence in fences:
            for owner, method in _DOC_CALL.findall(fence):
                if not hasattr(_DOC_TARGETS[owner], method):
                    missing.append(f"{doc.name}: {owner}.{method}()")
    assert not missing, f"consumer docs call methods that do not exist: {sorted(set(missing))}"
```

Import `Agent, Wica, WicaConfig, World` from `wica` at the top of the test module. If the regex
produces a false positive on a local variable in a doc snippet (e.g. a user-defined `agent`
object), narrow `_DOC_CALL` rather than weakening the assertion.

## Step 11 — Consumer docs

- `INTEGRATING.md`, "v1 limits to design around": add three bullets — Command names are unique and
  `noop`/`cancel_command` are reserved (`ValueError`); World keys must be non-empty with no
  whitespace or `" < > &`; `wica.stop()`/`close()` must not be called from inside a sink, Command,
  or Event subscriber when Wica owns the loop (raises `RuntimeError`; hand it to another thread).
- `INTEGRATING.md`, "Configuration": one sentence that the config dataclasses are frozen and
  validate their invariants on construction; use `dataclasses.replace` to derive variants.
- `README.md`, "Commands — the agent acting out": one sentence that names are unique and the two
  reserved ones exist.

## Step 12 — Close out

1. `uv run ruff check .` · `uv run ruff format .` · `uv run pyright` · `uv run pytest` ·
   `uv run pytest tests-e2e -k "fake or example"` — all green.
2. If a provider key is available: `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest
   tests-e2e -k anthropic'` to confirm real tool-call ids pass the key grammar (they are
   alphanumeric with `_`; if a provider ever emits something else, the fallback covers it).
3. `specs/_fixes.md`: delete the six implemented sections and the "Low priority" section. If only
   "Tracking" remains, delete the file and remove the first bullet of `specs/_todo.md` that points
   at it.
4. Flip `commands.md`, `wica.md`, `agent.md`, `world.md`, `config.md` from `Updated` back to
   `Implemented` (status line + `specs/_index.md` row).
5. Mark this plan `Done` (status line + `plans/_index.md` row).

## Verification

- `uv run ruff check .` · `uv run ruff format .` · `uv run pyright` · `uv run pytest`
- `uv run pytest tests-e2e -k "fake or example"`

## Out of scope

- Async shutdown (`astop`/`aclose`) — recorded as a `wica.md` open question in step 1d, not built.
- Escaping keys instead of validating them — rejected (D4).
- Enforcing the `agent:` prefix in the World — rejected (D4, convention only).
- Deep-freezing `model_kwargs` — not needed; it is data, not an invariant.
- Anything in `specs/_todo.md` (history compaction, reaction entries, streaming).
