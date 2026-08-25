import asyncio
import re
import threading
import time
from collections.abc import Iterator
from datetime import timedelta

import pytest

from wica.content import Content, ImagePart, TextPart
from wica.world import World, WorldEntry

WAIT_TIMEOUT = 2.0


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A real event loop running in a background thread — the World dispatches its reactive
    callbacks onto it, the same way the Wica facade owns it in production."""
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    yield event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join()
    event_loop.close()


@pytest.fixture
def world(loop: asyncio.AbstractEventLoop) -> Iterator[World]:
    """A started World on the running loop. Mutation (update) is only allowed while started."""
    w = World(loop)
    w.start()
    yield w
    w.stop()


def flatten(content: Content) -> str:
    return "".join(part.to_string() for part in content)


def identity_serialize(value, previous_value) -> Content:
    return [TextPart(str(value))]


def image_serialize(value: bytes | None, previous_value: bytes | None) -> Content:
    return [ImagePart(value, "image/png")] if value is not None else []


def text_archival_serialize(value: bytes | None, previous_value: bytes | None) -> Content:
    return [TextPart("a photo")]


class RecordingCallback:
    """Test double for a sync listener/on_trigger subscriber: records calls and signals an Event."""

    def __init__(self):
        self.calls: list[WorldEntry] = []
        self.event = threading.Event()

    def __call__(self, entry: WorldEntry) -> None:
        self.calls.append(entry)
        self.event.set()


# --- Data model (unguarded; needs no running loop) --------------------------


def test_register_then_get_returns_none(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    assert world.get("temp") is None


def test_update_changes_get_result(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 42)
    assert world.get("temp") == 42


def test_update_type_mismatch_raises_and_leaves_value_untouched(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 5)
    with pytest.raises(TypeError):
        world.update("temp", "not an int")
    assert world.get("temp") == 5
    assert '<entry key="temp" id="2">' in flatten(world.render_entry(world.get_entry("temp")))


def test_update_none_always_succeeds_regardless_of_type(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", None)
    assert world.get("temp") is None


def test_world_owns_mutable_values_and_returns_defensive_copies(world: World):
    world.register("state", dict, serialize_fn=identity_serialize)
    source = {"nested": [1]}

    world.update("state", source)
    source["nested"].append(2)
    fetched = world.get("state")
    fetched["nested"].append(3)
    entry = world.get_entry("state")
    entry.current.value["nested"].append(4)
    prompt_entry = world.get_prompt_entries()[0]
    prompt_entry.current.value["nested"].append(5)

    assert world.get("state") == {"nested": [1]}


def test_serializers_and_trigger_conditions_cannot_mutate_live_world_state(world: World):
    def serialize(value: dict | None, previous: dict | None) -> Content:
        if value is not None:
            value["changed_by_serializer"] = True
        if previous is not None:
            previous["changed_by_serializer"] = True
        return [TextPart(str(value))]

    def condition(old: dict | None, new: dict | None) -> bool:
        if old is not None:
            old["changed_by_condition"] = True
        if new is not None:
            new["changed_by_condition"] = True
        return False

    world.register(
        "state",
        dict,
        serialize_fn=serialize,
        triggers_llm_call=True,
        trigger_condition_fn=condition,
    )
    world.update("state", {"version": 1})
    world.update("state", {"version": 2})

    world.render_entry(world.get_entry("state"))

    assert world.get("state") == {"version": 2}


def test_listener_receives_value_copy_not_live_world_state(world: World):
    world.register("state", dict, serialize_fn=identity_serialize)
    delivered = threading.Event()

    def mutate(entry: WorldEntry) -> None:
        entry.current.value["changed_by_listener"] = True
        delivered.set()

    world.add_listener("state", mutate)
    world.update("state", {"original": True})

    assert delivered.wait(timeout=WAIT_TIMEOUT)
    assert world.get("state") == {"original": True}


def test_id_starts_at_one_on_register_and_increments_on_every_update(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    assert '<entry key="temp" id="1">' in flatten(world.render_entry(world.get_entry("temp")))

    world.update("temp", "a")
    assert '<entry key="temp" id="2">' in flatten(world.render_entry(world.get_entry("temp")))

    world.update("temp", "b")
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))


def test_clearing_advances_id_like_any_other_update(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "a")  # id=2
    world.update("temp", None)  # id=3, clearing still advances id
    assert world.get("temp") is None
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))

    world.update("temp", "c")  # id=4
    assert world.get("temp") == "c"
    assert '<entry key="temp" id="4">' in flatten(world.render_entry(world.get_entry("temp")))


def test_unregister_removes_entry(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    world.unregister("temp")
    with pytest.raises(KeyError):
        world.get("temp")
    with pytest.raises(KeyError):
        world.update("temp", "x")


def test_reregister_without_unregister_raises(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    with pytest.raises(ValueError):
        world.register("temp", str, serialize_fn=identity_serialize)


def test_reregister_after_unregister_continues_id_sequence(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)  # id=1
    world.update("temp", 5)  # id=2
    world.unregister("temp")

    world.register("temp", int, serialize_fn=identity_serialize)  # id=3
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))
    world.update("temp", 7)  # id=4
    assert '<entry key="temp" id="4">' in flatten(world.render_entry(world.get_entry("temp")))


def test_get_after_reregistration_with_different_type_just_returns_the_new_value(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 5)
    world.unregister("temp")

    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "now a string")

    # no handle to go stale — get() always reflects whatever is currently registered
    assert world.get("temp") == "now a string"


def test_get_entry_matches_get_and_registered_type(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 7)

    entry = world.get_entry("temp")
    assert entry.key == "temp"
    assert entry.type is int
    assert entry.current.value == 7
    assert entry.current.value == world.get("temp")


def test_get_entry_unregistered_raises(world: World):
    world.register("temp", int, serialize_fn=identity_serialize)
    world.unregister("temp")
    with pytest.raises(KeyError):
        world.get_entry("temp")


def test_bypass_coalescing_defaults_false_and_is_carried_onto_the_entry(world: World):
    world.register("plain", int, serialize_fn=identity_serialize)
    world.register("urgent", int, serialize_fn=identity_serialize, bypass_coalescing=True)

    # Carried onto the initial entry (like `type`) so the on_trigger snapshot is self-describing
    # without a config lookup.
    assert world.get_entry("plain").bypass_coalescing is False
    assert world.get_entry("urgent").bypass_coalescing is True

    # ...and preserved across updates (a new version, not a config change).
    world.update("urgent", 1)
    world.update("plain", 2)
    assert world.get_entry("urgent").bypass_coalescing is True
    assert world.get_entry("plain").bypass_coalescing is False


# --- Lifecycle guard --------------------------------------------------------


def test_update_before_start_raises(loop: asyncio.AbstractEventLoop):
    world = World(loop)  # not started
    world.register("temp", int, serialize_fn=identity_serialize)
    with pytest.raises(RuntimeError, match="World is not running"):
        world.update("temp", 1)


def test_update_after_stop_raises(loop: asyncio.AbstractEventLoop):
    world = World(loop)
    world.start()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 1)
    world.stop()
    with pytest.raises(RuntimeError, match="World is not running"):
        world.update("temp", 2)


def test_register_and_read_work_while_stopped(loop: asyncio.AbstractEventLoop):
    # Setup-before-start and reads-after-stop: the data model is unguarded, only mutation is gated.
    world = World(loop)  # never started
    assert world.is_running is False
    world.register("temp", str, serialize_fn=identity_serialize)
    world.start()
    world.update("temp", "hello")
    world.stop()
    # After stop, reads still work and return the last state (a held wica.world stays inspectable).
    assert world.get("temp") == "hello"
    assert world.get_entry("temp").current.value == "hello"
    assert "hello" in flatten(world.render_full_prompt())


# --- Listeners (loop-dispatched, sync/async, isolated) ----------------------


def test_listener_fires_on_update_but_not_for_unrelated_keys(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    world.register("other", str, serialize_fn=identity_serialize)
    callback = RecordingCallback()
    world.add_listener("temp", callback)

    world.update("other", "irrelevant")
    assert not callback.event.wait(timeout=0.2)

    world.update("temp", "hello")
    assert callback.event.wait(timeout=WAIT_TIMEOUT)
    assert callback.calls[-1].current.value == "hello"


def test_async_listener_runs(world: World):
    received: list[str] = []
    done = threading.Event()

    async def async_listener(entry: WorldEntry) -> None:
        received.append(entry.current.value)
        done.set()

    world.register("temp", str, serialize_fn=identity_serialize)
    world.add_listener("temp", async_listener)

    world.update("temp", "hi")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert received == ["hi"]


def test_remove_listener_stops_notifications(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    callback = RecordingCallback()
    world.add_listener("temp", callback)
    world.remove_listener("temp", callback)

    world.update("temp", "hello")
    assert not callback.event.wait(timeout=0.2)
    assert callback.calls == []


def test_blocking_sync_listener_does_not_stall_other_dispatch(world: World):
    # A blocking sync listener is offloaded to the loop's thread pool, so a sibling listener still
    # fires while it's blocked — the loop is never stalled.
    world.register("temp", str, serialize_fn=identity_serialize)
    release = threading.Event()
    blocked_started = threading.Event()
    fast_fired = threading.Event()

    def blocking_listener(entry: WorldEntry) -> None:
        blocked_started.set()
        release.wait(timeout=WAIT_TIMEOUT)

    def fast_listener(entry: WorldEntry) -> None:
        fast_fired.set()

    world.add_listener("temp", blocking_listener)
    world.add_listener("temp", fast_listener)

    world.update("temp", "hello")
    assert blocked_started.wait(timeout=WAIT_TIMEOUT)
    assert fast_fired.wait(timeout=WAIT_TIMEOUT)  # fires despite the sibling still blocked
    release.set()


def test_raising_listener_is_isolated_from_siblings(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    fired = threading.Event()

    def boom(entry: WorldEntry) -> None:
        raise RuntimeError("listener failure")

    def survivor(entry: WorldEntry) -> None:
        fired.set()

    world.add_listener("temp", boom)
    world.add_listener("temp", survivor)

    world.update("temp", "hello")  # the raising listener must not starve the survivor
    assert fired.wait(timeout=WAIT_TIMEOUT)
    assert world.get("temp") == "hello"  # update itself never saw the exception


def test_update_does_not_block_on_slow_callback(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    release = threading.Event()
    started = threading.Event()

    def slow_listener(entry: WorldEntry) -> None:
        started.set()
        release.wait(timeout=WAIT_TIMEOUT)

    world.add_listener("temp", slow_listener)

    world.update("temp", "hello")
    assert world.get("temp") == "hello"  # update() already returned

    assert started.wait(timeout=WAIT_TIMEOUT)
    release.set()


def test_snapshot_handed_to_listener_is_unaffected_by_later_update(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    seen: list[str] = []
    got_two = threading.Event()
    lock = threading.Lock()

    def listener(entry: WorldEntry) -> None:
        with lock:
            seen.append(entry.current.value)
            if len(seen) == 2:
                got_two.set()

    world.add_listener("temp", listener)

    world.update("temp", "first")
    world.update("temp", "second")

    assert got_two.wait(timeout=WAIT_TIMEOUT)
    # Each dispatch carried its own immutable snapshot — neither was mutated in place to the
    # other's value (which a shared live-entry reference would have caused).
    assert set(seen) == {"first", "second"}


# --- Trigger (the on_trigger Event) -----------------------------------------


def test_on_trigger_fires_without_condition_fn(world: World):
    world.register("temp", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    subscriber = RecordingCallback()
    world.on_trigger.subscribe(subscriber)

    world.update("temp", "hello")
    assert subscriber.event.wait(timeout=WAIT_TIMEOUT)
    assert subscriber.calls[-1].current.value == "hello"


def test_on_trigger_subscriber_cannot_mutate_live_world_state(world: World):
    world.register("state", dict, serialize_fn=identity_serialize, triggers_llm_call=True)
    delivered = threading.Event()

    def mutate(entry: WorldEntry) -> None:
        entry.current.value["changed_by_subscriber"] = True
        delivered.set()

    world.on_trigger.subscribe(mutate)
    world.update("state", {"original": True})

    assert delivered.wait(timeout=WAIT_TIMEOUT)
    assert world.get("state") == {"original": True}


def test_trigger_condition_fn_gates_the_emit(world: World):
    world.register(
        "temp",
        int,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        trigger_condition_fn=lambda old, new: new is not None and new > 10,
    )
    subscriber = RecordingCallback()
    world.on_trigger.subscribe(subscriber)

    world.update("temp", 5)
    assert not subscriber.event.wait(timeout=0.2)

    world.update("temp", 20)
    assert subscriber.event.wait(timeout=WAIT_TIMEOUT)
    assert subscriber.calls[-1].current.value == 20


def test_trigger_condition_fn_suppresses_noop_updates(world: World):
    world.register(
        "temp",
        str,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        trigger_condition_fn=lambda old, new: old != new,
    )
    subscriber = RecordingCallback()
    world.on_trigger.subscribe(subscriber)

    world.update("temp", "same")
    assert subscriber.event.wait(timeout=WAIT_TIMEOUT)
    subscriber.event.clear()

    world.update("temp", "same")
    assert not subscriber.event.wait(timeout=0.2)


def test_unregister_never_emits_trigger(world: World):
    world.register("temp", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    subscriber = RecordingCallback()
    world.on_trigger.subscribe(subscriber)

    world.unregister("temp")
    assert not subscriber.event.wait(timeout=0.2)


def test_update_without_triggers_llm_call_never_emits(world: World):
    world.register("temp", str, serialize_fn=identity_serialize, triggers_llm_call=False)
    subscriber = RecordingCallback()
    world.on_trigger.subscribe(subscriber)

    world.update("temp", "hello")
    assert not subscriber.event.wait(timeout=0.2)


# --- serialize_fn / previous -------------------------------------------------


def test_serialize_fn_receives_correct_previous_value(world: World):
    seen = []

    def recording_serialize(value, previous_value) -> Content:
        seen.append((value, previous_value))
        return [TextPart(str(value))]

    world.register("temp", str, serialize_fn=recording_serialize)
    world.update("temp", "A")
    world.render_entry(world.get_entry("temp"))
    world.update("temp", "B")
    world.render_entry(world.get_entry("temp"))
    world.update("temp", None)
    world.render_entry(world.get_entry("temp"))

    assert seen == [("A", None), ("B", "A"), (None, "B")]


# --- TTL --------------------------------------------------------------------


def test_ttl_resets_value_to_none_and_advances_id_without_triggering(world: World):
    world.register(
        "temp",
        str,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        ttl=timedelta(milliseconds=100),
    )
    subscriber = RecordingCallback()
    world.on_trigger.subscribe(subscriber)

    world.update("temp", "hello")  # id=2
    assert subscriber.event.wait(timeout=WAIT_TIMEOUT)  # the explicit update itself triggers
    subscriber.event.clear()

    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline and world.get("temp") is not None:
        time.sleep(0.01)

    assert world.get("temp") is None
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))
    assert not subscriber.event.wait(timeout=0.2)  # but the TTL-driven reset does not


def test_ttl_expiry_for_superseded_version_does_not_clobber_fresh_value(world: World):
    world.register("temp", str, serialize_fn=identity_serialize, ttl=timedelta(seconds=10))
    world.update("temp", "hello")  # id=2, arms a TTL timer bound to id=2
    world.update("temp", "fresh")  # id=3 (cancels id=2's timer, arms id=3's)

    # Simulate id=2's timer firing late — after "fresh" already landed. It must no-op,
    # since the entry's current id is now 3, not 2. Driving the private callback directly
    # because there's no public way to trigger an already-fired, stale timer.
    world._ttl_expire("temp", 2)
    assert world.get("temp") == "fresh"


def test_ttl_restart_postpones_reset(world: World):
    world.register("temp", str, serialize_fn=identity_serialize, ttl=timedelta(milliseconds=200))
    world.update("temp", "hello")

    time.sleep(0.12)
    world.update("temp", "hello-again")  # restarts the TTL window

    time.sleep(0.12)
    assert world.get("temp") == "hello-again"  # original 200ms window would've expired by now

    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline and world.get("temp") is not None:
        time.sleep(0.01)
    assert world.get("temp") is None


def test_stop_cancels_pending_ttl_timer(loop: asyncio.AbstractEventLoop):
    world = World(loop)
    world.start()
    world.register("temp", str, serialize_fn=identity_serialize, ttl=timedelta(milliseconds=100))
    world.update("temp", "hello")
    world.stop()  # cancels the armed TTL timer

    time.sleep(0.2)  # past the TTL — a live timer would have reset the value
    assert world.get("temp") == "hello"  # timer was cancelled, value untouched

    world.start()
    assert world.get("temp") is None  # restart reconciles the elapsed wall-clock deadline


def test_restart_restores_remaining_ttl(loop: asyncio.AbstractEventLoop):
    world = World(loop)
    world.start()
    world.register("temp", str, serialize_fn=identity_serialize, ttl=timedelta(milliseconds=300))
    world.update("temp", "hello")
    time.sleep(0.05)
    world.stop()
    time.sleep(0.05)

    world.start()
    assert world.get("temp") == "hello"

    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline and world.get("temp") is not None:
        time.sleep(0.01)
    assert world.get("temp") is None
    world.stop()


# --- Rendering --------------------------------------------------------------


def test_render_entry_format(world: World):
    world.register("user_profile", str, serialize_fn=lambda v, p: [TextPart("Jane is logged in.")])
    world.update("user_profile", "irrelevant raw value")

    rendered = flatten(world.render_entry(world.get_entry("user_profile")))
    match = re.match(
        r'^<entry key="user_profile" id="2">\n'
        r"Jane is logged in\.\n"
        r"Updated: (?P<ts>.+)\n"
        r"</entry>\n$",
        rendered,
    )
    assert match is not None


def test_render_entry_body_parts_are_returned_unwrapped(world: World):
    world.register("photo", bytes, serialize_fn=image_serialize)
    world.update("photo", b"\x89PNG")

    content = world.render_entry(world.get_entry("photo"))
    assert content == [
        TextPart('<entry key="photo" id="2">\n'),
        ImagePart(b"\x89PNG", "image/png"),
        TextPart(f"\nUpdated: {world.get_entry('photo').current.timestamp.isoformat()}\n</entry>\n"),
    ]


def test_render_entry_uses_archival_serialize_fn_when_requested(world: World):
    world.register(
        "photo",
        bytes,
        serialize_fn=image_serialize,
        archival_serialize_fn=text_archival_serialize,
    )
    world.update("photo", b"\x89PNG")
    entry = world.get_entry("photo")

    fresh = world.render_entry(entry)
    assert any(isinstance(part, ImagePart) for part in fresh)

    archival = world.render_entry(entry, archival=True)
    assert archival == [
        TextPart(f'<entry key="photo" id="{entry.current.id}">\n'),
        TextPart("a photo"),
        TextPart(f"\nUpdated: {entry.current.timestamp.isoformat()}\n</entry>\n"),
    ]


def test_render_entry_archival_falls_back_to_serialize_fn_when_absent(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "value")
    entry = world.get_entry("temp")

    assert world.render_entry(entry, archival=True) == world.render_entry(entry, archival=False)


def test_render_entry_raises_when_entry_key_no_longer_registered(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    stale_entry = world.get_entry("temp")
    world.unregister("temp")

    with pytest.raises(KeyError):
        world.render_entry(stale_entry)


def test_render_entry_serialize_fn_override_used_and_survives_unregister(world: World):
    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "raw value")
    entry = world.get_entry("temp")

    # The override replaces the registered serialize_fn for the body; the <entry …> envelope stays.
    rendered = flatten(
        world.render_entry(entry, serialize_fn=lambda v, p: [TextPart("OVERRIDDEN")])
    )
    assert '<entry key="temp" id="2">' in rendered
    assert "OVERRIDDEN" in rendered
    assert "raw value" not in rendered

    # It also works from a stale snapshot after the key is unregistered (the override skips the
    # config lookup) — this is what lets the Agent re-render a retired command entry from history.
    world.unregister("temp")
    rendered_after = flatten(
        world.render_entry(entry, serialize_fn=lambda v, p: [TextPart(f"body={v}")])
    )
    assert "body=raw value" in rendered_after


def test_render_entry_ignores_include_in_prompt(world: World):
    world.register("hidden", str, serialize_fn=identity_serialize, include_in_prompt=False)
    world.update("hidden", "secret")
    assert "secret" in flatten(world.render_entry(world.get_entry("hidden")))


def test_get_prompt_entries_omits_excluded_and_orders_by_timestamp(world: World):
    world.register("first", str, serialize_fn=identity_serialize, include_in_prompt=True)
    world.register("hidden", str, serialize_fn=identity_serialize, include_in_prompt=False)
    world.register("second", str, serialize_fn=identity_serialize, include_in_prompt=True)
    world.update("first", "1")
    world.update("hidden", "secret")
    world.update("second", "2")

    entries = world.get_prompt_entries()
    assert [e.key for e in entries] == ["first", "second"]
    assert all(isinstance(e, WorldEntry) for e in entries)

    world.update("first", "1-updated")
    entries = world.get_prompt_entries()
    assert [e.key for e in entries] == ["second", "first"]


def test_render_full_prompt_omits_excluded_entries(world: World):
    world.register("shown", str, serialize_fn=identity_serialize, include_in_prompt=True)
    world.register("hidden", str, serialize_fn=identity_serialize, include_in_prompt=False)
    world.update("shown", "visible-value")
    world.update("hidden", "secret-value")

    prompt = flatten(world.render_full_prompt())
    assert "visible-value" in prompt
    assert "secret-value" not in prompt


def test_render_full_prompt_orders_by_timestamp(world: World):
    world.register("first", str, serialize_fn=identity_serialize)
    world.register("second", str, serialize_fn=identity_serialize)
    world.update("first", "1")
    world.update("second", "2")

    prompt = flatten(world.render_full_prompt())
    assert prompt.index('key="first"') < prompt.index('key="second"')

    world.update("first", "1-updated")
    prompt = flatten(world.render_full_prompt())
    assert prompt.index('key="second"') < prompt.index('key="first"')


def test_render_full_prompt_block_matches_render_entry(world: World):
    world.register("solo", str, serialize_fn=identity_serialize)
    world.update("solo", "value")

    assert flatten(world.render_full_prompt()) == flatten(
        world.render_entry(world.get_entry("solo"))
    )


def test_render_full_prompt_separates_entries_with_a_single_newline(world: World):
    world.register("first", str, serialize_fn=identity_serialize)
    world.register("second", str, serialize_fn=identity_serialize)
    world.update("first", "1")
    world.update("second", "2")

    prompt = flatten(world.render_full_prompt())
    # Entries self-separate: one entry's `</entry>` and the next `<entry ...>` sit on adjacent
    # lines — never glued together, and never with a blank line between them.
    assert '</entry>\n<entry key="second"' in prompt
    assert "</entry>\n\n<entry" not in prompt
    assert "</entry><entry" not in prompt


def test_render_full_prompt_places_multimodal_part_between_entries(world: World):
    world.register("photo", bytes, serialize_fn=image_serialize)
    world.update("photo", b"\x89PNG")

    content = world.render_full_prompt()
    assert any(isinstance(part, ImagePart) for part in content)
