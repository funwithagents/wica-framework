import re
import threading
import time
from datetime import timedelta

import pytest

from wica.content import Content, ImagePart, TextPart
from wica.world import World, WorldEntry, get_world

WAIT_TIMEOUT = 2.0


def flatten(content: Content) -> str:
    return "".join(part.to_string() for part in content)


def identity_serialize(value, previous_value) -> Content:
    return [TextPart(str(value))]


def image_serialize(value: bytes | None, previous_value: bytes | None) -> Content:
    return [ImagePart(value, "image/png")] if value is not None else []


def text_archival_serialize(value: bytes | None, previous_value: bytes | None) -> Content:
    return [TextPart("a photo")]


class RecordingCallback:
    """Test double for a listener/trigger handler: records calls and signals an Event."""

    def __init__(self):
        self.calls: list[WorldEntry] = []
        self.event = threading.Event()

    def __call__(self, entry: WorldEntry) -> None:
        self.calls.append(entry)
        self.event.set()


def test_register_then_get_returns_none():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    assert world.get("temp") is None


def test_update_changes_get_result():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 42)
    assert world.get("temp") == 42


def test_update_type_mismatch_raises_and_leaves_value_untouched():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 5)
    with pytest.raises(TypeError):
        world.update("temp", "not an int")
    assert world.get("temp") == 5
    assert '<entry key="temp" id="2">' in flatten(world.render_entry(world.get_entry("temp")))


def test_update_none_always_succeeds_regardless_of_type():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", None)
    assert world.get("temp") is None


def test_id_starts_at_one_on_register_and_increments_on_every_update():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    assert '<entry key="temp" id="1">' in flatten(world.render_entry(world.get_entry("temp")))

    world.update("temp", "a")
    assert '<entry key="temp" id="2">' in flatten(world.render_entry(world.get_entry("temp")))

    world.update("temp", "b")
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))


def test_clearing_advances_id_like_any_other_update():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "a")  # id=2
    world.update("temp", None)  # id=3, clearing still advances id
    assert world.get("temp") is None
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))

    world.update("temp", "c")  # id=4
    assert world.get("temp") == "c"
    assert '<entry key="temp" id="4">' in flatten(world.render_entry(world.get_entry("temp")))


def test_unregister_removes_entry():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    world.unregister("temp")
    with pytest.raises(KeyError):
        world.get("temp")
    with pytest.raises(KeyError):
        world.update("temp", "x")


def test_reregister_without_unregister_raises():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    with pytest.raises(ValueError):
        world.register("temp", str, serialize_fn=identity_serialize)


def test_reregister_after_unregister_continues_id_sequence():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)  # id=1
    world.update("temp", 5)  # id=2
    world.unregister("temp")

    world.register("temp", int, serialize_fn=identity_serialize)  # id=3
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))
    world.update("temp", 7)  # id=4
    assert '<entry key="temp" id="4">' in flatten(world.render_entry(world.get_entry("temp")))


def test_get_after_reregistration_with_different_type_just_returns_the_new_value():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 5)
    world.unregister("temp")

    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "now a string")

    # no handle to go stale — get() always reflects whatever is currently registered
    assert world.get("temp") == "now a string"


def test_get_entry_matches_get_and_registered_type():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.update("temp", 7)

    entry = world.get_entry("temp")
    assert entry.key == "temp"
    assert entry.type is int
    assert entry.current.value == 7
    assert entry.current.value == world.get("temp")


def test_get_entry_unregistered_raises():
    world = World()
    world.register("temp", int, serialize_fn=identity_serialize)
    world.unregister("temp")
    with pytest.raises(KeyError):
        world.get_entry("temp")


def test_listener_fires_on_update_but_not_for_unrelated_keys():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    world.register("other", str, serialize_fn=identity_serialize)
    callback = RecordingCallback()
    world.add_listener("temp", callback)

    world.update("other", "irrelevant")
    assert not callback.event.wait(timeout=0.2)

    world.update("temp", "hello")
    assert callback.event.wait(timeout=WAIT_TIMEOUT)
    assert callback.calls[-1].current.value == "hello"


def test_remove_listener_stops_notifications():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    callback = RecordingCallback()
    world.add_listener("temp", callback)
    world.remove_listener("temp", callback)

    world.update("temp", "hello")
    assert not callback.event.wait(timeout=0.2)
    assert callback.calls == []


def test_trigger_handler_invoked_without_condition_fn():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    handler = RecordingCallback()
    world.set_trigger_handler(handler)

    world.update("temp", "hello")
    assert handler.event.wait(timeout=WAIT_TIMEOUT)
    assert handler.calls[-1].current.value == "hello"


def test_trigger_condition_fn_gates_handler():
    world = World()
    world.register(
        "temp",
        int,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        trigger_condition_fn=lambda old, new: new is not None and new > 10,
    )
    handler = RecordingCallback()
    world.set_trigger_handler(handler)

    world.update("temp", 5)
    assert not handler.event.wait(timeout=0.2)

    world.update("temp", 20)
    assert handler.event.wait(timeout=WAIT_TIMEOUT)
    assert handler.calls[-1].current.value == 20


def test_trigger_condition_fn_suppresses_noop_updates():
    world = World()
    world.register(
        "temp",
        str,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        trigger_condition_fn=lambda old, new: old != new,
    )
    handler = RecordingCallback()
    world.set_trigger_handler(handler)

    world.update("temp", "same")
    assert handler.event.wait(timeout=WAIT_TIMEOUT)
    handler.event.clear()

    world.update("temp", "same")
    assert not handler.event.wait(timeout=0.2)


def test_unregister_never_invokes_trigger_handler():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    handler = RecordingCallback()
    world.set_trigger_handler(handler)

    world.unregister("temp")
    assert not handler.event.wait(timeout=0.2)


def test_update_without_triggers_llm_call_never_invokes_handler():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize, triggers_llm_call=False)
    handler = RecordingCallback()
    world.set_trigger_handler(handler)

    world.update("temp", "hello")
    assert not handler.event.wait(timeout=0.2)


def test_update_does_not_block_on_slow_callback():
    world = World()
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


def test_snapshot_handed_to_listener_is_unaffected_by_later_update():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    callback = RecordingCallback()
    world.add_listener("temp", callback)

    world.update("temp", "first")
    world.update("temp", "second")

    assert callback.event.wait(timeout=WAIT_TIMEOUT)
    assert callback.calls[0].current.value == "first"


def test_serialize_fn_receives_correct_previous_value():
    seen = []

    def recording_serialize(value, previous_value) -> Content:
        seen.append((value, previous_value))
        return [TextPart(str(value))]

    world = World()
    world.register("temp", str, serialize_fn=recording_serialize)
    world.update("temp", "A")
    world.render_entry(world.get_entry("temp"))
    world.update("temp", "B")
    world.render_entry(world.get_entry("temp"))
    world.update("temp", None)
    world.render_entry(world.get_entry("temp"))

    assert seen == [("A", None), ("B", "A"), (None, "B")]


def test_ttl_resets_value_to_none_and_advances_id_without_triggering():
    world = World()
    world.register(
        "temp",
        str,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        ttl=timedelta(milliseconds=100),
    )
    handler = RecordingCallback()
    world.set_trigger_handler(handler)

    world.update("temp", "hello")  # id=2
    assert handler.event.wait(timeout=WAIT_TIMEOUT)  # the explicit update itself triggers
    handler.event.clear()

    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline and world.get("temp") is not None:
        time.sleep(0.01)

    assert world.get("temp") is None
    assert '<entry key="temp" id="3">' in flatten(world.render_entry(world.get_entry("temp")))
    assert not handler.event.wait(timeout=0.2)  # but the TTL-driven reset does not


def test_ttl_expiry_for_superseded_version_does_not_clobber_fresh_value():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize, ttl=timedelta(seconds=10))
    world.update("temp", "hello")  # id=2, arms a TTL timer bound to id=2
    world.update("temp", "fresh")  # id=3 (cancels id=2's timer, arms id=3's)

    # Simulate id=2's timer firing late — after "fresh" already landed. It must no-op,
    # since the entry's current id is now 3, not 2. Driving the private callback directly
    # because there's no public way to trigger an already-fired, stale timer.
    world._ttl_expire("temp", 2)
    assert world.get("temp") == "fresh"


def test_ttl_restart_postpones_reset():
    world = World()
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


def test_render_entry_format():
    world = World()
    world.register("user_profile", str, serialize_fn=lambda v, p: [TextPart("Jane is logged in.")])
    world.update("user_profile", "irrelevant raw value")

    rendered = flatten(world.render_entry(world.get_entry("user_profile")))
    match = re.match(
        r'^<entry key="user_profile" id="2">\n'
        r"Jane is logged in\.\n"
        r"Updated: (?P<ts>.+)\n"
        r"</entry>$",
        rendered,
    )
    assert match is not None


def test_render_entry_body_parts_are_returned_unwrapped():
    world = World()
    world.register("photo", bytes, serialize_fn=image_serialize)
    world.update("photo", b"\x89PNG")

    content = world.render_entry(world.get_entry("photo"))
    assert content == [
        TextPart('<entry key="photo" id="2">\n'),
        ImagePart(b"\x89PNG", "image/png"),
        TextPart(f"\nUpdated: {world.get_entry('photo').current.timestamp.isoformat()}\n</entry>"),
    ]


def test_render_entry_uses_archival_serialize_fn_when_requested():
    world = World()
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
        TextPart(f"\nUpdated: {entry.current.timestamp.isoformat()}\n</entry>"),
    ]


def test_render_entry_archival_falls_back_to_serialize_fn_when_absent():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    world.update("temp", "value")
    entry = world.get_entry("temp")

    assert world.render_entry(entry, archival=True) == world.render_entry(entry, archival=False)


def test_render_entry_raises_when_entry_key_no_longer_registered():
    world = World()
    world.register("temp", str, serialize_fn=identity_serialize)
    stale_entry = world.get_entry("temp")
    world.unregister("temp")

    with pytest.raises(KeyError):
        world.render_entry(stale_entry)


def test_render_entry_ignores_include_in_prompt():
    world = World()
    world.register("hidden", str, serialize_fn=identity_serialize, include_in_prompt=False)
    world.update("hidden", "secret")
    assert "secret" in flatten(world.render_entry(world.get_entry("hidden")))


def test_render_full_prompt_omits_excluded_entries():
    world = World()
    world.register("shown", str, serialize_fn=identity_serialize, include_in_prompt=True)
    world.register("hidden", str, serialize_fn=identity_serialize, include_in_prompt=False)
    world.update("shown", "visible-value")
    world.update("hidden", "secret-value")

    prompt = flatten(world.render_full_prompt())
    assert "visible-value" in prompt
    assert "secret-value" not in prompt


def test_render_full_prompt_orders_by_timestamp():
    world = World()
    world.register("first", str, serialize_fn=identity_serialize)
    world.register("second", str, serialize_fn=identity_serialize)
    world.update("first", "1")
    world.update("second", "2")

    prompt = flatten(world.render_full_prompt())
    assert prompt.index('key="first"') < prompt.index('key="second"')

    world.update("first", "1-updated")
    prompt = flatten(world.render_full_prompt())
    assert prompt.index('key="second"') < prompt.index('key="first"')


def test_render_full_prompt_block_matches_render_entry():
    world = World()
    world.register("solo", str, serialize_fn=identity_serialize)
    world.update("solo", "value")

    assert flatten(world.render_full_prompt()) == flatten(
        world.render_entry(world.get_entry("solo"))
    )


def test_render_full_prompt_places_multimodal_part_between_entries():
    world = World()
    world.register("photo", bytes, serialize_fn=image_serialize)
    world.update("photo", b"\x89PNG")

    content = world.render_full_prompt()
    assert any(isinstance(part, ImagePart) for part in content)


def test_get_world_singleton():
    a = get_world()
    b = get_world()
    assert a is b

    a.register("shared", str, serialize_fn=identity_serialize)
    a.update("shared", "hello")
    assert b.get("shared") == "hello"
