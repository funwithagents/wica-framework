from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from wica import CommandIssued, Wica
from wica.config import AgentConfig, WicaConfig
from wica.content import Content, TextPart
from wica.world import WorldEntry

WAIT_TIMEOUT = 2.0


def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except KeyError:
            pass
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


def identity_serialize(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


def fake_config(
    script: list[dict[str, Any]] | None = None,
    *,
    logging_level: str = "WARNING",
    delay_s: float = 0,
) -> WicaConfig:
    """A WicaConfig backed by the scripted provider: "fake" model (network-free, key-less)."""
    return WicaConfig(
        agent=AgentConfig(
            provider="fake",
            model="test",
            system_prompt="You are terse.",
            model_kwargs={"script": script or [{"text": "ok"}], "delay_s": delay_s},
        ),
        logging=logging_level,
    )


class RecordingSink:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.event = threading.Event()

    async def __call__(self, text: str) -> None:
        self.texts.append(text)
        self.event.set()


@pytest.fixture
def wica_factory() -> Iterator[Callable[..., Wica]]:
    created: list[Wica] = []

    def _make(config: WicaConfig, **kwargs: Any) -> Wica:
        w = Wica.init(config, **kwargs)
        created.append(w)
        return w

    yield _make
    for w in created:
        w.close()


def test_init_wires_world_and_agent_sharing_one_loop(wica_factory):
    wica = wica_factory(fake_config())

    assert wica.world is not None
    assert wica.agent is not None
    # One loop for the whole system — the World and Agent were handed the same object.
    assert wica.world._loop is wica.agent._loop
    assert wica.agent._world is wica.world


def test_init_applies_logging(wica_factory):
    try:
        wica_factory(fake_config(logging_level="DEBUG"))
        assert logging.getLogger("wica").level == logging.DEBUG
    finally:
        logging.getLogger("wica").setLevel(logging.WARNING)


def test_surfaced_events_are_the_same_objects(wica_factory):
    wica = wica_factory(fake_config())
    # Wica surfaces the underlying Events directly — no adapters, no emit shims.
    assert wica.on_world_trigger is wica.world.on_trigger
    assert wica.on_agent_trigger is wica.agent.on_trigger
    assert wica.on_agent_prompt is wica.agent.on_prompt
    assert wica.on_agent_command is wica.agent.on_command


def test_register_command_reaches_the_agent(wica_factory):
    wica = wica_factory(fake_config())

    async def wave() -> str:
        """Wave hello."""
        return "waved"

    wica.register_command(wave)
    assert "wave" in wica.agent._commands


def test_input_drives_world_and_agent_triggers_and_a_command_event(wica_factory):
    sink = RecordingSink()
    wica = wica_factory(
        fake_config([{"tool_calls": [{"name": "wave", "args": {}}]}, {"text": "hi"}]),
        output_sink=sink,
    )

    async def wave() -> str:
        """Wave hello."""
        return "waved"

    world_triggers: list[str] = []
    agent_triggers: list[str] = []
    commands: list[CommandIssued] = []
    wica.on_world_trigger.subscribe(lambda e: world_triggers.append(e.key))
    wica.on_agent_trigger.subscribe(lambda e: agent_triggers.append(e.key))
    wica.on_agent_command.subscribe(commands.append)

    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.register_command(wave)
    wica.start()

    wica.world.update("speech", "hello robot")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    # The raw World trigger fired for the input; the filtered Agent trigger observed it too.
    assert "speech" in world_triggers
    assert "speech" in agent_triggers
    # The model issued the wave Command; on_agent_command carried a CommandIssued at dispatch.
    assert CommandIssued("wave", {}) in commands


def test_multiple_subscribers_fire_and_a_raising_one_does_not_block_others(wica_factory):
    wica = wica_factory(fake_config())

    seen: list[str] = []
    fired = threading.Event()

    def boom(entry: WorldEntry) -> None:
        raise RuntimeError("subscriber failure")

    def recorder(entry: WorldEntry) -> None:
        seen.append(entry.key)
        fired.set()

    wica.on_world_trigger.subscribe(boom)
    wica.on_world_trigger.subscribe(recorder)

    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.start()

    wica.world.update("speech", "hi")
    assert fired.wait(timeout=WAIT_TIMEOUT)  # the raising sibling didn't starve the recorder
    assert seen == ["speech"]


def test_stop_with_an_in_flight_command_tears_down_cleanly(wica_factory):
    # Teardown is agent→world→loop: agent.stop() cancels the in-flight command while the World is
    # still running, so its terminal write is tolerated rather than raising, and stop() completes.
    sink = RecordingSink()
    wica = wica_factory(
        fake_config([{"tool_calls": [{"name": "block_forever", "args": {}}]}]),
        output_sink=sink,
    )
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    wica.world.register("go", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.register_command(block_forever)
    wica.start()

    wica.world.update("go", "start")
    key = "agent:command:fake_call_0_0"
    wait_until(lambda: wica.world.get_entry(key).current.value.state == "running")

    wica.stop()  # must not raise, even with a command in flight; joins the owned loop thread last

    # agent.stop() drains the cancellation while the World is still running, so the terminal write
    # landed before the World stopped — the command reads cancelled, not left dangling.
    assert wica.world.get_entry(key).current.value.state == "cancelled"
    # After stop the loop thread is gone (owned loop) and the World is torn down.
    assert wica._loop_thread is None
    assert wica.world.is_running is False


def test_update_after_stop_raises(wica_factory):
    wica = wica_factory(fake_config())
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.start()
    wica.stop()

    with pytest.raises(RuntimeError, match="World is not running"):
        wica.world.update("speech", "too late")


def test_owned_wica_can_stop_and_start_again(wica_factory):
    sink = RecordingSink()
    wica = wica_factory(
        fake_config([{"text": "first cycle"}, {"text": "second cycle"}]),
        output_sink=sink,
        coalesce_window=0,
    )
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    wica.start()
    first_thread = wica._loop_thread
    wica.world.update("speech", "one")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert sink.texts == ["first cycle"]

    wica.stop()
    assert wica.is_running is False
    assert wica.world.is_running is False
    assert not wica._loop.is_closed()
    assert wica._loop_thread is None

    sink.event.clear()
    wica.start()
    second_thread = wica._loop_thread
    assert second_thread is not None and second_thread is not first_thread
    wica.world.update("speech", "two")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert sink.texts == ["first cycle", "second cycle"]
    assert wica.world.get("speech") == "two"


def test_start_and_stop_are_idempotent(wica_factory):
    wica = wica_factory(fake_config())
    wica.start()
    thread = wica._loop_thread

    wica.start()
    assert wica._loop_thread is thread

    wica.stop()
    wica.stop()
    assert wica.is_running is False


def test_close_is_terminal_and_closes_an_owned_loop(wica_factory):
    wica = wica_factory(fake_config())
    wica.start()

    wica.close()
    wica.close()

    assert wica.is_running is False
    assert wica._loop.is_closed()
    with pytest.raises(RuntimeError, match="Wica is closed"):
        wica.start()


def test_close_before_start_closes_the_owned_loop(wica_factory):
    wica = wica_factory(fake_config())

    wica.close()

    assert wica._loop.is_closed()


def test_injected_loop_wica_can_restart_without_owning_the_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    sink = RecordingSink()
    wica = Wica.init(
        fake_config([{"text": "one"}, {"text": "two"}]),
        output_sink=sink,
        coalesce_window=0,
        loop=loop,
    )
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    try:
        wica.start()
        wica.world.update("speech", "first")
        assert sink.event.wait(timeout=WAIT_TIMEOUT)
        wica.stop()

        sink.event.clear()
        wica.start()
        wica.world.update("speech", "second")
        assert sink.event.wait(timeout=WAIT_TIMEOUT)

        assert sink.texts == ["one", "two"]
        wica.close()
        assert not loop.is_closed()
    finally:
        wica.close()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=WAIT_TIMEOUT)
        loop.close()


def test_stop_cancels_reasoning_before_a_restart(wica_factory):
    sink = RecordingSink()
    wica = wica_factory(
        fake_config([{"text": "must not escape the stopped cycle"}], delay_s=0.5),
        output_sink=sink,
        coalesce_window=0,
    )
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.start()
    wica.world.update("speech", "begin slow inference")
    wait_until(lambda: len(wica.agent.model.calls) == 1)

    wica.stop()
    wica.start()
    time.sleep(0.6)

    assert sink.texts == []
