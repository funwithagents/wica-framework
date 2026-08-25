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
    script: list[dict[str, Any]] | None = None, *, logging_level: str = "WARNING"
) -> WicaConfig:
    """A WicaConfig backed by the scripted provider: "fake" model (network-free, key-less)."""
    return WicaConfig(
        agent=AgentConfig(
            provider="fake",
            model="test",
            system_prompt="You are terse.",
            model_kwargs={"script": script or [{"text": "ok"}], "delay_s": 0},
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
        w.stop()
        # stop() only closes a loop it actually started; close an owned-but-never-started one too.
        if w._owns_loop and not w._loop.is_closed():
            w._loop.close()


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
    assert wica._loop_thread is not None and not wica._loop_thread.is_alive()
    assert wica.world.is_running is False


def test_update_after_stop_raises(wica_factory):
    wica = wica_factory(fake_config())
    wica.world.register("speech", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    wica.start()
    wica.stop()

    with pytest.raises(RuntimeError, match="World is not running"):
        wica.world.update("speech", "too late")
