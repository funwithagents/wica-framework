from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from wica import Command, CommandIssued, Wica
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

    def _make(
        config: WicaConfig, *, output_sink: RecordingSink | None = None, **kwargs: Any
    ) -> Wica:
        # The sink is wired after construction (build, then wire, then start — see specs/wica.md,
        # "Output wiring is delegated"); the factory just folds that step in for the tests.
        w = Wica.init(config, **kwargs)
        if output_sink is not None:
            w.set_output_sink(output_sink)
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


def test_output_wiring_is_set_after_init_and_delegated_to_the_agent(wica_factory):
    """Build, then wire, then start: the sink and the output Command are set on the facade after
    construction (so the objects they belong to can be built against the Wica first) and reach
    the Agent — free text goes to the sink as private reasoning, the user-facing text goes
    through the output Command, and its name is reserved from register_command."""
    wica = wica_factory(
        fake_config(
            [
                {
                    "text": "thinking",
                    "tool_calls": [{"name": "speak", "args": {"text": "hello"}}],
                },
                {"tool_calls": [{"name": "noop", "args": {}}]},
            ]
        ),
        coalesce_window=0,
    )
    spoken: list[str] = []
    spoke = threading.Event()

    async def speak(text: str) -> str:
        """Say something to the user."""
        spoken.append(text)
        spoke.set()
        return "spoken"

    sink = RecordingSink()
    wica.set_output_sink(sink)
    wica.set_output_command(speak)
    assert wica.agent.output_command_name == "speak"
    with pytest.raises(ValueError, match="output Command"):
        wica.register_command(speak)
    wica.world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    wica.world.update("input", "greet")

    assert spoke.wait(timeout=WAIT_TIMEOUT)
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert spoken == ["hello"]
    assert sink.texts == ["thinking"]


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
    texts: list[str] = []
    wica.on_agent_text.subscribe(texts.append)
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
    assert [(c.name, c.args) for c in commands] == [("wave", {})]
    # The step's free text was observable on on_agent_text as well as delivered to the sink.
    assert texts == ["hi"]


def test_multiple_subscribers_fire_and_a_raising_one_does_not_block_others(
    wica_factory,
):
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

    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()

    wica.world.update("speech", "hi")
    assert fired.wait(
        timeout=WAIT_TIMEOUT
    )  # the raising sibling didn't starve the recorder
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

    wica.world.register(
        "go", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
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
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
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
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

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
        coalesce_window=0,
        loop=loop,
    )
    wica.set_output_sink(sink)
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
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


def _no_consecutive_human_messages(messages: list[BaseMessage]) -> bool:
    return not any(
        isinstance(a, HumanMessage) and isinstance(b, HumanMessage)
        for a, b in zip(messages, messages[1:])
    )


def test_stop_cancels_reasoning_before_a_restart(wica_factory):
    # The fake consumes a script step only once its delay elapses, so the call cancelled by stop()
    # consumes nothing and this reply belongs to the post-restart step.
    sink = RecordingSink()
    wica = wica_factory(
        fake_config([{"text": "only the post-restart step replies"}], delay_s=0.5),
        output_sink=sink,
        coalesce_window=0,
    )
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    prompts: list[list[BaseMessage]] = []
    wica.on_agent_prompt.subscribe(prompts.append)
    wica.start()
    wica.world.update("speech", "begin slow inference")
    wait_until(lambda: len(wica.agent.model.calls) == 1)

    wica.stop()
    wica.start()
    time.sleep(0.6)

    assert sink.texts == []

    # The cancelled step's observation stays in history; the next step renders it merged with its
    # own observation, never as back-to-back user messages.
    wica.world.update("speech", "after the restart")
    assert sink.event.wait(WAIT_TIMEOUT)
    assert sink.texts == ["only the post-restart step replies"]
    assert len(prompts) == 2
    assert _no_consecutive_human_messages(prompts[1])


# --- Safe shutdown from the loop thread --------------------------------


def test_stop_from_an_output_sink_is_rejected_and_the_system_keeps_running(
    wica_factory,
):
    wica_ref: list[Wica] = []
    errors: list[BaseException] = []
    done = threading.Event()

    async def sink(text: str) -> None:
        try:
            wica_ref[0].stop()
        except BaseException as exc:  # noqa: BLE001 - we want the exact error
            errors.append(exc)
        done.set()

    wica = wica_factory(
        fake_config([{"text": "hi"}]), output_sink=sink, coalesce_window=0
    )
    wica_ref.append(wica)
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    wica.world.update("speech", "hello")
    assert done.wait(WAIT_TIMEOUT)
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "event-loop thread" in str(errors[0])
    assert wica.is_running and wica.world.is_running  # nothing was torn down
    wica.stop()  # from the test thread: clean
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
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
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


# --- Reserved names and duplicate rejection reach through the facade ----


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


def test_is_running_polled_from_the_loop_thread_does_not_deadlock_stop():
    # Regression: `is_running` used to take the lifecycle lock, which stop() holds for the whole
    # shutdown (Agent drain on the loop, then join of the loop thread). A Command polling
    # `wica.is_running` on the loop thread then blocked the loop against that lock, the drain
    # could never run, and the join never returned. Built without the closing fixture so a
    # regression fails the test instead of hanging the session's teardown.
    wica = Wica.init(
        fake_config([{"tool_calls": [{"name": "poll", "args": {}}]}]),
        coalesce_window=0,
    )
    seen_running = threading.Event()

    async def poll() -> str:
        """Polls the facade's is_running until cancelled."""
        while True:
            if wica.is_running:
                seen_running.set()
            await asyncio.sleep(0)

    wica.world.register(
        "go", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.register_command(poll)
    wica.start()
    wica.world.update("go", "start")
    assert seen_running.wait(timeout=WAIT_TIMEOUT)

    stopper = threading.Thread(target=wica.stop, daemon=True)
    stopper.start()
    stopper.join(timeout=WAIT_TIMEOUT)
    assert not stopper.is_alive(), (
        "stop() deadlocked against a loop-thread is_running read"
    )
    assert wica.is_running is False
    wica.close()
