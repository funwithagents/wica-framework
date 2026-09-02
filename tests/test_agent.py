from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from wica.agent import (
    Agent,
    AssistantTextRecord,
    CommandIssued,
    NoReactionRecord,
    ObservationRecord,
    _command_ack,
    _NOOP_ACK,
    _NOOP_COMMAND_NAME,
)
from wica.command import Command
from wica.config import AgentConfig
from wica.content import Content, TextPart
from wica.world import World

WAIT_TIMEOUT = 2.0


def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except KeyError:
            pass  # e.g. a World key that hasn't been registered by the background step yet
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


def identity_serialize(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


class ProgrammableChatModel(BaseChatModel):
    """A hand-written BaseChatModel whose responses come from an arbitrary async `respond`
    callable — for deterministic control over .tool_calls *and blocking model calls* that the
    data-scripted `provider: "fake"` model (wica.fake_model.FakeChatModel) can't give us: its
    responses are JSON config, so it can't await a test-controlled event mid-call. Injected via
    the Agent `model=` override seam. Named to avoid colliding with that library FakeChatModel."""

    respond: Callable[[list[BaseMessage]], Awaitable[AIMessage]] | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError("ProgrammableChatModel is async-only")

    async def _agenerate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(messages)
        assert self.respond is not None, (
            "ProgrammableChatModel.respond must be set before use"
        )
        message = await self.respond(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self, tools, *, tool_choice=None, **kwargs: Any
    ) -> Runnable[Any, AIMessage]:
        return self


def text_response(text: str) -> Callable[[list[BaseMessage]], Awaitable[AIMessage]]:
    async def respond(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content=text)

    return respond


def tool_call_response(
    calls: list[tuple[str, dict[str, Any], str]],
) -> Callable[[list[BaseMessage]], Awaitable[AIMessage]]:
    async def respond(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args, "id": call_id}
                for name, args, call_id in calls
            ],
        )

    return respond


def sequence(
    *responders: Callable[[list[BaseMessage]], Awaitable[AIMessage]],
) -> Callable[[list[BaseMessage]], Awaitable[AIMessage]]:
    remaining = list(responders)

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        responder = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return await responder(messages)

    return respond


class RecordingSink:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.event = threading.Event()

    async def __call__(self, text: str) -> None:
        self.texts.append(text)
        self.event.set()


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=WAIT_TIMEOUT)
    new_loop.close()


@pytest.fixture
def world(loop: asyncio.AbstractEventLoop) -> Iterator[World]:
    w = World(loop)
    w.start()
    yield w
    w.stop()


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


def make_agent(
    model: BaseChatModel,
    *,
    world: World,
    loop: asyncio.AbstractEventLoop,
    system_prompt: str = "You are terse.",
    **kwargs: Any,
) -> Agent:
    """Construct an Agent over an injected (fully-scripted) model. The Agent takes an AgentConfig
    and builds its own model from it; the `model=` override lets these tests supply the bespoke
    ProgrammableChatModel the loop is driven over. The prompt is carried on the config."""
    config = AgentConfig(provider="fake", model="test", system_prompt=system_prompt)
    return Agent(config, world=world, loop=loop, model=model, **kwargs)


def human_texts(message: BaseMessage) -> str:
    return message.text


def test_text_only_response_updates_sink_and_history(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("Hello there"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    world.update("input", "hi")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert sink.texts == ["Hello there"]
    assert len(agent._history) == 2
    assert isinstance(agent._history[0], ObservationRecord)
    assert [e.key for e in agent._history[0].entries] == ["input"]
    assert agent._history[1] == AssistantTextRecord("Hello there")

    agent.stop()


def test_tool_call_dispatches_then_completes_and_retriggers(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("The sum is 3"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        await asyncio.sleep(0.05)  # slow enough that "running" is observable below
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("input", "add 1 and 2")

    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")
    entry = world.get_entry(key)
    assert entry.current.value.name == "add"
    assert entry.current.value.args == {"a": 1, "b": 2}

    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert sink.texts == ["The sum is 3"]
    assert len(model.calls) == 2

    second_call_messages = model.calls[1]
    ai_messages = [m for m in second_call_messages if isinstance(m, AIMessage)]
    # the past command is a real tool_call — not "Called add …" text
    assert any(
        c["name"] == "add" and c["args"] == {"a": 1, "b": 2}
        for m in ai_messages
        for c in m.tool_calls
    )
    # its tool_result is a fixed ack pointing at the command entry, never the outcome
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("call1")
        for m in tool_messages
    )
    # the outcome (3) is delivered by the command's World entry, rendered into the observation
    observation = "".join(
        str(m.content) for m in second_call_messages if isinstance(m, HumanMessage)
    )
    assert "Called add(a=1, b=2) → 3" in observation

    with pytest.raises(KeyError):
        world.get_entry(key)

    agent.stop()


def test_past_commands_render_as_native_tool_calls_not_prose(loop, world, sink):
    # Regression: past commands must be re-rendered as the model's own native tool_calls, not as
    # a "Calling foo(...)…" assistant text block — otherwise the model imitates that prose and
    # emits command descriptions as plain text instead of issuing real tool calls.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("the sum is 3"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("input", "add them")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    messages = agent._render_messages()
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    assert any(c["name"] == "add" for m in ai_messages for c in m.tool_calls)
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("call1")
        for m in tool_messages
    )
    # no *assistant* message renders the call as prose (the outcome lives in the observation, a
    # user-role HumanMessage, so there is nothing for the model to imitate as its own output)
    assert not any(
        "Calling add" in str(m.content) or "Called add" in str(m.content)
        for m in ai_messages
    )

    agent.stop()


def test_tool_failure_surfaces_into_world_and_next_step(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("explode", {}, "call1")]),
            text_response("Sorry, that failed"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def explode() -> str:
        """Always raises."""
        await asyncio.sleep(0.05)  # slow enough that "running" is observable below
        raise ValueError("boom")

    agent.register_command(explode)
    agent.start()

    world.update("input", "explode please")

    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    second_call_messages = model.calls[1]
    # the tool_result is the fixed ack; the failure is delivered by the command's World entry
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("call1")
        for m in tool_messages
    )
    observation = "".join(
        str(m.content) for m in second_call_messages if isinstance(m, HumanMessage)
    )
    assert "Called explode() → failed: boom" in observation

    agent.stop()


def test_running_command_shown_as_in_progress_to_a_concurrent_step(loop, world, sink):
    # Regression (specs/_todo.md, "dance then say hello during dance"): while a command is still
    # running, a new input starts a fresh step; that step must be told the command is NOT finished.
    # The running command renders as an in-progress observation entry, and its tool_result is a
    # fixed ack — not a completed result that would read as "the call returned".
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()

    async def dance() -> str:
        """Blocks until released — models a long-running action."""
        await block.wait()
        return "done dancing"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("dance", {}, "call1")]),
            text_response("hi"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(dance)
    agent.start()

    world.update("input", "dance for me")
    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")
    wait_until(
        lambda: agent._busy is False
    )  # step 1 finished; dance still running in background

    # A new input arrives while dance is still running -> a concurrent step runs.
    world.update("input", "say hi")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert sink.texts == ["hi"]
    assert len(model.calls) == 2

    step2 = model.calls[1]
    # the running command is shown as an in-progress observation entry (the model knows it's not
    # done and could reason about / cancel it), rather than looking completed
    observation = "".join(str(m.content) for m in step2 if isinstance(m, HumanMessage))
    assert "Calling dance()… (still running — not finished)" in observation
    # its tool_result is the fixed ack, never a completed result
    tool_messages = [m for m in step2 if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("call1")
        for m in tool_messages
    )
    assert world.get_entry(key).current.value.state == "running"

    block.set()  # let dance finish before teardown
    agent.stop()


def test_parallel_tool_calls_independent_keys_and_mixed_status_line(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    slow_release = asyncio.Event()

    async def fast_tool() -> str:
        """Resolves immediately."""
        return "fast-result"

    async def slow_tool() -> str:
        """Blocks until released."""
        await slow_release.wait()
        return "slow-result"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("fast_tool", {}, "fast"), ("slow_tool", {}, "slow")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(fast_tool)
    agent.register_command(slow_tool)
    agent.start()

    world.update("input", "run both")

    fast_key = "agent:command:fast"
    slow_key = "agent:command:slow"
    wait_until(lambda: world.get_entry(slow_key).current.value.state == "running")

    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert len(model.calls) == 2

    second_call_messages = model.calls[1]
    call_names = {
        c["name"]
        for m in second_call_messages
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    }
    assert {"fast_tool", "slow_tool"} <= call_names
    tool_contents = {
        m.tool_call_id: m.content
        for m in second_call_messages
        if isinstance(m, ToolMessage)
    }
    assert tool_contents.get("fast") == _command_ack("fast")
    assert tool_contents.get("slow") == _command_ack("slow")
    # the mixed status is delivered by the command entries in the observation: fast completed,
    # slow was still running when this prompt was built
    observation = "".join(
        str(m.content) for m in second_call_messages if isinstance(m, HumanMessage)
    )
    assert "Called fast_tool() → fast-result" in observation
    assert "Calling slow_tool()… (still running — not finished)" in observation

    with pytest.raises(KeyError):
        world.get_entry(fast_key)
    assert world.get_entry(slow_key).current.value.state == "running"

    slow_release.set()  # let slow finish before teardown
    agent.stop()


def _is_unregistered(world: World, key: str) -> bool:
    try:
        world.get_entry(key)
        return False
    except KeyError:
        return True


def test_dropped_command_completion_persists_until_observed(loop, world, sink):
    # A turn issues two commands. The first completes and starts a second step; while that step is
    # in flight (busy), the second command completes — its trigger is dropped by the single-in-flight
    # loop. The dropped completion is NOT eagerly retired: it stays as current World state and is
    # rendered into history (then retired) by the next step that observes it, so the completed
    # Command is never silently lost — the invariant "always in history or current state".
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    release_slow = asyncio.Event()
    release_step2 = asyncio.Event()
    step2_running = threading.Event()

    async def fast() -> str:
        """Completes immediately."""
        return "fast-result"

    async def slow() -> str:
        """Completes only once released."""
        await release_slow.wait()
        return "slow-result"

    async def respond2(messages: list[BaseMessage]) -> AIMessage:
        step2_running.set()
        await release_step2.wait()
        return AIMessage(content="done")

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("fast", {}, "fast"), ("slow", {}, "slow")]),
            respond2,
            text_response("ack"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(fast)
    agent.register_command(slow)
    agent.start()

    world.update("input", "run both")

    # fast completes -> step 2 starts and blocks, so the loop is busy.
    assert step2_running.wait(timeout=WAIT_TIMEOUT)

    # Let slow finish: its completion trigger arrives while busy and is dropped (no step runs).
    loop.call_soon_threadsafe(release_slow.set)
    wait_until(
        lambda: world.get_entry("agent:command:slow").current.value.state == "complete"
    )

    # Finish step 2. fast was observed and retired by it; slow's completion was dropped, so no
    # third step ran for it — and it must persist as current terminal state, not be lost.
    loop.call_soon_threadsafe(release_step2.set)
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert sink.texts == ["done"]
    assert len(model.calls) == 2
    assert _is_unregistered(world, "agent:command:fast")
    assert world.get_entry("agent:command:slow").current.value.state == "complete"

    # A fresh input starts step 3, which observes slow: its outcome renders into that step's
    # prompt, and only then is the entry retired — the completion reaches history before it goes.
    sink.event.clear()
    world.update("input", "poke")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert len(model.calls) == 3
    step3_observation = "".join(
        str(m.content) for m in model.calls[2] if isinstance(m, HumanMessage)
    )
    assert "Called slow() → slow-result" in step3_observation
    wait_until(lambda: _is_unregistered(world, "agent:command:slow"))

    agent.stop()


def test_single_in_flight_trigger_dropped_and_logged(loop, world, sink, caplog):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    hold = asyncio.Event()
    started = threading.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        started.set()
        await hold.wait()
        return AIMessage(content="finally")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    with caplog.at_level(logging.INFO, logger="wica.agent"):
        world.update("input", "first")
        assert started.wait(timeout=WAIT_TIMEOUT)

        world.update("input", "second")
        world.update("input", "third")
        time.sleep(0.2)  # let any (incorrect) extra steps have a chance to start

        loop.call_soon_threadsafe(hold.set)
        assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert sink.texts == ["finally"]
    assert len(model.calls) == 1
    assert sum(1 for r in caplog.records if "dropping trigger" in r.message) == 2
    assert len(agent._history) == 2  # one ObservationRecord + one AssistantTextRecord

    agent.stop()


def test_full_bundle_capture_includes_passive_entries(loop, world, sink):
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=False)
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    world.update("b", "b-value")
    world.update("a", "a-value")

    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert len(agent._history) == 2
    observation = agent._history[0]
    assert isinstance(observation, ObservationRecord)
    assert {e.key for e in observation.entries} == {"a", "b"}

    agent.stop()


def test_freshness_flips_at_bundle_boundary(loop, world, sink):
    def fresh_serialize(value: Any, previous: Any) -> Content:
        return [TextPart(f"FRESH:{value}")]

    def archival_serialize(value: Any, previous: Any) -> Content:
        return [TextPart(f"ARCHIVAL:{value}")]

    world.register(
        "note",
        str,
        serialize_fn=fresh_serialize,
        archival_serialize_fn=archival_serialize,
        triggers_llm_call=True,
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response("first"), text_response("second"))
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    world.update("note", "one")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    sink.event.clear()

    world.update("note", "two")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    messages = agent._render_messages()
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(human_messages) == 2
    assert "ARCHIVAL:one" in human_texts(human_messages[0])
    assert "FRESH:two" in human_texts(human_messages[1])

    agent.stop()


def test_on_prompt_event_fires_with_the_messages_the_model_receives(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    captured: list[list[BaseMessage]] = []
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.on_prompt.subscribe(captured.append)
    agent.start()

    world.update("input", "hello")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert len(captured) == 1  # one emit per step
    assert captured[0] == model.calls[0]  # exactly the messages handed to the model

    agent.stop()


def test_on_prompt_subscriber_cannot_mutate_messages_sent_to_model(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    def mutate(messages: list[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, list):
                message.content.clear()

    agent.on_prompt.subscribe(mutate)
    agent.start()

    world.update("input", "hello")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert model.calls[0]
    assert any("hello" in human_texts(message) for message in model.calls[0])

    agent.stop()


def test_on_trigger_event_fires_with_the_entry_that_started_the_step(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    triggers: list[str] = []
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.on_trigger.subscribe(lambda entry: triggers.append(entry.key))
    agent.start()

    world.update("input", "hello")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert triggers == ["input"]

    agent.stop()


def test_on_command_event_fires_with_a_command_issued_at_dispatch(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    commands: list[CommandIssued] = []
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.on_command.subscribe(commands.append)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("input", "add them")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert commands == [CommandIssued("add", {"a": 1, "b": 2})]

    agent.stop()


def test_on_command_subscriber_cannot_mutate_dispatched_arguments(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("read_payload", {"payload": {"value": 1}}, "call1")]),
            text_response("done"),
        )
    )
    invoked: list[int] = []
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    def mutate(command: CommandIssued) -> None:
        command.args["payload"]["value"] = 99

    agent.on_command.subscribe(mutate)

    async def read_payload(payload: dict[str, int]) -> int:
        """Read a nested payload."""
        invoked.append(payload["value"])
        return payload["value"]

    agent.register_command(read_payload)
    agent.start()

    world.update("input", "add them")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert invoked == [1]

    agent.stop()


def test_raising_on_prompt_subscriber_does_not_abort_the_step(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("still replied"))

    def boom(messages: list[BaseMessage]) -> None:
        raise RuntimeError("subscriber failure")

    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.on_prompt.subscribe(boom)  # Event.emit isolates a raising subscriber
    agent.start()

    world.update("input", "hello")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert sink.texts == [
        "still replied"
    ]  # the raising subscriber didn't break the step
    assert len(model.calls) == 1

    agent.stop()


def test_stop_unsubscribes_from_the_world_trigger(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    agent.stop()

    world.update("input", "should not trigger")
    time.sleep(0.2)
    assert model.calls == []


def test_cancel_command_marks_cancelled_and_retriggers(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("block_forever", {}, "call1")]),
            text_response("cancelled that for you"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "block please")

    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    agent.cancel_command("call1")
    wait_until(lambda: sink.event.is_set())

    second_call_messages = model.calls[1]
    # the tool_result is the fixed ack; the cancellation is delivered by the command's World entry
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("call1")
        for m in tool_messages
    )
    observation = "".join(
        str(m.content) for m in second_call_messages if isinstance(m, HumanMessage)
    )
    assert "Called block_forever() → cancelled" in observation

    agent.cancel_command("does-not-exist")  # no-op, must not raise

    agent.stop()


def test_stop_cancels_running_tool_without_triggering_new_step(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    model = ProgrammableChatModel(
        respond=tool_call_response([("block_forever", {}, "call1")])
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "block please")

    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    agent.stop()

    # The World is still running (only the Agent stopped), so the cancellation's terminal write
    # lands — this is the agent-before-world teardown order Wica enforces.
    wait_until(lambda: world.get_entry(key).current.value.state == "cancelled")
    assert (
        len(model.calls) == 1
    )  # no second call: the Agent already unsubscribed from on_trigger


def _prompt_contains(model: ProgrammableChatModel, needle: str) -> bool:
    return any(
        needle in str(m.content)
        for call in model.calls
        for m in call
        if isinstance(m, HumanMessage)
    )


def _wait_for_render(model: ProgrammableChatModel, world: World, needle: str) -> None:
    """Wait until some model prompt has rendered `needle`, nudging the agent with fresh inputs so
    that a terminal command entry left in the World (e.g. a completion whose own trigger the
    single-in-flight loop dropped) is guaranteed to be observed by a later step. Once rendered, the
    snapshot is in the model's call history for good — even after the entry is retired."""
    deadline = time.monotonic() + WAIT_TIMEOUT
    i = 0
    while time.monotonic() < deadline:
        if _prompt_contains(model, needle):
            return
        world.update("input", f"status check {i}")
        i += 1
        time.sleep(0.02)
    raise AssertionError(f"no prompt rendered {needle!r} within {WAIT_TIMEOUT}s")


def test_model_can_cancel_a_running_command(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("block_forever", {}, "target")]),
            tool_call_response([("cancel_command", {"call_id": "target"}, "cancel1")]),
            text_response("stopped it"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(block_forever)
    agent.start()
    # cancel_command is a WICA-native Command the Agent auto-registers at start — no app wiring.
    assert "cancel_command" in agent._commands

    # Step 1: the model dispatches the long-running command, then the agent goes idle.
    world.update("input", "start working")
    key = "agent:command:target"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    # Step 2: a fresh input drives a step where the model observes it still running and issues
    # cancel_command(call_id="target"). The cancellation lands as the command's World entry going
    # terminal — the causally-correct point — and reaches the model as an observation.
    world.update("input", "actually, stop")
    _wait_for_render(model, world, "Called block_forever() → cancelled")

    agent.stop()


def test_cancel_command_action_is_lenient_on_full_entry_key(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("block_forever", {}, "t2")]),
            text_response("ok"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "go")
    key = "agent:command:t2"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    # The model may copy the entry key verbatim; the handler strips the agent:command: prefix and
    # still finds the running task — the "cancelling t2" result proves the lookup hit (a miss would
    # read "not a running command").
    future = asyncio.run_coroutine_threadsafe(
        agent._cancel_command_action("agent:command:t2"), loop
    )
    assert future.result(timeout=WAIT_TIMEOUT) == "cancelling t2"
    _wait_for_render(model, world, "Called block_forever() → cancelled")

    agent.stop()


def test_cancel_command_action_is_a_noop_for_unknown_id(loop, world, sink):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    future = asyncio.run_coroutine_threadsafe(
        agent._cancel_command_action("nope"), loop
    )
    result = future.result(timeout=WAIT_TIMEOUT)
    assert "not a running command" in result

    agent.stop()


def test_burst_of_triggers_coalesces_into_one_step(loop, world, sink):
    # Two triggers within the coalescing window run a *single* step whose observation captures
    # both — but on_trigger still fires once per collected trigger. See specs/agent.md.
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=text_response("ok"))
    triggers: list[str] = []
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0.3, output_sink=sink
    )
    agent.on_trigger.subscribe(lambda entry: triggers.append(entry.key))
    agent.start()

    world.update("a", "a-value")
    world.update("b", "b-value")  # joins the same window → same step
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    time.sleep(0.3)  # let any (incorrect) second step have a chance to run

    assert len(model.calls) == 1
    # dispatch order across the loop isn't guaranteed, so compare as a set
    assert sorted(triggers) == ["a", "b"]
    observation = agent._history[0]
    assert isinstance(observation, ObservationRecord)
    assert {e.key for e in observation.entries} == {"a", "b"}

    agent.stop()


def test_zero_window_fires_immediately_and_drops_while_busy(loop, world, sink):
    # coalesce_window=0 is the pre-coalescing behavior: each trigger fires at once (no wait), and a
    # trigger arriving while a step is in flight is dropped, not coalesced.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    hold = asyncio.Event()
    started = threading.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        started.set()
        await hold.wait()
        return AIMessage(content="done")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0, output_sink=sink
    )
    agent.start()

    t0 = time.monotonic()
    world.update("input", "first")
    assert started.wait(timeout=WAIT_TIMEOUT)
    assert time.monotonic() - t0 < 0.15  # no window wait, unlike the 0.2s default

    world.update("input", "second")  # arrives while busy → dropped
    time.sleep(0.2)
    loop.call_soon_threadsafe(hold.set)
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert len(model.calls) == 1
    assert len([r for r in agent._history if isinstance(r, ObservationRecord)]) == 1

    agent.stop()


def test_bypass_coalescing_flushes_the_window_early(loop, world, sink):
    # A long window would make a plain trigger wait ~1s; a bypass_coalescing trigger flushes the
    # window early, carrying along whatever was already batched.
    world.register("ctx", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register(
        "urgent",
        str,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        bypass_coalescing=True,
    )
    model = ProgrammableChatModel(respond=text_response("ok"))
    triggers: list[str] = []
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=1.0, output_sink=sink
    )
    agent.on_trigger.subscribe(lambda entry: triggers.append(entry.key))
    agent.start()

    world.update("ctx", "context")  # opens the (long) window
    wait_until(
        lambda: agent._window_timer is not None
    )  # ctx is now batched, window open

    t_urgent = time.monotonic()
    world.update("urgent", "stop!")  # bypass → flush now, pulling ctx forward
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert time.monotonic() - t_urgent < 0.5  # far under the 1.0s window → early flush

    assert len(model.calls) == 1
    assert sorted(triggers) == ["ctx", "urgent"]
    observation = agent._history[0]
    assert isinstance(observation, ObservationRecord)
    assert {e.key for e in observation.entries} == {"ctx", "urgent"}

    agent.stop()


def test_bypass_trigger_arriving_while_busy_is_still_dropped(loop, world, sink):
    # bypass_coalescing skips the *wait*, not the single-in-flight *drop*: an urgent trigger landing
    # while a step is in flight is dropped like any other (barge-in/interruption is deferred).
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register(
        "urgent",
        str,
        serialize_fn=identity_serialize,
        triggers_llm_call=True,
        bypass_coalescing=True,
    )
    hold = asyncio.Event()
    started = threading.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        started.set()
        await hold.wait()
        return AIMessage(content="done")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0, output_sink=sink
    )
    agent.start()

    world.update("input", "go")
    assert started.wait(timeout=WAIT_TIMEOUT)  # step in flight → busy

    world.update("urgent", "stop!")  # bypass, but busy → dropped
    time.sleep(0.2)
    loop.call_soon_threadsafe(hold.set)
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert len(model.calls) == 1  # urgent did not start a second step
    assert len([r for r in agent._history if isinstance(r, ObservationRecord)]) == 1

    agent.stop()


# --- Command object, output Command, noop, and system-prompt composition ---------------------


def text_and_tool_response(
    text: str, calls: list[tuple[str, dict[str, Any], str]]
) -> Callable[[list[BaseMessage]], Awaitable[AIMessage]]:
    async def respond(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(
            content=text,
            tool_calls=[
                {"name": name, "args": args, "id": call_id}
                for name, args, call_id in calls
            ],
        )

    return respond


def test_register_command_accepts_a_command_object(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("plus", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def add(a: int, b: int) -> int:
        return a + b

    # A Command lets us override the name without register_command kwargs.
    agent.register_command(Command(add, name="plus", description="Add two numbers."))
    agent.start()

    world.update("input", "add them")
    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "complete")
    assert world.get_entry(key).current.value.result == "3"

    agent.stop()


def test_system_prompt_composes_persona_and_runtime_primer_without_output_clause(
    loop, world, sink
):
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(
        model, world=world, loop=loop, system_prompt="You are terse.", output_sink=sink
    )

    prompt = agent.system_prompt
    # Persona is preserved verbatim, at the front.
    assert prompt.startswith("You are terse.")
    # The always-on primer is appended (perception + noop guidance present).
    assert "observations of your World" in prompt
    assert _NOOP_COMMAND_NAME in prompt
    # No output Command → default text-reply clause, not the private-reasoning one.
    assert "write your answer as ordinary text" in prompt
    assert "private reasoning" not in prompt


def test_output_command_is_the_user_channel_free_text_is_private(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoken: list[str] = []
    spoke = threading.Event()

    async def speak(text: str) -> str:
        """Say something to the user."""
        spoken.append(text)
        spoke.set()
        return "spoken"

    # Step 1: think in free text and speak via the output Command. Step 2 (re-triggered by the
    # output command completing) ends the turn with noop.
    model = ProgrammableChatModel(
        respond=sequence(
            text_and_tool_response(
                "thinking about it", [("speak", {"text": "hello"}, "s1")]
            ),
            tool_call_response([(_NOOP_COMMAND_NAME, {}, "n1")]),
        )
    )
    agent = make_agent(
        model, world=world, loop=loop, output_sink=sink, output_command=speak
    )

    # The output clause names the command and marks free text private.
    assert "speak" in agent.system_prompt
    assert "private reasoning" in agent.system_prompt

    agent.start()
    world.update("input", "greet the user")

    assert spoke.wait(timeout=WAIT_TIMEOUT)
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    # The Command carried the user-facing text; the sink got the private free-text reasoning.
    assert spoken == ["hello"]
    assert sink.texts == ["thinking about it"]
    # The output command's completion re-triggered a step (default flag), which the model ended
    # with noop — so the model was called at least twice.
    wait_until(lambda: len(model.calls) >= 2)

    agent.stop()


def test_noop_takes_no_action_and_does_not_retrigger(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=tool_call_response([(_NOOP_COMMAND_NAME, {}, "n1")])
    )
    issued: list[CommandIssued] = []
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.on_command.subscribe(issued.append)
    agent.start()

    world.update("input", "nothing to do here")
    wait_until(lambda: len(model.calls) == 1)
    time.sleep(0.2)  # give any (erroneous) re-trigger a chance to fire

    # No World command entry was created, nothing spoke, no re-trigger. on_command *does* fire for
    # noop (observability — it is a command the model issued, just not a World action).
    assert not any(
        e.key.startswith("agent:command:") for e in world.get_prompt_entries()
    )
    assert sink.texts == []
    assert issued == [CommandIssued("noop", {})]
    assert len(model.calls) == 1
    # History records the declined reaction as a dedicated NoReactionRecord.
    assert any(isinstance(r, NoReactionRecord) for r in agent._history)

    agent.stop()


def test_noop_renders_as_native_call_with_plain_ack(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    # First input → noop; a later input drives a second step whose prompt contains the rendered
    # noop from history (native tool call + plain ack).
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([(_NOOP_COMMAND_NAME, {}, "n1")]),
            text_response("ok"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1)

    world.update("input", "second")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    second_prompt = model.calls[1]
    # The past noop re-renders as the model's own native tool call...
    assert any(
        c["name"] == _NOOP_COMMAND_NAME
        for m in second_prompt
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    )
    # ...paired with a plain acknowledgement tool_result (not an entry-pointer ack).
    assert any(
        m.tool_call_id == "n1" and m.content == _NOOP_ACK
        for m in second_prompt
        if isinstance(m, ToolMessage)
    )

    agent.stop()
