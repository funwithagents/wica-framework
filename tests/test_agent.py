from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from wica.agent import Agent, AssistantTextRecord, ObservationRecord
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


class FakeChatModel(BaseChatModel):
    """A hand-written BaseChatModel with fully scripted async responses, for
    deterministic control over .tool_calls that langchain_core's built-in fakes don't
    give us."""

    respond: Callable[[list[BaseMessage]], Awaitable[AIMessage]] | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError("FakeChatModel is async-only")

    async def _agenerate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(messages)
        assert self.respond is not None, "FakeChatModel.respond must be set before use"
        message = await self.respond(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs: Any) -> Runnable[Any, AIMessage]:
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
            tool_calls=[{"name": name, "args": args, "id": call_id} for name, args, call_id in calls],
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
def loop():
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=WAIT_TIMEOUT)
    new_loop.close()


@pytest.fixture
def world():
    return World()


@pytest.fixture
def sink():
    return RecordingSink()


def human_texts(message: BaseMessage) -> str:
    return message.text


def test_text_only_response_updates_sink_and_history(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = FakeChatModel(respond=text_response("Hello there"))
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
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
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = FakeChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("The sum is 3"),
        )
    )
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)

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
    assert any(
        "Called add(a=1, b=2) → 3" in human_texts(m)
        for m in second_call_messages
        if isinstance(m, HumanMessage)
    )

    with pytest.raises(KeyError):
        world.get_entry(key)

    agent.stop()


def test_tool_failure_surfaces_into_world_and_next_step(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = FakeChatModel(
        respond=sequence(
            tool_call_response([("explode", {}, "call1")]),
            text_response("Sorry, that failed"),
        )
    )
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)

    async def explode() -> str:
        """Always raises."""
        await asyncio.sleep(0.05)  # slow enough that "running" is observable below
        raise ValueError("boom")

    agent.register_command(explode)
    agent.start()

    world.update("input", "explode please")

    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    second_call_messages = model.calls[1]
    assert any(
        "Called explode() → failed: boom" in human_texts(m)
        for m in second_call_messages
        if isinstance(m, HumanMessage)
    )

    agent.stop()


def test_parallel_tool_calls_independent_keys_and_mixed_status_line(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    slow_release = asyncio.Event()

    async def fast_tool() -> str:
        """Resolves immediately."""
        return "fast-result"

    async def slow_tool() -> str:
        """Blocks until released."""
        await slow_release.wait()
        return "slow-result"

    model = FakeChatModel(
        respond=sequence(
            tool_call_response([("fast_tool", {}, "fast"), ("slow_tool", {}, "slow")]),
            text_response("done"),
        )
    )
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
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
    flattened = " ".join(human_texts(m) for m in second_call_messages if isinstance(m, HumanMessage))
    assert "Called fast_tool() → fast-result" in flattened
    assert "Calling slow_tool()…" in flattened

    with pytest.raises(KeyError):
        world.get_entry(fast_key)
    assert world.get_entry(slow_key).current.value.state == "running"

    agent.stop()


def test_single_in_flight_trigger_dropped_and_logged(loop, world, sink, caplog):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    hold = asyncio.Event()
    started = threading.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        started.set()
        await hold.wait()
        return AIMessage(content="finally")

    model = FakeChatModel(respond=respond)
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
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
    model = FakeChatModel(respond=text_response("ok"))
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
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
    model = FakeChatModel(respond=sequence(text_response("first"), text_response("second")))
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
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


def test_stop_clears_trigger_handler(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = FakeChatModel(respond=text_response("hi"))
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
    agent.start()
    agent.stop()

    world.update("input", "should not trigger")
    time.sleep(0.2)
    assert model.calls == []


def test_cancel_command_marks_cancelled_and_retriggers(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    model = FakeChatModel(
        respond=sequence(
            tool_call_response([("block_forever", {}, "call1")]),
            text_response("cancelled that for you"),
        )
    )
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "block please")

    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    agent.cancel_command("call1")
    wait_until(lambda: sink.event.is_set())

    second_call_messages = model.calls[1]
    assert any(
        "Called block_forever() → cancelled" in human_texts(m)
        for m in second_call_messages
        if isinstance(m, HumanMessage)
    )

    agent.cancel_command("does-not-exist")  # no-op, must not raise

    agent.stop()


def test_stop_cancels_running_tool_without_triggering_new_step(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    block = asyncio.Event()

    async def block_forever() -> str:
        """Blocks until cancelled."""
        await block.wait()
        return "unreachable"

    model = FakeChatModel(respond=tool_call_response([("block_forever", {}, "call1")]))
    agent = Agent(model, system_prompt="You are terse.", world=world, loop=loop, output_sink=sink)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "block please")

    key = "agent:command:call1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    agent.stop()

    wait_until(lambda: world.get_entry(key).current.value.state == "cancelled")
    assert len(model.calls) == 1  # no second call: the trigger handler was already cleared
