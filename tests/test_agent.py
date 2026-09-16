from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from opentelemetry.trace import format_trace_id
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
from wica.instrumentation import (
    CommandTrace,
    ReactionTrace,
    TokenUsage,
    reaction_latency,
)
from wica.world import World, WorldEntry

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


def _recorder[T](items: list[T], event: threading.Event) -> Callable[[T], None]:
    """An Event subscriber that appends the payload and signals — the common pattern for waiting
    on a single instrumentation Event emission from the test thread."""

    def record(item: T) -> None:
        items.append(item)
        event.set()

    return record


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
    output_sink: Callable[[str], Awaitable[None]] | None = None,
    output_command: Callable[..., Any] | Command | None = None,
    **kwargs: Any,
) -> Agent:
    """Construct an Agent over an injected (fully-scripted) model. The Agent takes an AgentConfig
    and builds its own model from it; the `model=` override lets these tests supply the bespoke
    ProgrammableChatModel the loop is driven over. The prompt is carried on the config. The output
    sink / output Command are not constructor arguments (build, then wire — see specs/agent.md,
    "Output wiring"); this helper folds the wiring step in for brevity."""
    config = AgentConfig(provider="fake", model="test", system_prompt=system_prompt)
    agent = Agent(config, world=world, loop=loop, model=model, **kwargs)
    if output_sink is not None:
        agent.set_output_sink(output_sink)
    if output_command is not None:
        agent.set_output_command(output_command)
    return agent


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
    assert [e.entry.key for e in agent._history[0].entries] == ["input"]
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
    assert {e.entry.key for e in observation.entries} == {"a", "b"}

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

    assert commands == [CommandIssued("add", {"a": 1, "b": 2}, "call1")]

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
    assert {e.entry.key for e in observation.entries} == {"a", "b"}

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
    assert {e.entry.key for e in observation.entries} == {"ctx", "urgent"}

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
    # …including, in the perception + acting part (before the output-mode clause), that a
    # response's tool calls run concurrently, so sequencing dependent actions is the model's job.
    assert "run concurrently" in prompt
    assert "in a later step" in prompt
    assert prompt.index("run concurrently") < prompt.index(
        "write your answer as ordinary text"
    )
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
    assert issued == [CommandIssued("noop", {}, "n1")]
    assert not world.is_registered(
        "agent:command:n1"
    )  # a call_id, but no entry behind it
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


# --- History renders from captured serializers ------------------------------------------


def _step(
    model: ProgrammableChatModel, world: World, key: str, value: str, n_calls: int
) -> None:
    """Trigger a step by updating `key` and wait until the model has been called n_calls times."""
    world.update(key, value)
    wait_until(lambda: len(model.calls) >= n_calls)


def _rendered_user_text(model: ProgrammableChatModel, call_index: int) -> str:
    return "".join(
        human_texts(m) for m in model.calls[call_index] if isinstance(m, HumanMessage)
    )


def test_history_renders_an_unregistered_entry_from_its_captured_serializer(
    loop, world, sink
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, output_sink=sink, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"mood={v}")])
    agent.start()
    world.update("mood", "happy")
    _step(model, world, "speech", "one", 1)
    assert "mood=happy" in _rendered_user_text(model, 0)

    world.unregister("mood")  # would KeyError at render time before D3
    _step(model, world, "speech", "two", 2)
    # observation 1 still renders as it did, from its captured serializer
    assert "mood=happy" in _rendered_user_text(model, 1)
    agent.stop()


def test_reregistering_a_key_does_not_rewrite_earlier_observations(loop, world, sink):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, output_sink=sink, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"OLD:{v}")])
    agent.start()
    world.update("mood", "happy")
    _step(model, world, "speech", "one", 1)
    first_render = _rendered_user_text(model, 0)

    world.unregister("mood")
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"NEW:{v}")])
    world.update("mood", "calm")
    _step(model, world, "speech", "two", 2)
    # The first observation is byte-identical to what step one sent (cache-stable prefix) …
    assert _rendered_user_text(model, 1).startswith(first_render.split("</entry>")[0])
    assert "OLD:happy" in _rendered_user_text(model, 1)
    # … and the new observation uses the new serializer.
    assert "NEW:calm" in _rendered_user_text(model, 1)
    assert "NEW:happy" not in _rendered_user_text(model, 1)
    agent.stop()


def test_application_entry_under_agent_command_prefix_uses_its_own_serializer(
    loop, world, sink
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, output_sink=sink, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register(
        "agent:command:mine", str, serialize_fn=lambda v, p: [TextPart(f"MINE:{v}")]
    )
    agent.start()
    world.update("agent:command:mine", "x")
    _step(model, world, "speech", "one", 1)
    assert "MINE:x" in _rendered_user_text(model, 0)  # not "(no command)" / not a crash
    agent.stop()


# --- WICA-owned call ids ----------------------------------------------------------------


def world_keys_seen(model: ProgrammableChatModel) -> list[str]:
    text = "".join(
        human_texts(m)
        for call in model.calls
        for m in call
        if isinstance(m, HumanMessage)
    )
    return re.findall(r'<entry key="([^"]*)"', text)


def test_invalid_provider_call_id_gets_a_safe_world_key_but_keeps_its_message_id(
    loop, world, sink
):
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    bad_id = 'call "quoted"'
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, bad_id)]),
            text_response("done"),
        )
    )
    agent = make_agent(
        model, world=world, loop=loop, output_sink=sink, coalesce_window=0
    )
    agent.register_command(add)
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    agent.start()
    world.update("speech", "go")
    wait_until(lambda: len(model.calls) >= 2)  # completion re-triggered a second step
    keys = [k for k in world_keys_seen(model) if k.startswith("agent:command:")]
    assert keys and all('"' not in k for k in keys)
    ai = next(m for m in model.calls[1] if isinstance(m, AIMessage))
    assert ai.tool_calls[0]["id"] == bad_id  # the provider's id is what the model sees
    agent.stop()


def test_colliding_provider_call_ids_get_distinct_entries(loop, world, sink):
    release = threading.Event()

    async def slow() -> str:
        """Block until released."""
        await asyncio.get_running_loop().run_in_executor(None, release.wait)
        return "done"

    model = ProgrammableChatModel(respond=tool_call_response([("slow", {}, "same")]))
    agent = make_agent(
        model, world=world, loop=loop, output_sink=sink, coalesce_window=0
    )
    agent.register_command(slow)
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    agent.start()
    world.update("speech", "one")
    wait_until(lambda: world.is_registered("agent:command:same"))
    world.update("speech", "two")  # second step, same provider id, first still running
    wait_until(lambda: len(model.calls) >= 2)
    wait_until(lambda: len(agent._command_keys) == 2)
    assert "agent:command:same" in agent._command_keys
    release.set()
    agent.stop()


# --- Reserved names and duplicate rejection ---------------------------------------------


def _dummy_agent(loop, world, sink, **kwargs) -> Agent:
    return make_agent(
        ProgrammableChatModel(respond=text_response("ok")),
        world=world,
        loop=loop,
        output_sink=sink,
        **kwargs,
    )


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
def test_set_output_command_rejects_reserved_names(loop, world, sink, name):
    agent = _dummy_agent(loop, world, sink)
    before = agent.system_prompt
    with pytest.raises(ValueError, match="reserved"):
        agent.set_output_command(_named(name))
    assert agent.output_command_name is None
    assert agent.system_prompt == before


def test_register_command_rejects_the_output_commands_name(loop, world, sink):
    agent = _dummy_agent(loop, world, sink, output_command=_named("say"))
    with pytest.raises(ValueError, match="output Command"):
        agent.register_command(_named("say"))


def test_restart_keeps_exactly_one_of_each_builtin(loop, world, sink):
    agent = _dummy_agent(loop, world, sink, output_command=_named("say"))
    agent.register_command(_named("wave"))
    agent.start()
    agent.stop()
    agent.start()
    assert sorted(agent._commands) == ["cancel_command", "noop", "say", "wave"]
    agent.stop()


# --- Output wiring: set_output_command / set_output_sink (specs/agent.md "Output wiring") -------


def test_set_output_command_rejects_an_already_registered_name_unchanged(
    loop, world, sink
):
    agent = _dummy_agent(loop, world, sink)
    agent.register_command(_named("wave"))
    before = agent.system_prompt
    with pytest.raises(ValueError, match="already registered"):
        agent.set_output_command(_named("wave"))
    assert agent.output_command_name is None
    assert agent.system_prompt == before
    assert list(agent.commands) == ["wave"]


def test_set_output_command_after_start_attaches_immediately_and_dispatches(
    loop, world, sink
):
    """Setting while running is permitted: the Command is bound at once, the next step's system
    prompt carries the output clause, and a call to it dispatches like any Command."""
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

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("speak", {"text": "hello"}, "s1")]),
            tool_call_response([(_NOOP_COMMAND_NAME, {}, "n1")]),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    assert "speak" not in agent.commands
    assert "ordinary text" in agent.system_prompt

    agent.set_output_command(speak)
    assert "speak" in agent.commands
    assert agent.output_command_name == "speak"

    world.update("input", "greet the user")
    assert spoke.wait(timeout=WAIT_TIMEOUT)
    assert spoken == ["hello"]
    # The step's prompt (rendered after the set) names the output Command.
    system = model.calls[0][0]
    assert "speak" in system.text and "private reasoning" in system.text
    agent.stop()


def test_replacing_the_output_command_detaches_the_previous_one(loop, world, sink):
    agent = _dummy_agent(loop, world, sink, output_command=_named("say"))
    agent.start()
    assert "say" in agent.commands

    agent.set_output_command(_named("speak"))
    assert "say" not in agent.commands
    assert "speak" in agent.commands
    assert agent.output_command_name == "speak"
    assert "speak" in agent.system_prompt and "calling say" not in agent.system_prompt
    # The old name is free again; the new one is reserved.
    agent.register_command(_named("say"))
    with pytest.raises(ValueError, match="output Command"):
        agent.register_command(_named("speak"))
    agent.stop()


def test_setting_the_same_name_again_swaps_the_command_object(loop, world, sink):
    first = _named("say")
    second = _named("say")
    agent = _dummy_agent(loop, world, sink, output_command=first)
    agent.start()
    assert agent.commands["say"] is first
    agent.set_output_command(second)
    assert agent.commands["say"] is second
    assert list(agent.commands).count("say") == 1
    agent.stop()


def test_clearing_the_output_command_restores_free_text_mode(loop, world, sink):
    agent = _dummy_agent(loop, world, sink, output_command=_named("say"))
    agent.start()
    assert "private reasoning" in agent.system_prompt

    agent.set_output_command(None)
    assert agent.output_command_name is None
    assert "say" not in agent.commands
    assert "ordinary text" in agent.system_prompt
    assert "private reasoning" not in agent.system_prompt
    agent.register_command(_named("say"))  # no longer reserved
    agent.stop()


def test_set_output_sink_after_start_takes_effect_next_step_and_none_silences(
    loop, world
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(model, world=world, loop=loop, coalesce_window=0)
    agent.start()

    late = RecordingSink()
    agent.set_output_sink(late)
    world.update("input", "one")
    assert late.event.wait(timeout=WAIT_TIMEOUT)
    assert late.texts == ["hi"]

    agent.set_output_sink(None)
    world.update("input", "two")
    wait_until(lambda: len(model.calls) == 2)
    assert late.texts == ["hi"]  # the second step's text went to the no-op sink
    agent.stop()


def test_command_finishing_after_the_world_stopped_does_not_raise(loop, world, sink):
    # A Command can complete after the World paused (injected-loop shutdown, or a Command that
    # swallows its cancellation). Its terminal write then has nowhere to go: it must be dropped and
    # logged, not raised into the task — the cancel branch already tolerated this; the complete and
    # failed branches must too.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=tool_call_response([("finish_late", {}, "late1")])
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    release = asyncio.Event()

    async def finish_late() -> str:
        """Completes only once released — by then the World is stopped."""
        await release.wait()
        return "done"

    agent.register_command(finish_late)
    agent.start()
    world.update("input", "go")
    key = "agent:command:late1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")
    task = agent._running_tasks[key]

    world.stop()
    loop.call_soon_threadsafe(release.set)
    wait_until(task.done)

    assert task.exception() is None, (
        "terminal write into a stopped World escaped the task"
    )
    # The entry keeps its last recorded state: the completion could not be written.
    assert world.get_entry(key).current.value.state == "running"
    agent.stop()


def test_unknown_command_name_fails_with_a_clear_error(loop, world, sink):
    # A model naming a tool the Agent never bound fails the entry with a readable error (not the
    # bare KeyError repr "'ghost'"), delivered to the next step like any Command failure.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("ghost", {}, "g1")]),
            text_response("noted"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    world.update("input", "summon")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    observation = "".join(
        str(m.content) for m in model.calls[1] if isinstance(m, HumanMessage)
    )
    assert "Called ghost() → failed: unknown command 'ghost'" in observation
    agent.stop()


# --- Step failure handling: model-call and output-sink errors, well-formed history ---------


async def _explode(messages: list[BaseMessage]) -> AIMessage:
    raise ConnectionError("provider down")


def _no_consecutive_human_messages(messages: list[BaseMessage]) -> bool:
    return not any(
        isinstance(a, HumanMessage) and isinstance(b, HumanMessage)
        for a, b in zip(messages, messages[1:])
    )


def test_model_call_failure_is_logged_and_the_next_step_runs(loop, world, sink, caplog):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(_explode, text_response("recovered"))
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "first")
        wait_until(lambda: len(model.calls) == 1 and not agent._busy)
        world.update("input", "second")
        assert sink.event.wait(timeout=WAIT_TIMEOUT)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("model call failed" in r.message for r in errors)
    assert any(
        r.exc_info and isinstance(r.exc_info[1], ConnectionError) for r in errors
    )
    assert sink.texts == ["recovered"]

    agent.stop()


def test_output_sink_failure_is_logged_and_commands_still_dispatch(loop, world, caplog):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def bad_sink(text: str) -> None:
        raise RuntimeError("sink broke")

    # Step 1 speaks (into the raising sink) and issues wave; wave's completion re-triggers step 2,
    # which ends the chain with noop.
    model = ProgrammableChatModel(
        respond=sequence(
            text_and_tool_response("doing it", [("wave", {}, "w1")]),
            tool_call_response([(_NOOP_COMMAND_NAME, {}, "n1")]),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=bad_sink)
    waved = threading.Event()

    async def wave() -> str:
        """Wave."""
        waved.set()
        return "waved"

    agent.register_command(wave)
    agent.start()

    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "hello")
        assert waved.wait(timeout=WAIT_TIMEOUT)
        wait_until(
            lambda: len(model.calls) >= 2
        )  # wave's completion re-triggers a step

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("output sink raised" in r.message for r in errors)
    assert any(r.exc_info and isinstance(r.exc_info[1], RuntimeError) for r in errors)
    # The utterance is still in history: the next prompt carries it as assistant text.
    assert any(
        isinstance(m, AIMessage) and "doing it" in str(m.content)
        for m in model.calls[1]
    )

    agent.stop()


def test_failed_step_observation_merges_into_the_next_prompt(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=sequence(_explode, text_response("ok")))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "second")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    second = model.calls[1]
    assert _no_consecutive_human_messages(second)
    humans = [m for m in second if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    text = human_texts(humans[0])
    assert "first" in text and "second" in text  # both observations, one message
    assert text.index("first") < text.index("second")  # older observation first

    agent.stop()


def test_merged_observations_keep_their_own_freshness(loop, world, sink):
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
    model = ProgrammableChatModel(respond=sequence(_explode, text_response("ok")))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    world.update("note", "one")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("note", "two")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    humans = [m for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    text = human_texts(humans[0])
    assert "ARCHIVAL:one" in text and "FRESH:two" in text

    agent.stop()


def test_empty_model_response_does_not_split_the_next_prompt(loop, world, sink):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response(""), text_response("ok"))
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "second")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    assert _no_consecutive_human_messages(model.calls[1])
    assert len([m for m in model.calls[1] if isinstance(m, HumanMessage)]) == 1

    agent.stop()


def test_commands_mapping_lists_the_bound_set_read_only(loop, world, sink):
    async def speak(text: str) -> str:
        """Speak."""
        return "ok"

    async def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    agent = make_agent(
        ProgrammableChatModel(respond=text_response("x")),
        world=world,
        loop=loop,
        output_sink=sink,
        output_command=speak,
    )
    agent.register_command(add)
    assert agent.output_command_name == "speak"
    assert list(agent.commands) == ["add"]  # builtins attach at start()
    agent.start()
    assert set(agent.commands) == {"add", "speak", "noop", "cancel_command"}
    assert agent.commands["add"].tool.description == "Add."
    with pytest.raises(TypeError):
        agent.commands["ghost"] = agent.commands["add"]  # type: ignore[index]
    agent.stop()


def test_on_command_fires_once_the_execution_entry_exists_and_carries_its_call_id(
    loop, world, sink
):
    """A subscriber can follow a Command it did not know the id of: it attaches a listener on
    agent:command:<call_id> from inside the on_command handler and sees running -> complete."""
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    registered_at_emit: list[bool] = []
    states: list[str] = []
    terminal = threading.Event()

    def follow(command: CommandIssued) -> None:
        key = f"agent:command:{command.call_id}"
        registered_at_emit.append(world.is_registered(key))

        async def on_update(
            entry,
        ) -> None:  # async: delivered in version order on the loop
            states.append(entry.current.value.state)
            if entry.current.value.is_terminal():
                terminal.set()

        world.add_listener(key, on_update)

    agent.on_command.subscribe(follow)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        await asyncio.sleep(0.05)
        return a + b

    agent.register_command(add)
    agent.start()
    world.update("input", "add them")
    assert terminal.wait(timeout=WAIT_TIMEOUT)
    assert registered_at_emit == [True]
    assert states == ["running", "complete"]
    agent.stop()


def test_command_with_triggers_on_completion_false_does_not_retrigger(
    loop, world, sink
):
    """The completion wakes no step, but its outcome is not lost: the next input-driven step
    observes the terminal entry (rendered into its prompt) and retires it."""
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("ping", {}, "p1")]),
            text_response("second step"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    terminal = threading.Event()

    async def on_update(entry) -> None:
        if entry.current.value.is_terminal():
            terminal.set()

    agent.on_command.subscribe(
        lambda c: world.add_listener(f"agent:command:{c.call_id}", on_update)
    )

    async def ping() -> str:
        """Ping."""
        return "pong"

    agent.register_command(Command(ping, triggers_on_completion=False))
    agent.start()
    world.update("input", "ping it")
    assert terminal.wait(timeout=WAIT_TIMEOUT)
    time.sleep(0.2)  # give any (erroneous) re-trigger a chance to fire
    assert len(model.calls) == 1

    world.update("input", "anything else?")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert len(model.calls) == 2
    humans = [m for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert any("Called ping() → pong" in str(m.content) for m in humans)
    assert not any(
        e.key.startswith("agent:command:") for e in world.get_prompt_entries()
    )
    agent.stop()


@pytest.mark.parametrize("explicit_false", [False, True])
def test_output_command_retrigger_follows_its_own_flag_or_the_default(
    loop, world, sink, explicit_false
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoke = threading.Event()

    async def speak(text: str) -> str:
        """Say something to the user."""
        spoke.set()
        return "spoken"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("speak", {"text": "hello"}, "s1")]),
            tool_call_response([(_NOOP_COMMAND_NAME, {}, "n1")]),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    if explicit_false:
        agent.set_output_command(Command(speak, triggers_on_completion=False))
    else:
        agent.set_output_command(speak)  # bare callable: the module default (True)
    agent.start()
    world.update("input", "greet")
    assert spoke.wait(timeout=WAIT_TIMEOUT)
    if explicit_false:
        time.sleep(0.2)
        assert len(model.calls) == 1  # speak -> idle
    else:
        wait_until(lambda: len(model.calls) == 2)  # speak re-triggered, ended by noop
    agent.stop()


def test_on_text_fires_with_the_step_text_before_the_sink(loop, world):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    seen: list[str] = []
    delivered = threading.Event()

    async def ordered_sink(text: str) -> None:
        seen.append(f"sink:{text}")
        delivered.set()

    model = ProgrammableChatModel(
        respond=sequence(text_response("hello"), text_response(""))
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=ordered_sink)
    agent.on_text.subscribe(lambda text: seen.append(f"event:{text}"))
    agent.start()
    world.update("input", "hi")
    assert delivered.wait(timeout=WAIT_TIMEOUT)
    assert seen == ["event:hello", "sink:hello"]
    # An empty response emits nothing.
    world.update("input", "again")
    wait_until(lambda: len(model.calls) == 2 and not agent._busy)
    assert seen == ["event:hello", "sink:hello"]
    agent.stop()


def test_on_text_fires_even_when_the_sink_raises(loop, world, caplog):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    texts: list[str] = []
    got = threading.Event()

    async def bad_sink(text: str) -> None:
        raise RuntimeError("sink broke")

    agent = make_agent(
        ProgrammableChatModel(respond=text_response("still observed")),
        world=world,
        loop=loop,
        output_sink=bad_sink,
    )

    def record(text: str) -> None:
        texts.append(text)
        got.set()

    agent.on_text.subscribe(record)
    agent.start()
    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "hi")
        assert got.wait(timeout=WAIT_TIMEOUT)
        wait_until(
            lambda: any("output sink raised" in r.message for r in caplog.records)
        )
    assert texts == ["still observed"]
    agent.stop()


# --- Instrumentation: ReactionTrace, on_trigger_dropped, on_command_ended, spans ---------------


def test_reaction_ended_carries_the_phase_stamps_and_usage(loop, world, sink):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert len(traces) == 1
    trace = traces[0]
    assert trace.reaction_id == 1
    assert trace.outcome == "ok"
    assert trace.text_length == 2
    assert trace.usage == TokenUsage(3, 2, None)
    assert trace.noop is False
    assert trace.command_call_ids == ()
    assert len(trace.triggers) == 1
    assert trace.triggers[0].key == "prompt"
    assert trace.triggers[0].is_command_completion is False
    t = trace.triggers[0]
    assert trace.prompt_ready_at is not None
    assert trace.model_started_at is not None
    assert trace.model_ended_at is not None
    assert (
        t.written_at
        <= t.arrived_at
        <= trace.window_opened_at
        <= trace.window_closed_at
        <= trace.prompt_ready_at
        <= trace.model_started_at
        <= trace.model_ended_at
        <= trace.ended_at
    )
    assert trace.sink_duration is not None and trace.sink_duration >= 0
    assert trace.trace_id is not None
    assert re.fullmatch(r"[0-9a-f]{32}", trace.trace_id)

    agent.stop()


def test_reaction_ended_reports_an_empty_response(loop, world, sink):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response(""))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert traces[0].outcome == "empty"
    assert traces[0].sink_duration is None

    agent.stop()


def test_reaction_ended_reports_a_model_error(loop, world, sink):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("boom")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[0].outcome == "model_error"
    assert traces[0].error == "boom"
    assert traces[0].model_ended_at is not None

    done.clear()
    world.update("prompt", "y")  # the agent is free again — a second trace follows
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert len(traces) == 2

    agent.stop()


def test_reaction_ended_reports_cancellation_on_stop(loop, world, sink):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    prompt_fired = threading.Event()
    block = asyncio.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        await block.wait()
        return AIMessage(content="never")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.on_prompt.subscribe(lambda m: prompt_fired.set())
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert prompt_fired.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[0].outcome == "cancelled"
    assert traces[0].model_ended_at is None


def test_dropped_trigger_fires_on_trigger_dropped(loop, world, sink):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()
    started = threading.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        started.set()
        await block.wait()
        return AIMessage(content="done")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    dropped: list[WorldEntry] = []
    got_drop = threading.Event()
    agent.on_trigger_dropped.subscribe(_recorder(dropped, got_drop))
    agent.start()

    world.update("prompt", "first")
    assert started.wait(timeout=WAIT_TIMEOUT)

    world.update("prompt", "second")
    assert got_drop.wait(timeout=WAIT_TIMEOUT)
    assert dropped[0].current.value == "second"

    loop.call_soon_threadsafe(block.set)
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    agent.stop()


def test_coalesced_reaction_lists_every_trigger(loop, world, sink):
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0.2, output_sink=sink
    )
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("a", "1")
    world.update("b", "2")
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert len(traces[0].triggers) == 2
    assert traces[0].coalescing_wait >= 0.15

    agent.stop()


def test_command_ended_carries_reaction_id_and_duration(loop, world, sink):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    command_traces: list[CommandTrace] = []
    got_command = threading.Event()
    agent.on_command_ended.subscribe(_recorder(command_traces, got_command))
    reaction_traces: list[ReactionTrace] = []
    done = threading.Event()

    def on_reaction(t: ReactionTrace) -> None:
        reaction_traces.append(t)
        if len(reaction_traces) == 2:
            done.set()

    agent.on_reaction_ended.subscribe(on_reaction)
    agent.start()

    world.update("prompt", "add them")
    assert got_command.wait(timeout=WAIT_TIMEOUT)
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert len(command_traces) == 1
    ct = command_traces[0]
    assert ct.name == "add"
    assert ct.reaction_id == 1
    assert ct.state == "complete"
    assert ct.duration >= 0

    assert len(reaction_traces) == 2
    follow_up = reaction_traces[1]
    assert follow_up.triggers[0].is_command_completion is True
    assert reaction_latency(follow_up) is None

    agent.stop()


def test_span_tree_follows_the_reaction(loop, world, sink, spans):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    reaction_traces: list[ReactionTrace] = []
    done = threading.Event()

    def on_reaction(t: ReactionTrace) -> None:
        reaction_traces.append(t)
        if len(reaction_traces) == 2:
            done.set()

    agent.on_reaction_ended.subscribe(on_reaction)
    agent.start()

    world.update("prompt", "add them")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    finished = spans.get_finished_spans()
    by_name: dict[str, list[Any]] = {}
    for s in finished:
        by_name.setdefault(s.name, []).append(s)

    update_spans = sorted(by_name["wica.world.update"], key=lambda s: s.start_time)
    reaction_spans = sorted(by_name["wica.agent.reaction"], key=lambda s: s.start_time)
    model_spans = by_name["wica.agent.model"]
    (command_span,) = by_name["wica.agent.command"]

    assert len(update_spans) == 2
    assert len(reaction_spans) == 2
    assert len(model_spans) == 2

    first_reaction, second_reaction = reaction_spans
    first_update, second_update = update_spans

    assert first_reaction.parent is not None
    assert first_reaction.parent.span_id == first_update.context.span_id

    model_by_parent = {s.parent.span_id: s for s in model_spans if s.parent is not None}
    assert first_reaction.context.span_id in model_by_parent
    assert second_reaction.context.span_id in model_by_parent

    assert command_span.parent is not None
    assert command_span.parent.span_id == first_reaction.context.span_id

    assert second_reaction.parent is not None
    assert second_reaction.parent.span_id == second_update.context.span_id
    assert second_update.parent is not None
    assert second_update.parent.span_id == command_span.context.span_id

    assert reaction_traces[0].trace_id == format_trace_id(
        first_reaction.context.trace_id
    )


def test_spans_opened_inside_a_command_nest_under_the_command_span(
    loop, world, sink, spans
):
    from opentelemetry import trace as otel_trace_module

    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        with otel_trace_module.get_tracer("app").start_as_current_span(
            "tts.synthesize"
        ):
            pass
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("prompt", "add them")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    finished = spans.get_finished_spans()
    (tts_span,) = [s for s in finished if s.name == "tts.synthesize"]
    (command_span,) = [s for s in finished if s.name == "wica.agent.command"]
    assert tts_span.parent is not None
    assert tts_span.parent.span_id == command_span.context.span_id


def test_coalesced_reaction_links_the_other_triggers(loop, world, sink, spans):
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0.2, output_sink=sink
    )
    agent.start()

    world.update("a", "1")
    world.update("b", "2")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    finished = spans.get_finished_spans()
    update_spans = sorted(
        (s for s in finished if s.name == "wica.world.update"),
        key=lambda s: s.start_time,
    )
    (reaction_span,) = [s for s in finished if s.name == "wica.agent.reaction"]
    first_update, second_update = update_spans

    assert len(reaction_span.links) == 1
    assert reaction_span.links[0].context.span_id == second_update.context.span_id
    assert reaction_span.parent is not None
    assert reaction_span.parent.span_id == first_update.context.span_id
