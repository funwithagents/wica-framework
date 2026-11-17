from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from typing import Any, Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool as langchain_tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from opentelemetry.trace import format_trace_id
from pydantic import Field

from wica.agent import (
    Agent,
    AssistantTextRecord,
    CommandIssued,
    ObservationRecord,
    _command_ack,
    _EMPTY_OBSERVATION,
    _EMPTY_OBSERVATION_BLOCK,
    _assemble,
    _was_truncated,
    mark_cache_breakpoint,
)
from wica.command import Command
from wica.config import AgentConfig, ConfigError
from wica.content import Content, TextPart
from wica.text_output import OutputStream
from wica.guards import (
    REJECTED_TOOL_RESULT,
    ModelResponse,
    ResponseRejection,
    Retry,
)
from wica.instrumentation import (
    CommandTrace,
    ReactionTrace,
    TokenUsage,
    now,
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
    # One per call, in order: the tool names and tool_choice the call was bound with.
    bindings: list[dict[str, Any]] = Field(default_factory=list)

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
        self.bindings.append(
            {"tools": kwargs.get("tools"), "tool_choice": kwargs.get("tool_choice")}
        )
        assert self.respond is not None, (
            "ProgrammableChatModel.respond must be set before use"
        )
        message = await self.respond(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self, tools, *, tool_choice=None, **kwargs: Any
    ) -> Runnable[Any, AIMessage]:
        # Bound like a real integration: the tool names and choice reach every call as kwargs,
        # which _agenerate records in `bindings`.
        return self.bind(
            tools=[convert_to_openai_tool(tool)["function"]["name"] for tool in tools],
            tool_choice=tool_choice,
        )


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


class RecordingText:
    """Records the model's private text — what it writes outside an output tag, which is all of
    its text when no output Command is set — as an `on_private_text` subscriber."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.event = threading.Event()

    def __call__(self, text: str) -> None:
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
def private() -> RecordingText:
    return RecordingText()


def make_agent(
    model: BaseChatModel,
    *,
    world: World,
    loop: asyncio.AbstractEventLoop,
    system_prompt: str = "You are terse.",
    private_text: Callable[[str], None] | None = None,
    output_command: Callable[..., Any] | Command | None = None,
    history_reactions: int | None = None,
    history_budget: float | None = None,
    context_window: int | None = None,
    model_timeout: float | None = 10.0,
    max_retries: int = 1,
    max_next_steps: int | None = 1,
    **kwargs: Any,
) -> Agent:
    """Construct an Agent over an injected (fully-scripted) model. The Agent takes an AgentConfig
    and builds its own model from it; the `model=` override lets these tests supply the bespoke
    ProgrammableChatModel the loop is driven over. The prompt is carried on the config. The output
    Command is not a constructor argument (build, then wire — see specs/agent.md, "Output
    wiring"); this helper folds the wiring step in for brevity, and subscribes `private_text` to
    `on_private_text`."""
    config = AgentConfig(
        provider="fake",
        model="test",
        system_prompt=system_prompt,
        history_reactions=history_reactions,
        history_budget=history_budget,
        context_window=context_window,
        model_timeout=model_timeout,
        max_retries=max_retries,
        max_next_steps=max_next_steps,
    )
    agent = Agent(config, world=world, loop=loop, model=model, **kwargs)
    if private_text is not None:
        agent.on_private_text.subscribe(private_text)
    if output_command is not None:
        agent.set_output_command(output_command)
    return agent


def human_texts(message: BaseMessage) -> str:
    return message.text


def test_text_only_response_is_private_text_kept_in_history(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("Hello there"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    world.update("input", "hi")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert private.texts == ["Hello there"]
    assert len(agent._history) == 2
    assert isinstance(agent._history[0], ObservationRecord)
    assert [e.entry.key for e in agent._history[0].entries] == ["input"]
    assert agent._history[1] == AssistantTextRecord("Hello there")

    agent.stop()


def test_tool_call_dispatches_then_completes_and_retriggers(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("The sum is 3"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        await asyncio.sleep(0.05)  # slow enough that "running" is observable below
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("input", "add 1 and 2")

    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")
    entry = world.get_entry(key)
    assert entry.current.value.name == "add"
    assert entry.current.value.args == {"a": 1, "b": 2}

    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert private.texts == ["The sum is 3"]
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
        m.tool_call_id == "call1" and m.content == _command_ack("c1")
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


def test_past_commands_render_as_native_tool_calls_not_prose(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("input", "add them")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    messages = agent._render_messages()
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    assert any(c["name"] == "add" for m in ai_messages for c in m.tool_calls)
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("c1")
        for m in tool_messages
    )
    # no *assistant* message renders the call as prose (the outcome lives in the observation, a
    # user-role HumanMessage, so there is nothing for the model to imitate as its own output)
    assert not any(
        "Calling add" in str(m.content) or "Called add" in str(m.content)
        for m in ai_messages
    )

    agent.stop()


def test_tool_failure_surfaces_into_world_and_next_step(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("explode", {}, "call1")]),
            text_response("Sorry, that failed"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)

    async def explode() -> str:
        """Always raises."""
        await asyncio.sleep(0.05)  # slow enough that "running" is observable below
        raise ValueError("boom")

    agent.register_command(explode)
    agent.start()

    world.update("input", "explode please")

    assert private.event.wait(timeout=WAIT_TIMEOUT)

    second_call_messages = model.calls[1]
    # the tool_result is the fixed ack; the failure is delivered by the command's World entry
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("c1")
        for m in tool_messages
    )
    observation = "".join(
        str(m.content) for m in second_call_messages if isinstance(m, HumanMessage)
    )
    assert "Called explode() → failed: boom" in observation

    agent.stop()


def test_running_command_shown_as_in_progress_to_a_concurrent_step(
    loop, world, private
):
    # Regression (analysis/_todo.md, "dance then say hello during dance"): while a command is still
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(dance)
    agent.start()

    world.update("input", "dance for me")
    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")
    wait_until(
        lambda: agent._busy is False
    )  # step 1 finished; dance still running in background

    # A new input arrives while dance is still running -> a concurrent step runs.
    world.update("input", "say hi")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert private.texts == ["hi"]
    assert len(model.calls) == 2

    step2 = model.calls[1]
    # the running command is shown as an in-progress observation entry (the model knows it's not
    # done and could reason about / cancel it), rather than looking completed
    observation = "".join(str(m.content) for m in step2 if isinstance(m, HumanMessage))
    assert "Calling dance()… (still running — not finished)" in observation
    # its tool_result is the fixed ack, never a completed result
    tool_messages = [m for m in step2 if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("c1")
        for m in tool_messages
    )
    assert world.get_entry(key).current.value.state == "running"

    block.set()  # let dance finish before teardown
    agent.stop()


def test_parallel_tool_calls_independent_keys_and_mixed_status_line(
    loop, world, private
):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(fast_tool)
    agent.register_command(slow_tool)
    agent.start()

    world.update("input", "run both")

    fast_key = "agent:command:c1"
    slow_key = "agent:command:c2"
    wait_until(lambda: world.get_entry(slow_key).current.value.state == "running")

    assert private.event.wait(timeout=WAIT_TIMEOUT)
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
    assert tool_contents.get("fast") == _command_ack("c1")
    assert tool_contents.get("slow") == _command_ack("c2")
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


def test_dropped_command_completion_persists_until_observed(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(fast)
    agent.register_command(slow)
    agent.start()

    world.update("input", "run both")

    # fast completes -> step 2 starts and blocks, so the loop is busy.
    assert step2_running.wait(timeout=WAIT_TIMEOUT)

    # Let slow finish: its completion trigger arrives while busy and is dropped (no step runs).
    loop.call_soon_threadsafe(release_slow.set)
    wait_until(
        lambda: world.get_entry("agent:command:c2").current.value.state == "complete"
    )

    # Finish step 2. fast was observed and retired by it; slow's completion was dropped, so no
    # third step ran for it — and it must persist as current terminal state, not be lost.
    loop.call_soon_threadsafe(release_step2.set)
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert private.texts == ["done"]
    assert len(model.calls) == 2
    assert _is_unregistered(world, "agent:command:c1")
    assert world.get_entry("agent:command:c2").current.value.state == "complete"

    # A fresh input starts step 3, which observes slow: its outcome renders into that step's
    # prompt, and only then is the entry retired — the completion reaches history before it goes.
    private.event.clear()
    world.update("input", "poke")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert len(model.calls) == 3
    step3_observation = "".join(
        str(m.content) for m in model.calls[2] if isinstance(m, HumanMessage)
    )
    assert "Called slow() → slow-result" in step3_observation
    wait_until(lambda: _is_unregistered(world, "agent:command:c2"))

    agent.stop()


def test_single_in_flight_trigger_dropped_and_logged(loop, world, private, caplog):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    with caplog.at_level(logging.INFO, logger="wica.agent"):
        world.update("input", "first")
        assert started.wait(timeout=WAIT_TIMEOUT)

        world.update("input", "second")
        world.update("input", "third")
        time.sleep(0.2)  # let any (incorrect) extra steps have a chance to start

        loop.call_soon_threadsafe(hold.set)
        assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert private.texts == ["finally"]
    assert len(model.calls) == 1
    assert sum(1 for r in caplog.records if "dropping trigger" in r.message) == 2
    assert len(agent._history) == 2  # one ObservationRecord + one AssistantTextRecord

    agent.stop()


def test_full_bundle_capture_includes_passive_entries(loop, world, private):
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=False)
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    world.update("b", "b-value")
    world.update("a", "a-value")

    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert len(agent._history) == 2
    observation = agent._history[0]
    assert isinstance(observation, ObservationRecord)
    assert {e.entry.key for e in observation.entries} == {"a", "b"}

    agent.stop()


def test_freshness_flips_at_bundle_boundary(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    world.update("note", "one")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    private.event.clear()

    world.update("note", "two")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    messages = agent._render_messages()
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(human_messages) == 2
    assert "ARCHIVAL:one" in human_texts(human_messages[0])
    assert "FRESH:two" in human_texts(human_messages[1])

    agent.stop()


def test_on_prompt_event_fires_with_the_messages_the_model_receives(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    captured: list[list[BaseMessage]] = []
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.on_prompt.subscribe(captured.append)
    agent.start()

    world.update("input", "hello")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert len(captured) == 1  # one emit per step
    assert captured[0] == model.calls[0]  # exactly the messages handed to the model

    agent.stop()


def test_on_prompt_subscriber_cannot_mutate_messages_sent_to_model(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)

    def mutate(messages: list[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, list):
                message.content.clear()

    agent.on_prompt.subscribe(mutate)
    agent.start()

    world.update("input", "hello")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert model.calls[0]
    assert any("hello" in human_texts(message) for message in model.calls[0])

    agent.stop()


def test_on_trigger_event_fires_with_the_entry_that_started_the_step(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    triggers: list[str] = []
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.on_trigger.subscribe(lambda entry: triggers.append(entry.key))
    agent.start()

    world.update("input", "hello")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert triggers == ["input"]

    agent.stop()


def test_on_command_event_fires_with_a_command_issued_at_dispatch(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.on_command.subscribe(commands.append)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    agent.start()

    world.update("input", "add them")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert commands == [CommandIssued("add", {"a": 1, "b": 2}, "c1")]

    agent.stop()


def test_on_command_subscriber_cannot_mutate_dispatched_arguments(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)

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
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert invoked == [1]

    agent.stop()


def test_raising_on_prompt_subscriber_does_not_abort_the_step(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("still replied"))

    def boom(messages: list[BaseMessage]) -> None:
        raise RuntimeError("subscriber failure")

    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.on_prompt.subscribe(boom)  # Event.emit isolates a raising subscriber
    agent.start()

    world.update("input", "hello")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert private.texts == [
        "still replied"
    ]  # the raising subscriber didn't break the step
    assert len(model.calls) == 1

    agent.stop()


def test_stop_unsubscribes_from_the_world_trigger(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()
    agent.stop()

    world.update("input", "should not trigger")
    time.sleep(0.2)
    assert model.calls == []


def test_cancel_command_marks_cancelled_and_retriggers(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "block please")

    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    agent.cancel_command("c1")
    wait_until(lambda: private.event.is_set())

    second_call_messages = model.calls[1]
    # the tool_result is the fixed ack; the cancellation is delivered by the command's World entry
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert any(
        m.tool_call_id == "call1" and m.content == _command_ack("c1")
        for m in tool_messages
    )
    observation = "".join(
        str(m.content) for m in second_call_messages if isinstance(m, HumanMessage)
    )
    assert "Called block_forever() → cancelled" in observation

    agent.cancel_command("does-not-exist")  # no-op, must not raise

    agent.stop()


def test_stop_cancels_running_tool_without_triggering_new_step(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "block please")

    key = "agent:command:c1"
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


def test_model_can_cancel_a_running_command(loop, world, private):
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
            tool_call_response([("cancel_command", {"call_id": "c1"}, "cancel1")]),
            text_response("stopped it"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(block_forever)
    agent.start()
    # cancel_command is a WICA-native Command the Agent auto-registers at start — no app wiring.
    assert "cancel_command" in agent._commands

    # Step 1: the model dispatches the long-running command, then the agent goes idle.
    world.update("input", "start working")
    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    # Step 2: a fresh input drives a step where the model observes it still running and issues
    # cancel_command(call_id="c1"). The cancellation lands as the command's World entry going
    # terminal — the causally-correct point — and reaches the model as an observation.
    world.update("input", "actually, stop")
    _wait_for_render(model, world, "Called block_forever() → cancelled")

    agent.stop()


def test_cancel_command_action_is_lenient_on_full_entry_key(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.register_command(block_forever)
    agent.start()

    world.update("input", "go")
    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")

    # The model may copy the entry key verbatim; the handler strips the agent:command: prefix and
    # still finds the running task — the "cancelling c1" result proves the lookup hit (a miss would
    # read "not a running command").
    future = asyncio.run_coroutine_threadsafe(
        agent._cancel_command_action("agent:command:c1"), loop
    )
    assert future.result(timeout=WAIT_TIMEOUT) == "cancelling c1"
    _wait_for_render(model, world, "Called block_forever() → cancelled")

    agent.stop()


def test_cancel_command_action_is_a_noop_for_unknown_id(loop, world, private):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    future = asyncio.run_coroutine_threadsafe(
        agent._cancel_command_action("nope"), loop
    )
    result = future.result(timeout=WAIT_TIMEOUT)
    assert "not a running command" in result

    agent.stop()


def test_burst_of_triggers_coalesces_into_one_step(loop, world, private):
    # Two triggers within the coalescing window run a *single* step whose observation captures
    # both — but on_trigger still fires once per collected trigger. See specs/agent.md.
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=text_response("ok"))
    triggers: list[str] = []
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0.3, private_text=private
    )
    agent.on_trigger.subscribe(lambda entry: triggers.append(entry.key))
    agent.start()

    world.update("a", "a-value")
    world.update("b", "b-value")  # joins the same window → same step
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    time.sleep(0.3)  # let any (incorrect) second step have a chance to run

    assert len(model.calls) == 1
    # dispatch order across the loop isn't guaranteed, so compare as a set
    assert sorted(triggers) == ["a", "b"]
    observation = agent._history[0]
    assert isinstance(observation, ObservationRecord)
    assert {e.entry.key for e in observation.entries} == {"a", "b"}

    agent.stop()


def test_zero_window_fires_immediately_and_drops_while_busy(loop, world, private):
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
        model, world=world, loop=loop, coalesce_window=0, private_text=private
    )
    agent.start()

    t0 = time.monotonic()
    world.update("input", "first")
    assert started.wait(timeout=WAIT_TIMEOUT)
    assert time.monotonic() - t0 < 0.15  # no window wait, unlike the 0.2s default

    world.update("input", "second")  # arrives while busy → dropped
    time.sleep(0.2)
    loop.call_soon_threadsafe(hold.set)
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert len(model.calls) == 1
    assert len([r for r in agent._history if isinstance(r, ObservationRecord)]) == 1

    agent.stop()


def test_bypass_coalescing_flushes_the_window_early(loop, world, private):
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
        model, world=world, loop=loop, coalesce_window=1.0, private_text=private
    )
    agent.on_trigger.subscribe(lambda entry: triggers.append(entry.key))
    agent.start()

    world.update("ctx", "context")  # opens the (long) window
    wait_until(
        lambda: agent._window_timer is not None
    )  # ctx is now batched, window open

    t_urgent = time.monotonic()
    world.update("urgent", "stop!")  # bypass → flush now, pulling ctx forward
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert time.monotonic() - t_urgent < 0.5  # far under the 1.0s window → early flush

    assert len(model.calls) == 1
    assert sorted(triggers) == ["ctx", "urgent"]
    observation = agent._history[0]
    assert isinstance(observation, ObservationRecord)
    assert {e.entry.key for e in observation.entries} == {"ctx", "urgent"}

    agent.stop()


def test_bypass_trigger_arriving_while_busy_is_still_dropped(loop, world, private):
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
        model, world=world, loop=loop, coalesce_window=0, private_text=private
    )
    agent.start()

    world.update("input", "go")
    assert started.wait(timeout=WAIT_TIMEOUT)  # step in flight → busy

    world.update("urgent", "stop!")  # bypass, but busy → dropped
    time.sleep(0.2)
    loop.call_soon_threadsafe(hold.set)
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert len(model.calls) == 1  # urgent did not start a second step
    assert len([r for r in agent._history if isinstance(r, ObservationRecord)]) == 1

    agent.stop()


# --- Command object, output Command, doing nothing, and system-prompt composition ------------


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


def test_register_command_accepts_a_command_object(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("plus", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)

    async def add(a: int, b: int) -> int:
        return a + b

    # A Command lets us override the name without register_command kwargs.
    agent.register_command(Command(add, name="plus", description="Add two numbers."))
    agent.start()

    world.update("input", "add them")
    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "complete")
    assert world.get_entry(key).current.value.result == "3"

    agent.stop()


def test_system_prompt_without_an_output_command_is_the_persona_then_the_primer(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        system_prompt="You are terse.",
        private_text=private,
    )

    prompt = agent.system_prompt
    # Persona is preserved verbatim, at the front.
    assert prompt.startswith("You are terse.")
    # Perception + acting, including that a response's tool calls run concurrently, so sequencing
    # dependent actions is the model's job.
    assert "observations of your World" in prompt
    assert "run concurrently" in prompt
    assert "in a later step" in prompt
    # No output Command: no tag is taught, and the model is told nobody reads its text.
    assert "Nobody reads the text you write" in prompt
    assert "is heard" not in prompt
    # The silence clause: doing nothing is a tag in the text, not a tool.
    assert "write <noop/> as your whole response" in prompt
    assert "noop is not a tool" in prompt
    assert (
        prompt.index("run concurrently")
        < prompt.index("Nobody reads the text you write")
        < prompt.index("<noop/>")
    )
    # No continuation clause without an output: there is no speech to continue after.
    assert "<next>" not in prompt
    assert "end of the delivery" not in prompt


def test_system_prompt_teaches_the_output_tag_its_attributes_and_the_note(
    loop, world, private
):
    async def say(
        text: OutputStream,
        emotion: Literal["happy", "sad"] = "happy",
        voice: str = "default",
    ) -> None:
        """Speak out loud."""

    agent = _dummy_agent(loop, world, private, output_command=say)

    prompt = agent.system_prompt
    # The output clause: the tag, that it is not a tool, the Command's description, its
    # attributes with their allowed values, and an example carrying each default.
    assert "<say>Hello there!</say>" in prompt
    assert "Only the text inside the tag reaches the user" in prompt
    assert "say is not a tool" in prompt
    assert "What say does: Speak out loud." in prompt
    assert "emotion (one of: happy, sad); voice" in prompt
    assert '<say emotion="happy" voice="default">Hello there!</say>' in prompt
    assert "A new say tag replaces the one still being delivered" in prompt
    assert "Nobody reads the text you write" not in prompt
    # The continuation clause, with its neutral example, comes last.
    assert "<next>look it up, then tell them what I found</next>" in prompt
    assert (
        prompt.index("run concurrently")
        < prompt.index("<say>Hello there!</say>")
        < prompt.index("write <noop/> as your whole response")
        < prompt.index("<next>")
    )


@pytest.mark.parametrize(
    ("max_next_steps", "taught"), [(0, False), (1, True), (None, True)]
)
def test_system_prompt_teaches_the_note_unless_it_is_turned_off(
    loop, world, private, max_next_steps, taught
):
    def say(text: str) -> None:
        """Speak out loud."""

    agent = _dummy_agent(
        loop, world, private, output_command=say, max_next_steps=max_next_steps
    )

    prompt = agent.system_prompt
    assert "<say>Hello there!</say>" in prompt
    assert "optional attributes" not in prompt  # the Command declares none
    assert ("<next>" in prompt) is taught
    # Turned off, the model is still told the end of its output does not call it again.
    assert (
        "The end of the delivery of your own say tag does not call you again; a command "
        "finishing does." in prompt
    ) is not taught


def test_an_output_is_a_command_execution_and_text_outside_its_tag_is_private(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    release = asyncio.Event()

    async def speak(text: str) -> str:
        """Say something to the user."""
        await release.wait()
        return "spoken"

    model = ProgrammableChatModel(
        respond=text_response("thinking about it <speak>hello</speak>")
    )
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=speak,
        coalesce_window=0,
    )
    issued: list[CommandIssued] = []
    ended: list[CommandTrace] = []
    agent.on_command.subscribe(issued.append)
    agent.on_command_ended.subscribe(ended.append)
    agent.start()
    world.update("input", "greet the user")

    # While it is delivered the output is a running Command entry, reported on on_command with
    # its whole text; the text outside the tag went to on_private_text.
    key = "agent:command:c1"
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert issued == [CommandIssued("speak", {"text": "hello"}, "c1", is_output=True)]
    running = world.get_entry(key).current.value
    assert (running.name, running.args, running.state) == (
        "speak",
        {"text": "hello"},
        "running",
    )
    assert private.texts == ["thinking about it"]
    # History keeps the response as written, as text: no tool call stands for the speech.
    assert agent._history[1:] == [
        AssistantTextRecord("thinking about it <speak>hello</speak>")
    ]

    loop.call_soon_threadsafe(release.set)
    wait_until(lambda: world.get_entry(key).current.value.state == "complete")
    assert world.get_entry(key).current.value.result == "spoken"
    assert [(t.call_id, t.name, t.state, t.reaction_id) for t in ended] == [
        ("c1", "speak", "complete", 1)
    ]
    time.sleep(0.2)  # give any (erroneous) wake a chance to fire
    assert len(model.calls) == 1  # the end of the output woke nothing

    # The next step observes the finished output like any Command, then retires its entry.
    world.update("input", "again")
    wait_until(lambda: len(model.calls) == 2)
    assert 'speak output: "hello" — delivered: spoken.' in _rendered_user_text(model, 1)
    wait_until(lambda: not world.is_registered(key))

    agent.stop()


@pytest.mark.parametrize(("name", "args"), [("say", {"text": "hello"}), ("noop", {})])
def test_calling_the_output_command_or_noop_as_a_tool_names_an_unknown_command(
    loop, world, private, name, args
):
    """Neither is bound as a tool: the unknown_command guard sends the call back, and the retry
    speaks through the tag."""
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoken: list[str] = []
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([(name, args, "t1")]),
            text_response("<say>hello</say>"),
        )
    )
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=_say_command(spoken),
        coalesce_window=0,
    )
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()

    world.update("input", "greet the user")
    wait_until(lambda: spoken == ["hello"])
    agent.stop()

    [rejection] = rejections
    assert f"There is no command named {name!r}" in rejection.feedback
    assert "The commands you can call are: cancel_command." in rejection.feedback
    assert len(model.calls) == 2


def test_noop_issues_no_command_and_stays_in_history_as_text(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response("<noop/>"), text_response("ok"))
    )
    issued: list[CommandIssued] = []
    traces: list[ReactionTrace] = []
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.on_command.subscribe(issued.append)
    agent.on_reaction_ended.subscribe(traces.append)
    agent.start()

    world.update("input", "nothing to do here")
    wait_until(lambda: len(traces) == 1)
    time.sleep(0.2)  # give any (erroneous) re-trigger a chance to fire

    # No Command was issued, no World entry created, nothing private reported, no re-trigger:
    # the reaction record's flag is what reports the choice.
    assert traces[0].noop is True and traces[0].outcome == "ok"
    assert traces[0].command_call_ids == ()
    assert issued == []
    assert not any(
        e.key.startswith("agent:command:") for e in world.get_prompt_entries()
    )
    assert private.texts == []
    assert len(model.calls) == 1
    assert agent._history[1:] == [AssistantTextRecord("<noop/>")]

    # A later step sees the choice as the assistant's own text — no tool call, no tool_result.
    world.update("input", "second")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    second_prompt = model.calls[1]
    assert [m.text for m in second_prompt if isinstance(m, AIMessage)] == ["<noop/>"]
    assert not any(m.tool_calls for m in second_prompt if isinstance(m, AIMessage))
    assert not any(isinstance(m, ToolMessage) for m in second_prompt)
    # Private text alone, with no output Command: nothing reached anyone, nothing was done.
    assert traces[1].noop is True

    agent.stop()


# --- History renders from captured serializers ------------------------------------------


def _step(
    agent: Agent,
    model: ProgrammableChatModel,
    world: World,
    key: str,
    value: str,
    n_calls: int,
) -> None:
    """Trigger a step by updating `key` and wait until the model has been called n_calls times
    and the reaction has ended — so the next `_step`'s trigger is not dropped as in-flight."""
    world.update(key, value)
    wait_until(lambda: len(model.calls) >= n_calls and not agent._busy)


def _rendered_user_text(model: ProgrammableChatModel, call_index: int) -> str:
    return "".join(
        human_texts(m) for m in model.calls[call_index] if isinstance(m, HumanMessage)
    )


def test_history_renders_an_unregistered_entry_from_its_captured_serializer(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"mood={v}")])
    agent.start()
    world.update("mood", "happy")
    _step(agent, model, world, "speech", "one", 1)
    assert "mood=happy" in _rendered_user_text(model, 0)

    world.unregister("mood")  # would KeyError at render time before D3
    _step(agent, model, world, "speech", "two", 2)
    # observation 1 still renders as it did, from its captured serializer
    assert "mood=happy" in _rendered_user_text(model, 1)
    agent.stop()


def test_reregistering_a_key_does_not_rewrite_earlier_observations(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"OLD:{v}")])
    agent.start()
    world.update("mood", "happy")
    _step(agent, model, world, "speech", "one", 1)
    first_render = _rendered_user_text(model, 0)

    world.unregister("mood")
    world.register("mood", str, serialize_fn=lambda v, p: [TextPart(f"NEW:{v}")])
    world.update("mood", "calm")
    _step(agent, model, world, "speech", "two", 2)
    # The first observation is byte-identical to what step one sent (cache-stable prefix) …
    assert _rendered_user_text(model, 1).startswith(first_render.split("</entry>")[0])
    assert "OLD:happy" in _rendered_user_text(model, 1)
    # … and the new observation uses the new serializer.
    assert "NEW:calm" in _rendered_user_text(model, 1)
    assert "NEW:happy" not in _rendered_user_text(model, 1)
    agent.stop()


def test_application_entry_under_agent_command_prefix_uses_its_own_serializer(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register(
        "agent:command:mine", str, serialize_fn=lambda v, p: [TextPart(f"MINE:{v}")]
    )
    agent.start()
    world.update("agent:command:mine", "x")
    _step(agent, model, world, "speech", "one", 1)
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
    loop, world, private
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
        model, world=world, loop=loop, private_text=private, coalesce_window=0
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


def test_call_ids_are_minted_in_order_whatever_the_provider_id(loop, world, private):
    # The model is pointed at a short id WICA mints (c1, c2, …), never the provider's: a model
    # copying a long opaque id into cancel_command garbles it. A provider reusing an id is harmless.
    release = threading.Event()

    async def slow() -> str:
        """Block until released."""
        await asyncio.get_running_loop().run_in_executor(None, release.wait)
        return "done"

    model = ProgrammableChatModel(
        respond=tool_call_response([("slow", {}, "toolu_01Same")])
    )
    issued: list[CommandIssued] = []
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    agent.on_command.subscribe(issued.append)
    agent.register_command(slow)
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    agent.start()
    world.update("speech", "one")
    wait_until(lambda: world.is_registered("agent:command:c1"))
    wait_until(lambda: agent._busy is False)
    world.update("speech", "two")  # second step, same provider id, first still running
    wait_until(lambda: world.is_registered("agent:command:c2"))
    assert [c.call_id for c in issued] == ["c1", "c2"]
    # The native pair keeps the provider's id; its acknowledgement points at the minted one.
    tool_messages = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in tool_messages] == [
        ("toolu_01Same", _command_ack("c1"))
    ]
    release.set()
    agent.stop()


def test_minted_ids_count_outputs_skip_taken_keys_and_survive_a_restart(
    loop, world, private
):
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = ProgrammableChatModel(
        respond=sequence(
            text_and_tool_response(
                "<say>Adding.</say>", [("add", {"a": 1, "b": 2}, "p1")]
            ),
            text_response("<noop/>"),
            tool_call_response([("add", {"a": 2, "b": 2}, "p2")]),
            text_response("<noop/>"),
        )
    )
    issued: list[CommandIssued] = []
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=_say_command([]),
        coalesce_window=0,
    )
    agent.on_command.subscribe(issued.append)
    agent.register_command(add)
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    # Someone else already holds agent:command:c2: the Agent skips that number.
    world.register("agent:command:c2", str, serialize_fn=identity_serialize)
    agent.start()
    world.update("input", "add 1 and 2")
    wait_until(lambda: len(model.calls) == 2 and not agent._busy)
    agent.stop()
    agent.start()
    world.update("input", "add 2 and 2")
    wait_until(lambda: len(model.calls) == 4 and not agent._busy)
    # An output takes an id like a tool call does; a <noop/> takes none.
    assert [(c.name, c.call_id) for c in issued] == [
        ("say", "c1"),
        ("add", "c3"),
        ("add", "c4"),
    ]
    agent.stop()


# --- Reserved names and duplicate rejection ---------------------------------------------


def _dummy_agent(loop, world, private, **kwargs) -> Agent:
    return make_agent(
        ProgrammableChatModel(respond=text_response("ok")),
        world=world,
        loop=loop,
        private_text=private,
        **kwargs,
    )


def _named(name: str, description: str = "A test command.") -> Command:
    """A Command usable both as a registered Command and as an output Command (its one
    parameter is the text)."""

    def fn(text: str = "") -> str:
        return "x"

    return Command(fn, name=name, description=description)


@pytest.mark.parametrize("name", ["noop", "next", "cancel_command"])
def test_register_command_rejects_reserved_names(loop, world, private, name):
    agent = _dummy_agent(loop, world, private)
    with pytest.raises(ValueError, match="reserved"):
        agent.register_command(_named(name))


def test_register_command_rejects_duplicate_names(loop, world, private):
    agent = _dummy_agent(loop, world, private)
    agent.register_command(_named("wave"))
    with pytest.raises(ValueError, match="already registered"):
        agent.register_command(_named("wave"))


@pytest.mark.parametrize("name", ["noop", "next", "cancel_command"])
def test_set_output_command_rejects_reserved_names(loop, world, private, name):
    agent = _dummy_agent(loop, world, private)
    before = agent.system_prompt
    with pytest.raises(ValueError, match="reserved"):
        agent.set_output_command(_named(name))
    assert agent.output_command_name is None
    assert agent.system_prompt == before


def test_register_command_rejects_the_output_commands_name(loop, world, private):
    agent = _dummy_agent(loop, world, private, output_command=_named("say"))
    with pytest.raises(ValueError, match="output Command"):
        agent.register_command(_named("say"))


def test_restart_keeps_exactly_one_builtin_and_the_output_command_out_of_the_tools(
    loop, world, private
):
    agent = _dummy_agent(loop, world, private, output_command=_named("say"))
    agent.register_command(_named("wave"))
    agent.start()
    agent.stop()
    agent.start()
    assert sorted(agent.commands) == ["cancel_command", "wave"]
    assert agent.output_command_name == "say"
    agent.stop()


def test_set_output_command_rejects_a_name_that_is_not_a_tag_name_or_a_wrapped_tool(
    loop, world, private
):
    @langchain_tool
    def search(query: str) -> str:
        """Search the web."""
        return query

    agent = _dummy_agent(loop, world, private)
    before = agent.system_prompt
    with pytest.raises(ValueError, match="not a valid tag name"):
        agent.set_output_command(_named("say it"))
    with pytest.raises(ValueError, match="built from a callable"):
        agent.set_output_command(Command(search))
    assert agent.output_command_name is None
    assert agent.system_prompt == before


# --- Output wiring: set_output_command (specs/agent.md "Output wiring") ------------------------


def test_set_output_command_rejects_an_already_registered_name_unchanged(
    loop, world, private
):
    agent = _dummy_agent(loop, world, private)
    agent.register_command(_named("wave"))
    before = agent.system_prompt
    with pytest.raises(ValueError, match="already registered"):
        agent.set_output_command(_named("wave"))
    assert agent.output_command_name is None
    assert agent.system_prompt == before
    assert list(agent.commands) == ["wave"]


def test_set_output_command_after_start_takes_effect_at_the_next_step(
    loop, world, private
):
    """Setting while running is permitted: the next step's system prompt teaches the tag and its
    text is delivered — and the model binding is untouched, since the output Command is no tool."""
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

    model = ProgrammableChatModel(respond=text_response("<speak>hello</speak>"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    agent.start()
    world.update("input", "before the voice")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: not agent._busy)
    assert private.texts == ["<speak>hello</speak>"]  # no output Command yet: private
    assert "Nobody reads the text you write" in agent.system_prompt

    agent.set_output_command(speak)
    assert agent.output_command_name == "speak"
    assert list(agent.commands) == ["cancel_command"]

    world.update("input", "greet the user")
    assert spoke.wait(timeout=WAIT_TIMEOUT)
    assert spoken == ["hello"]
    assert private.texts == ["<speak>hello</speak>"]  # nothing private this time
    # The step's prompt (rendered after the set) teaches the tag; the binding did not change.
    assert "<speak>Hello there!</speak>" in model.calls[1][0].text
    assert "<speak>Hello there!</speak>" not in model.calls[0][0].text
    assert [b["tools"] for b in model.bindings] == [["cancel_command"]] * 2
    agent.stop()


def test_replacing_the_output_command_frees_the_previous_name(loop, world, private):
    agent = _dummy_agent(loop, world, private, output_command=_named("say"))
    agent.start()
    assert "<say>Hello there!</say>" in agent.system_prompt

    agent.set_output_command(_named("speak"))
    assert agent.output_command_name == "speak"
    assert "<speak>Hello there!</speak>" in agent.system_prompt
    assert "<say>" not in agent.system_prompt
    # The old name is free again; the new one is taken.
    agent.register_command(_named("say"))
    with pytest.raises(ValueError, match="output Command"):
        agent.register_command(_named("speak"))
    agent.stop()


def test_setting_the_same_name_again_swaps_the_command_object(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    first: list[str] = []
    second: list[str] = []
    model = ProgrammableChatModel(respond=text_response("<say>hello</say>"))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=_say_command(first),
        coalesce_window=0,
    )
    agent.start()
    agent.set_output_command(_say_command(second))
    world.update("input", "greet the user")
    wait_until(lambda: second == ["hello"])
    assert first == []
    agent.stop()


def test_clearing_the_output_command_makes_all_text_private_again(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoken: list[str] = []
    model = ProgrammableChatModel(respond=text_response("<say>hello</say>"))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=_say_command(spoken),
        coalesce_window=0,
    )
    agent.start()
    assert "<say>Hello there!</say>" in agent.system_prompt

    agent.set_output_command(None)
    assert agent.output_command_name is None
    assert "<say>" not in agent.system_prompt
    assert "<next>" not in agent.system_prompt
    assert "Nobody reads the text you write" in agent.system_prompt

    world.update("input", "greet the user")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert private.texts == ["<say>hello</say>"]  # no tag is an output any more
    assert spoken == []
    assert not any(
        e.key.startswith("agent:command:") for e in world.get_prompt_entries()
    )
    agent.register_command(_named("say"))  # no longer the output Command's name
    agent.stop()


# --- The model binding: the output Command is not a tool, and no tool choice is forced ----


def _say_command(spoken: list[str]) -> Command:
    """An output Command `say` taking the whole text of each output."""

    async def say(text: str) -> str:
        """Say something to the user."""
        spoken.append(text)
        return "spoken"

    return Command(say)


def test_the_output_command_is_not_bound_as_a_tool_and_no_tool_choice_is_forced(
    loop, world, private
):
    """The model speaks in its text, so the binding holds only the Commands issued as tool calls
    and carries no tool_choice (which several self-hosted servers ignore)."""
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoken: list[str] = []
    model = ProgrammableChatModel(respond=text_response("<say>hello</say>"))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=_say_command(spoken),
        coalesce_window=0,
    )
    agent.register_command(_named("wave"))
    agent.start()
    world.update("input", "greet the user")
    wait_until(lambda: spoken == ["hello"])
    agent.stop()

    [binding] = model.bindings
    assert binding["tool_choice"] is None
    assert set(binding["tools"]) == {"wave", "cancel_command"}


def test_command_finishing_after_the_world_stopped_does_not_raise(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    release = asyncio.Event()

    async def finish_late() -> str:
        """Completes only once released — by then the World is stopped."""
        await release.wait()
        return "done"

    agent.register_command(finish_late)
    agent.start()
    world.update("input", "go")
    key = "agent:command:c1"
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


def test_unknown_command_name_fails_with_a_clear_error(loop, world, private):
    # Without the unknown_command guard, a model naming a tool the Agent never bound fails the
    # entry with a readable error (not the bare KeyError repr "'ghost'"), delivered to the next step
    # like any Command failure.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("ghost", {}, "g1")]),
            text_response("noted"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.remove_response_guard("unknown_command")
    agent.start()

    world.update("input", "summon")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    observation = "".join(
        str(m.content) for m in model.calls[1] if isinstance(m, HumanMessage)
    )
    assert "Called ghost() → failed: unknown command 'ghost'" in observation
    agent.stop()


# --- Step failure handling: model-call and output Command errors, well-formed history ------


async def _explode(messages: list[BaseMessage]) -> AIMessage:
    raise ConnectionError("provider down")


def _no_consecutive_human_messages(messages: list[BaseMessage]) -> bool:
    return not any(
        isinstance(a, HumanMessage) and isinstance(b, HumanMessage)
        for a, b in zip(messages, messages[1:])
    )


def test_model_call_failure_is_logged_and_the_next_step_runs(
    loop, world, private, caplog
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(_explode, text_response("recovered"))
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "first")
        wait_until(lambda: len(model.calls) == 1 and not agent._busy)
        world.update("input", "second")
        assert private.event.wait(timeout=WAIT_TIMEOUT)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("model call failed" in r.message for r in errors)
    assert any(
        r.exc_info and isinstance(r.exc_info[1], ConnectionError) for r in errors
    )
    assert private.texts == ["recovered"]

    agent.stop()


def test_a_failing_output_command_ends_as_a_failed_entry_and_commands_still_dispatch(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def say(text: str) -> str:
        """Say something to the user."""
        raise RuntimeError("voice broke")

    # Step 1 speaks (into the raising output Command) and issues wave; wave's completion
    # re-triggers step 2, which does nothing.
    model = ProgrammableChatModel(
        respond=sequence(
            text_and_tool_response("<say>doing it</say>", [("wave", {}, "w1")]),
            text_response("<noop/>"),
        )
    )
    agent = make_agent(
        model, world=world, loop=loop, output_command=say, coalesce_window=0
    )
    ended: list[CommandTrace] = []
    agent.on_command_ended.subscribe(ended.append)
    waved = threading.Event()

    async def wave() -> str:
        """Wave."""
        waved.set()
        return "waved"

    agent.register_command(wave)
    agent.start()

    world.update("input", "hello")
    assert waved.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: len(model.calls) >= 2)  # wave's completion re-triggers a step

    assert ("say", "failed") in [(t.name, t.state) for t in ended]
    # The utterance is still in history as written, and the failure is observed like any
    # Command's: as its World entry.
    second = model.calls[1]
    assert [m.text for m in second if isinstance(m, AIMessage)] == [
        "<say>doing it</say>"
    ]
    assert 'say output: "doing it" — failed: voice broke' in _rendered_user_text(
        model, 1
    )

    agent.stop()


def test_failed_step_observation_merges_into_the_next_prompt(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=sequence(_explode, text_response("ok")))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()
    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "second")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    second = model.calls[1]
    assert _no_consecutive_human_messages(second)
    humans = [m for m in second if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    text = human_texts(humans[0])
    assert "first" in text and "second" in text  # both observations, one message
    assert text.index("first") < text.index("second")  # older observation first

    agent.stop()


def test_merged_observations_keep_their_own_freshness(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()
    world.update("note", "one")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("note", "two")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    humans = [m for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    text = human_texts(humans[0])
    assert "ARCHIVAL:one" in text and "FRESH:two" in text

    agent.stop()


def test_empty_model_response_does_not_split_the_next_prompt(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response(""), text_response("ok"))
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()
    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "second")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert _no_consecutive_human_messages(model.calls[1])
    assert len([m for m in model.calls[1] if isinstance(m, HumanMessage)]) == 1

    agent.stop()


def test_commands_mapping_lists_the_bound_set_read_only(loop, world, private):
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
        private_text=private,
        output_command=speak,
    )
    agent.register_command(add)
    assert agent.output_command_name == "speak"
    assert list(agent.commands) == ["add"]  # builtins attach at start()
    agent.start()
    # The output Command is not a tool: it is named apart, never listed among the Commands.
    assert set(agent.commands) == {"add", "cancel_command"}
    assert agent.output_command_name == "speak"
    assert agent.commands["add"].tool.description == "Add."
    with pytest.raises(TypeError):
        agent.commands["ghost"] = agent.commands["add"]  # type: ignore[index]
    agent.stop()


def test_on_command_fires_once_the_execution_entry_exists_and_carries_its_call_id(
    loop, world, private
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
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
    loop, world, private
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
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
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert len(model.calls) == 2
    humans = [m for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert any("Called ping() → pong" in str(m.content) for m in humans)
    assert not any(
        e.key.startswith("agent:command:") for e in world.get_prompt_entries()
    )
    agent.stop()


@pytest.mark.parametrize("triggers_on_completion", [True, False])
def test_the_end_of_an_output_wakes_no_step_whatever_its_completion_flag(
    loop, world, private, triggers_on_completion
):
    """An output's wake is the model's <next> note, never the Command's triggers_on_completion."""
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def speak(text: str) -> str:
        """Say something to the user."""
        return "spoken"

    model = ProgrammableChatModel(respond=text_response("<speak>hello</speak>"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    agent.set_output_command(
        Command(speak, triggers_on_completion=triggers_on_completion)
    )
    ended: list[CommandTrace] = []
    agent.on_command_ended.subscribe(ended.append)
    agent.start()
    world.update("input", "greet")
    wait_until(lambda: [t.state for t in ended] == ["complete"])
    time.sleep(0.2)  # give any (erroneous) wake a chance to fire
    assert len(model.calls) == 1
    assert [(t.is_output, t.triggers_step) for t in ended] == [(True, False)]
    agent.stop()


def test_the_record_of_a_commands_end_says_whether_it_is_an_output_and_wakes_a_step(
    loop, world, private
):
    """What a harness waiting for the Agent to settle reads, from the Event alone: a tool-call
    Command triggers a step by its flag, an output only when its response carried a note."""
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoken: list[str] = []
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("wave", {}, "w1")]),
            text_response("<say>One moment.</say><next>then report</next>"),
            text_response("<say>Done.</say>"),
        )
    )
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=_say_command(spoken),
        coalesce_window=0,
    )
    agent.register_command(_named("wave"))
    ended: list[CommandTrace] = []
    issued: list[CommandIssued] = []
    agent.on_command_ended.subscribe(ended.append)
    agent.on_command.subscribe(issued.append)
    agent.start()
    world.update("input", "wave, then tell me")
    wait_until(lambda: spoken == ["One moment.", "Done."])
    wait_until(lambda: not agent._busy and len(ended) == 3)

    assert {t.call_id: (t.is_output, t.triggers_step) for t in ended} == {
        "c1": (False, True),  # wave: a tool call, triggers_on_completion
        "c2": (True, True),  # its response carried a note
        "c3": (True, False),  # no note: the chain ends
    }
    assert [(c.call_id, c.is_output) for c in issued] == [
        ("c1", False),
        ("c2", True),
        ("c3", True),
    ]
    time.sleep(0.2)
    assert len(model.calls) == 3
    agent.stop()


def test_on_private_text_carries_only_the_text_outside_the_output_tags(loop, world):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    spoken: list[str] = []
    texts: list[str] = []
    traces: list[ReactionTrace] = []
    model = ProgrammableChatModel(
        respond=sequence(
            text_response("an aside <say>hello</say>"),
            text_response(""),
            text_response("<say>only speech</say>"),
        )
    )
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        output_command=_say_command(spoken),
        coalesce_window=0,
    )
    agent.on_private_text.subscribe(texts.append)
    agent.on_reaction_ended.subscribe(traces.append)
    agent.start()
    for count, value in enumerate(["hi", "again", "once more"], start=1):
        world.update("input", value)
        wait_until(lambda: len(traces) == count)
    wait_until(lambda: spoken == ["hello", "only speech"])
    # An empty response and one that is all speech emit nothing.
    assert texts == ["an aside"]
    agent.stop()


# --- Instrumentation: ReactionTrace, on_trigger_dropped, on_command_ended, spans ---------------


def test_reaction_ended_carries_the_phase_stamps_and_usage(loop, world, private):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, private_text=private)
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
    assert trace.noop is True  # private text only, no output Command: a decline
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
    assert trace.first_output_at is None  # no output Command: nothing was delivered
    assert trace.trace_id is not None
    assert re.fullmatch(r"[0-9a-f]{32}", trace.trace_id)

    agent.stop()


def test_reaction_ended_reports_an_empty_response(loop, world, private):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response(""))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert traces[0].outcome == "empty"
    assert traces[0].noop is True  # no text and no tool call counts as doing nothing
    assert traces[0].text_length == 0

    agent.stop()


def test_reaction_ended_reports_a_model_error(loop, world, private):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("boom")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, private_text=private)
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


def test_model_call_past_its_deadline_ends_the_step_as_model_timeout(
    loop, world, private
):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    hang = (
        asyncio.Event()
    )  # never set: the first call hangs until the deadline cancels it
    calls = 0

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        nonlocal calls
        calls += 1
        if calls == 1:
            await hang.wait()
        return AIMessage(content="back")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, model_timeout=0.05
    )
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "first")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[0].outcome == "model_timeout"
    assert traces[0].error == "model call exceeded its 0.05 s deadline"
    assert traces[0].model_ended_at is not None
    assert private.texts == []

    # The Agent is free again; the timed-out observation merges into the next prompt.
    done.clear()
    world.update("prompt", "second")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[1].outcome == "ok"
    assert private.texts == ["back"]
    user_text = _rendered_user_text(model, 1)
    assert "first" in user_text and "second" in user_text

    agent.stop()


def test_a_providers_own_timeout_error_is_a_model_error(loop, world, private):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        raise TimeoutError("read timed out")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[0].outcome == "model_error"
    assert traces[0].error == "read timed out"

    agent.stop()


@pytest.mark.parametrize("model_timeout", [None, 5.0])
def test_a_call_within_its_deadline_completes(loop, world, private, model_timeout):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        await asyncio.sleep(0.1)
        return AIMessage(content="slow but fine")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, model_timeout=model_timeout
    )
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("prompt", "x")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[0].outcome == "ok"
    assert private.texts == ["slow but fine"]

    agent.stop()


def test_reaction_ended_reports_cancellation_on_stop(loop, world, private):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    prompt_fired = threading.Event()
    block = asyncio.Event()

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        await block.wait()
        return AIMessage(content="never")

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, private_text=private)
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


def test_dropped_trigger_fires_on_trigger_dropped(loop, world, private):
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)
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
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    agent.stop()


def test_coalesced_reaction_lists_every_trigger(loop, world, private):
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0.2, private_text=private
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


def test_command_ended_carries_reaction_id_and_duration(loop, world, private):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)

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


def test_span_tree_follows_the_reaction(loop, world, private, spans):
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("done"),
        )
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)

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
    loop, world, private, spans
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
    agent = make_agent(model, world=world, loop=loop, private_text=private)

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
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    finished = spans.get_finished_spans()
    (tts_span,) = [s for s in finished if s.name == "tts.synthesize"]
    (command_span,) = [s for s in finished if s.name == "wica.agent.command"]
    assert tts_span.parent is not None
    assert tts_span.parent.span_id == command_span.context.span_id


def test_an_output_span_is_a_command_span_under_the_reaction(
    loop, world, private, spans
):
    """An output is traced like a Command: its span opens during the model call as a child of
    the reaction (a sibling of the model span), what the output Command opens nests under it,
    and the model span marks the first output. There is no sink span."""
    from opentelemetry import trace as otel_trace_module

    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def say(text: str) -> str:
        """Speak out loud."""
        with otel_trace_module.get_tracer("app").start_as_current_span(
            "tts.synthesize"
        ):
            pass
        return "said"

    model = ProgrammableChatModel(respond=text_response("<say>hello</say>"))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        output_command=say,
        coalesce_window=0,
    )
    ended: list[CommandTrace] = []
    agent.on_command_ended.subscribe(ended.append)
    agent.start()
    world.update("prompt", "greet")
    wait_until(lambda: [t.state for t in ended] == ["complete"])
    agent.stop()

    finished = spans.get_finished_spans()
    assert "wica.agent.sink" not in {s.name for s in finished}
    (reaction_span,) = [s for s in finished if s.name == "wica.agent.reaction"]
    (model_span,) = [s for s in finished if s.name == "wica.agent.model"]
    (command_span,) = [s for s in finished if s.name == "wica.agent.command"]
    (tts_span,) = [s for s in finished if s.name == "tts.synthesize"]
    assert model_span.parent is not None and command_span.parent is not None
    assert model_span.parent.span_id == reaction_span.context.span_id
    assert command_span.parent.span_id == reaction_span.context.span_id
    assert command_span.attributes["wica.command.name"] == "say"
    assert command_span.attributes["wica.command.state"] == "complete"
    assert tts_span.parent is not None
    assert tts_span.parent.span_id == command_span.context.span_id
    assert "wica.first_output" in [event.name for event in model_span.events]


def test_coalesced_reaction_links_the_other_triggers(loop, world, private, spans):
    world.register("a", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    world.register("b", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, coalesce_window=0.2, private_text=private
    )
    agent.start()

    world.update("a", "1")
    world.update("b", "2")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
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


# --- Omitted entries (serializer returns []) --------------------------------


def _hide_when_none(value: Any, previous: Any) -> Content:
    return [] if value is None else [TextPart(f"mood={value}")]


def test_observation_leaves_out_entries_whose_serializer_returns_empty(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    world.register("mood", str, serialize_fn=_hide_when_none)
    agent.start()

    _step(agent, model, world, "speech", "one", 1)
    first = _rendered_user_text(model, 0)
    assert 'key="speech"' in first
    assert "mood" not in first  # no hollow <entry key="mood"> block either

    world.update("mood", "happy")
    _step(agent, model, world, "speech", "two", 2)
    second = _rendered_user_text(model, 1)
    assert 'key="mood"' in second
    assert "mood=happy" in second
    # the archival re-render of observation 1 still omits it — its snapshot held None
    assert second.count('key="mood"') == 1
    agent.stop()


def test_all_omitted_observation_renders_as_a_placeholder_user_message(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0
    )
    world.register("silent", str, serialize_fn=lambda v, p: [], triggers_llm_call=True)
    agent.start()

    _step(agent, model, world, "silent", "x", 1)
    humans = [m for m in model.calls[0] if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    assert humans[0].content == [{"type": "text", "text": _EMPTY_OBSERVATION}]
    assert isinstance(
        model.calls[0][1], HumanMessage
    )  # a user turn follows the system prompt

    _step(agent, model, world, "silent", "y", 2)
    assert _no_consecutive_human_messages(model.calls[1])
    humans = [m for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert [m.content for m in humans] == [[_EMPTY_OBSERVATION_BLOCK]] * 2
    agent.stop()


def test_merged_all_omitted_observations_share_one_placeholder(loop, world, private):
    world.register("silent", str, serialize_fn=lambda v, p: [], triggers_llm_call=True)
    world.register("mood", str, serialize_fn=_hide_when_none)
    model = ProgrammableChatModel(
        respond=sequence(text_response(""), text_response(""), text_response("ok"))
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    world.update("silent", "a")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("silent", "b")  # empty response above: observations 1 and 2 merge
    wait_until(lambda: len(model.calls) == 2 and not agent._busy)
    humans = [m for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert [m.content for m in humans] == [[_EMPTY_OBSERVATION_BLOCK]]

    world.update("mood", "calm")
    world.update(
        "silent", "c"
    )  # real blocks replace the placeholder in the merged message
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    humans = [m for m in model.calls[2] if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    assert isinstance(humans[0].content, list)
    assert _EMPTY_OBSERVATION_BLOCK not in humans[0].content
    assert "mood=calm" in human_texts(humans[0])
    agent.stop()


# --- History window (specs/agent.md, "History window") -----------------------------------


def _run_text_steps(
    agent: Agent, world: World, private: RecordingText, inputs: list[str]
) -> None:
    """Drive one text-only step per input, waiting for each reply before the next update."""
    for value in inputs:
        private.event.clear()
        world.update("input", value)
        assert private.event.wait(timeout=WAIT_TIMEOUT)
        wait_until(lambda: agent._busy is False)


def _assert_tool_stream_valid(messages: list[BaseMessage]) -> None:
    """Every ToolMessage answers a tool_call of the AIMessage right before it, and nothing
    assistant-side precedes the first user turn — what a provider requires of the stream."""
    assert isinstance(messages[1], HumanMessage)
    open_ids: set[str] = set()
    for message in messages[1:]:
        if isinstance(message, AIMessage):
            open_ids = {c["id"] for c in message.tool_calls if c["id"]}
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in open_ids, message
            open_ids.discard(message.tool_call_id)
        else:
            open_ids = set()


def test_history_window_grows_to_twice_the_limit_then_cuts_back_to_it(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("ok"))
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, history_reactions=2
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.start()

    _run_text_steps(agent, world, private, ["one", "two", "three", "four"])

    def observations(messages: list[BaseMessage]) -> list[str]:
        return [human_texts(m) for m in messages if isinstance(m, HumanMessage)]

    # Steps 1-3: below 2X reactions, nothing is cut — history keeps growing and the message
    # right after the system prompt is byte-identical across calls (the cacheable prefix).
    assert len(observations(prompts[2])) == 3
    assert prompts[1][1] == prompts[2][1]
    assert "one" in human_texts(prompts[2][1])
    # Step 4 reaches 2X = 4: cut back to the newest X = 2 (step 3 and the current step 4).
    kept = observations(prompts[3])
    assert len(kept) == 2
    assert "three" in kept[0] and "one" not in kept[0] and "two" not in kept[0]
    assert "four" in kept[1]
    assert prompts[3][1] != prompts[2][1]
    # The cut removed the records from storage, not just from the rendered prompt.
    assert [
        r for r in agent._history if isinstance(r, ObservationRecord)
    ].__len__() == 2

    agent.stop()


def test_history_window_counts_failed_and_noop_steps_as_reactions(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            _explode,
            text_response("<noop/>"),
            text_response("three"),
            text_response("four"),
        )
    )
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, history_reactions=2
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.start()

    world.update("input", "one")  # model raises: a reaction all the same
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "two")  # noop: a reaction all the same
    wait_until(lambda: len(model.calls) == 2 and not agent._busy)
    _run_text_steps(agent, world, private, ["three", "four"])

    # Four reactions (error, noop, text, text) reach 2X = 4 → only steps 3 and 4 remain, so the
    # failed step's observation (which would have merged into step 2's) and the noop are gone.
    step4 = prompts[3]
    humans = [human_texts(m) for m in step4 if isinstance(m, HumanMessage)]
    assert len(humans) == 2
    assert "three" in humans[0] and "one" not in humans[0] and "two" not in humans[0]
    assert AssistantTextRecord("<noop/>") not in agent._history
    assert [m.text for m in step4 if isinstance(m, AIMessage)] == ["three"]

    agent.stop()


def test_history_window_cut_keeps_a_steps_tool_calls_with_their_results(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 1}, "call1")]),
            text_response("two"),
            tool_call_response([("add", {"a": 2, "b": 2}, "call3")]),
            text_response("four"),
        )
    )
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, history_reactions=2
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.register_command(add)
    agent.start()

    world.update("input", "add 1 and 1")  # step 1 dispatches call1; completion → step 2
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: agent._busy is False)
    private.event.clear()
    world.update("input", "add 2 and 2")  # step 3 dispatches call3; completion → step 4
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert private.texts == ["two", "four"]
    assert len(prompts) == 4

    for messages in prompts:
        _assert_tool_stream_valid(messages)
    # Step 4 hit 2X: steps 1-2 are gone as a whole — call1's tool_call, its ack and its outcome —
    # while step 3's tool_call travels with its ack into the kept prefix.
    step4 = prompts[3]
    assert isinstance(step4[1], HumanMessage) and "add 2 and 2" in human_texts(step4[1])
    assert isinstance(step4[2], AIMessage)
    assert [c["id"] for c in step4[2].tool_calls] == ["call3"]
    assert isinstance(step4[3], ToolMessage) and step4[3].tool_call_id == "call3"
    assert "call1" not in "".join(str(m.content) for m in step4)
    assert "Called add(a=2, b=2) → 4" in human_texts(step4[4])

    agent.stop()


def test_command_outcome_still_renders_after_its_dispatch_was_cut(loop, world, private):
    # With X = 1 every second step cuts the previous one, so a slow Command's dispatch is gone
    # before its outcome is observed: the entry must still render (as an orphan), and the stream
    # must stay valid without the tool_call it answers.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    block = asyncio.Event()

    async def dance() -> str:
        """Blocks until released."""
        await block.wait()
        return "done dancing"

    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("dance", {}, "call1")]),
            text_response("still going"),
            text_response("finished"),
        )
    )
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, history_reactions=1
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.register_command(dance)
    agent.start()

    world.update("input", "dance")
    key = "agent:command:c1"
    wait_until(lambda: world.get_entry(key).current.value.state == "running")
    wait_until(lambda: agent._busy is False)
    world.update(
        "input", "how is it going?"
    )  # step 2 cuts step 1 while dance still runs
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    private.event.clear()
    loop.call_soon_threadsafe(block.set)  # completion → step 3 cuts step 2
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    assert private.texts == ["still going", "finished"]

    step2, step3 = prompts[1], prompts[2]
    for messages in (step2, step3):
        _assert_tool_stream_valid(messages)
        assert not any(isinstance(m, (AIMessage, ToolMessage)) for m in messages)
        assert len(messages) == 2
    assert "still running" in human_texts(step2[1])
    assert "done dancing" in human_texts(step3[1])

    agent.stop()


def test_unbounded_history_keeps_every_reaction_by_default(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("ok"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.start()

    _run_text_steps(agent, world, private, ["one", "two", "three", "four", "five"])

    humans = [m for m in model.calls[4] if isinstance(m, HumanMessage)]
    assert len(humans) == 5
    assert "one" in human_texts(humans[0])
    assert sum(isinstance(r, ObservationRecord) for r in agent._history) == 5
    assert "most recent reactions" not in agent.system_prompt

    agent.stop()


def test_history_window_clause_sits_between_perception_and_output_clauses(
    loop, world, private
):
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, history_reactions=3
    )

    prompt = agent.system_prompt
    assert "most recent reactions" in prompt
    assert (
        prompt.index("run concurrently")
        < prompt.index("most recent reactions")
        < prompt.index("Nobody reads the text you write")
        < prompt.index("never use <noop/>")
    )

    async def say(text: str) -> None:
        """Speak."""

    agent.set_output_command(say)  # recomposition keeps the clause, in the same place
    prompt = agent.system_prompt
    assert prompt.index("run concurrently") < prompt.index("most recent reactions")
    assert prompt.index("most recent reactions") < prompt.index(
        "<say>Hello there!</say>"
    )


# --- History budget (specs/agent.md, "History budget") -----------------------------------


def usage_responses(
    input_tokens: Sequence[int | None], text: str = "ok"
) -> Callable[[list[BaseMessage]], Awaitable[AIMessage]]:
    """One text reply per call, each reporting the given input token count (None = no usage
    reported, as a provider that returns none). The last figure repeats once the list is spent."""
    remaining = list(input_tokens)

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        tokens = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if tokens is None:
            return AIMessage(content=text)
        return AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": tokens,
                "output_tokens": 1,
                "total_tokens": tokens + 1,
            },
        )

    return respond


def _observation_texts(messages: list[BaseMessage]) -> list[str]:
    return [human_texts(m) for m in messages if isinstance(m, HumanMessage)]


def test_history_budget_cuts_to_the_low_water_mark_from_reported_usage(
    loop, world, private
):
    # window 1000 × budget 0.5 → high 500, low 250. The first report (100) is the fixed part +
    # one reaction; six reactions reporting 600 → (600 - 100) / 5 = 100 each → keep
    # floor((250 - 100) / 100) = 1 newest; the 7th prompt then starts at reaction 6.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=usage_responses([100, 200, 300, 400, 450, 600, 300])
    )
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        history_budget=0.5,
        context_window=1000,
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.start()

    _run_text_steps(agent, world, private, ["r1", "r2", "r3", "r4", "r5", "r6", "r7"])

    # Under the mark (calls 1-5) nothing is cut: prompt 6 still holds all six reactions.
    assert len(_observation_texts(prompts[5])) == 6
    assert sum(isinstance(r, ObservationRecord) for r in agent._history) == 2
    # Call 6 reported 600 > 500: cut to 1 reaction at the end of step 6, so prompt 7 holds
    # reaction 6 and the new 7 — and call 7's 300 (< 500) triggers no further cut.
    kept = _observation_texts(prompts[6])
    assert [t.split("\n")[1] for t in kept] == ["r6", "r7"]
    for messages in prompts:
        _assert_tool_stream_valid(messages)

    agent.stop()


def test_history_budget_charges_the_fixed_part_once_not_per_reaction(
    loop, world, private
):
    # A prompt dominated by its fixed part: window 100k × 0.5 → high 50k, low 25k; the first
    # report is 10k fixed + 1k → 11k, then +1k per reaction up to 51k at 41 reactions. A plain
    # average (51k / 41 ≈ 1.24k) would keep 20; charging the fixed part once keeps
    # floor((25k - 11k) / 1k) = 14, so the prompt after the cut is 10k + 15k = 25k — on the mark.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    reports = [10_000 + 1_000 * n for n in range(1, 42)] + [25_000]
    model = ProgrammableChatModel(respond=usage_responses(reports))
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        history_budget=0.5,
        context_window=100_000,
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.start()

    _run_text_steps(agent, world, private, [f"r{n}" for n in range(1, 43)])

    assert (
        len(_observation_texts(prompts[40])) == 41
    )  # nothing cut before the 41st report
    kept = _observation_texts(prompts[41])
    assert len(kept) == 15  # 14 kept by the cut + the new reaction 42
    assert kept[0].split("\n")[1] == "r28"

    agent.stop()


def test_history_budget_warns_when_the_fixed_part_leaves_no_room(
    loop, world, private, caplog
):
    # Fixed part ~400 (first report) against a low mark of 250: nothing fits; keep one, warn.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=usage_responses([400, 450, 520]))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        history_budget=0.5,
        context_window=1000,
    )
    agent.start()

    with caplog.at_level(logging.WARNING, logger="wica.agent"):
        _run_text_steps(agent, world, private, ["r1", "r2", "r3"])

    assert sum(isinstance(r, ObservationRecord) for r in agent._history) == 1
    assert any("leaves no room" in r.message for r in caplog.records)

    agent.stop()


def test_history_budget_ignores_calls_without_usage(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=usage_responses([None]))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        history_budget=0.1,
        context_window=10,
    )
    agent.start()

    _run_text_steps(agent, world, private, ["r1", "r2", "r3", "r4"])

    assert sum(isinstance(r, ObservationRecord) for r in agent._history) == 4

    agent.stop()


def test_history_budget_cannot_cut_a_single_reaction_and_warns(
    loop, world, private, caplog
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=usage_responses([900]))
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        history_budget=0.5,
        context_window=1000,
    )
    agent.start()

    with caplog.at_level(logging.WARNING, logger="wica.agent"):
        _run_text_steps(agent, world, private, ["r1"])

    assert sum(isinstance(r, ObservationRecord) for r in agent._history) == 1
    assert any(
        "history budget" in r.message and "cannot be cut" in r.message
        for r in caplog.records
    )

    agent.stop()


def test_history_budget_and_fixed_window_compose(loop, world, private):
    # The fixed cap still applies at step start (2X = 6 → keep 3) while the budget never trips.
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=usage_responses([10]))
    prompts: list[list[BaseMessage]] = []
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        private_text=private,
        history_reactions=3,
        history_budget=0.5,
        context_window=1000,
    )
    agent.on_prompt.subscribe(prompts.append)
    agent.start()

    _run_text_steps(agent, world, private, ["r1", "r2", "r3", "r4", "r5", "r6"])

    assert len(_observation_texts(prompts[4])) == 5
    assert len(_observation_texts(prompts[5])) == 3

    agent.stop()


def test_history_budget_requires_a_known_context_window(loop, world, private):
    model = ProgrammableChatModel(respond=text_response("hi"))  # profile: None
    with pytest.raises(ConfigError, match="context_window"):
        make_agent(model, world=world, loop=loop, history_budget=0.5)

    # The model profile supplies the window when the config does not…
    profiled = ProgrammableChatModel(
        respond=text_response("hi"), profile={"max_input_tokens": 2000}
    )
    agent = make_agent(profiled, world=world, loop=loop, history_budget=0.5)
    assert (agent._history_high, agent._history_low) == (1000, 500)
    # …and the config override wins over it.
    agent = make_agent(
        profiled, world=world, loop=loop, history_budget=0.5, context_window=400
    )
    assert (agent._history_high, agent._history_low) == (200, 100)
    # A window alone (no budget) leaves the budget off and needs no profile.
    agent = make_agent(model, world=world, loop=loop, context_window=400)
    assert agent._history_high is None


def test_history_budget_alone_turns_on_the_window_clause(loop, world, private):
    model = ProgrammableChatModel(respond=text_response("hi"))
    agent = make_agent(
        model, world=world, loop=loop, history_budget=0.5, context_window=1000
    )
    assert "most recent reactions" in agent.system_prompt


# --- prompt caching (specs/agent.md, "Prompt caching") -------------------------------------------


def _breakpoints(messages: list[BaseMessage]) -> list[int]:
    """Indexes of the messages carrying an Anthropic cache_control block."""
    return [
        i
        for i, m in enumerate(messages)
        if isinstance(m.content, list)
        and any(isinstance(b, dict) and "cache_control" in b for b in m.content)
    ]


def test_breakpoint_sits_before_the_newest_observation_and_copies():
    first = [
        SystemMessage(content="sys"),
        HumanMessage(content=[{"type": "text", "text": "o1"}]),
    ]
    marked = mark_cache_breakpoint(first)
    assert _breakpoints(marked) == [
        0
    ]  # first step: the system prompt (tools ahead of it)
    assert first[0].content == "sys"  # the caller's messages are untouched

    later = [
        *first,
        AIMessage(content="", tool_calls=[{"name": "say", "args": {}, "id": "c1"}]),
        ToolMessage(content="ack", tool_call_id="c1"),
        HumanMessage(content=[{"type": "text", "text": "o2"}]),
    ]
    marked = mark_cache_breakpoint(later)
    assert _breakpoints(marked) == [
        3
    ]  # the last tool_result, not the fresh observation
    assert marked[3].content == [
        {"type": "text", "text": "ack", "cache_control": {"type": "ephemeral"}}
    ]


@pytest.mark.parametrize(
    ("provider", "marked"), [("anthropic", True), ("openai", False)]
)
def test_only_anthropic_requests_carry_the_breakpoint(
    loop, world, private, provider, marked
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("add", {"a": 1, "b": 2}, "call1")]),
            text_response("The sum is 3"),
        )
    )
    config = AgentConfig(
        provider=provider, model="test", system_prompt="You are terse.", api_key="k"
    )
    agent = Agent(config, world=world, loop=loop, model=model)
    agent.on_private_text.subscribe(private)
    prompts: list[list[BaseMessage]] = []
    agent.on_prompt.subscribe(prompts.append)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent.register_command(add)
    agent.start()
    world.update("input", "add 1 and 2")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    sent = model.calls[1]  # the step the add completion woke
    tool_index = next(i for i, m in enumerate(sent) if isinstance(m, ToolMessage))
    assert _breakpoints(sent) == ([tool_index] if marked else [])
    assert all(
        _breakpoints(p) == [] for p in prompts
    )  # observers see the plain rendering


# --- Response guards: a rejected response is retried, only the accepted one is kept ------------
#
# See specs/agent.md ("Response guards"). The guards' own rules are in test_guards.py.


def _with_usage(message: AIMessage, input_tokens: int) -> AIMessage:
    message.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": 1,
        "total_tokens": input_tokens + 1,
    }
    return message


def _rejecting(word: str, *, show: bool) -> Callable[[ModelResponse], Retry | None]:
    """An application guard rejecting any text containing `word`."""

    def guard(response: ModelResponse) -> Retry | None:
        return (
            Retry(f"never say {word}", show_response=show)
            if word in response.text
            else None
        )

    guard.__name__ = f"no_{word}"
    return guard


def _texts(message: BaseMessage) -> list[str]:
    content = message.content
    if isinstance(content, str):
        return [content]
    return [b if isinstance(b, str) else str(b.get("text", "")) for b in content]


def test_a_rejected_response_is_retried_and_only_the_accepted_one_is_kept(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    answers = iter(
        [
            AIMessage(
                content="", tool_calls=[{"name": "ghost", "args": {}, "id": "g1"}]
            ),
            AIMessage(content="fine"),
            AIMessage(content="second"),
        ]
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        return _with_usage(next(answers), 10 * (len(messages)))

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    prompts: list[list[BaseMessage]] = []
    issued: list[CommandIssued] = []
    rejections: list[ResponseRejection] = []
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_prompt.subscribe(prompts.append)
    agent.on_command.subscribe(issued.append)
    agent.on_response_rejected.subscribe(rejections.append)
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("input", "summon")
    assert done.wait(timeout=WAIT_TIMEOUT)

    # Two model calls, one prompt event; on_private_text and on_command see only the accepted response.
    assert len(model.calls) == 2
    assert len(prompts) == 1
    assert private.texts == ["fine"]
    assert issued == []
    # The retry showed the rejected call, answered it, then said why.
    retry = model.calls[1]
    assert retry[: len(model.calls[0])] == model.calls[0]
    shown, result, correction = retry[len(model.calls[0]) :]
    assert isinstance(shown, AIMessage) and shown.tool_calls[0]["name"] == "ghost"
    assert isinstance(result, ToolMessage) and result.tool_call_id == "g1"
    assert result.content == REJECTED_TOOL_RESULT
    assert isinstance(correction, HumanMessage)
    assert "There is no command named 'ghost'" in correction.text
    [rejection] = rejections
    assert (rejection.attempt, rejection.show_response, rejection.will_retry) == (
        1,
        True,
        True,
    )
    assert rejection.response.commands[0].name == "ghost"
    [trace] = traces
    assert trace.outcome == "ok"
    assert trace.attempts == 2
    assert trace.rejections == (rejection.feedback,)
    assert trace.guard_duration is not None
    first, second = (10 * len(c) for c in model.calls)
    assert trace.usage == TokenUsage(first + second, 2, None)

    # The next step's prompt reads as if the model had answered right the first time.
    world.update("input", "again")
    wait_until(lambda: len(model.calls) == 3)
    next_prompt = model.calls[2]
    rendered = " ".join(t for m in next_prompt for t in _texts(m))
    assert "ghost" not in rendered and "rejected" not in rendered
    assert [m.text for m in next_prompt if isinstance(m, AIMessage)] == ["fine"]
    agent.stop()


def test_a_reminder_retry_hides_the_response_and_extends_the_newest_observation(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response("bad words"), text_response("kind words"))
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.add_response_guard(_rejecting("bad", show=False))
    agent.start()

    world.update("input", "hello")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert private.texts == ["kind words"]
    first, retry = model.calls
    assert len(retry) == len(first)
    assert retry[:-1] == first[:-1]
    assert not any(isinstance(m, AIMessage) for m in retry)
    assert _texts(retry[-1])[: len(_texts(first[-1]))] == _texts(first[-1])
    assert "never say bad" in _texts(retry[-1])[-1]
    assert _no_consecutive_human_messages(retry)
    agent.stop()


def test_several_rejections_make_one_correction_shown_if_any_asks(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response("bad and ugly"), text_response("nice"))
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.add_response_guard(_rejecting("bad", show=False))
    agent.add_response_guard(_rejecting("ugly", show=True))
    agent.start()

    world.update("input", "hello")
    assert private.event.wait(timeout=WAIT_TIMEOUT)

    retry = model.calls[1]
    assert isinstance(retry[-2], AIMessage) and retry[-2].text == "bad and ugly"
    assert "- never say bad\n- never say ugly" in retry[-1].text
    agent.stop()


def test_out_of_retries_the_reaction_ends_rejected_and_nothing_is_used(
    loop, world, private, caplog
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(respond=text_response("bad"))
    agent = make_agent(model, world=world, loop=loop, private_text=private)
    agent.add_response_guard(_rejecting("bad", show=True))
    rejections: list[ResponseRejection] = []
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_response_rejected.subscribe(rejections.append)
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    with caplog.at_level(logging.WARNING, logger="wica.agent"):
        world.update("input", "first")
        assert done.wait(timeout=WAIT_TIMEOUT)

    assert len(model.calls) == 2  # 1 + max_retries
    assert private.texts == []
    assert [r.will_retry for r in rejections] == [True, False]
    [trace] = traces
    assert (trace.outcome, trace.attempts, trace.error) == (
        "rejected",
        2,
        "never say bad",
    )
    levels = [r.levelno for r in caplog.records if "rejected" in r.getMessage()]
    assert levels == [logging.WARNING, logging.ERROR]

    # The rejected step left only its Observation, merged into the next prompt.
    model.respond = text_response("good")
    world.update("input", "second")
    assert private.event.wait(timeout=WAIT_TIMEOUT)
    last = model.calls[-1]
    assert not any(isinstance(m, AIMessage) for m in last)
    assert _no_consecutive_human_messages(last)
    assert "first" in last[-1].text and "second" in last[-1].text
    agent.stop()


def test_with_no_retries_the_first_rejection_ends_the_reaction(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=tool_call_response([("ghost", {}, "g1"), ("wave", {}, "w1")])
    )
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, max_retries=0
    )
    waved: list[str] = []

    async def wave() -> str:
        """Wave."""
        waved.append("waved")
        return "waved"

    agent.register_command(wave)
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("input", "go")
    assert done.wait(timeout=WAIT_TIMEOUT)
    time.sleep(0.1)

    assert len(model.calls) == 1
    assert traces[0].outcome == "rejected"
    # The valid call beside the unknown one was not used either.
    assert traces[0].command_call_ids == ()
    assert waved == []
    agent.stop()


def test_a_retry_past_its_deadline_ends_the_reaction_as_a_timeout(loop, world, private):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def slow(messages: list[BaseMessage]) -> AIMessage:
        await asyncio.sleep(1.0)
        return AIMessage(content="too late")

    model = ProgrammableChatModel(respond=sequence(text_response("bad"), slow))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, model_timeout=0.2
    )
    agent.add_response_guard(_rejecting("bad", show=True))
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()

    world.update("input", "go")
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert (traces[0].outcome, traces[0].attempts) == ("model_timeout", 2)
    assert private.texts == []
    agent.stop()


def test_a_raising_guard_accepts_and_an_async_guard_is_awaited(
    loop, world, private, caplog
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = ProgrammableChatModel(
        respond=sequence(text_response("bad"), text_response("good"))
    )
    agent = make_agent(model, world=world, loop=loop, private_text=private)

    def broken(response: ModelResponse) -> Retry | None:
        raise RuntimeError("guard bug")

    async def judge(response: ModelResponse) -> Retry | None:
        await asyncio.sleep(0)
        return (
            Retry("not that", show_response=False) if response.text == "bad" else None
        )

    agent.add_response_guard(broken)
    agent.add_response_guard(judge)
    agent.start()

    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "go")
        assert private.event.wait(timeout=WAIT_TIMEOUT)

    assert private.texts == ["good"]
    assert len(model.calls) == 2
    assert any("'broken' raised" in r.getMessage() for r in caplog.records)
    agent.stop()


def test_guards_are_listed_added_and_removed_by_name(loop, world):
    agent = make_agent(ProgrammableChatModel(), world=world, loop=loop)
    assert list(agent.response_guards) == [
        "incomplete_response",
        "unknown_command",
        "invalid_args",
        "output_command_required",
    ]

    agent.add_response_guard(lambda r: None, name="mine")
    agent.remove_response_guard("invalid_args")
    assert list(agent.response_guards) == [
        "incomplete_response",
        "unknown_command",
        "output_command_required",
        "mine",
    ]
    with pytest.raises(ValueError, match="already added"):
        agent.add_response_guard(lambda r: None, name="mine")
    with pytest.raises(ValueError, match="no response guard"):
        agent.remove_response_guard("invalid_args")


# --- Streaming the model call (specs/agent.md "Streaming the model call") ------------------------


def _tool_chunk(
    index: int, args: str, *, name: str | None = None, call_id: str | None = None
) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            tool_call_chunk(name=name, args=args, id=call_id, index=index)
        ],
    )


def _usage_chunk(input_tokens: int, output_tokens: int) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def _sum(chunks: Sequence[AIMessageChunk]) -> AIMessageChunk:
    summed = chunks[0]
    for chunk in chunks[1:]:
        summed = summed + chunk
    return summed


type Stream = Callable[[], AsyncIterator[AIMessageChunk]]


def chunks_stream(*chunks: AIMessageChunk) -> Stream:
    async def stream() -> AsyncIterator[AIMessageChunk]:
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    return stream


class StreamingChatModel(BaseChatModel):
    """A streaming double: each call plays the next scripted stream of chunks (the last one
    repeats). Like ProgrammableChatModel, injected through the Agent's `model=` seam."""

    streams: list[Stream] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-stream"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError("StreamingChatModel only streams")

    async def _astream(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        index = min(len(self.calls), len(self.streams) - 1)
        self.calls.append(messages)
        async for chunk in self.streams[index]():
            yield ChatGenerationChunk(message=chunk)

    def bind_tools(
        self, tools, *, tool_choice=None, **kwargs: Any
    ) -> Runnable[Any, AIMessage]:
        return self.bind(tool_choice=tool_choice)


def test_assemble_sums_split_text_tool_calls_and_usage():
    summed = _sum(
        [
            AIMessageChunk(content="Turning it "),
            AIMessageChunk(content="red."),
            _tool_chunk(0, '{"col', name="set_light", call_id="c1"),
            _tool_chunk(0, 'or": "red"}'),
            _tool_chunk(1, "", name="wave", call_id="c2"),
            _usage_chunk(30, 12),
        ]
    )

    message = _assemble(summed)

    assert type(message) is AIMessage
    assert message.text == "Turning it red."
    assert message.tool_calls == [
        {
            "name": "set_light",
            "args": {"color": "red"},
            "id": "c1",
            "type": "tool_call",
        },
        {"name": "wave", "args": {}, "id": "c2", "type": "tool_call"},
    ]
    assert message.invalid_tool_calls == []
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] == 30
    assert message.usage_metadata["output_tokens"] == 12


def test_assemble_drops_a_call_cut_mid_arguments_that_langchain_would_close(caplog):
    summed = _sum(
        [
            _tool_chunk(0, '{"text": "Once upon', name="say", call_id="c1"),
            _tool_chunk(0, " a time, in"),
        ]
    )
    # LangChain's own lenient assembly turns the cut call into a valid one.
    assert summed.tool_calls[0]["args"] == {"text": "Once upon a time, in"}

    with caplog.at_level(logging.WARNING, logger="wica"):
        message = _assemble(summed)

    assert message.tool_calls == []
    [invalid] = message.invalid_tool_calls
    assert invalid["name"] == "say"
    assert invalid["args"] == '{"text": "Once upon a time, in'
    assert invalid["id"] == "c1"
    assert invalid["error"] is not None and "not complete JSON" in invalid["error"]
    assert "malformed tool call 'say'" in caplog.text


def test_assemble_rejects_arguments_that_are_not_an_object():
    message = _assemble(_tool_chunk(0, "[1, 2]", name="say", call_id="c1"))
    assert message.tool_calls == []
    [invalid] = message.invalid_tool_calls
    assert invalid["error"] == "arguments are a JSON list, not an object"


def test_assemble_keeps_list_content_such_as_a_thinking_block():
    thinking = {"type": "thinking", "thinking": "hm", "signature": "s", "index": 0}
    message = _assemble(_sum([AIMessageChunk(content=[thinking])]))
    assert message.content == [thinking]


def _streaming_agent(
    streams: Sequence[Stream], loop, world, private, **kwargs: Any
) -> tuple[Agent, StreamingChatModel, list[ReactionTrace], threading.Event]:
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    model = StreamingChatModel(streams=list(streams))
    agent = make_agent(
        model, world=world, loop=loop, private_text=private, coalesce_window=0, **kwargs
    )
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    return agent, model, traces, done


def test_a_streamed_response_is_used_once_whole(loop, world, private):
    """Split text is reported once, whole; a split tool call is dispatched once with all its
    arguments; the usage from the last chunk lands in the reaction record."""
    lights: list[str] = []

    async def set_light(color: str) -> str:
        """Set the light's color."""
        lights.append(color)
        return "ok"

    agent, _, traces, done = _streaming_agent(
        [
            chunks_stream(
                AIMessageChunk(content="Turning "),
                AIMessageChunk(content="it red."),
                _tool_chunk(0, '{"color"', name="set_light", call_id="c1"),
                _tool_chunk(0, ': "red"}'),
                _usage_chunk(40, 9),
            ),
            chunks_stream(AIMessageChunk(content="<noop/>")),
        ],
        loop,
        world,
        private,
    )
    agent.register_command(set_light)
    agent.start()

    world.update("input", "make it red")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: lights == ["red"])
    agent.stop()

    assert private.texts == ["Turning it red."]
    assert traces[0].outcome == "ok"
    assert traces[0].usage == TokenUsage(40, 9, None)
    assert len(traces[0].command_call_ids) == 1


def test_a_streamed_call_cut_by_the_cap_is_not_dispatched_and_is_retried(
    loop, world, private
):
    """A call whose arguments were cut is malformed: not executed, and the incomplete_response
    guard asks for it again — as a reminder, since the broken call cannot be replayed."""
    notes: list[str] = []

    async def write_note(text: str) -> str:
        """Write a note."""
        notes.append(text)
        return "written"

    agent, model, traces, done = _streaming_agent(
        [
            chunks_stream(
                _tool_chunk(
                    0, '{"text": "Once upon a ti', name="write_note", call_id="s1"
                )
            ),
            chunks_stream(
                _tool_chunk(
                    0, '{"text": "Once upon a time."}', name="write_note", call_id="s2"
                )
            ),
            chunks_stream(AIMessageChunk(content="<noop/>")),
        ],
        loop,
        world,
        private,
    )
    agent.register_command(write_note)
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()

    world.update("input", "note the story")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(
        lambda: len(traces) == 2
    )  # the note's completion woke a step that did nothing
    agent.stop()

    assert notes == ["Once upon a time."]
    assert traces[0].outcome == "ok"
    assert traces[0].attempts == 2
    assert len(model.calls) == 3
    [rejection] = rejections
    assert rejection.show_response is False
    [malformed] = rejection.response.malformed
    assert (malformed.name, malformed.raw_args) == (
        "write_note",
        '{"text": "Once upon a ti',
    )
    assert "not complete JSON" in malformed.error
    assert rejection.feedback.startswith("Your call to `write_note` was not executed")
    retry_prompt = model.calls[1]
    assert not any(isinstance(m, AIMessage) for m in retry_prompt)
    assert "Your call to `write_note` was not executed" in human_texts(retry_prompt[-1])


def test_an_output_is_delivered_while_the_response_is_still_streaming(
    loop, world, private
):
    """The output Command gets the first words before the model has finished: the entry opens
    with an empty text, is filled once the body is complete, and first_output_at stamps it."""
    heard: list[str] = []
    first_piece_at: list[Any] = []
    text_at_first_piece: list[str] = []

    async def say(text: OutputStream) -> str:
        """Speak out loud."""
        async for piece in text:
            if not heard:
                first_piece_at.append(now())
                text_at_first_piece.append(world.get("agent:command:c1").args["text"])
            heard.append(piece)
        return "said"

    agent, _, traces, done = _streaming_agent(
        [
            _slow_stream(
                AIMessageChunk(content="hm <say>Once"),
                0.15,
                AIMessageChunk(content=" upon a time.</say>"),
            )
        ],
        loop,
        world,
        private,
        output_command=say,
    )
    issued: list[CommandIssued] = []
    ended: list[CommandTrace] = []
    agent.on_command.subscribe(issued.append)
    agent.on_command_ended.subscribe(ended.append)
    agent.start()
    world.update("input", "tell me a story")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: [t.state for t in ended] == ["complete"])
    agent.stop()

    trace = traces[0]
    assert "".join(heard) == "Once upon a time."
    assert heard[0] == "Once"
    assert text_at_first_piece == [""]
    assert issued == [
        CommandIssued("say", {"text": "Once upon a time."}, "c1", is_output=True)
    ]
    assert private.texts == ["hm"]
    assert trace.command_call_ids == ("c1",)
    assert trace.first_content_at is not None and trace.model_ended_at is not None
    assert trace.first_output_at is not None
    assert trace.first_content_at <= trace.first_output_at < trace.model_ended_at
    assert first_piece_at[0] < trace.model_ended_at
    assert (trace.model_ended_at - trace.first_output_at).total_seconds() >= 0.1


def test_a_response_that_spoke_drops_a_cut_call_without_a_retry(loop, world, private):
    """Once an output has started the response cannot be taken back: no guard runs, the words
    are kept, and the malformed call beside them is dropped."""
    spoken: list[str] = []
    lights: list[str] = []

    async def set_light(color: str) -> str:
        """Set the light's color."""
        lights.append(color)
        return "ok"

    agent, model, traces, done = _streaming_agent(
        [
            chunks_stream(
                AIMessageChunk(content="<say>On it.</say>"),
                _tool_chunk(0, '{"color": "re', name="set_light", call_id="c1"),
            )
        ],
        loop,
        world,
        private,
        output_command=_say_command(spoken),
    )
    agent.register_command(set_light)
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()

    world.update("input", "make it red")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: spoken == ["On it."])
    agent.stop()

    assert lights == []
    assert rejections == []
    assert traces[0].outcome == "ok"
    assert traces[0].attempts == 1
    assert len(traces[0].command_call_ids) == 1  # the output alone
    assert len(model.calls) == 1


def _failing_after_first_chunk() -> AsyncIterator[AIMessageChunk]:
    async def stream() -> AsyncIterator[AIMessageChunk]:
        yield AIMessageChunk(content="Hel")
        raise ConnectionError("stream broken")

    return stream()


def _stalling_after_first_chunk() -> AsyncIterator[AIMessageChunk]:
    async def stream() -> AsyncIterator[AIMessageChunk]:
        yield AIMessageChunk(content="Hel")
        await asyncio.Event().wait()
        yield AIMessageChunk(content="lo")

    return stream()


def _empty_stream() -> AsyncIterator[AIMessageChunk]:
    async def stream() -> AsyncIterator[AIMessageChunk]:
        return
        yield  # an async generator that yields nothing

    return stream()


@pytest.mark.parametrize(
    ("stream", "outcome", "error"),
    [
        (_failing_after_first_chunk, "model_error", "stream broken"),
        (
            _stalling_after_first_chunk,
            "model_timeout",
            "model call exceeded its 0.1 s deadline",
        ),
        (_empty_stream, "model_error", "No generation chunks were returned"),
    ],
)
def test_a_stream_that_does_not_complete_discards_its_partial_message(
    loop, world, private, stream, outcome, error
):
    agent, _, traces, done = _streaming_agent(
        [stream], loop, world, private, model_timeout=0.1
    )
    agent.start()

    world.update("input", "hello")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    assert traces[0].outcome == outcome
    assert traces[0].error is not None and error in traces[0].error
    assert private.texts == []
    assert not any(isinstance(r, AssistantTextRecord) for r in agent._history)


def test_without_the_malformed_guard_a_cut_call_is_dropped_and_the_rest_used(
    loop, world, private
):
    agent, model, traces, done = _streaming_agent(
        [
            chunks_stream(
                AIMessageChunk(content="On it."),
                _tool_chunk(0, '{"color": "re', name="set_light", call_id="c1"),
            )
        ],
        loop,
        world,
        private,
    )
    lights: list[str] = []

    async def set_light(color: str) -> str:
        """Set the light's color."""
        lights.append(color)
        return "ok"

    agent.register_command(set_light)
    agent.remove_response_guard("incomplete_response")
    agent.start()

    world.update("input", "make it red")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    assert private.texts == ["On it."]
    assert lights == []
    assert traces[0].outcome == "ok"
    assert traces[0].attempts == 1
    assert traces[0].command_call_ids == ()
    assert len(model.calls) == 1


def test_a_response_with_a_malformed_call_is_never_shown_in_a_retry(
    loop, world, private
):
    """A guard asking to show the rejected response gets a reminder instead when the response
    holds a malformed call: the broken call cannot be replayed to a provider."""
    agent, model, traces, done = _streaming_agent(
        [
            chunks_stream(
                AIMessageChunk(content="On it."),
                _tool_chunk(0, '{"color": "re', name="set_light", call_id="c1"),
            ),
            chunks_stream(AIMessageChunk(content="Done.")),
        ],
        loop,
        world,
        private,
    )
    agent.remove_response_guard("incomplete_response")
    agent.add_response_guard(
        lambda r: Retry("Say it differently.") if r.malformed else None,
        name="shows",
    )
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()

    world.update("input", "make it red")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    assert private.texts == ["Done."]
    assert rejections[0].show_response is False
    retry_prompt = model.calls[1]
    assert not any(isinstance(m, AIMessage) for m in retry_prompt)
    assert "Say it differently." in human_texts(retry_prompt[-1])


def _slow_stream(*steps: AIMessageChunk | float) -> Stream:
    """A stream that sleeps for each number and yields each chunk."""

    async def stream() -> AsyncIterator[AIMessageChunk]:
        for step in steps:
            if isinstance(step, AIMessageChunk):
                yield step
            else:
                await asyncio.sleep(step)

    return stream


def test_first_content_is_stamped_at_the_first_chunk_carrying_an_answer(
    loop, world, private
):
    """A leading chunk with only metadata or reasoning does not count; the stamp splits the model
    latency into the wait and the arrival."""
    thinking = {"type": "thinking", "thinking": "hm", "index": 0}
    agent, _, traces, done = _streaming_agent(
        [
            _slow_stream(
                AIMessageChunk(content=""),
                AIMessageChunk(content=[thinking]),
                0.1,
                AIMessageChunk(content="Hel"),
                0.1,
                AIMessageChunk(content="lo."),
            )
        ],
        loop,
        world,
        private,
    )
    agent.start()
    world.update("input", "hello")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    trace = traces[0]
    assert private.texts == ["Hello."]
    assert trace.model_started_at is not None and trace.model_ended_at is not None
    assert trace.first_content_at is not None
    assert trace.model_started_at < trace.first_content_at < trace.model_ended_at
    assert trace.time_to_first_content is not None and trace.generation_time is not None
    assert trace.time_to_first_content >= 0.05
    assert trace.generation_time >= 0.05


def test_first_content_counts_a_tool_call_and_is_the_last_attempts(
    loop, world, private
):
    """A tool-call chunk is content. After a retry the stamp is the accepted attempt's, so
    generation_time is the used response's and time_to_first_content includes the rejected one."""
    notes: list[str] = []

    async def write_note(text: str) -> str:
        """Write a note."""
        notes.append(text)
        return "written"

    agent, _, traces, done = _streaming_agent(
        [
            # Rejected by unknown_command, after 0.1 s.
            _slow_stream(_tool_chunk(0, "{}", name="ghost", call_id="g1"), 0.1),
            _slow_stream(
                0.1,
                _tool_chunk(0, '{"text": "Hello', name="write_note", call_id="s1"),
                _tool_chunk(0, ' there."}'),
            ),
            chunks_stream(AIMessageChunk(content="<noop/>")),
        ],
        loop,
        world,
        private,
    )
    agent.register_command(write_note)
    agent.start()
    world.update("input", "greet")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: notes == ["Hello there."])
    agent.stop()

    trace = traces[0]
    assert trace.attempts == 2
    assert trace.time_to_first_content is not None and trace.generation_time is not None
    # First attempt (0.1 s) + the second's wait before its tool call (0.1 s).
    assert trace.time_to_first_content >= 0.1
    assert trace.generation_time < trace.time_to_first_content / 2


def test_first_content_survives_a_failed_stream_and_is_none_when_nothing_arrived(
    loop, world, private
):
    agent, _, traces, done = _streaming_agent(
        [_failing_after_first_chunk, _empty_stream], loop, world, private
    )
    agent.start()
    world.update("input", "one")
    assert done.wait(timeout=WAIT_TIMEOUT)
    done.clear()
    world.update("input", "two")
    wait_until(lambda: len(traces) == 2)
    agent.stop()

    assert traces[0].outcome == "model_error"
    assert traces[0].first_content_at is not None
    assert traces[1].outcome == "model_error"
    assert traces[1].first_content_at is None
    assert traces[1].time_to_first_content is None


def test_a_model_without_a_streaming_path_stamps_first_content_at_its_whole_message(
    loop, world, private
):
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        await asyncio.sleep(0.1)
        return AIMessage(content="hi")

    agent = make_agent(
        ProgrammableChatModel(respond=respond),
        world=world,
        loop=loop,
        private_text=private,
    )
    traces: list[ReactionTrace] = []
    done = threading.Event()
    agent.on_reaction_ended.subscribe(_recorder(traces, done))
    agent.start()
    world.update("input", "hello")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    trace = traces[0]
    assert trace.time_to_first_content is not None and trace.generation_time is not None
    assert trace.time_to_first_content >= 0.05
    assert trace.generation_time < trace.time_to_first_content / 2


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"stop_reason": "max_tokens"}, True),  # Anthropic
        ({"finish_reason": "length"}, True),  # OpenAI, Mistral, the Hub
        ({"finish_reason": "MAX_TOKENS"}, True),  # Gemini
        ({"finish_reason": "stop"}, False),
        ({"stop_reason": "tool_use"}, False),
        ({"finish_reason": "tool_calls"}, False),
        ({}, False),
    ],
)
def test_a_response_is_truncated_when_its_stop_reason_is_the_output_cap(
    metadata, expected
):
    assert (
        _was_truncated(AIMessage(content="x", response_metadata=metadata)) is expected
    )


def _cut_at_the_cap(tokens: int) -> AIMessageChunk:
    """The closing chunk of a response the provider stopped at its output cap."""
    return AIMessageChunk(
        content="",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": tokens,
            "total_tokens": 10 + tokens,
        },
    )


def test_a_well_formed_call_in_a_response_cut_at_the_cap_is_retried(
    loop, world, private
):
    """A provider may return a cut call well-formed, with empty arguments. The stop reason marks
    the response truncated: incomplete_response alone corrects it (invalid_args stays silent
    about the cut call), as a reminder, with the cap as the length to stay under."""
    notes: list[str] = []

    async def write_note(text: str) -> str:
        """Write a note."""
        notes.append(text)
        return "written"

    agent, model, traces, done = _streaming_agent(
        [
            chunks_stream(
                _tool_chunk(0, "", name="write_note", call_id="s1"), _cut_at_the_cap(60)
            ),
            chunks_stream(
                _tool_chunk(
                    0, '{"text": "A robot woke up."}', name="write_note", call_id="s2"
                )
            ),
            chunks_stream(AIMessageChunk(content="<noop/>")),
        ],
        loop,
        world,
        private,
    )
    agent.register_command(write_note)
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()

    world.update("input", "note a story")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(
        lambda: len(traces) == 2
    )  # the note's completion woke a step that did nothing
    agent.stop()

    assert notes == ["A robot woke up."]
    assert traces[0].attempts == 2
    [rejection] = rejections
    assert rejection.response.truncated is True
    assert rejection.response.output_tokens == 60
    assert rejection.show_response is False
    assert rejection.feedback.startswith(
        "Your response was cut off at the output length limit while you were writing the call "
        "to `write_note`"
    )
    assert "keep it under 60 characters" in rejection.feedback
    assert "invalid" not in rejection.feedback
    assert not any(isinstance(m, AIMessage) for m in model.calls[1])


def test_text_cut_at_the_cap_is_accepted(loop, world, private):
    agent, model, traces, done = _streaming_agent(
        [chunks_stream(AIMessageChunk(content="Once upon a"), _cut_at_the_cap(4))],
        loop,
        world,
        private,
    )
    agent.start()
    world.update("input", "tell me a story")
    assert done.wait(timeout=WAIT_TIMEOUT)
    agent.stop()

    assert private.texts == ["Once upon a"]
    assert traces[0].outcome == "ok"
    assert traces[0].attempts == 1


def test_an_output_cut_at_the_cap_is_delivered_as_far_as_it_got(loop, world, private):
    """The cap closes the tag: what was written is said, and nothing is retried."""
    spoken: list[str] = []
    agent, model, traces, done = _streaming_agent(
        [chunks_stream(AIMessageChunk(content="<say>Once upon a"), _cut_at_the_cap(4))],
        loop,
        world,
        private,
        output_command=_say_command(spoken),
    )
    agent.start()
    world.update("input", "tell me a story")
    assert done.wait(timeout=WAIT_TIMEOUT)
    wait_until(lambda: spoken == ["Once upon a"])
    agent.stop()

    assert traces[0].outcome == "ok"
    assert traces[0].attempts == 1
    assert len(model.calls) == 1
    assert agent._history[1:] == [AssistantTextRecord("<say>Once upon a")]


# --- the output Command as a tool call (specs/agent.md, "The output Command as a tool call") ---


def _tool_mode_agent(
    model: ProgrammableChatModel,
    loop: asyncio.AbstractEventLoop,
    world: World,
    output: Command,
    *,
    force: bool = True,
    max_next_steps: int | None = 1,
    commands: Sequence[Callable[..., Any] | Command] = (),
) -> Agent:
    world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    agent = make_agent(
        model,
        world=world,
        loop=loop,
        coalesce_window=0,
        max_next_steps=max_next_steps,
    )
    for command in commands:
        agent.register_command(command)
    agent.set_output_command(output, as_tool=True, force=force)
    return agent


def _emotional_say(spoken: list[tuple[str, str]], delay: float = 0.0) -> Command:
    async def say(
        text: OutputStream, emotion: Literal["happy", "sad", "neutral"] = "neutral"
    ) -> str:
        """Say something to the user."""
        words = await text.read()
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            text.heard = words[:5]
            raise
        spoken.append((words, emotion))
        return "spoken"

    return Command(say)


def test_tool_mode_binds_the_output_and_noop_forced_and_delivers_a_call(loop, world):
    spoken: list[tuple[str, str]] = []
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("say", {"text": "Hi!", "emotion": "happy"}, "t1")]),
            tool_call_response([("noop", {}, "t2")]),
        )
    )
    agent = _tool_mode_agent(
        model, loop, world, _emotional_say(spoken), commands=(_named("wave"),)
    )
    issued: list[CommandIssued] = []
    traces: list[ReactionTrace] = []
    agent.on_command.subscribe(issued.append)
    agent.on_reaction_ended.subscribe(traces.append)
    agent.start()
    world.update("input", "hello")
    wait_until(lambda: spoken == [("Hi!", "happy")] and len(traces) == 2)

    assert model.bindings[0]["tool_choice"] == "any"
    assert sorted(model.bindings[0]["tools"]) == [
        "cancel_command",
        "noop",
        "say",
        "wave",
    ]
    assert sorted(agent.commands) == [
        "cancel_command",
        "wave",
    ]  # WICA's tools are not listed
    assert issued == [
        CommandIssued("say", {"emotion": "happy", "text": "Hi!"}, "c1", is_output=True)
    ]
    # The end of the output woke a step (the second call), which declined with the noop tool.
    assert [(t.noop, t.command_call_ids) for t in traces] == [
        (False, ("c1",)),
        (True, ()),
    ]
    # History: native tool calls, each with its tool result — noop's a plain acknowledgement.
    third = agent._render_messages()
    calls = [m for m in third if isinstance(m, AIMessage)]
    assert [c.tool_calls[0]["name"] for c in calls] == ["say", "noop"]
    acks = [m.content for m in third if isinstance(m, ToolMessage)]
    assert acks[1] == "Acknowledged — no action taken."
    agent.stop()


def test_tool_mode_unforced_with_force_false(loop, world):
    model = ProgrammableChatModel(respond=tool_call_response([("noop", {}, "t1")]))
    agent = _tool_mode_agent(model, loop, world, _emotional_say([]), force=False)
    agent.start()
    world.update("input", "hello")
    wait_until(lambda: len(model.bindings) == 1)
    assert model.bindings[0]["tool_choice"] is None
    assert "say" in model.bindings[0]["tools"]
    agent.stop()


def test_tool_mode_wakes_after_each_output_under_the_cap(loop, world):
    spoken: list[tuple[str, str]] = []
    say_again = tool_call_response([("say", {"text": "More."}, "t")])
    model = ProgrammableChatModel(respond=sequence(say_again, say_again, say_again))
    agent = _tool_mode_agent(
        model, loop, world, _emotional_say(spoken), max_next_steps=1
    )
    ended: list[CommandTrace] = []
    agent.on_command_ended.subscribe(ended.append)
    agent.start()
    world.update("input", "talk")
    wait_until(lambda: len(ended) == 2)
    time.sleep(0.2)
    # The first output woke one step; that step's output did not wake another (the cap).
    assert len(model.calls) == 2
    assert [(t.is_output, t.triggers_step) for t in ended] == [
        (True, True),
        (True, False),
    ]
    agent.stop()


def test_tool_mode_rejects_a_plain_text_reply_and_retries(loop, world):
    spoken: list[tuple[str, str]] = []
    model = ProgrammableChatModel(
        respond=sequence(
            text_response("Hello there!"),
            tool_call_response([("say", {"text": "Hello there!"}, "t1")]),
            tool_call_response([("noop", {}, "t2")]),
        )
    )
    agent = _tool_mode_agent(model, loop, world, _emotional_say(spoken))
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()
    world.update("input", "hello")
    wait_until(lambda: spoken == [("Hello there!", "neutral")])
    assert [r.feedback for r in rejections] == [
        "Your response called no command, so the user heard nothing: your free text is "
        "private. Call `say` to reply to the user, or `noop` to do nothing."
    ]
    agent.stop()


def test_tool_mode_rejects_an_attribute_outside_its_values(loop, world):
    spoken: list[tuple[str, str]] = []
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("say", {"text": "Hi", "emotion": "angry"}, "t1")]),
            tool_call_response([("say", {"text": "Hi", "emotion": "sad"}, "t2")]),
            tool_call_response([("noop", {}, "t3")]),
        )
    )
    agent = _tool_mode_agent(model, loop, world, _emotional_say(spoken))
    rejections: list[ResponseRejection] = []
    agent.on_response_rejected.subscribe(rejections.append)
    agent.start()
    world.update("input", "hello")
    wait_until(lambda: spoken == [("Hi", "sad")])
    assert len(rejections) == 1 and "emotion" in rejections[0].feedback
    agent.stop()


def test_tool_mode_cut_off_output_keeps_what_was_heard_in_its_call(loop, world):
    spoken: list[tuple[str, str]] = []
    model = ProgrammableChatModel(
        respond=sequence(
            tool_call_response([("say", {"text": "Once upon a time."}, "t1")]),
            tool_call_response([("say", {"text": "Okay."}, "t2")]),
            tool_call_response([("noop", {}, "t3")]),
        )
    )
    agent = _tool_mode_agent(
        model, loop, world, _emotional_say(spoken, delay=0.5), max_next_steps=0
    )
    agent.start()
    world.update("input", "story")
    wait_until(lambda: len(model.calls) == 1)
    time.sleep(0.1)
    world.update("input", "stop")  # its say cuts off the running story (one voice)
    wait_until(lambda: spoken == [("Okay.", "neutral")])
    calls = [
        m.tool_calls[0]
        for m in agent._render_messages()
        if isinstance(m, AIMessage) and m.tool_calls
    ]
    assert [c["args"]["text"] for c in calls] == ["Once ", "Okay."]
    agent.stop()


def test_tool_mode_primer_teaches_the_call_and_the_noop_tool(loop, world):
    agent = _tool_mode_agent(ProgrammableChatModel(), loop, world, _emotional_say([]))
    prompt = agent.system_prompt
    assert "must be said by calling say" in prompt
    assert "call noop to do nothing" in prompt
    assert "<say>" not in prompt and "<next>" not in prompt and "<noop/>" not in prompt


def test_switching_the_output_mode_while_running_rebinds_the_model(loop, world):
    model = ProgrammableChatModel(respond=text_response("<noop/>"))
    agent = _tool_mode_agent(model, loop, world, _emotional_say([]))
    agent.set_output_command(_emotional_say([]))  # back to tag mode before start
    agent.start()
    world.update("input", "one")
    wait_until(lambda: len(model.calls) == 1)
    assert "say" not in (model.bindings[0]["tools"] or [])
    assert model.bindings[0]["tool_choice"] is None
    model.respond = tool_call_response([("noop", {}, "t1")])
    agent.set_output_command(_emotional_say([]), as_tool=True)
    world.update("input", "two")
    wait_until(lambda: len(model.calls) == 2)
    assert "say" in model.bindings[1]["tools"]
    assert model.bindings[1]["tool_choice"] == "any"
    agent.stop()
