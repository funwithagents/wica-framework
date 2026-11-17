from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from wica import Command, CommandIssued, Wica
from wica.config import AgentConfig, ConfigError, WicaConfig
from wica.content import Content, TextPart
from wica.instrumentation import CommandTrace, ReactionTrace
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


class RecordingVoice:
    """A recording output Command: `say` receives the whole text of each output."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.event = threading.Event()

    async def say(self, text: str) -> str:
        """Say something to the user."""
        self.texts.append(text)
        self.event.set()
        return "said"


@pytest.fixture
def wica_factory() -> Iterator[Callable[..., Wica]]:
    created: list[Wica] = []

    def _make(
        config: WicaConfig, *, voice: RecordingVoice | None = None, **kwargs: Any
    ) -> Wica:
        # The output Command is wired after construction (build, then wire, then start — see
        # specs/wica.md, "Output wiring is delegated"); the factory just folds that step in.
        w = Wica.init(config, **kwargs)
        if voice is not None:
            w.set_output_command(voice.say)
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
    """Build, then wire, then start: the output Command is set on the facade after construction
    (so the objects it belongs to can be built against the Wica first) and reaches the Agent —
    the body of its tag goes through it, the text outside is reported on on_agent_private_text,
    and its name is taken from register_command."""
    wica = wica_factory(
        fake_config([{"text": "thinking <speak>hello</speak>"}]),
        coalesce_window=0,
    )
    spoken: list[str] = []
    spoke = threading.Event()
    private: list[str] = []
    thought = threading.Event()

    async def speak(text: str) -> str:
        """Say something to the user."""
        spoken.append(text)
        spoke.set()
        return "spoken"

    def on_private(text: str) -> None:
        private.append(text)
        thought.set()

    wica.set_output_command(speak)
    wica.on_agent_private_text.subscribe(on_private)
    assert wica.agent.output_command_name == "speak"
    with pytest.raises(ValueError, match="output Command"):
        wica.register_command(speak)
    wica.world.register(
        "input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    wica.world.update("input", "greet")

    assert spoke.wait(timeout=WAIT_TIMEOUT)
    assert thought.wait(timeout=WAIT_TIMEOUT)
    assert spoken == ["hello"]
    assert private == ["thinking"]


def test_set_output_command_validates_and_clears_through_the_facade(wica_factory):
    wica = wica_factory(fake_config())

    def needs_an_attribute(text: str, emotion: str) -> None:
        """An attribute without a default: not an output's signature."""

    def speak(text: str) -> None:
        """Say something to the user."""

    with pytest.raises(ValueError, match="output Command"):
        wica.set_output_command(needs_an_attribute)
    with pytest.raises(ValueError, match="reserved"):
        wica.set_output_command(Command(speak, name="next"))
    assert wica.agent.output_command_name is None

    wica.set_output_command(speak)
    assert wica.agent.output_command_name == "speak"
    assert "<speak>" in wica.agent.system_prompt
    wica.set_output_command(None)
    assert wica.agent.output_command_name is None
    assert "<speak>" not in wica.agent.system_prompt
    # The output sink is gone: the output Command is the one output path.
    assert not hasattr(wica, "set_output_sink")


def test_surfaced_events_are_the_same_objects(wica_factory):
    wica = wica_factory(fake_config())
    # Wica surfaces the underlying Events directly — no adapters, no emit shims.
    assert wica.on_world_trigger is wica.world.on_trigger
    assert wica.on_agent_trigger is wica.agent.on_trigger
    assert wica.on_agent_prompt is wica.agent.on_prompt
    assert wica.on_agent_command is wica.agent.on_command
    assert wica.on_agent_private_text is wica.agent.on_private_text
    assert not hasattr(wica, "on_agent_text")


def test_facade_surfaces_the_instrumentation_events(wica_factory):
    wica = wica_factory(fake_config())
    assert wica.on_agent_reaction_ended is wica.agent.on_reaction_ended
    assert wica.on_agent_trigger_dropped is wica.agent.on_trigger_dropped
    assert wica.on_agent_command_ended is wica.agent.on_command_ended
    assert wica.on_agent_response_rejected is wica.agent.on_response_rejected


def test_response_guards_are_delegated_to_the_agent(wica_factory):
    wica = wica_factory(fake_config())
    assert list(wica.response_guards) == [
        "incomplete_response",
        "unknown_command",
        "invalid_args",
        "output_command_required",
        "garbled_text",
    ]
    wica.remove_response_guard("invalid_args")
    wica.add_response_guard(lambda response: None, name="mine")
    assert list(wica.response_guards) == list(wica.agent.response_guards)
    assert list(wica.response_guards) == [
        "incomplete_response",
        "unknown_command",
        "output_command_required",
        "garbled_text",
        "mine",
    ]


def test_fake_flow_yields_reaction_and_command_traces(wica_factory):
    wica = wica_factory(
        fake_config(
            [
                {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 2}}]},
                {"text": "done"},
            ]
        ),
        coalesce_window=0,
    )

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    wica.register_command(add)
    reaction_traces: list[ReactionTrace] = []
    command_traces: list[CommandTrace] = []
    done = threading.Event()

    def on_reaction(t: ReactionTrace) -> None:
        reaction_traces.append(t)
        if len(reaction_traces) == 2:
            done.set()

    wica.on_agent_reaction_ended.subscribe(on_reaction)
    wica.on_agent_command_ended.subscribe(command_traces.append)
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()

    wica.world.update("speech", "add them")
    assert done.wait(timeout=WAIT_TIMEOUT)

    assert [t.reaction_id for t in reaction_traces] == [1, 2]
    assert len(command_traces) == 1
    assert command_traces[0].reaction_id == 1
    assert command_traces[0].state == "complete"


def test_register_command_reaches_the_agent(wica_factory):
    wica = wica_factory(fake_config())

    async def wave() -> str:
        """Wave hello."""
        return "waved"

    wica.register_command(wave)
    assert "wave" in wica.agent._commands


def test_input_drives_world_and_agent_triggers_and_a_command_event(wica_factory):
    wica = wica_factory(
        fake_config([{"tool_calls": [{"name": "wave", "args": {}}]}, {"text": "hi"}]),
    )

    async def wave() -> str:
        """Wave hello."""
        return "waved"

    world_triggers: list[str] = []
    agent_triggers: list[str] = []
    commands: list[CommandIssued] = []
    texts: list[str] = []
    thought = threading.Event()

    def on_private(text: str) -> None:
        texts.append(text)
        thought.set()

    wica.on_agent_private_text.subscribe(on_private)
    wica.on_world_trigger.subscribe(lambda e: world_triggers.append(e.key))
    wica.on_agent_trigger.subscribe(lambda e: agent_triggers.append(e.key))
    wica.on_agent_command.subscribe(commands.append)

    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.register_command(wave)
    wica.start()

    wica.world.update("speech", "hello robot")
    assert thought.wait(timeout=WAIT_TIMEOUT)

    # The raw World trigger fired for the input; the filtered Agent trigger observed it too.
    assert "speech" in world_triggers
    assert "speech" in agent_triggers
    # The model issued the wave Command; on_agent_command carried a CommandIssued at dispatch.
    assert [(c.name, c.args) for c in commands] == [("wave", {})]
    # With no output Command set, the step's text is private: on_agent_private_text carried it.
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
    wica = wica_factory(
        fake_config([{"tool_calls": [{"name": "block_forever", "args": {}}]}]),
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
    key = "agent:command:c1"
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
    voice = RecordingVoice()
    wica = wica_factory(
        fake_config(
            [{"text": "<say>first cycle</say>"}, {"text": "<say>second cycle</say>"}]
        ),
        voice=voice,
        coalesce_window=0,
    )
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    wica.start()
    first_thread = wica._loop_thread
    wica.world.update("speech", "one")
    assert voice.event.wait(timeout=WAIT_TIMEOUT)
    assert voice.texts == ["first cycle"]

    wica.stop()
    assert wica.is_running is False
    assert wica.world.is_running is False
    assert not wica._loop.is_closed()
    assert wica._loop_thread is None

    voice.event.clear()
    wica.start()
    second_thread = wica._loop_thread
    assert second_thread is not None and second_thread is not first_thread
    wica.world.update("speech", "two")
    assert voice.event.wait(timeout=WAIT_TIMEOUT)

    assert voice.texts == ["first cycle", "second cycle"]
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
    voice = RecordingVoice()
    wica = Wica.init(
        fake_config([{"text": "<say>one</say>"}, {"text": "<say>two</say>"}]),
        coalesce_window=0,
        loop=loop,
    )
    wica.set_output_command(voice.say)
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    try:
        wica.start()
        wica.world.update("speech", "first")
        assert voice.event.wait(timeout=WAIT_TIMEOUT)
        wica.stop()

        voice.event.clear()
        wica.start()
        wica.world.update("speech", "second")
        assert voice.event.wait(timeout=WAIT_TIMEOUT)

        assert voice.texts == ["one", "two"]
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
    voice = RecordingVoice()
    wica = wica_factory(
        fake_config(
            [{"text": "<say>only the post-restart step replies</say>"}], delay_s=0.5
        ),
        voice=voice,
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

    assert voice.texts == []

    # The cancelled step's observation stays in history; the next step renders it merged with its
    # own observation, never as back-to-back user messages.
    wica.world.update("speech", "after the restart")
    assert voice.event.wait(WAIT_TIMEOUT)
    assert voice.texts == ["only the post-restart step replies"]
    assert len(prompts) == 2
    assert _no_consecutive_human_messages(prompts[1])


# --- Safe shutdown from the loop thread --------------------------------


def test_stop_from_an_output_command_is_rejected_and_the_system_keeps_running(
    wica_factory,
):
    wica_ref: list[Wica] = []
    errors: list[BaseException] = []
    done = threading.Event()

    async def say(text: str) -> None:
        """Say something to the user."""
        try:
            wica_ref[0].stop()
        except BaseException as exc:  # noqa: BLE001 - we want the exact error
            errors.append(exc)
        done.set()

    wica = wica_factory(fake_config([{"text": "<say>hi</say>"}]), coalesce_window=0)
    wica.set_output_command(say)
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


def test_model_timeout_from_config_bounds_the_model_call(wica_factory):
    # The fake model sleeps 5 s per call; the config's 0.1 s deadline abandons the step.
    config = fake_config([{"text": "<say>too late</say>"}], delay_s=5)
    config = WicaConfig(agent=dataclasses.replace(config.agent, model_timeout=0.1))
    voice = RecordingVoice()
    wica = wica_factory(config, voice=voice, coalesce_window=0)
    traces: list[ReactionTrace] = []
    done = threading.Event()

    def on_reaction(t: ReactionTrace) -> None:
        traces.append(t)
        done.set()

    wica.on_agent_reaction_ended.subscribe(on_reaction)
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()

    wica.world.update("speech", "hello")
    assert done.wait(timeout=WAIT_TIMEOUT)
    assert traces[0].outcome == "model_timeout"
    assert voice.texts == []


# --- openai-chat-completions-compatible: a local server, played by an httpx MockTransport --------


class ChatCompletionsServer:
    """An in-process stand-in for a server implementing OpenAI's Chat Completions API: records each
    request and streams one assistant text, then a usage chunk, as server-sent events (the Agent
    streams every call). No socket is opened."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        head = {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 0}
        head["model"] = "local"
        chunks = [
            {
                **head,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": self.text},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **head,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                **head,
                "choices": [],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 4,
                    "total_tokens": 124,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(
            200,
            content=body + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )


def compatible_config(server: ChatCompletionsServer, **agent: Any) -> WicaConfig:
    return WicaConfig(
        agent=AgentConfig(
            provider="openai-chat-completions-compatible",
            model="qwen2.5-7b-instruct",
            base_url="http://127.0.0.1:8080/v1",
            system_prompt="You are terse.",
            model_kwargs={
                "http_async_client": httpx.AsyncClient(
                    transport=httpx.MockTransport(server.handler)
                )
            },
            **agent,
        )
    )


def run_one_reaction(wica: Wica) -> ReactionTrace:
    traces: list[ReactionTrace] = []
    done = threading.Event()

    def on_reaction(trace: ReactionTrace) -> None:
        traces.append(trace)
        done.set()

    wica.on_agent_reaction_ended.subscribe(on_reaction)
    wica.world.register(
        "speech", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    wica.world.update("speech", "Hello robot!")
    assert done.wait(timeout=WAIT_TIMEOUT)
    return traces[0]


def test_a_compatible_server_is_reached_at_its_base_url_with_a_placeholder_key(
    wica_factory, monkeypatch: pytest.MonkeyPatch
):
    """The provider talks to the configured server, reports its usage, and with no key configured
    sends a placeholder — never the OPENAI_API_KEY meant for OpenAI. See specs/config.md
    ("Providers")."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-meant-for-openai")
    server = ChatCompletionsServer("<say>Hello there!</say>")
    voice = RecordingVoice()
    wica = wica_factory(compatible_config(server), voice=voice, coalesce_window=0)

    trace = run_one_reaction(wica)

    assert voice.event.wait(timeout=WAIT_TIMEOUT)
    assert voice.texts == ["Hello there!"]
    assert trace.outcome == "ok"
    assert trace.usage is not None
    assert (trace.usage.input_tokens, trace.usage.output_tokens) == (120, 4)
    assert trace.usage.cache_read_tokens == 100
    request = server.requests[0]
    assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "qwen2.5-7b-instruct"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][-1]["role"] == "user"
    # The output Command is not a tool, and no tool choice is sent (several self-hosted servers
    # ignore it): the model speaks through the tag in its text.
    assert [tool["function"]["name"] for tool in body["tools"]] == ["cancel_command"]
    assert "tool_choice" not in body
    authorization = request.headers["authorization"]
    assert "sk-meant-for-openai" not in authorization
    assert authorization.startswith("Bearer ") and authorization != "Bearer "


def test_a_compatible_server_gets_the_configured_key(wica_factory):
    server = ChatCompletionsServer("Hi.")
    wica = wica_factory(
        compatible_config(server, api_key="local-secret"), coalesce_window=0
    )

    run_one_reaction(wica)

    assert server.requests[0].headers["authorization"] == "Bearer local-secret"


def test_a_history_budget_on_a_compatible_server_needs_the_server_context_size():
    """A local model has no LangChain profile: the error names context_window and the server's own
    context size — see specs/agent.md ("Provider-agnostic model, from config")."""
    server = ChatCompletionsServer("unused")
    with pytest.raises(ConfigError, match="context_window.*server's own context size"):
        Wica.init(compatible_config(server, history_budget=0.5)).close()
