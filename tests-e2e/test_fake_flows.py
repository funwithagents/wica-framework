"""Deterministic scripted whole-flow tests over the `provider: "fake"` model.

Unlike the live-provider e2e tests (parametrized over PROVIDER_CONFIGS, skipping without a key),
these are network-free and key-less, so they **always run** — a deterministic complement in the
same full-loop tier. They drive the Agent's whole step loop over a scripted model whose output the
test fully controls, asserting the exact sequence of Commands and utterances a real, non-
deterministic provider could never pin. See specs/fake-provider.md.

The file is named so `pytest tests-e2e -k fake` selects the whole suite.

One case (`test_scripted_flow_through_wica_init`) runs on the `wica.testing` toolkit, as a demonstration
that the Harness carries its weight; the others **deliberately stay hand-rolled** (their own
`wait_until` / per-key listeners / `RecordingVoice`): they pin the very seams the Probe stands on,
so a toolkit bug must not be able to mask a framework bug (specs/wica-tester.md, "Relation to
existing tests").
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from wica import OutputStream, ReactionTrace, Wica
from wica.agent import Agent
from wica.config import AgentConfig, WicaConfig
from wica.content import Content, TextPart
from wica.fake_model import FakeChatModel
from wica.testing import TestRun
from wica.world import World, WorldEntry

WAIT_TIMEOUT = 5.0


def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except KeyError:
            pass
        time.sleep(0.02)
    raise AssertionError(f"condition not met within {timeout}s")


def identity_serialize(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


def find_command_entry(world: World) -> WorldEntry | None:
    for entry in world.get_prompt_entries():
        if entry.key.startswith("agent:command:"):
            return entry
    return None


def message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


class RecordingVoice:
    """A hand-rolled output Command: `say` records each output's pieces as they stream."""

    def __init__(self) -> None:
        self.pieces: list[list[str]] = []
        self.texts: list[str] = []
        self.event = threading.Event()

    async def say(self, text: OutputStream) -> str:
        """Speak out loud."""
        pieces = [piece async for piece in text]
        self.pieces.append(pieces)
        self.texts.append("".join(pieces))
        self.event.set()
        return "said"


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    yield event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(timeout=WAIT_TIMEOUT)
    event_loop.close()


def test_scripted_flow_through_wica_init():
    """The whole loop, driven through the real entrypoint: WicaConfig -> Wica.init. An input
    triggers a step that issues a scripted Command; its completion re-triggers a step that speaks a
    scripted line through the output Command's tag. Observed through `wica.testing.TestRun` — the
    one fake-flow case on the toolkit."""
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test double.",
                "model_kwargs": {
                    "delay_s": 0,
                    "script": [
                        {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 2}}]},
                        {"text": "<say>The sum is 4.</say>"},
                    ],
                    "default": {"text": ""},
                },
            }
        }
    )

    wica = Wica.init(config, coalesce_window=0.0)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    async def add(a: int, b: int) -> int:
        """Add two integers and return their sum."""
        await asyncio.sleep(0.05)  # long enough for the running state to be observed
        return a + b

    voice = RecordingVoice()
    wica.register_command(add)
    wica.set_output_command(voice.say)
    wica.start()
    run = TestRun(wica, scenario="scripted flow", sut_label="fake")
    try:
        run.set("prompt", "Add 2 and 2.")

        # Step 1: the scripted tool call is dispatched and tracked as a running World entry, then
        # completes with the sum.
        assert run.expect_action("add", where={"a": 2, "b": 2}, state="running").ok
        assert run.expect_action("add", state="complete", result_contains="4").ok

        # Step 2: the command completion re-triggered a step that spoke the scripted line.
        assert run.expect_output(contains="The sum is 4.").ok
        assert run.wait_for("idle", within=WAIT_TIMEOUT).ok
        assert voice.texts == ["The sum is 4."]
        assert [
            e.args
            for e in run.timeline()
            if e.kind == "command_issued" and e.name == "say"
        ] == [{"text": "The sum is 4."}]

        # Introspection: the step-2 prompt observed the completed command (its "4" result rendered
        # into the observation), proving the loop fed the outcome back to the model.
        model = wica.agent.model
        assert isinstance(model, FakeChatModel)
        assert len(model.calls) == 2  # the end of the speech woke no third step
        assert any("4" in message_text(m) for m in model.calls[-1])
        # And the Harness measured the turn: two reactions for one input, nothing dropped.
        metrics, dropped = run.metrics()
        assert dropped == 0
        assert metrics["reactions_per_input"].min == 2
    finally:
        run.close()
        wica.close()


def test_scripted_flow_direct_construction(loop: asyncio.AbstractEventLoop):
    """The non-facade seam: an Agent built directly from a provider: "fake" AgentConfig, on a World
    the test owns, drives the loop and streams its output to the output Command."""
    world = World(loop)
    world.start()
    world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )

    config = AgentConfig(
        provider="fake",
        model="scripted",
        system_prompt="You are a test double.",
        model_kwargs={"script": [{"text": "<say>hello there</say>"}], "delay_s": 0},
    )
    voice = RecordingVoice()
    agent = Agent(config, world=world, loop=loop, coalesce_window=0.0)
    agent.set_output_command(voice.say)

    agent.start()
    try:
        world.update("prompt", "hi")
        assert voice.event.wait(timeout=WAIT_TIMEOUT)
        assert voice.texts == ["hello there"]
        # The fake streams word by word, so the voice got the utterance in pieces.
        assert voice.pieces == [["hello ", "there"]]
        # The model saw the rendered input prompt on its single call.
        model = agent.model
        assert isinstance(model, FakeChatModel)
        assert len(model.calls) == 1
        assert any("hi" in message_text(m) for m in model.calls[0])
    finally:
        agent.stop()
        world.stop()


def test_output_command_flow_through_the_setters():
    """With an output Command set (after init, through the facade's setter), the body of its tag is
    delivered by the Command while the text outside is the model's private text; the `<next>`
    note makes the end of the speech wake a step, which sees the note and the finished speech
    and ends the turn with `<noop/>`."""
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test double.",
                "model_kwargs": {
                    "delay_s": 0,
                    "script": [
                        {
                            "text": "thinking about it <speak>hello</speak>"
                            "<next>check on them</next>"
                        },
                        {"text": "<noop/>"},
                    ],
                    "default": {"text": ""},
                },
            }
        }
    )

    spoken: list[str] = []
    private: list[str] = []
    reactions: list[ReactionTrace] = []

    async def speak(text: str) -> str:
        """Say something to the user."""
        spoken.append(text)
        return "spoken"

    wica = Wica.init(config, coalesce_window=0.0)
    wica.set_output_command(speak)
    wica.on_agent_private_text.subscribe(private.append)
    wica.on_agent_reaction_ended.subscribe(reactions.append)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    try:
        wica.world.update("prompt", "Greet the user.")
        wait_until(lambda: len(reactions) == 2)
        # The Command carried the user-facing text; the text outside the tag stayed private.
        assert spoken == ["hello"]
        assert private == ["thinking about it"]
        # The end of the speech woke the second step, which saw its note and the speech's outcome.
        model = wica.agent.model
        assert isinstance(model, FakeChatModel)
        second_prompt = " ".join(message_text(m) for m in model.calls[1])
        assert "You planned to do next: check on them" in second_prompt
        assert 'speak output: "hello" — delivered: spoken.' in second_prompt
        assert [r.noop for r in reactions] == [False, True]
        time.sleep(0.2)  # give any (erroneous) further wake a chance to fire
        assert len(model.calls) == 2
    finally:
        wica.close()


def test_noop_flow_takes_no_action():
    """A scripted `<noop/>` declares no reaction: no Command issued, no command entry, nothing
    spoken, no re-trigger — the reaction record's flag reports it."""
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test double.",
                "model_kwargs": {
                    "delay_s": 0,
                    "script": [{"text": "<noop/>"}],
                    "default": {"text": ""},
                },
            }
        }
    )

    voice = RecordingVoice()
    issued: list[Any] = []
    reactions: list[ReactionTrace] = []
    wica = Wica.init(config, coalesce_window=0.0)
    wica.set_output_command(voice.say)
    wica.on_agent_command.subscribe(issued.append)
    wica.on_agent_reaction_ended.subscribe(reactions.append)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    try:
        wica.world.update("prompt", "Nothing to do.")
        model = wica.agent.model
        assert isinstance(model, FakeChatModel)
        wait_until(lambda: len(reactions) == 1)
        time.sleep(0.2)  # give any (erroneous) re-trigger a chance to fire

        assert reactions[0].noop is True
        assert issued == []  # <noop/> is not a Command
        assert find_command_entry(wica.world) is None  # and created no World entry
        assert voice.texts == []  # nothing spoken
        assert len(model.calls) == 1  # no re-trigger
    finally:
        wica.close()


def test_history_budget_flow_cuts_after_a_scripted_over_budget_report():
    """The usage-driven history cut, through the JSON config path: a scripted usage figure over
    the budget's high-water mark at step 3 makes the Agent drop the oldest reactions, so the
    prompt of step 4 holds fewer observations than step 3's — see specs/agent.md ("History
    budget") and specs/fake-provider.md ("Token usage")."""
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test double.",
                "history_budget": 0.5,
                "context_window": 1600,  # high 800, low 400
                "model_kwargs": {
                    "delay_s": 0,
                    "script": [
                        {
                            "text": "<say>one</say>",
                            "usage": {"input_tokens": 100, "output_tokens": 1},
                        },
                        {
                            "text": "<say>two</say>",
                            "usage": {"input_tokens": 300, "output_tokens": 1},
                        },
                        # fixed part ~100 (first report); 3 reactions at 900 → (900 - 100)
                        # / 2 = 400 each → keep floor((400 - 100) / 400) = 0 → at least one,
                        # so reaction 3 stays and reactions 1-2 go.
                        {
                            "text": "<say>three</say>",
                            "usage": {"input_tokens": 900, "output_tokens": 1},
                        },
                        {
                            "text": "<say>four</say>",
                            "usage": {"input_tokens": 500, "output_tokens": 1},
                        },
                    ],
                    "default": {"text": ""},
                },
            }
        }
    )

    voice = RecordingVoice()
    prompts: list[list[Any]] = []
    reactions: list[ReactionTrace] = []
    wica = Wica.init(config, coalesce_window=0.0)
    wica.set_output_command(voice.say)
    wica.on_agent_prompt.subscribe(prompts.append)
    wica.on_agent_reaction_ended.subscribe(reactions.append)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    try:
        for i, text in enumerate(["first", "second", "third", "fourth"], start=1):
            wica.world.update("prompt", text)
            wait_until(lambda: len(reactions) == i and len(voice.texts) == i)

        def observations(messages: list[Any]) -> list[str]:
            return [m.text for m in messages if m.type == "human"]

        assert len(observations(prompts[2])) == 3  # nothing cut before the report
        kept = observations(
            prompts[3]
        )  # after the cut: reaction 3 and the new reaction 4
        assert len(kept) == 2
        assert "third" in kept[0] and "fourth" in kept[1]
        assert "most recent reactions" in wica.agent.system_prompt
    finally:
        wica.close()
