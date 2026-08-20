"""Deterministic scripted whole-flow tests over the `provider: "fake"` model.

Unlike the live-provider e2e tests (parametrized over PROVIDER_CONFIGS, skipping without a key),
these are network-free and key-less, so they **always run** — a deterministic complement in the
same full-loop tier. They drive the Agent's whole step loop over a scripted model whose output the
test fully controls, asserting the exact sequence of Commands and utterances a real, non-
deterministic provider could never pin. See specs/fake-provider.md.

The file is named so `pytest tests-e2e -k fake` selects the whole suite.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from wica.agent import Agent
from wica.config import WicaConfig
from wica.content import Content, TextPart
from wica.fake_model import FakeChatModel
from wica.world import World, WorldEntry, get_world

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


class RecordingSink:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.event = threading.Event()

    async def __call__(self, text: str) -> None:
        if text.strip():
            self.texts.append(text)
            self.event.set()


def test_scripted_flow_through_config_path():
    """The whole loop, driven through the exact WicaConfig -> Agent.from_config path production
    uses: an input triggers a step that issues a scripted Command; its completion re-triggers a
    step that speaks a scripted line."""
    world = get_world()
    world.register("prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True)

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
                        {"text": "The sum is 4."},
                    ],
                    "default": {"text": ""},
                },
            }
        }
    )

    sink = RecordingSink()
    agent = Agent.from_config(
        config.agent, world=world, output_sink=sink, coalesce_window=0.0
    )

    async def add(a: int, b: int) -> int:
        """Add two integers and return their sum."""
        await asyncio.sleep(0.05)  # gives the poll below a chance to observe "running"
        return a + b

    agent.register_command(add)
    agent.start()
    try:
        world.update("prompt", "Add 2 and 2.")

        # Step 1: the scripted tool call is dispatched and tracked as a running World entry.
        wait_until(lambda: find_command_entry(world) is not None)
        running = find_command_entry(world)
        assert running is not None
        assert running.current.value.state == "running"

        # Capture the terminal command outcome via the key's listener (avoids racing the Agent's
        # own cleanup, which unregisters the key after folding it into history).
        terminal: list[WorldEntry] = []
        done = threading.Event()

        def on_update(entry: WorldEntry) -> None:
            if entry.current.value.is_terminal():
                terminal.append(entry)
                done.set()

        world.add_listener(running.key, on_update)
        assert done.wait(timeout=WAIT_TIMEOUT)
        assert terminal[0].current.value.state == "complete"
        assert "4" in (terminal[0].current.value.result or "")

        # Step 2: the command completion re-triggered a step that spoke the scripted line.
        assert sink.event.wait(timeout=WAIT_TIMEOUT)
        assert sink.texts == ["The sum is 4."]

        # Introspection: the step-2 prompt observed the completed command (its "4" result rendered
        # into the observation), proving the loop fed the outcome back to the model.
        model = agent.model
        assert isinstance(model, FakeChatModel)
        assert len(model.calls) >= 2
        assert any("4" in message_text(m) for m in model.calls[-1])
    finally:
        agent.stop()


def test_scripted_flow_direct_construction():
    """The non-config seam: a FakeChatModel passed straight to Agent(model=...) drives the loop."""
    world = get_world()
    world.register("prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    model = FakeChatModel(script=[{"text": "hello there"}], delay_s=0)
    sink = RecordingSink()
    agent = Agent(
        model=model,
        system_prompt="You are a test double.",
        world=world,
        output_sink=sink,
        coalesce_window=0.0,
    )

    agent.start()
    try:
        world.update("prompt", "hi")
        assert sink.event.wait(timeout=WAIT_TIMEOUT)
        assert sink.texts == ["hello there"]
        # The model saw the rendered input prompt on its single call.
        assert len(model.calls) == 1
        assert any("hi" in message_text(m) for m in model.calls[0])
    finally:
        agent.stop()
