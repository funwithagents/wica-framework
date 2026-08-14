from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from wica.content import Content, TextPart
from wica.world import World, WorldEntry, get_world

from support import PROVIDER_CONFIGS, real_agent

WAIT_TIMEOUT = 15.0


def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except KeyError:
            pass
        time.sleep(0.05)
    raise AssertionError(f"condition not met within {timeout}s")


def identity_serialize(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


def find_command_entry(world: World) -> WorldEntry | None:
    for entry in world.get_prompt_entries():
        if entry.key.startswith("agent:command:"):
            return entry
    return None


class RecordingSink:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.event = threading.Event()

    async def __call__(self, text: str) -> None:
        self.texts.append(text)
        self.event.set()


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_plain_text_round_trip(config_path: Path):
    world = get_world()
    world.register("prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    sink = RecordingSink()
    agent = real_agent(config_path, world=world, output_sink=sink)
    agent.start()
    try:
        world.update("prompt", "Say hello in one short sentence.")
        assert sink.event.wait(timeout=WAIT_TIMEOUT)
        assert sink.texts
        assert sink.texts[0].strip()
    finally:
        agent.stop()


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_real_tool_calling_round_trip(config_path: Path):
    world = get_world()
    world.register("prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    sink = RecordingSink()
    agent = real_agent(config_path, world=world, output_sink=sink)

    async def add(a: int, b: int) -> int:
        """Add two integers and return their sum."""
        await asyncio.sleep(0.1)  # gives the polling loop below a chance to see "running"
        return a + b

    agent.register_command(add)
    agent.start()
    try:
        world.update("prompt", "What is 2 + 2? Use the add tool.")

        wait_until(lambda: find_command_entry(world) is not None)
        running_entry = find_command_entry(world)
        assert running_entry is not None
        assert running_entry.current.value.state == "running"

        # Listen on this specific key so the terminal update is captured via the
        # immutable snapshot handed to the listener — reading world.get_entry(key)
        # again would race the Agent's own cleanup, which unregisters the key right
        # after folding its terminal value into history (see commands.md decision on
        # command-execution World keys).
        terminal: list[WorldEntry] = []
        done = threading.Event()

        def on_update(entry: WorldEntry) -> None:
            if entry.current.value.is_terminal():
                terminal.append(entry)
                done.set()

        world.add_listener(running_entry.key, on_update)
        assert done.wait(timeout=WAIT_TIMEOUT)
        assert terminal[0].current.value.state == "complete"
        assert "4" in (terminal[0].current.value.result or "")
        # Deliberately not asserting the model also speaks a follow-up text reply here: whether
        # free text follows a completed tool call is model-decided, not a WICA guarantee (see
        # agent.md open question #2) — asserting it would make this test flaky on real
        # non-determinism unrelated to the tool-calling round trip under test. The sink/output
        # path itself is already covered by test_plain_text_round_trip.
    finally:
        agent.stop()
