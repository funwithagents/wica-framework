"""Plain helpers for the `wica.testing` tests (the fixture itself is in conftest.py). Named
`sut_support`, not `support`, so it cannot shadow `tests/support.py` under pytest's prepend import
mode (neither directory is a package)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Literal

from wica import OutputStream
from wica.content import Content, TextPart


def identity_serialize(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


def describe_timeline(events: list[Any]) -> str:
    """One line per `SutEvent`, for assertion messages — what a CI failure needs to be read."""
    return "\n".join(
        f"{e.t:.3f} {e.kind} name={e.name} key={e.key} state={e.state} text={e.text!r}"
        + (f" outcome={e.reaction.outcome}" if e.reaction is not None else "")
        for e in events
    )


ADD_THEN_TEXT: list[dict[str, Any]] = [
    {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 2}}]},
    {"text": "The sum is 4."},
]


async def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    await asyncio.sleep(0.05)  # long enough for a listener to see "running"
    return a + b


class Voice:
    """A recording `say` output Command (a streaming one, with an `emotion` attribute) whose
    delivery takes `delay` seconds — `first` seconds for the first utterance, when given:
    `sut.set_output_command(voice.say)`. `said` holds what was delivered to its end."""

    def __init__(self, delay: float = 0.0, *, first: float | None = None) -> None:
        self.delay = delay
        self.first = first
        self.said: list[str] = []
        voice = self

        async def say(
            text: OutputStream, emotion: Literal["happy", "sad", "neutral"] = "neutral"
        ) -> str:
            """Speak out loud."""
            pause = voice.delay if voice.first is None else voice.first
            voice.first = None
            words = await text.read()
            await asyncio.sleep(pause)
            voice.said.append(words)
            return "Said it."

        self.say = say
