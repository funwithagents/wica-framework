"""Fast, deterministic unit tests for the conversation demo's own pieces.

These pin what the demo contributes on top of the `wica.contrib.gradio` components: its
`display_entry` hook (`app.py`), the Speaking panel's model (`speaking.py`) and the robot's `say`
Command driving it — without any network or a browser. The generic transcript, prompt history and
World table are the contrib's and are tested in `tests/contrib/`; the full end-to-end wiring
(World entries + Commands + presenters, driven by a scripted fake model) is covered separately in
tests-e2e/test_example_flow.py.

See specs/conversation-demo.md ("Reused from the Gradio contrib").
"""

from __future__ import annotations

import asyncio

import pytest

from examples.conversation_demo import app
from examples.conversation_demo.app import Robot, display_entry
from examples.conversation_demo.speaking import SpeakingSlot
from tests.support import command_entry, make_entry
from wica.contrib.gradio import EntryDisplay

# --- the demo's display_entry hook ------------------------------------------------------


def test_demo_display_entry_labels_the_robot_entries():
    assert display_entry(make_entry("speech_input", "hi")) == EntryDisplay('🗣️ "hi"')
    assert display_entry(make_entry("closest_user", "alice")) == EntryDisplay(
        "👤 Closest user detected: alice"
    )
    assert display_entry(make_entry("closest_user", None)) == EntryDisplay(
        "👤 Closest user gone"
    )
    # The voice: channel title, spoken text as body.
    assert display_entry(
        command_entry("c1", "say", {"text": "hi there"}, "running")
    ) == EntryDisplay("🗣️ say", "hi there")
    # Anything else is left to the generic default.
    assert display_entry(make_entry("emotion", "happy")) is None
    assert display_entry(command_entry("c2", "dance", {}, "running")) is None


# --- SpeakingSlot and the say Command -------------------------------------------------


def test_speaking_slot_tracks_words_and_ending():
    slot = SpeakingSlot()
    assert slot.read() is None

    token = slot.start("hello big world")
    slot.advance(token)
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("hello", "big", "world")
    assert speaking.spoken == 1
    assert speaking.state == "speaking"
    assert speaking.text == "hello big world"

    slot.advance(token)
    slot.cancelled(token)
    speaking = slot.read()
    assert speaking is not None
    assert (speaking.spoken, speaking.state) == (2, "cancelled")


def test_speaking_slot_ignores_a_stale_utterance():
    """An older `say` ending after a newer one started must not stamp the newer one."""
    slot = SpeakingSlot()
    old = slot.start("one two")
    new = slot.start("three four")
    slot.advance(old)
    slot.complete(old)
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("three", "four")
    assert (speaking.spoken, speaking.state) == (0, "speaking")
    slot.complete(new)
    assert slot.read().state == "complete"  # type: ignore[union-attr]


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch, unstarted_wica) -> Robot:
    """The robot over an unstarted Wica's World and a fresh Speaking slot — `say` touches only the
    slot, so nothing needs to run."""
    monkeypatch.setattr(app, "_SAY_WORD_DELAY_S", 0.01)
    return Robot(unstarted_wica(None).world, SpeakingSlot())


def test_say_drives_the_speaking_slot_to_complete(robot: Robot):
    async def run() -> str:
        return await robot.say("hi there friend")

    assert asyncio.run(run()) == "Said it."
    speaking = robot.speaking.read()
    assert speaking is not None
    assert (speaking.spoken, speaking.state) == (3, "complete")


def test_say_marks_the_slot_cancelled_when_cut_off(robot: Robot):
    """Barge-in: cancelling the say task mid-sentence leaves the words spoken so far and marks
    the utterance cancelled — and the cancellation still propagates to the caller."""

    async def run() -> None:
        task = asyncio.create_task(robot.say("one two three four five six"))
        await asyncio.sleep(0.025)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.state == "cancelled"
    assert 0 < speaking.spoken < 6
