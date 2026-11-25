"""Fast, deterministic unit tests for the conversation demo's own pieces.

These pin what the demo contributes on top of the `wica.contrib.gradio` components: its
`display_entry` hook (`app.py`), the Speaking panel's model (`speaking.py`) and the robot's `say`
output Command driving it from the words the model writes — without any network or a browser. The generic transcript, prompt history and
World table are the contrib's and are tested in `tests/contrib/`; the full end-to-end wiring
(World entries + Commands + presenters, driven by a scripted fake model) is covered separately in
tests-e2e/test_example_flow.py.

See specs/conversation-demo.md ("Reused from the Gradio contrib").
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from examples.conversation_demo import app
from examples.conversation_demo.app import Robot, display_entry
from examples.conversation_demo.speaking import SpeakingSlot
from tests.support import command_entry, make_entry
from wica import OutputStream
from wica.contrib.gradio import EntryDisplay
from wica.text_output import output_signature

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
    slot.extend(old, "one two and more")
    slot.complete(old)
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("three", "four")
    assert (speaking.spoken, speaking.state) == (0, "speaking")
    slot.complete(new)
    assert slot.read().state == "complete"  # type: ignore[union-attr]


def test_speaking_slot_grows_as_the_words_arrive():
    """The model is still writing the sentence: the words to come grow, the spoken count and
    the state are kept."""
    slot = SpeakingSlot()
    token = slot.start("hello big")
    slot.advance(token)
    slot.extend(token, "hello big wide world")
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("hello", "big", "wide", "world")
    assert (speaking.spoken, speaking.state) == (1, "speaking")


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch, unstarted_wica) -> Robot:
    """The robot over an unstarted Wica's World and a fresh Speaking slot — `say` touches only the
    slot, so nothing needs to run."""
    monkeypatch.setattr(app, "_SAY_WORD_DELAY_S", 0.01)
    return Robot(unstarted_wica(None).world, SpeakingSlot())


def stream_of(*pieces: str, close: bool = True) -> OutputStream:
    """An output's text as the framework hands it to `say`: pieces pushed, then the end."""
    stream = OutputStream()
    for piece in pieces:
        stream.push(piece)
    if close:
        stream.close()
    return stream


def test_say_is_a_streaming_output_command_without_attributes(robot: Robot):
    """`say` takes the words as they are written — what makes the robot start speaking before
    the response is complete — and declares no tag attribute."""
    signature = output_signature(robot.say)
    assert signature.streams
    assert signature.attributes == {}


def test_say_drives_the_speaking_slot_to_complete(robot: Robot):
    async def run() -> None:
        # Pieces cut mid-word, as a model's chunks are.
        return await robot.say(stream_of("hi th", "ere fri", "end"))

    assert asyncio.run(run()) is None
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.text == "hi there friend"
    assert (speaking.spoken, speaking.state) == (3, "complete")


def test_say_starts_on_the_first_words_and_waits_for_the_rest(robot: Robot):
    """Streaming: the robot speaks the words it has while the model is still writing — never a
    word that may still be growing — then waits for more and finishes once the text ends."""

    async def run() -> None:
        stream = stream_of("one two thr", close=False)
        task = asyncio.create_task(robot.say(stream))
        await asyncio.sleep(0.1)  # far longer than the three words would take
        speaking = robot.speaking.read()
        assert speaking is not None
        # "thr" is not a whole word yet: two words spoken, the sentence still in progress.
        assert (speaking.spoken, speaking.state) == (2, "speaking")
        assert not task.done()

        stream.push("ee four")
        stream.close()
        assert await task is None

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.words == ("one", "two", "three", "four")
    assert (speaking.spoken, speaking.state) == (4, "complete")


def test_say_with_no_words_leaves_the_panel_alone(robot: Robot):
    token = robot.speaking.start("before")
    robot.speaking.complete(token)

    async def run() -> None:
        return await robot.say(stream_of("  "))

    assert asyncio.run(run()) is None
    speaking = robot.speaking.read()
    assert speaking is not None and speaking.text == "before"


def test_say_marks_the_slot_cancelled_and_reports_what_was_heard(robot: Robot):
    """Barge-in: cancelling the say task mid-sentence leaves the words spoken so far, marks the
    utterance cancelled and sets `heard` to exactly those words (what the robot's history
    keeps) — and the cancellation still propagates to the caller."""
    stream = stream_of("one two three four five six")

    async def run() -> None:
        task = asyncio.create_task(robot.say(stream))
        await asyncio.sleep(0.025)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.state == "cancelled"
    assert 0 < speaking.spoken < 6
    assert stream.heard == " ".join(speaking.words[: speaking.spoken])


def test_say_cut_off_while_waiting_for_more_words_reports_what_was_heard(robot: Robot):
    stream = stream_of("one two ", close=False)

    async def run() -> None:
        task = asyncio.create_task(robot.say(stream))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert (speaking.spoken, speaking.state) == (2, "cancelled")
    assert stream.heard == "one two"


def test_persona_describes_the_say_tag_and_no_removed_mechanism():
    """The persona under test (the benchmark's too) must teach the tag, not a `say` call, and
    must not tell the model to call `noop`."""
    persona = (Path(app.__file__).parent / "prompts" / "wica.md").read_text()
    assert "<say>" in persona and "</say>" in persona
    assert "say(" not in persona
    assert "noop" not in persona
