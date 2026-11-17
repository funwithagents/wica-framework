"""Deterministic end-to-end tests of the conversation demo, over the `provider: "fake"` model.

Like tests-e2e/test_fake_flows.py these are network-free and key-less, so they **always run** — but
here the system under test is the *example itself*: they stand up the real demo through the same
Gradio-free steps `main()` uses — `build_system(config)`, then `wire(wica, speaking)`, then
`start()` — playing the UI's role themselves (a `TranscriptLog`, a `ReactionLog` and a
`SpeakingSlot` of their own, exactly what `build_ui` would create), against a scripted fake config,
and assert the example's actual World entries, Commands, and presenters turn inputs into the
expected transcript (with live Command state), reaction history, Speaking panel state and World
state. They build no Gradio page, but the presenters come from `wica.contrib.gradio`, so the
`dev` group's Gradio is needed to import them.

See specs/conversation-demo.md, specs/gradio-contrib.md and specs/fake-provider.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from examples.conversation_demo import app
from examples.conversation_demo.app import build_system, display_entry, wire
from examples.conversation_demo.speaking import SpeakingSlot
from wica import Wica, World
from wica.config import WicaConfig
from wica.contrib.gradio import ReactionLog, TranscriptLog

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


def items_titled(
    conversation: list[dict[str, Any]], prefix: str
) -> list[dict[str, Any]]:
    """Items whose title starts with `prefix` — a Command item's title gains a state suffix
    (✅ / ❌ / ⏹) once it ends, so match on the prefix."""
    return [
        m
        for m in conversation
        if str(m.get("metadata", {}).get("title", "")).startswith(prefix)
    ]


def fake_config(script: list[dict[str, Any]]) -> WicaConfig:
    return WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test robot.",
                "model_kwargs": {
                    "delay_s": 0,
                    "script": script,
                    "default": {"text": "<noop/>"},
                },
            }
        }
    )


@pytest.fixture
def fast_say(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_SAY_WORD_DELAY_S", 0.05)


@dataclass(frozen=True)
class Stack:
    """The stood-up demo: the running Wica, its World, and the presenters the test owns."""

    wica: Wica
    world: World
    transcript: TranscriptLog
    reactions: ReactionLog
    speaking: SpeakingSlot


def stand_up(config: WicaConfig) -> Stack:
    """Build → (the test as the UI) → wire → start, in the order `main()` uses."""
    wica, world, config_error = build_system(config)
    assert wica is not None and config_error is None  # fake provider needs no key
    transcript = TranscriptLog(wica, display_entry)
    reactions = ReactionLog(wica, display_entry)
    speaking = SpeakingSlot()
    wire(wica, speaking)
    wica.start()
    return Stack(
        wica=wica,
        world=world,
        transcript=transcript,
        reactions=reactions,
        speaking=speaking,
    )


def test_speech_input_drives_say_action_and_world_state(fast_say: None):
    """A spoken input triggers a reasoning step whose scripted response speaks through a `<say>`
    tag (delivered to the `say` output Command), keeps a private note and sets an emotion; the
    emotion Command's completion wakes a step the model declines. The whole example wiring
    (World entries + Commands + presenters) is exercised end to end."""
    config = fake_config(
        [
            {
                "text": "considering the greeting <say>hi there</say>",
                "tool_calls": [{"name": "set_emotion", "args": {"emotion": "happy"}}],
            },
            {"text": "<noop/>"},
        ]
    )

    handle = stand_up(config)
    try:
        transcript = handle.transcript
        handle.world.update("speech_input", "hello")

        # The robot spoke the scripted line through `say` (its item flips to complete once the
        # words are out) and the emotion Command landed in the World.
        def spoke_and_felt() -> bool:
            say = items_titled(transcript.snapshot().conversation, "🗣️ say")
            emotion = handle.world.get_entry("emotion").current.value
            return (
                bool(say)
                and say[0]["metadata"]["title"].endswith("✅")
                and emotion == "happy"
            )

        wait_until(spoke_and_felt)

        snap = transcript.snapshot()
        convo = snap.conversation

        # Input shown on the user side, through the demo's display hook.
        assert {"role": "user", "content": '🗣️ "hello"'} in convo

        # One reaction group opened, and the step's outputs nest under it.
        groups = [m for m in convo if m.get("metadata", {}).get("id") == "reaction-1"]
        assert len(groups) == 1

        # The say item: one for the one tag, channel title with the outcome, the full text plus
        # the result as body.
        assert len(items_titled(convo, "🗣️ say")) == 1
        say = items_titled(convo, "🗣️ say")[0]
        assert say["metadata"]["title"] == "🗣️ say ✅"
        assert say["content"] == "hi there\n→ Said it."
        assert say["metadata"]["parent_id"] == "reaction-1"
        assert say["metadata"]["duration"] >= 0.0

        action = items_titled(convo, "🦾 set_emotion")
        assert action and action[0]["metadata"]["parent_id"] == "reaction-1"
        assert action[0]["metadata"]["title"] == "🦾 set_emotion ✅"
        assert action[0]["content"] == (
            "set_emotion(emotion='happy')\n→ Now showing emotion: happy."
        )
        # What the model wrote outside the tag is private, not the voice.
        thought = items_titled(convo, "💭 private text")
        assert thought and thought[0]["content"] == "considering the greeting"
        assert thought[0]["metadata"]["parent_id"] == "reaction-1"

        # The Speaking panel saw the whole utterance go out.
        speaking = handle.speaking.read()
        assert speaking is not None
        assert speaking.text == "hi there"
        assert (speaking.spoken, speaking.state) == (2, "complete")

        # The reaction history captured this step's prompt, labelled by the speech trigger, and
        # the prompt text carries the rendered World (the speech entry the model saw).
        reactions = handle.reactions.snapshot()
        assert reactions.count >= 1
        assert reactions.choices[0][0].endswith('— 🗣️ "hello"')
        assert "hello" in (handle.reactions.prompt_text(0) or "")

        # Its instrumentation starts pending and resolves once the reaction ends — it must have
        # ended well before this point, since the say output it opened already completed
        # (spoke_and_felt() above waited for that). It timed the first word.
        instrumentation = handle.reactions.instrumentation_text(0)
        assert instrumentation is not None and "in progress" not in instrumentation
        assert "reaction 1 — ok" in instrumentation
        assert "time to first output" in instrumentation

        # The end of the speech wakes nothing (no <next> note); the emotion Command's completion
        # does, so a second reaction appears, labelled by it — and the model declines it: a
        # 🚫 noop item under that reaction, read from its record (no Command is issued).
        wait_until(lambda: handle.reactions.snapshot().count >= 2)
        assert (
            handle.reactions.snapshot()
            .choices[1][0]
            .endswith("— 🦾 set_emotion finished")
        )
        wait_until(
            lambda: bool(items_titled(transcript.snapshot().conversation, "🚫 noop"))
        )
        noop = items_titled(transcript.snapshot().conversation, "🚫 noop")
        assert len(noop) == 1
        assert noop[0]["metadata"]["parent_id"] == "reaction-2"
        assert handle.reactions.snapshot().count == 2
    finally:
        handle.wica.close()


def test_next_note_lets_the_robot_speak_again_once_it_has_spoken(fast_say: None):
    """With a `<next>` note the end of the speech wakes a step, shown as a reaction labelled by
    the finished `say`, in which the robot speaks again: two say items, one per reaction."""
    config = fake_config(
        [
            {"text": "<say>first thing</say><next>then the second thing</next>"},
            {"text": "<say>second thing</say>"},
        ]
    )

    handle = stand_up(config)
    try:
        transcript = handle.transcript
        handle.world.update("speech_input", "hello")

        def spoke_twice() -> bool:
            say = items_titled(transcript.snapshot().conversation, "🗣️ say")
            return len(say) == 2 and all(
                item["metadata"]["title"] == "🗣️ say ✅" for item in say
            )

        wait_until(spoke_twice)
        say = items_titled(transcript.snapshot().conversation, "🗣️ say")
        assert [(m["content"], m["metadata"]["parent_id"]) for m in say] == [
            ("first thing\n→ Said it.", "reaction-1"),
            ("second thing\n→ Said it.", "reaction-2"),
        ]
        choices = handle.reactions.snapshot().choices
        assert choices[1][0].endswith("— 🗣️ say finished")
        speaking = handle.speaking.read()
        assert speaking is not None
        assert (speaking.text, speaking.state) == ("second thing", "complete")
    finally:
        handle.wica.close()


def test_barge_in_cancels_the_running_say(fast_say: None):
    """A second input while the robot is still speaking starts a new step (the say output runs
    in the background, it is not a busy reasoning call) whose scripted reply cancels the speech:
    the transcript's say item flips to cancelled, a cancel_command item completes, and the
    Speaking panel shows the utterance cut off part-way."""
    long_text = "one two three four five six seven eight nine ten"
    config = fake_config(
        [
            {"text": f"<say>{long_text}</say>"},
            # WICA numbers the outputs and calls it opens (c1, c2, …), so the scripted step can
            # name the running say: it is the first one.
            {"tool_calls": [{"name": "cancel_command", "args": {"call_id": "c1"}}]},
        ]
    )

    handle = stand_up(config)
    try:
        transcript = handle.transcript
        handle.world.update("speech_input", "tell me a story")

        def say_running() -> bool:
            say = items_titled(transcript.snapshot().conversation, "🗣️ say")
            speaking = handle.speaking.read()
            return bool(say) and speaking is not None and speaking.spoken >= 1

        wait_until(say_running)
        # Reported with its full text while it runs.
        say = items_titled(transcript.snapshot().conversation, "🗣️ say")[0]
        assert say["content"] == long_text
        assert say["metadata"]["status"] == "pending"
        handle.world.update("speech_input", "stop!")

        def say_cancelled() -> bool:
            convo = transcript.snapshot().conversation
            say = items_titled(convo, "🗣️ say")
            cancel = items_titled(convo, "🦾 cancel_command")
            return (
                bool(say)
                and say[0]["metadata"]["title"] == "🗣️ say ⏹ cancelled"
                and bool(cancel)
                and cancel[0]["metadata"]["title"].endswith("✅")
            )

        wait_until(say_cancelled)

        convo = transcript.snapshot().conversation
        say = items_titled(convo, "🗣️ say")[0]
        assert "status" not in say["metadata"]  # spinner gone, item left open
        cancel = items_titled(convo, "🦾 cancel_command")[0]
        assert cancel["metadata"]["title"] == "🦾 cancel_command ✅"
        assert "c1" in cancel["content"]

        speaking = handle.speaking.read()
        assert speaking is not None
        assert speaking.state == "cancelled"
        assert 1 <= speaking.spoken < 10
        # Cut off, `say` reported the words it got to: the item (no result on a cancelled
        # Command) and the robot's World keep what was heard, not the whole text.
        heard = " ".join(long_text.split()[: speaking.spoken])
        assert say["content"] == heard
    finally:
        handle.wica.close()


def test_speaking_again_cuts_off_what_the_robot_was_saying(fast_say: None):
    """One voice: a reply spoken while the robot is still speaking replaces the earlier speech —
    no cancel_command needed. The first say item flips to cancelled as the new one appears, and
    the Speaking panel moves on to the new utterance."""
    long_text = "one two three four five six seven eight nine ten"
    config = fake_config(
        [{"text": f"<say>{long_text}</say>"}, {"text": "<say>all right</say>"}]
    )

    handle = stand_up(config)
    try:
        transcript = handle.transcript
        handle.world.update("speech_input", "tell me a story")

        def say_running() -> bool:
            say = items_titled(transcript.snapshot().conversation, "🗣️ say")
            speaking = handle.speaking.read()
            return bool(say) and speaking is not None and speaking.spoken >= 1

        wait_until(say_running)
        handle.world.update("speech_input", "stop!")

        def replaced() -> bool:
            say = items_titled(transcript.snapshot().conversation, "🗣️ say")
            return len(say) == 2 and say[1]["metadata"]["title"] == "🗣️ say ✅"

        wait_until(replaced)
        convo = transcript.snapshot().conversation
        first, second = items_titled(convo, "🗣️ say")
        assert first["metadata"]["title"] == "🗣️ say ⏹ cancelled"
        assert first["metadata"]["parent_id"] == "reaction-1"
        words = first["content"].split()
        assert 1 <= len(words) < 10 and words == long_text.split()[: len(words)]
        assert second["content"] == "all right\n→ Said it."
        assert second["metadata"]["parent_id"] == "reaction-2"
        assert not items_titled(convo, "🦾 cancel_command")

        speaking = handle.speaking.read()
        assert speaking is not None
        assert (speaking.text, speaking.state) == ("all right", "complete")
    finally:
        handle.wica.close()
