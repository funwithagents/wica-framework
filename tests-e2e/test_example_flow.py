"""Deterministic end-to-end tests of the conversation demo, over the `provider: "fake"` model.

Like tests-e2e/test_fake_flows.py these are network-free and key-less, so they **always run** — but
here the system under test is the *example itself*: they stand up the real demo through
`examples.conversation_demo.app.build_app` (the same entry point `main()` uses), against a scripted
fake config, and assert the example's actual World entries, Commands, and presenters turn inputs
into the expected transcript (with live Command state), Speaking panel state and World state. They
import no Gradio — `app` keeps the UI import lazy — so they need only the core deps.

See specs/conversation-demo.md and specs/fake-provider.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest

from examples.conversation_demo import app
from examples.conversation_demo.app import AppHandle, build_app
from wica.config import WicaConfig

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
                    "default": {"tool_calls": [{"name": "noop", "args": {}}]},
                },
            }
        }
    )


@pytest.fixture
def fast_say(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_SAY_WORD_DELAY_S", 0.05)


def test_speech_input_drives_say_action_and_world_state(fast_say: None):
    """A spoken input triggers a reasoning step whose scripted output speaks through the `say`
    output Command and sets an emotion; the completion of `say` re-triggers a step ended by noop.
    The whole example wiring (World entries + Commands + presenters) is exercised end to end."""
    config = fake_config(
        [
            {
                "text": "considering the greeting",
                "tool_calls": [
                    {"name": "say", "args": {"text": "hi there"}},
                    {"name": "set_emotion", "args": {"emotion": "happy"}},
                ],
            },
            {"tool_calls": [{"name": "noop", "args": {}}]},
        ]
    )

    handle: AppHandle = build_app(config)
    assert handle.wica is not None  # fake provider needs no key: the Agent is live
    try:
        transcript = handle.state.transcript
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

        # The say item: channel title with the outcome, the full text plus the result as body.
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
        # The model's free text became private reasoning, not the voice.
        thought = items_titled(convo, "💭 output sink")
        assert thought and thought[0]["content"] == "considering the greeting"

        # The Speaking panel saw the whole utterance go out.
        speaking = handle.state.speaking.read()
        assert speaking is not None
        assert speaking.text == "hi there"
        assert (speaking.spoken, speaking.state) == (2, "complete")

        # The prompt panel captured this step's prompt, labelled by the speech trigger.
        assert snap.prompt_count >= 1
        assert '🗣️ "hello"' in snap.prompt_choices[0][0]

        # `say` completing re-triggered a second step (ended by noop), so a second prompt appears.
        wait_until(lambda: transcript.snapshot().prompt_count >= 2)
        assert items_titled(transcript.snapshot().conversation, "🚫 noop")
    finally:
        handle.wica.close()


def test_barge_in_cancels_the_running_say(fast_say: None):
    """A second input while the robot is still speaking starts a new step (the say Command runs
    in the background, it is not a busy reasoning call) whose scripted reply cancels the speech:
    the transcript's say item flips to cancelled, a cancel_command item completes, and the
    Speaking panel shows the utterance cut off part-way."""
    long_text = "one two three four five six seven eight nine ten"
    config = fake_config(
        [
            {"tool_calls": [{"name": "say", "args": {"text": long_text}}]},
            # The fake model's tool-call ids are deterministic (fake_call_<step>_<i>), so the
            # scripted step can name the running say.
            {
                "tool_calls": [
                    {"name": "cancel_command", "args": {"call_id": "fake_call_0_0"}}
                ]
            },
        ]
    )

    handle: AppHandle = build_app(config)
    assert handle.wica is not None
    try:
        transcript = handle.state.transcript
        handle.world.update("speech_input", "tell me a story")

        def say_running() -> bool:
            say = items_titled(transcript.snapshot().conversation, "🗣️ say")
            speaking = handle.state.speaking.read()
            return bool(say) and speaking is not None and speaking.spoken >= 1

        wait_until(say_running)
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
        assert say["content"] == long_text  # no result on a cancelled Command
        cancel = items_titled(convo, "🦾 cancel_command")[0]
        assert cancel["metadata"]["title"] == "🦾 cancel_command ✅"
        assert "fake_call_0_0" in cancel["content"]

        speaking = handle.state.speaking.read()
        assert speaking is not None
        assert speaking.state == "cancelled"
        assert 1 <= speaking.spoken < 10
    finally:
        handle.wica.close()
