"""Deterministic end-to-end test of the conversation demo, over the `provider: "fake"` model.

Like tests-e2e/test_fake_flows.py this is network-free and key-less, so it **always runs** — but
here the system under test is the *example itself*: it stands up the real demo through
`examples.conversation_demo.app.build_app` (the same entry point `main()` uses), against a scripted
fake config, and asserts the example's actual World entries, Commands, and presenter callbacks turn
one spoken input into the expected transcript and World state. It imports no Gradio — `app` keeps
the UI import lazy — so it needs only the core deps.

See specs/conversation-demo.md and specs/fake-provider.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

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


def children_titled(
    conversation: list[dict[str, Any]], title: str
) -> list[dict[str, Any]]:
    return [m for m in conversation if m.get("metadata", {}).get("title") == title]


def test_speech_input_drives_say_action_and_world_state():
    """A spoken input triggers a reasoning step whose scripted output speaks through the `say`
    output Command and sets an emotion; the completion of `say` re-triggers a step ended by noop.
    The whole example wiring (World entries + Commands + presenter) is exercised end to end."""
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test robot.",
                "model_kwargs": {
                    "delay_s": 0,
                    "script": [
                        {
                            "text": "considering the greeting",
                            "tool_calls": [
                                {"name": "say", "args": {"text": "hi there"}},
                                {"name": "set_emotion", "args": {"emotion": "happy"}},
                            ],
                        },
                        {"tool_calls": [{"name": "noop", "args": {}}]},
                    ],
                    "default": {"text": ""},
                },
            }
        }
    )

    handle: AppHandle = build_app(config)
    assert handle.wica is not None  # fake provider needs no key: the Agent is live
    try:
        handle.world.update("speech_input", "hello")

        # The robot spoke the scripted line through `say` (streamed word by word to its final text)
        # and the emotion Command landed in the World.
        def spoke_and_felt() -> bool:
            convo = handle.state.snapshot().conversation
            say = children_titled(convo, "🗣️ say")
            emotion = handle.world.get_entry("emotion").current.value
            return bool(say) and say[0]["content"] == "hi there" and emotion == "happy"

        wait_until(spoke_and_felt)

        snap = handle.state.snapshot()
        convo = snap.conversation

        # Input shown on the user side.
        assert {"role": "user", "content": '🗣️ "hello"'} in convo

        # One reaction group opened, and the step's outputs nest under it.
        groups = [m for m in convo if m.get("metadata", {}).get("id") == "reaction-1"]
        assert len(groups) == 1

        say = children_titled(convo, "🗣️ say")[0]
        action = children_titled(convo, "🦾 set_emotion")
        thought = children_titled(convo, "💭 output sink")
        assert say["metadata"]["parent_id"] == "reaction-1"
        assert action and action[0]["metadata"]["parent_id"] == "reaction-1"
        assert action[0]["content"] == "set_emotion(emotion='happy')"
        # The model's free text became private reasoning, not the voice.
        assert thought and thought[0]["content"] == "considering the greeting"

        # The prompt panel captured this step's prompt, labelled by the speech trigger.
        assert snap.prompt_count >= 1
        assert '🗣️ "hello"' in snap.prompt_choices[0][0]

        # `say` completing re-triggered a second step (ended by noop), so a second prompt appears.
        wait_until(lambda: handle.state.snapshot().prompt_count >= 2)
        assert children_titled(handle.state.snapshot().conversation, "🚫 noop")
    finally:
        handle.wica.close()
