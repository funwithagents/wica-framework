"""Fast, deterministic unit tests for the conversation demo's presenters.

These pin the generic transcript (`examples/conversation_demo/transcript.py`), the Speaking
panel's model (`speaking.py`) and the demo's `display_entry` hook and the robot's `say` Command
(`app.py`) — the seam between the agent's instrumentation callbacks and the Gradio UI — without any
network or Gradio. Where a log needs a `Wica` (to read the output Command's name live, or to bind
World listeners) the tests build an **unstarted** one over the key-less `provider: "fake"` model.
They need only core deps (wica + langchain_core), so they run in the default `tests/` tier. The
full end-to-end wiring (World entries + Commands + presenter, driven by a scripted fake model) is
covered separately in tests-e2e/test_example_flow.py.

See specs/conversation-demo.md ("A reusable transcript").
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from examples.conversation_demo import app
from examples.conversation_demo.app import Robot, display_entry
from examples.conversation_demo.speaking import SpeakingSlot
from examples.conversation_demo.transcript import (
    EntryDisplay,
    TranscriptLog,
    _flatten_content,
    _render_message,
)
from wica import Command, CommandIssued, Wica, WorldEntry, WorldEntryVersion
from wica.agent import CommandExecution
from wica.config import WicaConfig


def make_entry(key: str, value: Any) -> WorldEntry:
    """A minimal WorldEntry snapshot as the presenter's callbacks receive it."""
    version = WorldEntryVersion(id=1, value=value, timestamp=datetime.now(timezone.utc))
    return WorldEntry(
        key=key,
        type=type(value) if value is not None else str,
        bypass_coalescing=False,
        current=version,
        previous=None,
    )


def command_entry(
    call_id: str, name: str, args: dict[str, Any], state: str, **kw: Any
) -> WorldEntry:
    """The agent:command:<call_id> entry snapshot a listener receives at a given state."""
    execution = CommandExecution(name=name, args=args, state=state, **kw)  # type: ignore[arg-type]
    return make_entry(f"agent:command:{call_id}", execution)


@pytest.fixture
def unstarted_wica() -> Iterator[Callable[[str | None], Wica]]:
    """A factory for an unstarted (no loop thread, no network) `Wica` over the fake provider, with
    an output Command of the given name set — what a log needs to read the voice's name live and
    to bind its per-Command World listeners. Closed on teardown."""
    created: list[Wica] = []

    def _make(output_command_name: str | None) -> Wica:
        config = WicaConfig.from_dict(
            {
                "agent": {
                    "provider": "fake",
                    "model": "scripted",
                    "system_prompt": "You are a test robot.",
                    "model_kwargs": {"delay_s": 0, "script": [{"text": ""}]},
                }
            }
        )
        wica = Wica.init(config)
        if output_command_name is not None:

            def voice(text: str) -> str:
                """The voice."""
                return text

            wica.set_output_command(Command(voice, name=output_command_name))
        created.append(wica)
        return wica

    yield _make
    for wica in created:
        wica.close()


def titles(conversation: list[dict[str, Any]]) -> list[str | None]:
    return [m.get("metadata", {}).get("title") for m in conversation]


def item_titled(conversation: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        m
        for m in conversation
        if str(m.get("metadata", {}).get("title", "")).startswith(prefix)
    ]
    assert len(matches) == 1, titles(conversation)
    return matches[0]


# --- pure rendering helpers ----------------------------------------------------------


def test_flatten_content_string_passthrough():
    assert _flatten_content("hello world") == "hello world"


def test_flatten_content_concatenates_text_blocks_without_extra_newlines():
    blocks = [{"type": "text", "text": "<entry>\n"}, {"type": "text", "text": "body"}]
    # Concatenated, not newline-joined, so the model-facing text is reproduced exactly.
    assert _flatten_content(blocks) == "<entry>\nbody"


def test_flatten_content_marks_image_blocks():
    assert _flatten_content([{"type": "image", "mime_type": "image/png"}]) == (
        "[image image/png]"
    )


def test_render_message_ai_with_text_and_tool_call():
    m = AIMessage(
        content="thinking about it",
        tool_calls=[
            {"name": "say", "args": {"text": "hi"}, "id": "c1", "type": "tool_call"}
        ],
    )
    out = _render_message(m)
    assert "### AI" in out
    assert "thinking about it" in out
    assert "tool_call say(text='hi')" in out
    assert "c1" in out


def test_render_message_tool_result_shows_call_id():
    out = _render_message(ToolMessage(content="4", tool_call_id="c1"))
    assert "result for [id=c1]" in out
    assert "4" in out


# --- display: the generic default and the demo's hook ---------------------------------


def test_default_display_is_generic_for_entries_and_commands(unstarted_wica):
    log = TranscriptLog(unstarted_wica("speak"))
    assert log.display(make_entry("temperature", 21)) == EntryDisplay(
        "⚡ temperature = 21"
    )
    assert log.display(
        command_entry("c1", "walk_to", {"room": "kitchen"}, "running")
    ) == (EntryDisplay("🦾 walk_to", "walk_to(room='kitchen')"))
    # The configured output Command is the voice, labelled as such without any hook.
    assert log.display(command_entry("c2", "speak", {"text": "hi"}, "running")) == (
        EntryDisplay("🗣️ speak", "speak(text='hi')")
    )


def test_hook_overrides_and_none_falls_back_to_default():
    def hook(entry: WorldEntry) -> EntryDisplay | None:
        if entry.key == "mood":
            return EntryDisplay(f"🙂 mood is {entry.current.value}")
        return None

    log = TranscriptLog(None, hook)
    assert log.display(make_entry("mood", "good")).label == "🙂 mood is good"
    assert log.display(make_entry("other", 1)).label == "⚡ other = 1"


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


# --- TranscriptLog: trigger / prompt / sink callbacks ---------------------------------


def test_on_trigger_shows_input_on_the_user_side_via_the_hook():
    log = TranscriptLog(None, display_entry)
    log.on_trigger(make_entry("speech_input", "hi"))
    assert log.snapshot().conversation == [{"role": "user", "content": '🗣️ "hi"'}]


def test_on_trigger_skips_command_retrigger_but_labels_the_next_prompt():
    """A command-completion re-trigger is not shown on the transcript (its item already carries
    the outcome), but its label is still recorded so the next prompt is tagged with it."""
    log = TranscriptLog(None)
    log.on_trigger(command_entry("7", "dance", {}, "complete", result="done"))
    assert log.snapshot().conversation == []  # nothing added on the transcript

    log.on_prompt([HumanMessage(content="prompt body")])
    label = log.snapshot().prompt_choices[0][0]
    assert "🦾 dance finished" in label


def test_on_prompt_records_prompt_and_opens_a_reaction_group():
    log = TranscriptLog(None, display_entry)
    log.on_trigger(make_entry("speech_input", "hello robot"))
    log.on_prompt([HumanMessage(content="hello robot")])
    snap = log.snapshot()

    assert snap.prompt_count == 1
    assert "hello robot" in (snap.newest_prompt_text or "")
    # A single reaction-group header was opened for the step, labelled by its trigger.
    group = [m for m in snap.conversation if m.get("metadata", {}).get("id")]
    assert len(group) == 1
    assert group[0]["metadata"]["id"] == "reaction-1"
    assert group[0]["metadata"]["title"] == '💬 reaction 1 · 🗣️ "hello robot"'


def test_output_sink_ignores_blank_and_records_private_reasoning():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    asyncio.run(log.output_sink("   "))
    assert titles(log.snapshot().conversation).count("💭 output sink") == 0

    asyncio.run(log.output_sink("private reasoning"))
    thought = item_titled(log.snapshot().conversation, "💭 output sink")
    assert thought["content"] == "private reasoning"
    assert thought["metadata"]["parent_id"] == "reaction-1"


# --- TranscriptLog: Command items and their live state --------------------------------


def test_on_command_creates_a_pending_item_under_the_current_reaction():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])  # opens reaction-1
    log.on_command(CommandIssued(name="dance", args={}, call_id="c1"))
    item = item_titled(log.snapshot().conversation, "🦾 dance")

    assert item["content"] == "dance()"
    assert item["metadata"]["parent_id"] == "reaction-1"
    assert (
        item["metadata"]["status"] == "pending"
    )  # in progress from the moment it's issued


def test_noop_is_shown_as_choosing_not_to_react_without_state():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    log.on_command(CommandIssued(name="noop", args={}, call_id="c2"))
    item = item_titled(log.snapshot().conversation, "🚫 noop")
    assert item["content"] == "chose not to react"
    assert "status" not in item["metadata"]  # no execution to follow


def test_command_item_follows_running_then_complete(unstarted_wica):
    """Over a real (unstarted) Wica the log labels the voice by the Agent's output Command and
    binds a listener on the Command's entry — which the Agent registers before emitting
    on_command, so the test registers it the same way."""
    wica = unstarted_wica("say")
    log = TranscriptLog(wica, display_entry)
    log.on_prompt([HumanMessage(content="hi")])
    wica.world.register(
        "agent:command:c1", CommandExecution, serialize_fn=lambda value, previous: []
    )
    log.on_command(CommandIssued(name="say", args={"text": "hi there"}, call_id="c1"))

    asyncio.run(
        log.on_command_update(
            command_entry("c1", "say", {"text": "hi there"}, "running")
        )
    )
    item = item_titled(log.snapshot().conversation, "🗣️ say")
    # Re-rendered through the hook: channel title, spoken text as body, still in progress.
    assert item["content"] == "hi there"
    assert item["metadata"]["status"] == "pending"

    asyncio.run(
        log.on_command_update(
            command_entry(
                "c1", "say", {"text": "hi there"}, "complete", result="Said it."
            )
        )
    )
    snap = log.snapshot()
    item = item_titled(snap.conversation, "🗣️ say")
    assert item["metadata"]["title"] == "🗣️ say ✅"
    # No status once it ends: the spinner goes, and the item stays open (Gradio collapses a
    # "done" item — see on_command_update).
    assert "status" not in item["metadata"]
    assert item["content"] == "hi there\n→ Said it."
    assert isinstance(item["metadata"]["duration"], float)
    assert len(snap.conversation) == 2  # edited in place, not appended


def test_command_item_shows_failure_and_cancellation():
    log = TranscriptLog(None)
    log.on_command(CommandIssued(name="dance", args={}, call_id="c1"))
    log.on_command(CommandIssued(name="say", args={"text": "x"}, call_id="c2"))

    asyncio.run(
        log.on_command_update(
            command_entry("c1", "dance", {}, "failed", error="tripped")
        )
    )
    asyncio.run(
        log.on_command_update(command_entry("c2", "say", {"text": "x"}, "cancelled"))
    )
    convo = log.snapshot().conversation

    failed = item_titled(convo, "🦾 dance")
    assert failed["metadata"]["title"] == "🦾 dance ❌ failed"
    assert failed["content"] == "dance()\n✗ tripped"
    assert "status" not in failed["metadata"]

    cancelled = item_titled(convo, "🦾 say")
    assert cancelled["metadata"]["title"] == "🦾 say ⏹ cancelled"
    assert cancelled["content"] == "say(text='x')"
    assert "status" not in cancelled["metadata"]


def test_command_update_for_unknown_or_foreign_entries_is_ignored():
    log = TranscriptLog(None)
    log.on_command(CommandIssued(name="dance", args={}, call_id="c1"))
    before = log.snapshot()
    asyncio.run(log.on_command_update(command_entry("zzz", "dance", {}, "complete")))
    asyncio.run(log.on_command_update(make_entry("agent:command:c1", None)))
    after = log.snapshot()
    assert after.conv_sig == before.conv_sig


def test_signature_tracks_in_place_state_edits_and_is_stable_otherwise():
    log = TranscriptLog(None)
    log.on_command(CommandIssued(name="dance", args={}, call_id="c1"))
    first = log.snapshot()
    assert log.snapshot().conv_sig == first.conv_sig  # nothing changed between ticks

    asyncio.run(log.on_command_update(command_entry("c1", "dance", {}, "complete")))
    assert log.snapshot().conv_sig != first.conv_sig  # a state flip is a real change


def test_prompt_text_indexing():
    log = TranscriptLog(None)
    assert log.prompt_text(None) is None
    assert log.prompt_text(0) is None  # nothing captured yet

    log.on_prompt([HumanMessage(content="first")])
    log.on_prompt([HumanMessage(content="second")])
    assert "first" in (log.prompt_text(0) or "")
    assert "second" in (log.prompt_text(1) or "")
    assert log.prompt_text(2) is None  # out of range


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
