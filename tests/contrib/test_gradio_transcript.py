"""Fast, deterministic tests of the contrib's conversation transcript (`TranscriptLog`) — the
seam between the Agent's instrumentation callbacks and the chatbot — driven the way the Agent
drives it, without a browser or a `gr.Blocks`. Where a log needs a `Wica` (to read the output
Command's name live, or to bind World listeners) the tests build an **unstarted** one over the
key-less `provider: "fake"` model. See specs/gradio-contrib.md ("Component 3", "Testing").
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from tests.support import command_entry, item_titled, make_entry, titles
from wica import CommandExecution, CommandIssued, WorldEntry
from wica.contrib.gradio import EntryDisplay, TranscriptLog


def robot_hook(entry: WorldEntry) -> EntryDisplay | None:
    """A demo-like application hook: labels one input and the voice, leaves the rest generic."""
    value = entry.current.value
    if entry.key == "speech_input":
        return EntryDisplay(f'🗣️ "{value}"')
    if isinstance(value, CommandExecution) and value.name == "say":
        return EntryDisplay("🗣️ say", str(value.args.get("text", "")))
    return None


# --- display: the generic default and the hook ---------------------------------------


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


# --- trigger / prompt / sink callbacks -----------------------------------------------


def test_on_trigger_shows_input_on_the_user_side_via_the_hook():
    log = TranscriptLog(None, robot_hook)
    log.on_trigger(make_entry("speech_input", "hi"))
    assert log.snapshot().conversation == [{"role": "user", "content": '🗣️ "hi"'}]


def test_on_trigger_skips_command_retrigger_but_labels_the_next_reaction():
    """A command-completion re-trigger is not shown on the transcript (its item already carries
    the outcome), but its label still titles the reaction group the next prompt opens."""
    log = TranscriptLog(None)
    log.on_trigger(command_entry("7", "dance", {}, "complete", result="done"))
    assert log.snapshot().conversation == []  # nothing added on the transcript

    log.on_prompt([HumanMessage(content="prompt body")])
    group = item_titled(log.snapshot().conversation, "💬 reaction 1")
    assert group["metadata"]["title"] == "💬 reaction 1 · 🦾 dance finished"


def test_on_prompt_opens_a_reaction_group_titled_by_the_trigger():
    log = TranscriptLog(None, robot_hook)
    log.on_trigger(make_entry("speech_input", "hello robot"))
    log.on_prompt([HumanMessage(content="hello robot")])
    snap = log.snapshot()

    # A single reaction-group header was opened for the step, labelled by its trigger.
    group = [m for m in snap.conversation if m.get("metadata", {}).get("id")]
    assert len(group) == 1
    assert group[0]["metadata"]["id"] == "reaction-1"
    assert group[0]["metadata"]["title"] == '💬 reaction 1 · 🗣️ "hello robot"'
    assert log.current_reaction_id() == "reaction-1"


def test_output_sink_ignores_blank_and_records_private_reasoning():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    asyncio.run(log.output_sink("   "))
    assert titles(log.snapshot().conversation).count("💭 output sink") == 0

    asyncio.run(log.output_sink("private reasoning"))
    thought = item_titled(log.snapshot().conversation, "💭 output sink")
    assert thought["content"] == "private reasoning"
    assert thought["metadata"]["parent_id"] == "reaction-1"


# --- Command items and their live state ----------------------------------------------


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
    log = TranscriptLog(wica, robot_hook)
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
