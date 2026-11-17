"""Fast, deterministic tests of the contrib's conversation transcript (`TranscriptLog`) — the
seam between the Agent's instrumentation callbacks and the chatbot — driven the way the Agent
drives it, without a browser or a `gr.Blocks`. Where a log needs a `Wica` (to read the output
Command's name live, or to bind World listeners) the tests build an **unstarted** one over the
key-less `provider: "fake"` model. See specs/gradio-contrib.md ("Component 3", "Testing").
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage

from tests.support import command_entry, item_titled, make_entry, titles
from wica import (
    Command,
    CommandExecution,
    CommandIssued,
    OutputStream,
    ReactionTrace,
    TextPart,
    Wica,
    WicaConfig,
    WorldEntry,
)
from wica.contrib.gradio import EntryDisplay, TranscriptLog


def reaction_trace(
    reaction_id: int, busy_time: float, *, noop: bool = False
) -> ReactionTrace:
    """A hand-built ReactionTrace for on_reaction_ended tests — only reaction_id and busy_time
    (derived from window_closed_at -> ended_at) matter to the transcript."""
    t0 = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
    return ReactionTrace(
        reaction_id=reaction_id,
        triggers=(),
        window_opened_at=t0,
        window_closed_at=t0,
        prompt_ready_at=t0,
        model_started_at=t0,
        model_ended_at=t0,
        outcome="ok",
        error=None,
        text_length=0,
        command_call_ids=(),
        noop=noop,
        usage=None,
        ended_at=t0 + timedelta(seconds=busy_time),
        trace_id=None,
        span_id=None,
    )


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


# --- trigger / prompt / private-text callbacks ---------------------------------------


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


def test_reaction_group_is_pending_until_the_reaction_ends():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    group = item_titled(log.snapshot().conversation, "💬 reaction 1")
    assert group["metadata"]["status"] == "pending"

    log.on_reaction_ended(reaction_trace(1, 1.2))
    group = item_titled(log.snapshot().conversation, "💬 reaction 1")
    assert "status" not in group["metadata"]
    assert group["metadata"]["duration"] == 1.2


def test_reaction_ended_for_an_unknown_id_is_ignored():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    before = log.snapshot()

    log.on_reaction_ended(reaction_trace(7, 1.0))

    after = log.snapshot()
    assert after.conv_sig == before.conv_sig


def test_private_text_ignores_blank_and_is_shown_under_the_reaction():
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    log.on_private_text("   ")
    assert titles(log.snapshot().conversation).count("💭 private text") == 0

    log.on_private_text("a note to myself")
    thought = item_titled(log.snapshot().conversation, "💭 private text")
    assert thought["content"] == "a note to myself"
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


def test_declined_reaction_is_shown_from_its_record_without_state():
    """A declined reaction issues no Command: the item comes from the reaction record's `noop`
    flag, nested under that reaction's group, and a record without the flag adds nothing."""
    log = TranscriptLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    log.on_reaction_ended(reaction_trace(1, 0.5))
    assert "🚫 noop" not in titles(log.snapshot().conversation)

    log.on_prompt([HumanMessage(content="a background change")])
    log.on_reaction_ended(reaction_trace(2, 0.5, noop=True))
    item = item_titled(log.snapshot().conversation, "🚫 noop")
    assert item["content"] == "chose not to react"
    assert item["metadata"]["parent_id"] == "reaction-2"
    assert "status" not in item["metadata"]  # no execution to follow


def test_output_item_is_created_from_an_entry_already_running(unstarted_wica):
    """An output is reported once its text is complete, its entry by then already `running`: no
    listener call will ever carry that state, so the log reads it when it attaches — the item
    is rendered through the hook at once, and its duration counts from that version."""
    wica = unstarted_wica("say")
    log = TranscriptLog(wica, robot_hook)
    log.on_prompt([HumanMessage(content="hi")])
    key = "agent:command:o1"
    wica.world.register(key, CommandExecution, serialize_fn=lambda value, previous: [])
    wica.start()  # the World takes updates only while running
    wica.world.update(
        key, CommandExecution(name="say", args={"text": "hi there"}, state="running")
    )
    running_at = wica.world.get_entry(key).current.timestamp

    log.on_command(CommandIssued(name="say", args={"text": "hi there"}, call_id="o1"))
    item = item_titled(log.snapshot().conversation, "🗣️ say")
    assert item["content"] == "hi there"  # through the hook, not `say(text=…)`
    assert item["metadata"]["status"] == "pending"

    ended = command_entry("o1", "say", {"text": "hi there"}, "cancelled")
    ended = replace(
        ended,
        current=replace(ended.current, timestamp=running_at + timedelta(seconds=2.5)),
    )
    asyncio.run(log.on_command_update(ended))
    item = item_titled(log.snapshot().conversation, "🗣️ say")
    assert item["metadata"]["title"] == "🗣️ say ⏹ cancelled"
    assert item["metadata"]["duration"] == 2.5
    assert "status" not in item["metadata"]


def test_output_item_is_created_from_an_entry_already_over(unstarted_wica):
    """A short output may be over before its text is reported: the item shows the outcome at
    once, timed from the running version before it."""
    wica = unstarted_wica("say")
    log = TranscriptLog(wica, robot_hook)
    key = "agent:command:o1"
    wica.world.register(key, CommandExecution, serialize_fn=lambda value, previous: [])
    wica.start()
    wica.world.update(
        key, CommandExecution(name="say", args={"text": "hi"}, state="running")
    )
    wica.world.update(
        key,
        CommandExecution(
            name="say", args={"text": "hi"}, state="complete", result="Said."
        ),
    )
    entry = wica.world.get_entry(key)
    assert entry.previous is not None
    ran = (entry.current.timestamp - entry.previous.timestamp).total_seconds()

    log.on_command(CommandIssued(name="say", args={"text": "hi"}, call_id="o1"))
    item = item_titled(log.snapshot().conversation, "🗣️ say")
    assert item["metadata"]["title"] == "🗣️ say ✅"
    assert item["content"] == "hi\n→ Said."
    assert "status" not in item["metadata"]
    assert item["metadata"]["duration"] == round(ran, 1)


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


# --- over a running Wica: the transcript of real reactions ---------------------------


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_transcript_of_a_spoken_reaction_then_a_declined_one():
    """The whole seam over a scripted model: a response with private text and two output tags
    gives one 🗣️ item per tag (each with its whole text and its outcome) and a 💭 item; a
    `<noop/>` response gives a 🚫 item under its own reaction and no Command item."""
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
                            "text": "They greeted me. <say>Hello there</say>"
                            '<say emotion="happy">How are you?</say>'
                        },
                        {"text": "<noop/>"},
                    ],
                },
            }
        }
    )
    wica = Wica.init(config)
    try:

        async def say(text: OutputStream, emotion: str = "neutral") -> str:
            """Speak to the person."""
            return f"Said {len((await text.read()).split())} words."

        wica.set_output_command(Command(say))
        for key in ("speech_input", "light"):
            wica.world.register(
                key,
                str,
                serialize_fn=lambda v, p, key=key: [TextPart(f"{key}: {v}")],
                triggers_llm_call=True,
            )
        log = TranscriptLog(wica)
        wica.start()

        def finished(count: int) -> bool:
            return (
                sum(
                    str(t).startswith("🗣️ say ✅")
                    for t in titles(log.snapshot().conversation)
                )
                == count
            )

        wica.world.update("speech_input", "hello")
        # Both outputs over and the reaction ended: the next input is not dropped as busy.
        _wait_for(
            lambda: (
                finished(2)
                and "status"
                not in item_titled(log.snapshot().conversation, "💬 reaction 1")[
                    "metadata"
                ]
            )
        )
        wica.world.update("light", "on")
        _wait_for(lambda: "🚫 noop" in titles(log.snapshot().conversation))
        convo = log.snapshot().conversation
    finally:
        wica.close()

    spoken = [
        m for m in convo if str(m.get("metadata", {}).get("title", "")).startswith("🗣️")
    ]
    assert [m["content"] for m in spoken] == [
        "say(text='Hello there')\n→ Said 2 words.",
        "say(emotion='happy', text='How are you?')\n→ Said 3 words.",
    ]
    assert all(m["metadata"]["parent_id"] == "reaction-1" for m in spoken)
    assert all("status" not in m["metadata"] for m in spoken)
    private = item_titled(convo, "💭 private text")
    assert private["content"] == "They greeted me."
    assert private["metadata"]["parent_id"] == "reaction-1"
    # The declined reaction: one item under its own group, and no Command item for it.
    noop = item_titled(convo, "🚫 noop")
    assert noop["metadata"]["parent_id"] == "reaction-2"
    assert [t for t in titles(convo) if str(t).startswith(("🗣️", "🦾"))] == [
        "🗣️ say ✅",
        "🗣️ say ✅",
    ]
