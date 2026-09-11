"""Fast, deterministic unit tests for the conversation demo's presenter.

These pin the `examples/conversation_demo/app_state.py` glue — the seam between the agent's
instrumentation callbacks and the Gradio UI — without any network, Gradio, or event loop. They
import only `app_state`, which needs core deps (wica + langchain_core), so they run in the default
`tests/` tier. The full end-to-end wiring (World entries + Commands + presenter, driven by a
scripted fake model) is covered separately in tests-e2e/test_example_flow.py.

See specs/conversation-demo.md.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from examples.conversation_demo.app_state import (
    DemoState,
    _describe_trigger,
    _flatten_content,
    _render_message,
)
from wica import CommandIssued, WorldEntry, WorldEntryVersion
from wica.agent import CommandExecution


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


def titles(conversation: list[dict[str, Any]]) -> list[str | None]:
    return [m.get("metadata", {}).get("title") for m in conversation]


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


def test_describe_trigger_labels_each_kind():
    assert _describe_trigger(make_entry("speech_input", "hi")) == '🗣️ "hi"'
    assert _describe_trigger(make_entry("closest_user", "alice")) == (
        "👤 Closest user detected: alice"
    )
    assert _describe_trigger(make_entry("closest_user", None)) == "👤 Closest user gone"
    execution = CommandExecution(name="dance", args={}, state="complete", result="done")
    assert _describe_trigger(make_entry("agent:command:1", execution)) == (
        "⚡ command finished: dance"
    )


# --- DemoState: trigger / prompt / command / sink callbacks ---------------------------


def test_on_trigger_shows_input_on_the_user_side():
    state = DemoState()
    state.on_trigger(make_entry("speech_input", "hi"))
    convo = state.snapshot().conversation
    assert convo == [{"role": "user", "content": '🗣️ "hi"'}]


def test_on_trigger_skips_command_retrigger_but_labels_the_next_prompt():
    """A command-completion re-trigger is not shown on the transcript (it's already visible as a
    command call), but its label is still recorded so the next prompt is tagged with it."""
    state = DemoState()
    execution = CommandExecution(name="dance", args={}, state="complete", result="done")
    state.on_trigger(make_entry("agent:command:7", execution))
    assert state.snapshot().conversation == []  # nothing added on the transcript

    state.on_prompt([HumanMessage(content="prompt body")])
    label = state.snapshot().prompt_choices[0][0]
    assert "command finished: dance" in label


def test_on_prompt_records_prompt_and_opens_a_reaction_group():
    state = DemoState()
    state.on_prompt([HumanMessage(content="hello robot")])
    snap = state.snapshot()

    assert snap.prompt_count == 1
    assert "hello robot" in (snap.newest_prompt_text or "")
    # A single reaction-group header was opened for the step.
    group = [m for m in snap.conversation if m.get("metadata", {}).get("id")]
    assert len(group) == 1
    assert group[0]["metadata"]["id"] == "reaction-1"
    assert group[0]["metadata"]["title"].startswith("💬 reaction 1")


def test_on_command_nests_actions_under_the_current_reaction():
    state = DemoState()
    state.on_prompt([HumanMessage(content="hi")])  # opens reaction-1
    state.on_command(CommandIssued(name="dance", args={}))
    convo = state.snapshot().conversation

    action = [m for m in convo if m.get("metadata", {}).get("title") == "🦾 dance"]
    assert len(action) == 1
    assert action[0]["content"] == "dance()"
    assert action[0]["metadata"]["parent_id"] == "reaction-1"


def test_on_command_skips_say_and_marks_noop():
    state = DemoState()
    state.on_prompt([HumanMessage(content="hi")])
    state.on_command(
        CommandIssued(name="say", args={"text": "hi"})
    )  # the voice, not a 🦾
    state.on_command(CommandIssued(name="noop", args={}))
    convo_titles = titles(state.snapshot().conversation)

    assert "🗣️ say" not in convo_titles  # say is rendered by say(), not on_command
    assert "🚫 noop" in convo_titles


def test_output_sink_ignores_blank_and_records_private_reasoning():
    state = DemoState()
    state.on_prompt([HumanMessage(content="hi")])
    asyncio.run(state.output_sink("   "))
    assert titles(state.snapshot().conversation).count("💭 output sink") == 0

    asyncio.run(state.output_sink("private reasoning"))
    convo = state.snapshot().conversation
    thought = [
        m for m in convo if m.get("metadata", {}).get("title") == "💭 output sink"
    ]
    assert len(thought) == 1
    assert thought[0]["content"] == "private reasoning"


# --- DemoState: say streaming + snapshot/prompt reads --------------------------------


def test_open_say_bubble_streams_content_and_changes_the_signature():
    state = DemoState()
    state.on_prompt([HumanMessage(content="hi")])
    message = state.open_say_bubble("Hello", state.current_reaction_id())
    assert message["metadata"]["title"] == "🗣️ say"
    assert message["metadata"]["parent_id"] == "reaction-1"

    sig_before = state.snapshot().conv_sig
    message["content"] = "Hello there"  # grow the bubble in place, as say() does
    snap = state.snapshot()
    assert snap.conv_sig != sig_before  # an in-place edit is a real change
    say_bubble = [
        m for m in snap.conversation if m.get("metadata", {}).get("title") == "🗣️ say"
    ]
    assert say_bubble[0]["content"] == "Hello there"


def test_open_say_bubble_honors_the_captured_reaction_not_the_current_one():
    """`say` captures its reaction before streaming; a newer reaction opening mid-stream must not
    adopt its words. open_say_bubble binds the parent passed in, not the current reaction — the fix
    for the streaming-say nesting bug (specs/_fixes.md)."""
    state = DemoState()
    state.on_prompt([HumanMessage(content="first")])  # reaction-1
    captured = state.current_reaction_id()
    state.on_prompt([HumanMessage(content="second")])  # reaction-2 opens mid-stream
    assert state.current_reaction_id() == "reaction-2"

    message = state.open_say_bubble("hi", captured)
    assert (
        message["metadata"]["parent_id"] == "reaction-1"
    )  # captured, not the current reaction-2


def test_snapshot_signature_is_stable_without_changes():
    state = DemoState()
    state.on_trigger(make_entry("speech_input", "hi"))
    first = state.snapshot()
    second = state.snapshot()
    assert first.conv_sig == second.conv_sig  # nothing changed between ticks


def test_prompt_text_indexing():
    state = DemoState()
    assert state.prompt_text(None) is None
    assert state.prompt_text(0) is None  # nothing captured yet

    state.on_prompt([HumanMessage(content="first")])
    state.on_prompt([HumanMessage(content="second")])
    assert "first" in (state.prompt_text(0) or "")
    assert "second" in (state.prompt_text(1) or "")
    assert state.prompt_text(2) is None  # out of range
