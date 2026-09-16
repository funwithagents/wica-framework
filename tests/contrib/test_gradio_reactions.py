"""Fast, deterministic tests of the contrib's reaction history: `render_messages`/
`render_reaction_trace` (what the panel shows) and `ReactionLog` (what it keeps), driven the way
the Agent drives it — a trigger, then the prompt, then (later) the reaction's trace — without a
browser. See specs/gradio-contrib.md ("Component 2", "Testing").
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.support import command_entry, make_entry, make_reaction_trace
from wica import TokenUsage, WorldEntry
from wica.contrib.gradio import (
    EntryDisplay,
    ReactionLog,
    render_messages,
    render_reaction_trace,
)
from wica.contrib.gradio.reactions import _flatten_content

# --- rendering -----------------------------------------------------------------------


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


def test_render_messages_ai_with_text_and_tool_call():
    m = AIMessage(
        content="thinking about it",
        tool_calls=[
            {"name": "say", "args": {"text": "hi"}, "id": "c1", "type": "tool_call"}
        ],
    )
    out = render_messages([m])
    assert "### AI" in out
    assert "thinking about it" in out
    assert "tool_call say(text='hi')" in out
    assert "c1" in out


def test_render_messages_tool_result_shows_call_id():
    out = render_messages([ToolMessage(content="4", tool_call_id="c1")])
    assert "result for [id=c1]" in out
    assert "4" in out


def test_render_messages_separates_messages_with_a_blank_line():
    out = render_messages([HumanMessage(content="first"), AIMessage(content="second")])
    assert out == "### HUMAN\nfirst\n\n### AI\nsecond"


def test_render_reaction_trace_shows_only_present_measures():
    cancelled = make_reaction_trace(
        outcome="cancelled",
        prompt_ready_at=None,
        model_started_at=None,
        model_ended_at=None,
        sink_duration=None,
    )
    out = render_reaction_trace(cancelled)
    assert "model latency" not in out
    assert "render time" not in out
    assert "sink duration" not in out
    assert (
        "busy time" in out
    )  # always present — derived from window_closed_at/ended_at only

    normal = make_reaction_trace()
    out = render_reaction_trace(normal)
    assert "model latency" in out
    assert "render time" in out
    assert "sink duration" in out


def test_render_reaction_trace_shows_usage_commands_noop_and_error():
    with_usage = make_reaction_trace(
        usage=TokenUsage(3, 2, None), command_call_ids=("c1",)
    )
    out = render_reaction_trace(with_usage)
    assert "tokens: 3 in / 2 out" in out
    assert "commands: c1" in out

    noop = make_reaction_trace(noop=True)
    assert "noop" in render_reaction_trace(noop)

    failed = make_reaction_trace(outcome="model_error", error="boom")
    assert "error: boom" in render_reaction_trace(failed)


# --- the log -------------------------------------------------------------------------


def hook(entry: WorldEntry) -> EntryDisplay | None:
    if entry.key == "speech_input":
        return EntryDisplay(f'🗣️ "{entry.current.value}"')
    return None


_LABEL = re.compile(r"^\d\d:\d\d:\d\d — (.*)$")


def trigger_part(label: str) -> str:
    match = _LABEL.match(label)
    assert match, label
    return match.group(1)


def test_empty_log_snapshot_and_text():
    log = ReactionLog(None)
    snap = log.snapshot()
    assert (snap.count, snap.choices) == (0, [])
    assert log.prompt_text(None) is None
    assert log.prompt_text(0) is None  # nothing captured yet
    assert log.instrumentation_text(None) is None
    assert log.instrumentation_text(0) is None  # out of range, not pending


def test_prompts_are_labelled_by_time_and_hook_rendered_trigger():
    log = ReactionLog(None, hook)
    log.on_prompt([HumanMessage(content="boot")])  # before any trigger
    log.on_trigger(make_entry("speech_input", "hi"))
    log.on_prompt([HumanMessage(content="after hi")])
    log.on_trigger(make_entry("emotion", "happy"))  # not handled by the hook → generic
    log.on_prompt([HumanMessage(content="after emotion")])

    snap = log.snapshot()
    assert snap.count == 3
    assert [i for _, i in snap.choices] == [0, 1, 2]
    assert [trigger_part(label) for label, _ in snap.choices] == [
        "start",
        '🗣️ "hi"',
        "⚡ emotion = 'happy'",
    ]
    assert "after emotion" in (log.prompt_text(2) or "")


def test_command_completion_retrigger_is_labelled_finished():
    log = ReactionLog(None)
    log.on_trigger(command_entry("7", "dance", {}, "complete", result="done"))
    log.on_prompt([HumanMessage(content="prompt body")])
    label, _ = log.snapshot().choices[0]
    assert trigger_part(label) == "🦾 dance finished"


def test_voice_is_labelled_from_the_agents_output_command(unstarted_wica):
    log = ReactionLog(unstarted_wica("say"))
    log.on_trigger(command_entry("1", "say", {"text": "x"}, "complete"))
    log.on_prompt([HumanMessage(content="p")])
    assert trigger_part(log.snapshot().choices[0][0]) == "🗣️ say finished"


def test_prompt_text_indexes_the_history_and_bounds_check():
    log = ReactionLog(None)
    log.on_prompt([HumanMessage(content="first")])
    log.on_prompt([HumanMessage(content="second")])
    assert "first" in (log.prompt_text(0) or "")
    assert "second" in (log.prompt_text(1) or "")
    assert log.prompt_text(2) is None  # out of range
    assert log.prompt_text(-1) is None


# --- the merge: instrumentation opens pending, is filled in by on_reaction_ended -----------


def test_instrumentation_is_pending_until_the_reaction_ends():
    log = ReactionLog(None)
    log.on_prompt([HumanMessage(content="hi")])

    pending = log.instrumentation_text(0)
    assert pending is not None and "in progress" in pending

    trace = make_reaction_trace(reaction_id=1)
    log.on_reaction_ended(trace)
    assert log.instrumentation_text(0) == render_reaction_trace(trace)


def test_on_reaction_ended_for_an_unknown_index_is_ignored():
    log = ReactionLog(None)
    log.on_prompt([HumanMessage(content="hi")])
    before = log.snapshot()

    log.on_reaction_ended(make_reaction_trace(reaction_id=7))

    assert log.snapshot() == before
    pending = log.instrumentation_text(0)
    assert pending is not None and "in progress" in pending
