"""Fast, deterministic tests of the contrib's prompt history: `render_messages` (what the panel
shows) and `PromptLog` (what it keeps), driven the way the Agent drives it — a trigger, then the
prompt — without a browser. See specs/gradio-contrib.md ("Component 2", "Testing").
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.support import command_entry, make_entry
from wica import WorldEntry
from wica.contrib.gradio import EntryDisplay, PromptLog, render_messages
from wica.contrib.gradio.prompts import _flatten_content

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
    log = PromptLog(None)
    snap = log.snapshot()
    assert (snap.count, snap.choices, snap.newest_text) == (0, [], None)
    assert log.text(None) is None
    assert log.text(0) is None  # nothing captured yet


def test_prompts_are_labelled_by_time_and_hook_rendered_trigger():
    log = PromptLog(None, hook)
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
    assert "after emotion" in (snap.newest_text or "")


def test_command_completion_retrigger_is_labelled_finished():
    log = PromptLog(None)
    log.on_trigger(command_entry("7", "dance", {}, "complete", result="done"))
    log.on_prompt([HumanMessage(content="prompt body")])
    label, _ = log.snapshot().choices[0]
    assert trigger_part(label) == "🦾 dance finished"


def test_voice_is_labelled_from_the_agents_output_command(unstarted_wica):
    log = PromptLog(unstarted_wica("say"))
    log.on_trigger(command_entry("1", "say", {"text": "x"}, "complete"))
    log.on_prompt([HumanMessage(content="p")])
    assert trigger_part(log.snapshot().choices[0][0]) == "🗣️ say finished"


def test_text_indexes_the_history_and_bounds_check():
    log = PromptLog(None)
    log.on_prompt([HumanMessage(content="first")])
    log.on_prompt([HumanMessage(content="second")])
    assert "first" in (log.text(0) or "")
    assert "second" in (log.text(1) or "")
    assert log.text(2) is None  # out of range
    assert log.text(-1) is None
