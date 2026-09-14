"""The prompt-history panel: every prompt sent to the model, browsable step by step.

A prompt exists only for the instant `on_agent_prompt` fires, on the agent loop, while the UI
reads on another thread. `PromptLog` keeps an append-only, thread-safe history of them, each
labelled by time and by the trigger that caused its step (rendered through the application's
`display_entry` hook, so the labels match the transcript's). `prompt_panel` shows that history:
a dropdown of steps and the selected prompt's exact text, snapping to the newest step when one
appears and otherwise leaving the user free to browse. See specs/gradio-contrib.md
("Component 2").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from wica import CommandExecution, Wica, WorldEntry
from wica.contrib.gradio.display import (
    DisplayEntry,
    display_entry_or_default,
    format_args,
)

_NO_PROMPT_YET = "(no prompt sent to the model yet)"


# --- rendering ---------------------------------------------------------------------------


def _flatten_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "image":
                parts.append(f"[image {block.get('mime_type', '')}]")
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    # Concatenate (don't newline-join): each World-rendered block already carries its own
    # newlines (an `<entry ...>\n` opener, a `\n...</entry>\n` closer), so this shows the exact
    # text the model receives when the provider merges adjacent blocks — no phantom blank lines.
    return "".join(parts)


def _render_message(m: BaseMessage) -> str:
    """Render one LangChain message to text. An assistant message that only issues a command has
    empty .content — the call lives in the structured .tool_calls field — so we render those (and
    the tool_call id on a tool result) explicitly, or the AI block looks blank."""
    lines = [f"### {m.type.upper()}"]
    if isinstance(m, ToolMessage):
        lines.append(
            f"🔧 result for [id={m.tool_call_id}] => {_flatten_content(m.content)}"
        )
        return "\n".join(lines)
    body = _flatten_content(m.content)
    if body:
        lines.append(body)
    if isinstance(m, AIMessage):
        for call in m.tool_calls:
            lines.append(
                f"🔧 tool_call {call['name']}({format_args(call['args'])})  [id={call['id']}]"
            )
    return "\n".join(lines)


def render_messages(messages: list[BaseMessage]) -> str:
    """The exact messages of one model call as readable text: a `### ROLE` header per message,
    content blocks flattened, tool calls and tool results shown explicitly."""
    return "\n\n".join(_render_message(m) for m in messages)


# --- the log -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSnapshot:
    """A consistent read of the history for one UI tick: how many prompts there are, the
    dropdown choices `(label, index)`, and the newest prompt's text (None when empty)."""

    count: int
    choices: list[tuple[str, int]]
    newest_text: str | None


class PromptLog:
    """Append-only, thread-safe history of the prompts sent to the model.

    Built over a `Wica`, it subscribes in its constructor to `on_agent_trigger` (to label the
    coming step by what triggered it) and `on_agent_prompt` (to record the step's exact messages).
    `wica=None` is the explore-only case: nothing subscribes and the history stays empty.

    Threading: the two handlers run on the agent loop, sequentially and in that order for each
    step (see agent.py `_run_step`), so the pending trigger label needs no lock; the history does,
    as the UI reads it on its own thread. Records are never mutated once appended."""

    def __init__(
        self, wica: Wica | None, display_entry: DisplayEntry | None = None
    ) -> None:
        self._wica = wica
        self._display_entry = display_entry
        self._lock = threading.Lock()
        self._prompts: list[dict[str, str]] = []
        self._last_trigger_label = "start"
        if wica is not None:
            wica.on_agent_trigger.subscribe(self.on_trigger)
            wica.on_agent_prompt.subscribe(self.on_prompt)

    @property
    def output_command_name(self) -> str | None:
        return None if self._wica is None else self._wica.agent.output_command_name

    def on_trigger(self, entry: WorldEntry) -> None:
        """A World entry triggered a step — remember its label for the prompt that follows. A
        Command-completion re-trigger is labelled `<item> finished`."""
        display = display_entry_or_default(
            entry, self._display_entry, self.output_command_name
        )
        if isinstance(entry.current.value, CommandExecution):
            self._last_trigger_label = f"{display.label} finished"
        else:
            self._last_trigger_label = display.label

    def on_prompt(self, messages: list[BaseMessage]) -> None:
        """Capture the exact messages sent to the model, labelled by time + trigger."""
        rendered = render_messages(messages)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        with self._lock:
            self._prompts.append(
                {"label": f"{stamp} — {self._last_trigger_label}", "text": rendered}
            )

    def snapshot(self) -> PromptSnapshot:
        with self._lock:
            return PromptSnapshot(
                count=len(self._prompts),
                choices=[(p["label"], i) for i, p in enumerate(self._prompts)],
                newest_text=self._prompts[-1]["text"] if self._prompts else None,
            )

    def text(self, index: int | None) -> str | None:
        """The exact text of the prompt at `index`, or None if out of range."""
        if index is None:
            return None
        with self._lock:
            if 0 <= index < len(self._prompts):
                return self._prompts[index]["text"]
        return None


# --- the panel ---------------------------------------------------------------------------


def prompt_panel(
    log: PromptLog, *, refresh_s: float = 0.2, lines: int = 16
) -> tuple[gr.Dropdown, gr.Textbox]:
    """Create the prompt-history dropdown + read-only text and the timer that keeps them live.
    Call inside a `gr.Blocks` context.

    The view snaps to the newest prompt only when a new step has appeared; between steps the
    timer sends bare `gr.update()`s so the dropdown and text are left untouched and the user can
    browse older prompts (the dropdown's `.change` shows the selected one)."""
    selector = gr.Dropdown(
        label="Prompt sent to the model (newest shown automatically)",
        choices=[],
        interactive=True,
    )
    view = gr.Textbox(
        show_label=False, lines=lines, interactive=False, value=_NO_PROMPT_YET
    )

    last_shown_count = 0

    def tick() -> tuple[Any, Any]:
        nonlocal last_shown_count
        snap = log.snapshot()
        if snap.count > last_shown_count:
            last_shown_count = snap.count
            newest = snap.count - 1
            return (
                gr.update(choices=snap.choices, value=newest),
                gr.update(value=snap.newest_text),
            )
        return gr.update(), gr.update()

    def on_select(index: int | None) -> Any:
        text = log.text(index)
        return gr.update() if text is None else text

    selector.change(on_select, inputs=selector, outputs=view)
    timer = gr.Timer(refresh_s)
    timer.tick(tick, outputs=[selector, view])
    return selector, view
