"""The reaction-history panel: every reaction the Agent ran, browsable step by step, with both its
prompt and its instrumentation.

A prompt exists only for the instant `on_agent_prompt` fires, on the agent loop; a reaction's
`ReactionTrace` exists only later, once `on_agent_reaction_ended` fires for it. Both describe the
same reasoning step from two angles, so `ReactionLog` keeps one append-only, thread-safe record per
reaction — opened by `on_agent_prompt`, filled in by `on_agent_reaction_ended` — and `reaction_panel`
shows both facets of whichever reaction is selected: a dropdown of reactions (labelled by time and
by the trigger that caused each one, rendered through the application's `display_entry` hook, so
the labels match the transcript's) with a "Prompt" tab and an "Instrumentation" tab underneath,
snapping to the newest reaction when one appears and otherwise leaving the user free to browse. See
specs/gradio-contrib.md ("Component 2").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from wica import CommandExecution, ReactionTrace, Wica, WorldEntry
from wica.contrib.gradio.display import (
    DisplayEntry,
    display_entry_or_default,
    format_args,
)
from wica.instrumentation import reaction_latency

_NO_REACTION_YET = "(no reaction yet)"
_INSTRUMENTATION_PENDING = (
    "(reaction still in progress — instrumentation appears once it ends)"
)


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


def render_reaction_trace(trace: ReactionTrace) -> str:
    """The reaction's ReactionTrace as readable text: outcome, the derived measures (only the ones
    whose underlying stamp exists — a cancelled reaction has no model_latency), token usage,
    dispatched commands, noop, the error on a model_error outcome, and the OpenTelemetry trace id
    when an SDK is installed. See specs/instrumentation.md ("Layer 1", "Metrics")."""
    lines = [f"reaction {trace.reaction_id} — {trace.outcome}"]
    latency = reaction_latency(trace)
    if latency is not None:
        lines.append(f"reaction latency: {latency:.3f}s")
    lines.append(f"coalescing wait: {trace.coalescing_wait:.3f}s")
    if trace.render_time is not None:
        lines.append(f"render time: {trace.render_time:.3f}s")
    if trace.model_latency is not None:
        lines.append(f"model latency: {trace.model_latency:.3f}s")
    if trace.sink_duration is not None:
        lines.append(f"sink duration: {trace.sink_duration:.3f}s")
    lines.append(f"busy time: {trace.busy_time:.3f}s")
    if trace.usage is not None:
        cached = (
            f" ({trace.usage.cache_read_tokens} cached)"
            if trace.usage.cache_read_tokens
            else ""
        )
        lines.append(
            f"tokens: {trace.usage.input_tokens} in / {trace.usage.output_tokens} out{cached}"
        )
    if trace.command_call_ids:
        lines.append(f"commands: {', '.join(trace.command_call_ids)}")
    if trace.noop:
        lines.append("noop")
    if trace.error is not None:
        lines.append(f"error: {trace.error}")
    if trace.trace_id is not None:
        lines.append(f"trace: {trace.trace_id}")
    return "\n".join(lines)


# --- the log -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactionSnapshot:
    """A consistent read of the history for one UI tick: how many reactions there are, and the
    dropdown choices (label, index). Unlike the old PromptSnapshot there is no "newest text" field
    — the panel reads whichever index it needs (newest or currently selected) directly, since the
    Instrumentation tab needs that for the *selected* index too, not just the newest."""

    count: int
    choices: list[tuple[str, int]]


class ReactionLog:
    """Append-only, thread-safe history of the framework's reasoning steps: one record per
    reaction, holding both the exact prompt sent to the model (from `on_agent_prompt`) and, once
    the reaction ends, its ReactionTrace (from `on_agent_reaction_ended`) — merged because they
    describe the same step from two angles. See specs/gradio-contrib.md ("Component 2").

    Built over a `Wica`, it subscribes in its constructor to `on_agent_trigger` (to label the
    coming reaction by what triggered it), `on_agent_prompt` (to open the record) and
    `on_agent_reaction_ended` (to fill in that same record's trace, matched by `reaction_id`).
    `wica=None` is the explore-only case: nothing subscribes and the history stays empty.

    Threading: the three handlers run on the agent loop; `on_prompt` and `on_reaction_ended` fire
    at different points of the same reaction's life (`on_prompt` opens the record, well before
    `on_reaction_ended` fills it in), so both take the lock. Records are appended once and mutated
    exactly once (to attach the trace); the UI reads them on its own thread."""

    def __init__(
        self, wica: Wica | None, display_entry: DisplayEntry | None = None
    ) -> None:
        self._wica = wica
        self._display_entry = display_entry
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._last_trigger_label = "start"
        if wica is not None:
            wica.on_agent_trigger.subscribe(self.on_trigger)
            wica.on_agent_prompt.subscribe(self.on_prompt)
            wica.on_agent_reaction_ended.subscribe(self.on_reaction_ended)

    @property
    def output_command_name(self) -> str | None:
        return None if self._wica is None else self._wica.agent.output_command_name

    def on_trigger(self, entry: WorldEntry) -> None:
        """A World entry triggered a step — remember its label for the reaction that follows. A
        Command-completion re-trigger is labelled `<item> finished`."""
        display = display_entry_or_default(
            entry, self._display_entry, self.output_command_name
        )
        if isinstance(entry.current.value, CommandExecution):
            self._last_trigger_label = f"{display.label} finished"
        else:
            self._last_trigger_label = display.label

    def on_prompt(self, messages: list[BaseMessage]) -> None:
        """Open this reaction's record: the exact messages sent to the model, labelled by time +
        trigger. Its instrumentation is filled in later by on_reaction_ended."""
        rendered = render_messages(messages)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        with self._lock:
            self._records.append(
                {
                    "label": f"{stamp} — {self._last_trigger_label}",
                    "prompt_text": rendered,
                    "trace": None,
                }
            )

    def on_reaction_ended(self, trace: ReactionTrace) -> None:
        """Attach this reaction's trace to the record on_prompt opened for it, matched by
        reaction_id (on_agent_prompt fires exactly once per reaction that starts, in the same
        order ReactionTrace.reaction_id counts them, so reaction_id - 1 is that record's index).
        An index out of range is a defensive no-op — it should not happen."""
        with self._lock:
            index = trace.reaction_id - 1
            if 0 <= index < len(self._records):
                self._records[index]["trace"] = trace

    def snapshot(self) -> ReactionSnapshot:
        with self._lock:
            return ReactionSnapshot(
                count=len(self._records),
                choices=[(r["label"], i) for i, r in enumerate(self._records)],
            )

    def prompt_text(self, index: int | None) -> str | None:
        """The exact prompt text at `index`, or None if out of range."""
        if index is None:
            return None
        with self._lock:
            if 0 <= index < len(self._records):
                return self._records[index]["prompt_text"]
        return None

    def instrumentation_text(self, index: int | None) -> str | None:
        """The rendered ReactionTrace at `index`, the pending placeholder if the reaction hasn't
        ended yet, or None if `index` is out of range — the same three-way split `prompt_text`
        would have if a reaction could have no prompt, made explicit here because "not ended yet"
        is a real, common state (unlike an unset prompt)."""
        if index is None:
            return None
        with self._lock:
            if not (0 <= index < len(self._records)):
                return None
            trace = self._records[index]["trace"]
        return (
            _INSTRUMENTATION_PENDING if trace is None else render_reaction_trace(trace)
        )


# --- the panel ---------------------------------------------------------------------------


def reaction_panel(
    log: ReactionLog, *, refresh_s: float = 0.2, lines: int = 16
) -> tuple[gr.Dropdown, gr.Textbox, gr.Textbox]:
    """Create the reaction-history dropdown, its two tabs (Prompt / Instrumentation), and the timer
    that keeps them live. Call inside a `gr.Blocks` context.

    The dropdown snaps to the newest reaction exactly as the old prompt_panel did — selecting it
    and showing both its (known) prompt and its (pending) instrumentation — and otherwise sends no
    update, so the user can browse older reactions freely. The one addition: every tick also
    re-reads the Instrumentation tab for whichever reaction is *currently selected* and, if its
    text changed (pending -> the rendered trace, once that reaction ends), pushes just that pane —
    never the dropdown, never the Prompt tab — so a reaction being watched updates in place once
    its trace lands, the same way the transcript's own reaction group loses its spinner in place.
    """
    selector = gr.Dropdown(
        label="Reaction (newest shown automatically)",
        choices=[],
        interactive=True,
    )
    with gr.Tabs():
        with gr.Tab("Prompt"):
            prompt_view = gr.Textbox(
                show_label=False, lines=lines, interactive=False, value=_NO_REACTION_YET
            )
        with gr.Tab("Instrumentation"):
            instrumentation_view = gr.Textbox(
                show_label=False, lines=lines, interactive=False, value=_NO_REACTION_YET
            )

    last_shown_count = 0
    last_instrumentation: dict[int, str] = {}

    def tick(selected: int | None) -> tuple[Any, Any, Any]:
        nonlocal last_shown_count
        snap = log.snapshot()
        if snap.count > last_shown_count:
            last_shown_count = snap.count
            newest = snap.count - 1
            instrumentation = log.instrumentation_text(newest)
            if instrumentation is not None:
                last_instrumentation[newest] = instrumentation
            return (
                gr.update(choices=snap.choices, value=newest),
                gr.update(value=log.prompt_text(newest)),
                gr.update(value=instrumentation),
            )
        if selected is not None:
            current = log.instrumentation_text(selected)
            if current is not None and last_instrumentation.get(selected) != current:
                last_instrumentation[selected] = current
                return gr.update(), gr.update(), gr.update(value=current)
        return gr.update(), gr.update(), gr.update()

    def on_select(index: int | None) -> tuple[Any, Any]:
        prompt = log.prompt_text(index)
        instrumentation = log.instrumentation_text(index)
        if instrumentation is not None and index is not None:
            last_instrumentation[index] = instrumentation
        return (
            gr.update() if prompt is None else prompt,
            gr.update() if instrumentation is None else instrumentation,
        )

    selector.change(
        on_select, inputs=selector, outputs=[prompt_view, instrumentation_view]
    )
    timer = gr.Timer(refresh_s)
    timer.tick(
        tick, inputs=selector, outputs=[selector, prompt_view, instrumentation_view]
    )
    return selector, prompt_view, instrumentation_view
