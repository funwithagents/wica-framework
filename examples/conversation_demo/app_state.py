"""Presenter state shared between the WICA agent callbacks and the Gradio UI.

`DemoState` owns the transcript- and prompt-history state that used to live as module globals in
`app.py`, and it is the single seam between two worlds that run on different threads:

  - the **agent loop** (a daemon thread Wica owns) fires the instrumentation Events, which land on
    the callback methods here (`on_trigger`/`on_command`/`on_prompt`/`output_sink`) plus the `say`
    output Command's streaming bubble (`open_say_bubble`);
  - the **Gradio UI thread** reads that state through `snapshot()` / `prompt_text()`.

Everything the two threads share is behind this object: a thread-safe `queue.Queue` of pending
transcript events, an append-only prompt history under a lock, and the reaction-grouping counters.
The UI file (`app_ui.py`) never touches these directly, and the app file (`app.py`) only wires the
callbacks up and hands `say` a bubble to grow — so neither reaches into the other's state through
globals.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from wica import CommandIssued, WorldEntry
from wica.agent import CommandExecution

# The output Command: the robot's user-facing voice. With it configured (output_command=say in
# Wica.init), the model's free text becomes private reasoning (shown as a 💭 thought via output_sink)
# and anything the robot actually says goes through this Command — a real, cancellable, observable
# Command like any other. See specs/agent.md ("Output") and specs/commands.md ("The output Command").
OUTPUT_COMMAND_NAME = "say"
# WICA auto-registers a `noop` Command the model calls to declare "no reaction" (it often ends an
# output-Command re-trigger chain). It fires on_command like any issued command, so the demo can
# show it — see on_command below. Matches wica.agent's _NOOP_COMMAND_NAME.
NOOP_COMMAND_NAME = "noop"


@dataclass(frozen=True)
class TranscriptSnapshot:
    """A consistent read of the transcript + prompt history for one UI tick.

    `conversation` is the full ordered message list (Gradio "messages" format); `conv_sig` changes
    whenever any message is added or edited (a `say` bubble grows in place), so the UI can skip
    re-pushing an unchanged transcript. `prompt_count`/`prompt_choices`/`newest_prompt_text` drive
    the prompt-history dropdown and its auto-snap to the newest step."""

    conversation: list[dict[str, Any]]
    conv_sig: tuple[Any, ...]
    prompt_count: int
    prompt_choices: list[tuple[str, int]]
    newest_prompt_text: str | None


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
    """Render one LangChain message to text for the prompt panel. An assistant message that only
    issues a command has empty .content — the call lives in the structured .tool_calls field — so
    we render those (and the tool_call id on a tool result) explicitly, or the AI block looks blank."""
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
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            lines.append(f"🔧 tool_call {call['name']}({args})  [id={call['id']}]")
    return "\n".join(lines)


def _describe_trigger(entry: WorldEntry) -> str:
    key = entry.key
    value = entry.current.value
    if key == "speech_input":
        return f'🗣️ "{value}"'
    if key == "closest_user":
        return (
            "👤 Closest user gone"
            if value is None
            else f"👤 Closest user detected: {value}"
        )
    if key.startswith("agent:command:"):
        name = value.name if isinstance(value, CommandExecution) else key
        return f"⚡ command finished: {name}"
    return f"⚡ {key} = {value!r}"


class DemoState:
    """Transcript + prompt-history state shared by the agent callbacks and the UI.

    Roles map to the two sides of the conversation:
     - "user"      → right side: the World entry the Agent actually observed (on_trigger).
     - "assistant" → left side:  the robot's spoken reply — the `say` output Command, labelled
                                 "🗣️ say" — plus its private free text (output_sink, labelled
                                 "💭 output sink") and its command calls (on_command, 🦾).

    Messages use Gradio's "messages" format ({"role", "content", optional "metadata"}). Two metadata
    features carry the structure: a metadata.title names the framework channel each assistant item
    came from ("🗣️ say", "💭 output sink", "🦾 <cmd>", "🚫 noop"); and metadata.id / metadata.parent_id
    nest every item of one reasoning step under a single "reaction" group header (opened in
    on_prompt), so the transcript reads as one collapsible group per reaction."""

    def __init__(self) -> None:
        # A single ordered queue keeps triggers, the robot's spoken replies, and its command calls
        # interleaved in the exact order they happened; the UI's tick drains it into _conversation.
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._conversation: list[dict[str, Any]] = []
        # Every reasoning step's prompt, append-only, each {"label": ..., "text": ...} — never
        # mutated in place, so readers can pull `text` outside the lock once they've read len.
        self._prompts: list[dict[str, str]] = []
        # The trigger that started the current step. on_trigger fires just before on_prompt on the
        # agent loop (see agent.py _run_step), so on_prompt reads this to label each captured prompt.
        self._last_trigger_label = "start"
        # Reaction grouping: each reasoning step (a "reaction") is one collapsible group in the
        # transcript. on_prompt (once per step) opens a new group — a parent assistant message with
        # an `id` — and the step's outputs (say, output sink, 🦾 actions, noop) are emitted as
        # children nested under it via metadata.parent_id. All these callbacks run on the single
        # agent loop, so the id is set/read consistently without a lock.
        self._reaction_count = 0
        self._current_reaction_id: str | None = None

    # --- transcript item construction ------------------------------------------------

    def _reaction_child(self, title: str, content: str) -> dict[str, Any]:
        """An assistant transcript item nested under the current reaction group (or top-level if no
        reaction is open, e.g. before the first step)."""
        metadata: dict[str, Any] = {"title": title}
        if self._current_reaction_id is not None:
            metadata["parent_id"] = self._current_reaction_id
        return {"role": "assistant", "content": content, "metadata": metadata}

    def current_reaction_id(self) -> str | None:
        """The reaction group currently open, or None before the first step. `say` captures this
        before its first `await` so that a later reaction opening mid-stream doesn't adopt its
        spoken words — see the parent_id argument to open_say_bubble."""
        return self._current_reaction_id

    def open_say_bubble(self, first_word: str, parent_id: str | None) -> dict[str, Any]:
        """Create the `say` transcript bubble under an explicitly given reaction group, enqueue it,
        and return it so the caller can grow its `content` in place word by word — tick appends this
        exact dict to the transcript, so mutating it streams the words into the UI (see `say` in
        app.py). The parent is passed in (not read from the current reaction) because `say` streams
        in the background: a step that opens a newer reaction while it speaks must not capture its
        words. The caller passes the reaction captured before it began streaming."""
        metadata: dict[str, Any] = {"title": "🗣️ say"}
        if parent_id is not None:
            metadata["parent_id"] = parent_id
        message: dict[str, Any] = {
            "role": "assistant",
            "content": first_word,
            "metadata": metadata,
        }
        self._events.put(message)
        return message

    # --- agent callbacks (the bridge) ------------------------------------------------

    async def output_sink(self, text: str) -> None:
        """The agent's free text for a step. With an output Command configured, this is no longer the
        robot's *voice* (that goes through `say`) but its private reasoning. Labelled by source
        ("output sink") so the demo makes plain which framework channel produced it, set apart from
        the spoken `say` reply above. Runs on the agent loop."""
        if text.strip():
            self._events.put(self._reaction_child("💭 output sink", text))

    def on_trigger(self, entry: WorldEntry) -> None:
        """A World entry triggered a step — show it on the input (right) side. Skip the robot's own
        command-completion re-triggers on the transcript (they're already shown as command calls on
        the left), but always record the label so on_prompt can tag this step's prompt correctly."""
        label = _describe_trigger(entry)
        self._last_trigger_label = label
        if entry.key.startswith("agent:command:"):
            return
        self._events.put({"role": "user", "content": label})

    def on_command(self, command: CommandIssued) -> None:
        """The robot issued a command — show it on the assistant (left) side. `say` is skipped here
        (it *is* the spoken reply, rendered by say() itself, not a 🦾 action); `noop` — the robot
        explicitly choosing not to react — is shown labelled by source like say/output_sink; every
        other command is a 🦾 action, italicised."""
        if command.name == OUTPUT_COMMAND_NAME:
            return
        if command.name == NOOP_COMMAND_NAME:
            self._events.put(self._reaction_child("🚫 noop", "chose not to react"))
            return
        rendered = ", ".join(f"{k}={v!r}" for k, v in command.args.items())
        self._events.put(
            self._reaction_child(f"🦾 {command.name}", f"{command.name}({rendered})")
        )

    def on_prompt(self, messages: list[BaseMessage]) -> None:
        """Debug hook — capture the exact messages sent to the model, appending to the prompt history
        labelled by time + the trigger that caused this step; and open this step's **reaction group**
        in the transcript so its outputs nest under one header. Fires once per reasoning step, on the
        agent loop, before the model call — so it precedes the step's say/output-sink/command events.
        """
        rendered = "\n\n".join(_render_message(m) for m in messages)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        trigger_label = self._last_trigger_label
        with self._lock:
            self._prompts.append(
                {"label": f"{stamp} — {trigger_label}", "text": rendered}
            )

        self._reaction_count += 1
        self._current_reaction_id = f"reaction-{self._reaction_count}"
        self._events.put(
            {
                "role": "assistant",
                "content": "",
                "metadata": {
                    "id": self._current_reaction_id,
                    "title": f"💬 reaction {self._reaction_count} · {trigger_label}",
                },
            }
        )

    # --- UI reads --------------------------------------------------------------------

    def snapshot(self) -> TranscriptSnapshot:
        """Drain pending transcript events into the conversation and return a consistent read of it
        plus the prompt history, for one UI tick. Called from the Gradio thread."""
        with self._lock:
            while True:
                try:
                    event = self._events.get_nowait()
                except queue.Empty:
                    break
                self._conversation.append(event)
            conversation = list(self._conversation)
            # A signature that changes whenever any message is added or edited (a `say` bubble grows
            # its content in place). The UI compares it so it only pushes a new chatbot value on a
            # real change.
            conv_sig = tuple(str(m.get("content", "")) for m in conversation)
            prompt_count = len(self._prompts)
            prompt_choices = [(p["label"], i) for i, p in enumerate(self._prompts)]
            newest_prompt_text = self._prompts[-1]["text"] if self._prompts else None
        return TranscriptSnapshot(
            conversation=conversation,
            conv_sig=conv_sig,
            prompt_count=prompt_count,
            prompt_choices=prompt_choices,
            newest_prompt_text=newest_prompt_text,
        )

    def prompt_text(self, index: int | None) -> str | None:
        """The exact text of the prompt at `index` in the history, or None if out of range — used by
        the prompt dropdown's selection handler."""
        if index is None:
            return None
        with self._lock:
            if 0 <= index < len(self._prompts):
                return self._prompts[index]["text"]
        return None
