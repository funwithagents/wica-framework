"""A generic, reusable conversation transcript over WICA's instrumentation.

`TranscriptLog` turns the framework's uniform signals into a chat transcript (Gradio "messages"
format) plus a browsable prompt history, without knowing anything about the application: what
the World entries are, what the Commands do, or what the robot is called. It consumes

  - `on_agent_trigger`  → the World entry a step observed, shown on the input (right) side;
  - `on_agent_prompt`   → the step's exact prompt (kept in the history) and the opening of the
                          step's **reaction group** in the transcript;
  - `on_agent_command`  → each Command the model issued, shown as an item under the reaction,
                          **with its live execution state**: the log listens to the Command's
                          `agent:command:<call_id>` World entry (the seam `CommandIssued.call_id`
                          exists for — see specs/agent.md "Instrumentation") and updates the item
                          in place as it goes running → complete / failed / cancelled;
  - the `output_sink`   → the model's free text for the step (💭).

The only application-specific knowledge — which icon and wording a given entry gets — enters
through one hook, `display_entry(entry: WorldEntry) -> EntryDisplay | None`. Because a Command's
execution *is* a World entry (value: `CommandExecution`), the same hook customises both sides of
the transcript; returning `None` falls back to the generic default. The log itself only knows the
framework's own names: `noop` (no World entry, so never through the hook) and the configured
output Command (read from the Agent at `attach`, labelled 🗣️ by default).

Threading: the handlers run on the agent loop (Events and listeners are dispatched there), the
UI reads through `snapshot()` / `prompt_text()` on its own thread. Everything shared sits behind
this object: a thread-safe queue of pending items, the append-only prompt history and the items
under a lock. See specs/conversation-demo.md ("A reusable transcript").
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from wica import CommandIssued, Wica, World, WorldEntry
from wica.agent import CommandExecution

# WICA auto-registers a `noop` Command the model calls to declare "no reaction" (it often ends an
# output-Command re-trigger chain). It fires on_command like any issued command but has no World
# entry behind it, so the log renders it directly. Matches wica.agent's _NOOP_COMMAND_NAME.
NOOP_COMMAND_NAME = "noop"
# The World-key prefix of a Command's execution entry (specs/commands.md, "Command execution as a
# World entry"); matches wica.agent's _COMMAND_KEY_PREFIX.
COMMAND_KEY_PREFIX = "agent:command:"

_STATE_SUFFIX = {"complete": "✅", "failed": "❌ failed", "cancelled": "⏹ cancelled"}


@dataclass(frozen=True)
class EntryDisplay:
    """How one World entry shows in the transcript. `label` is the one-line text with its icon:
    the input-side message for a triggering entry, the item title for a Command execution.
    `detail` is a Command item's body (ignored for the input side)."""

    label: str
    detail: str | None = None


DisplayEntry = Callable[[WorldEntry], "EntryDisplay | None"]


def format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


@dataclass(frozen=True)
class TranscriptSnapshot:
    """A consistent read of the transcript + prompt history for one UI tick.

    `conversation` is the full ordered message list (Gradio "messages" format); `conv_sig` changes
    whenever any message is added or edited in place (a Command item's title/status/content
    changing as it runs), so the UI can skip re-pushing an unchanged transcript.
    `prompt_count`/`prompt_choices`/`newest_prompt_text` drive the prompt-history dropdown and its
    auto-snap to the newest step."""

    conversation: list[dict[str, Any]]
    conv_sig: tuple[Any, ...]
    prompt_count: int
    prompt_choices: list[tuple[str, int]]
    newest_prompt_text: str | None


# --- prompt rendering ----------------------------------------------------------------


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
            lines.append(
                f"🔧 tool_call {call['name']}({format_args(call['args'])})  [id={call['id']}]"
            )
    return "\n".join(lines)


# --- the log ---------------------------------------------------------------------------


class TranscriptLog:
    """Generic transcript + prompt-history presenter over WICA's instrumentation.

    Roles map to the two sides of the conversation:
     - "user"      → right side: the World entry the Agent actually observed (on_trigger),
                     rendered by `display_entry`.
     - "assistant" → left side: every Command the model issued (on_command → its live state via
                     on_command_update), its free text (output_sink, "💭 output sink") and
                     "🚫 noop".

    Messages use Gradio's "messages" format ({"role", "content", optional "metadata"}). The
    metadata carries the structure: `title` names the item; `id`/`parent_id` nest every item of one
    reasoning step under a single "reaction" group header (opened in on_prompt); `status`
    ("pending" while it runs — a spinner; removed once it ends, so the item stays open) and
    `duration` show a Command item's execution state."""

    def __init__(
        self,
        display_entry: DisplayEntry | None = None,
        *,
        output_command_name: str | None = None,
    ) -> None:
        self._display_entry = display_entry
        self._world: World | None = None
        # Which Command is the voice (labelled 🗣️ by default). `attach` reads it from the Agent;
        # the keyword serves a log used without a Wica.
        self._output_command_name = output_command_name
        # A single ordered queue keeps triggers, Command items and free text interleaved in the
        # exact order they happened; the UI's snapshot drains it into _conversation.
        self._items: queue.Queue[dict[str, Any]] = queue.Queue()
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
        # an `id` — and the step's outputs are emitted as children nested under it via
        # metadata.parent_id. All these callbacks run on the single agent loop, so the id is
        # set/read consistently without a lock.
        self._reaction_count = 0
        self._current_reaction_id: str | None = None
        # Command items by call_id, so the execution listener can edit them in place, and when
        # each was issued (monotonic) for the duration shown on completion.
        self._command_items: dict[str, dict[str, Any]] = {}
        self._command_started: dict[str, float] = {}

    # --- wiring ------------------------------------------------------------------------

    def attach(self, wica: Wica) -> None:
        """Wire the log to a running system: subscribe the three instrumentation Events, bind the
        World the Command listeners are added on, and learn the output Command's name for the
        default rendering. The application passes `output_sink` to `Wica.init` itself (it is a
        constructor argument, not an Event). Until attached, on_command still creates items but
        nothing follows their state (unit tests drive on_command_update directly)."""
        self._world = wica.world
        self._output_command_name = wica.agent.output_command_name
        wica.on_agent_trigger.subscribe(self.on_trigger)
        wica.on_agent_prompt.subscribe(self.on_prompt)
        wica.on_agent_command.subscribe(self.on_command)

    @property
    def output_command_name(self) -> str | None:
        return self._output_command_name

    # --- rendering -----------------------------------------------------------------------

    def display(self, entry: WorldEntry) -> EntryDisplay:
        """Render an entry through the application hook, falling back to the generic default:
        `⚡ key = value` for any entry; for a Command execution `🦾 name` (🗣️ for the output
        Command) with `name(args)` as detail."""
        if self._display_entry is not None:
            custom = self._display_entry(entry)
            if custom is not None:
                return custom
        value = entry.current.value
        if isinstance(value, CommandExecution):
            return EntryDisplay(
                f"{self._command_icon(value.name)} {value.name}",
                f"{value.name}({format_args(value.args)})",
            )
        return EntryDisplay(f"⚡ {entry.key} = {value!r}")

    def _command_icon(self, name: str) -> str:
        return "🗣️" if name == self._output_command_name else "🦾"

    def _reaction_child(self, title: str, content: str) -> dict[str, Any]:
        """An assistant transcript item nested under the current reaction group (or top-level if no
        reaction is open, e.g. before the first step)."""
        metadata: dict[str, Any] = {"title": title}
        if self._current_reaction_id is not None:
            metadata["parent_id"] = self._current_reaction_id
        return {"role": "assistant", "content": content, "metadata": metadata}

    def current_reaction_id(self) -> str | None:
        """The reaction group currently open, or None before the first step."""
        return self._current_reaction_id

    # --- agent callbacks (the bridge) ------------------------------------------------

    async def output_sink(self, text: str) -> None:
        """The agent's free text for a step. With an output Command configured this is the model's
        private reasoning rather than its voice; labelled by source so the transcript makes plain
        which framework channel produced it. Runs on the agent loop."""
        if text.strip():
            self._items.put(self._reaction_child("💭 output sink", text))

    def on_trigger(self, entry: WorldEntry) -> None:
        """A World entry triggered a step — show it on the input (right) side, rendered by the
        hook. A Command-completion re-trigger is not shown (the Command item already carries its
        outcome) but still labels the step, so on_prompt tags this step's prompt correctly."""
        display = self.display(entry)
        if isinstance(entry.current.value, CommandExecution):
            self._last_trigger_label = f"{display.label} finished"
            return
        self._last_trigger_label = display.label
        self._items.put({"role": "user", "content": display.label})

    def on_command(self, command: CommandIssued) -> None:
        """The model issued a Command — create its item under the current reaction right away
        (pinning its position in issue order) marked in progress, and follow its execution entry
        so on_command_update can flip the item to its outcome. `noop` has no entry: rendered as
        the robot explicitly choosing not to react."""
        if command.name == NOOP_COMMAND_NAME:
            self._items.put(self._reaction_child("🚫 noop", "chose not to react"))
            return
        item = self._reaction_child(
            f"{self._command_icon(command.name)} {command.name}",
            f"{command.name}({format_args(command.args)})",
        )
        item["metadata"]["status"] = "pending"
        with self._lock:
            self._command_items[command.call_id] = item
            self._command_started[command.call_id] = time.monotonic()
        self._items.put(item)
        if self._world is not None:
            # The Event fires once the entry is registered and before its running value lands
            # (specs/agent.md, "Instrumentation"), so the listener sees the whole life. It must be
            # async: a sync listener is offloaded to a thread pool, which could deliver a
            # terminal state before `running`; an async one runs on the loop in version order.
            self._world.add_listener(
                f"{COMMAND_KEY_PREFIX}{command.call_id}", self.on_command_update
            )

    async def on_command_update(self, entry: WorldEntry) -> None:
        """World listener on a Command's execution entry: re-render its item through the hook on
        every state, and on a terminal state give it a ✅/❌/⏹ suffix, the result or error, and
        how long it ran. Edits the item dict in place — the snapshot's signature covers
        title/status/content, so the UI pushes the change.

        The `status` key is *removed* on a terminal state rather than set to "done": Gradio's
        thought component shows its spinner only for "pending", and auto-collapses an item the
        moment its status becomes "done" (the person could no longer read a finished `say` without
        expanding it). No status means no spinner and the item stays open."""
        value = entry.current.value
        if not isinstance(value, CommandExecution):
            return
        call_id = entry.key.removeprefix(COMMAND_KEY_PREFIX)
        display = self.display(entry)
        with self._lock:
            item = self._command_items.get(call_id)
            if item is None:
                return
            title, content = display.label, display.detail or ""
            metadata = item["metadata"]
            if value.is_terminal():
                title = f"{title} {_STATE_SUFFIX[value.state]}"
                if value.state == "complete" and value.result:
                    content = f"{content}\n→ {value.result}"
                elif value.state == "failed" and value.error:
                    content = f"{content}\n✗ {value.error}"
                metadata.pop("status", None)
                started = self._command_started.pop(call_id, None)
                if started is not None:
                    metadata["duration"] = round(time.monotonic() - started, 1)
                self._command_items.pop(call_id, None)
            else:
                metadata["status"] = "pending"
            metadata["title"] = title
            item["content"] = content

    def on_prompt(self, messages: list[BaseMessage]) -> None:
        """Capture the exact messages sent to the model, appending to the prompt history labelled
        by time + the trigger that caused this step; and open this step's **reaction group** in the
        transcript so its outputs nest under one header. Fires once per reasoning step, on the
        agent loop, before the model call — so it precedes the step's items."""
        rendered = "\n\n".join(_render_message(m) for m in messages)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        trigger_label = self._last_trigger_label
        with self._lock:
            self._prompts.append(
                {"label": f"{stamp} — {trigger_label}", "text": rendered}
            )

        self._reaction_count += 1
        self._current_reaction_id = f"reaction-{self._reaction_count}"
        self._items.put(
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
        """Drain pending items into the conversation and return a consistent read of it plus the
        prompt history, for one UI tick. Called from the UI thread."""
        with self._lock:
            while True:
                try:
                    item = self._items.get_nowait()
                except queue.Empty:
                    break
                self._conversation.append(item)
            conversation = list(self._conversation)
            # A signature that changes whenever any message is added or edited in place (a Command
            # item's title/status/content as it runs). The UI compares it so it only pushes a new
            # chatbot value on a real change.
            conv_sig = tuple(
                (
                    str(m.get("content", "")),
                    m.get("metadata", {}).get("title"),
                    m.get("metadata", {}).get("status"),
                )
                for m in conversation
            )
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
