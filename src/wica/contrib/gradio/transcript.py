"""The conversation transcript: WICA's reasoning loop as a chat, over its instrumentation.

`TranscriptLog` turns the framework's uniform signals into a chat transcript (Gradio "messages"
format) without knowing anything about the application: what the World entries are, what the
Commands do, or what the agent is called. It consumes

  - `on_agent_trigger`  → the World entry a step observed, shown on the input (right) side;
  - `on_agent_prompt`   → the opening of the step's **reaction group** in the transcript;
  - `on_agent_command`  → each Command the model issued, shown as an item under the reaction,
                          **with its live execution state**: the log listens to the Command's
                          `agent:command:<call_id>` World entry (the seam `CommandIssued.call_id`
                          exists for — see specs/agent.md "Instrumentation") and updates the item
                          in place as it goes running → complete / failed / cancelled;
  - `output_sink`       → its own async method, which the application sets on the `Wica`
                          (`wica.set_output_sink(log.output_sink)`): the model's free text for
                          the step (💭).

A log is built over a `Wica` and subscribes to its Events in the constructor — build the Wica,
then the log, then wire the sink (see specs/gradio-contrib.md, "Common shape"). `wica=None` is
the explore-only case (no Agent runs): nothing subscribes and Command items are never followed.

The only application-specific knowledge — which icon and wording a given entry gets — enters
through the `display_entry` hook (see display.py). The log itself only knows the framework's own
names: `noop` (no World entry, so never through the hook) and the configured output Command
(read live from the Agent, labelled 🗣️ by default — live, so the log may be built before
`set_output_command` runs).

Threading: the handlers run on the agent loop (Events and listeners are dispatched there), the
UI reads through `snapshot()` on its own thread. Everything shared sits behind this object: a
thread-safe queue of pending items and the items under a lock. `conversation_panel` is the
Gradio view over it. See specs/gradio-contrib.md ("Component 3").
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import gradio as gr
from langchain_core.messages import BaseMessage

from wica import CommandExecution, CommandIssued, ReactionTrace, Wica, World, WorldEntry
from wica.agent import _COMMAND_KEY_PREFIX, _NOOP_COMMAND_NAME
from wica.contrib.gradio.display import (
    DisplayEntry,
    EntryDisplay,
    display_entry_or_default,
    format_args,
)

_STATE_SUFFIX = {"complete": "✅", "failed": "❌ failed", "cancelled": "⏹ cancelled"}


@dataclass(frozen=True)
class TranscriptSnapshot:
    """A consistent read of the transcript for one UI tick.

    `conversation` is the full ordered message list (Gradio "messages" format); `conv_sig` changes
    whenever any message is added or edited in place (a Command item's title/status/content
    changing as it runs), so the UI can skip re-pushing an unchanged transcript."""

    conversation: list[dict[str, Any]]
    conv_sig: tuple[Any, ...]


class TranscriptLog:
    """Generic transcript presenter over WICA's instrumentation.

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
        self, wica: Wica | None, display_entry: DisplayEntry | None = None
    ) -> None:
        """Build the log over a `Wica`: subscribe the three instrumentation Events and keep the
        World the per-Command listeners are added on. The application then passes `output_sink`
        to `wica.set_output_sink`. With `wica=None` (explore-only mode, no Agent) nothing
        subscribes; `on_command` still creates items but nothing follows their state."""
        self._wica = wica
        self._display_entry = display_entry
        self._world: World | None = None if wica is None else wica.world
        # A single ordered queue keeps triggers, Command items and free text interleaved in the
        # exact order they happened; the UI's snapshot drains it into _conversation.
        self._items: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._conversation: list[dict[str, Any]] = []
        # The trigger that started the current step. on_trigger fires just before on_prompt on the
        # agent loop (see agent.py _run_step), so on_prompt reads this to title the reaction group.
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
        # Reaction group items by reaction_id, so on_reaction_ended can drop the spinner and add
        # the duration once the reaction finishes. See specs/instrumentation.md.
        self._reaction_items: dict[int, dict[str, Any]] = {}

        if wica is not None:
            wica.on_agent_trigger.subscribe(self.on_trigger)
            wica.on_agent_prompt.subscribe(self.on_prompt)
            wica.on_agent_command.subscribe(self.on_command)
            wica.on_agent_reaction_ended.subscribe(self.on_reaction_ended)

    @property
    def output_command_name(self) -> str | None:
        """Which Command is the voice (labelled 🗣️ by default) — read live from the Agent, so it
        is right whether the log or `set_output_command` came first."""
        return None if self._wica is None else self._wica.agent.output_command_name

    # --- rendering -----------------------------------------------------------------------

    def display(self, entry: WorldEntry) -> EntryDisplay:
        """Render an entry through the application hook, falling back to the generic default
        (see display.py)."""
        return display_entry_or_default(
            entry, self._display_entry, self.output_command_name
        )

    def _command_icon(self, name: str) -> str:
        return "🗣️" if name == self.output_command_name else "🦾"

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
        outcome) but still labels the step, so on_prompt titles this step's reaction correctly."""
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
        the agent explicitly choosing not to react."""
        if command.name == _NOOP_COMMAND_NAME:
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
                f"{_COMMAND_KEY_PREFIX}{command.call_id}", self.on_command_update
            )

    async def on_command_update(self, entry: WorldEntry) -> None:
        """World listener on a Command's execution entry: re-render its item through the hook on
        every state, and on a terminal state give it a ✅/❌/⏹ suffix, the result or error, and
        how long it ran. Edits the item dict in place — the snapshot's signature covers
        title/status/content, so the UI pushes the change.

        The `status` key is *removed* on a terminal state rather than set to "done": Gradio's
        thought component shows its spinner only for "pending", and auto-collapses an item the
        moment its status becomes "done" (the person could no longer read a finished item without
        expanding it). No status means no spinner and the item stays open."""
        value = entry.current.value
        if not isinstance(value, CommandExecution):
            return
        call_id = entry.key.removeprefix(_COMMAND_KEY_PREFIX)
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
        """Open this step's **reaction group** in the transcript so its outputs nest under one
        header, titled by the trigger that caused the step. Fires once per reasoning step, on the
        agent loop, before the model call — so it precedes the step's items. (The prompt itself
        is kept by `ReactionLog`, not here.) Opens *pending* (a spinner while the model thinks);
        `on_reaction_ended` drops it and shows the reaction's duration."""
        self._reaction_count += 1
        self._current_reaction_id = f"reaction-{self._reaction_count}"
        item = {
            "role": "assistant",
            "content": "",
            "metadata": {
                "id": self._current_reaction_id,
                "title": f"💬 reaction {self._reaction_count} · {self._last_trigger_label}",
                "status": "pending",
            },
        }
        with self._lock:
            self._reaction_items[self._reaction_count] = item
        self._items.put(item)

    def on_reaction_ended(self, reaction: ReactionTrace) -> None:
        """The reaction finished: drop the spinner and show how long the Agent was busy. Matched by
        reaction_id, which counts reactions exactly as on_prompt does (both start at 1 and advance
        once per reaction). Runs on the agent loop."""
        with self._lock:
            item = self._reaction_items.pop(reaction.reaction_id, None)
            if item is None:
                return
            item["metadata"].pop("status", None)
            item["metadata"]["duration"] = round(reaction.busy_time, 1)

    # --- UI reads --------------------------------------------------------------------

    def snapshot(self) -> TranscriptSnapshot:
        """Drain pending items into the conversation and return a consistent read of it, for one
        UI tick. Called from the UI thread."""
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
        return TranscriptSnapshot(conversation=conversation, conv_sig=conv_sig)


# --- the panel ---------------------------------------------------------------------------


def conversation_panel(
    log: TranscriptLog, *, refresh_s: float = 0.2, height: int = 420
) -> gr.Chatbot:
    """Create the transcript chatbot and the timer that keeps it live. Call inside a `gr.Blocks`
    context.

    The chatbot is deliberately NOT an output of the timer. Every event that lists a component as
    an output flips that component's loading status (pending → complete) even with
    show_progress="hidden", and gr.Chatbot's autoscroll effect re-runs on that flip: whenever the
    view is within ~100px of the bottom it schedules an unconditional scroll-to-bottom 300ms
    later. Driven by a 200ms timer that made scrolling up nearly impossible — the reader was
    yanked back before getting clear of the bottom, even with nothing new to show. So the timer
    only bumps a version `gr.State` when the snapshot's signature changes (new messages, or a
    Command item's title/status/content edited in place as it runs); Gradio fires the State's
    .change only on a real value change, and that event alone pushes the transcript. Autoscroll
    then follows genuinely new content and leaves reading alone."""
    chatbot = gr.Chatbot(label="Conversation", height=height)
    version = gr.State(0)

    last_sig: tuple[Any, ...] | None = None
    current_version = 0

    def tick() -> int:
        nonlocal last_sig, current_version
        sig = log.snapshot().conv_sig
        if sig != last_sig:
            last_sig = sig
            current_version += 1
        return current_version  # an unchanged int is a no-op for the gr.State

    def push() -> list[dict[str, Any]]:
        """Fires on version.change — i.e. only when the transcript really changed."""
        return log.snapshot().conversation

    timer = gr.Timer(refresh_s)
    timer.tick(tick, outputs=version)
    version.change(push, outputs=chatbot, show_progress="hidden")
    return chatbot
