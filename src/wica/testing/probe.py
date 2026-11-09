"""The Probe: the policy-free connection to a `Wica` under test. Implements specs/wica-tester.md,
"Probe — the connection" and "What the framework provides".

It inspects the SUT once through public surface only (`world.keys()` / `get_config()`,
`agent.commands`, `agent.output_command_name`), writes its Inputs through `world.update`, and
turns the facade's eight instrumentation Events — plus a per-key listener on each
`agent:command:<call_id>` entry as it appears — into one normalized stream of `SutEvent` records.
It knows no application Command by name and does not know whether a script or a model drives.

Threading: every handler here runs on the SUT's loop thread — the Event handlers because the
facade emits there, the per-key Command listener because it is *async* (the World offloads a sync
listener to its thread-pool executor, which would let a `command_state` record land after the
events that state change caused). So the timeline is causally ordered: a Command's `complete`
always precedes the step its completion triggers. Each handler only builds a record and hands it
to the subscribed listeners (each isolated), never blocks, never waits.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage

from wica.agent import CommandExecution, CommandIssued
from wica.command import Command
from wica.instrumentation import CommandTrace, ReactionTrace, now
from wica.testing.report import SutEvent
from wica.wica import Wica
from wica.world import WorldEntry, WorldEntryConfig

_logger = logging.getLogger(__name__)

_COMMAND_KEY_PREFIX = "agent:command:"
_NATIVE_COMMANDS = frozenset({"noop", "cancel_command"})


def _t() -> float:
    return now().timestamp()


class Probe:
    """The mirror connection to *sut*. `attach()` subscribes to the SUT's Events; `subscribe()` adds
    a listener to the resulting `SutEvent` stream; `set()` writes an Input and records it."""

    def __init__(self, sut: Wica, *, record_prompt_text: bool = False) -> None:
        self._sut = sut
        self._record_prompt_text = record_prompt_text
        self._lock = threading.Lock()
        self._listeners: list[Callable[[SutEvent], None]] = []
        self._key_listeners: dict[str, Callable[[WorldEntry], Any]] = {}
        self._attached = False

    # --- schema ----------------------------------------------------------------------------------

    @property
    def inputs(self) -> dict[str, WorldEntryConfig]:
        """The SUT entries the tester may write: every registered key that is not WICA's own
        `agent:*`. Re-read on each call (live)."""
        world = self._sut.world
        return {
            key: world.get_config(key)
            for key in world.keys()
            if not key.startswith("agent:")
        }

    @property
    def commands(self) -> dict[str, Command]:
        """The SUT's registered Commands minus WICA's native `noop` / `cancel_command`. The output
        Command is among them once the SUT has started (see `output_command_name`). Live."""
        return {
            name: command
            for name, command in self._sut.agent.commands.items()
            if name not in _NATIVE_COMMANDS
        }

    @property
    def output_command_name(self) -> str | None:
        return self._sut.agent.output_command_name

    # --- the stream ------------------------------------------------------------------------------

    def subscribe(self, listener: Callable[[SutEvent], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[SutEvent], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, event: SutEvent) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # one bad listener must not break the others
                _logger.exception("Probe listener raised on %s", event.kind)

    # --- acting ----------------------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Write the SUT's Input *key*: records `input_written` (with the version id the write
        will get) then calls `sut.world.update`, whose own validation (`KeyError`, `TypeError`,
        `RuntimeError` while stopped) is left to raise."""
        version_id = self._sut.world.get_entry(key).current.id + 1
        self._emit(
            SutEvent(kind="input_written", t=_t(), key=key, version_id=version_id)
        )
        self._sut.world.update(key, value)

    # --- perceiving ------------------------------------------------------------------------------

    def attach(self) -> None:
        """Subscribe to the SUT's eight Events (idempotent)."""
        if self._attached:
            return
        sut = self._sut
        sut.on_world_trigger.subscribe(self._on_world_trigger)
        sut.on_agent_trigger.subscribe(self._on_agent_trigger)
        sut.on_agent_trigger_dropped.subscribe(self._on_trigger_dropped)
        sut.on_agent_prompt.subscribe(self._on_prompt)
        sut.on_agent_command.subscribe(self._on_command)
        sut.on_agent_text.subscribe(self._on_text)
        sut.on_agent_reaction_ended.subscribe(self._on_reaction_ended)
        sut.on_agent_command_ended.subscribe(self._on_command_ended)
        self._attached = True

    def detach(self) -> None:
        """Unsubscribe everything, the per-key Command listeners still attached included."""
        if not self._attached:
            return
        sut = self._sut
        sut.on_world_trigger.unsubscribe(self._on_world_trigger)
        sut.on_agent_trigger.unsubscribe(self._on_agent_trigger)
        sut.on_agent_trigger_dropped.unsubscribe(self._on_trigger_dropped)
        sut.on_agent_prompt.unsubscribe(self._on_prompt)
        sut.on_agent_command.unsubscribe(self._on_command)
        sut.on_agent_text.unsubscribe(self._on_text)
        sut.on_agent_reaction_ended.unsubscribe(self._on_reaction_ended)
        sut.on_agent_command_ended.unsubscribe(self._on_command_ended)
        with self._lock:
            pending = dict(self._key_listeners)
            self._key_listeners.clear()
        for key, handler in pending.items():
            sut.world.remove_listener(key, handler)
        self._attached = False

    def _on_world_trigger(self, entry: WorldEntry) -> None:
        self._emit(
            SutEvent(
                kind="trigger_raw",
                t=_t(),
                key=entry.key,
                version_id=entry.current.id,
            )
        )

    def _on_agent_trigger(self, entry: WorldEntry) -> None:
        self._emit(
            SutEvent(
                kind="trigger_observed",
                t=_t(),
                key=entry.key,
                version_id=entry.current.id,
            )
        )

    def _on_trigger_dropped(self, entry: WorldEntry) -> None:
        self._emit(
            SutEvent(
                kind="trigger_dropped",
                t=_t(),
                key=entry.key,
                version_id=entry.current.id,
            )
        )

    def _on_prompt(self, messages: list[BaseMessage]) -> None:
        text = (
            "\n".join(str(m.content) for m in messages)
            if self._record_prompt_text
            else None
        )
        self._emit(
            SutEvent(kind="prompt", t=_t(), message_count=len(messages), text=text)
        )

    def _on_command(self, issued: CommandIssued) -> None:
        self._emit(
            SutEvent(
                kind="command_issued",
                t=_t(),
                name=issued.name,
                call_id=issued.call_id,
                args=dict(issued.args),
            )
        )
        if issued.name == "noop":
            return  # no World entry, no state to follow
        key = f"{_COMMAND_KEY_PREFIX}{issued.call_id}"
        call_id = issued.call_id

        # Async on purpose: the World runs an async listener as a task on the SUT's loop, queued
        # *before* the trigger the same write emits, so the `command_state` record lands in the
        # timeline before anything the completion causes (the re-triggered step, its text). A sync
        # listener is offloaded to the thread-pool executor and can land after them — seen on slow
        # CI runners, where the Harness watermark then skipped the text that followed.
        async def handler(entry: WorldEntry) -> None:
            execution = entry.current.value
            if not isinstance(execution, CommandExecution):
                return
            self._emit(
                SutEvent(
                    kind="command_state",
                    t=_t(),
                    key=key,
                    name=execution.name,
                    call_id=call_id,
                    args=dict(execution.args),
                    state=execution.state,
                    result=execution.result,
                    error=execution.error,
                )
            )
            if execution.is_terminal():
                with self._lock:
                    self._key_listeners.pop(key, None)
                # The Agent may already have unregistered the key; remove_listener tolerates that.
                self._sut.world.remove_listener(key, handler)

        with self._lock:
            self._key_listeners[key] = handler
        try:
            # on_agent_command fires once the entry is registered and before its running value
            # lands, so the listener sees running → terminal (specs/agent.md).
            self._sut.world.add_listener(key, handler)
        except KeyError:
            with self._lock:
                self._key_listeners.pop(key, None)

    def _on_text(self, text: str) -> None:
        self._emit(SutEvent(kind="output_text", t=_t(), text=text))

    def _on_reaction_ended(self, trace: ReactionTrace) -> None:
        self._emit(SutEvent(kind="reaction_ended", t=_t(), reaction=trace))

    def _on_command_ended(self, trace: CommandTrace) -> None:
        self._emit(
            SutEvent(
                kind="command_ended",
                t=_t(),
                name=trace.name,
                call_id=trace.call_id,
                state=trace.state,
                command=trace,
            )
        )
