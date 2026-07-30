from __future__ import annotations

import asyncio
import base64
import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from wica.config import AgentConfig
from wica.content import Content, ImagePart, TextPart
from wica.world import World, WorldEntry, get_world

_logger = logging.getLogger(__name__)


# A Command is WICA's unit of agent action on the World, backed under the hood by a
# LangChain tool (see specs/commands.md). CommandExecution is the value stored in the
# per-call `agent:command:<call_id>` World entry that tracks its lifecycle.
@dataclass(frozen=True)
class CommandExecution:
    name: str
    args: dict[str, Any]
    state: Literal["running", "complete", "failed", "cancelled"]
    result: str | None = None
    error: str | None = None

    def is_terminal(self) -> bool:
        return self.state != "running"


# History records — Observation captures the full include_in_prompt bundle at trigger
# time (not just the entry that fired), so passive include_in_prompt entries still reach
# the model. See specs/agent.md "History record shape".
@dataclass(frozen=True)
class ObservationRecord:
    entries: list[WorldEntry]


@dataclass(frozen=True)
class AssistantTextRecord:
    text: str


@dataclass(frozen=True)
class CommandRecord:
    call_id: str
    name: str
    args: dict[str, Any]


HistoryRecord = ObservationRecord | AssistantTextRecord | CommandRecord


def _format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_entry(entry: WorldEntry) -> str:
    """A log-friendly summary of a triggering entry that shows its *content*, not just its
    key/id — e.g. a bare `agent:command:c1` key is meaningless, but `dance() [complete]` isn't."""
    value = entry.current.value
    if isinstance(value, CommandExecution):
        summary = f"{value.name}({_format_args(value.args)}) [{value.state}]"
    elif value is None:
        summary = "None"
    else:
        summary = _truncate(repr(value), 80)
    return f"{entry.key}#{entry.current.id} = {summary}"


def _serialize_command_execution(
    value: CommandExecution | None, previous: CommandExecution | None
) -> Content:
    if value is None:
        return [TextPart("(no command)")]
    call = f"{value.name}({_format_args(value.args)})"
    if value.state == "running":
        text = f"Calling {call}… (still running — not finished)"
    elif value.state == "complete":
        text = f"Called {call} → {value.result}"
    elif value.state == "failed":
        text = f"Called {call} → failed: {value.error}"
    else:
        text = f"Called {call} → cancelled"
    return [TextPart(text)]


def _content_to_message_blocks(content: Content) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append(
                {
                    "type": "image",
                    "base64": base64.b64encode(part.data).decode("ascii"),
                    "mime_type": part.media_type,
                }
            )
        else:
            blocks.append({"type": "text", "text": part.to_string()})
    return blocks


_COMMAND_KEY_PREFIX = "agent:command:"

# cancel_command is a WICA-native control Command (not backed by an off-the-shelf tool): the Agent
# implements it directly and auto-registers it so the model can abort a Command it previously issued
# that is still running. It targets the command by its call_id — already on screen in the
# `<entry key="agent:command:<call_id>" …>` envelope — so nothing extra is rendered. See
# specs/commands.md ("cancel_command") and specs/agent.md ("Commands").
_CANCEL_COMMAND_NAME = "cancel_command"
_CANCEL_COMMAND_DESCRIPTION = (
    "Cancel a command you previously issued that is still running. Pass its call_id — the part "
    f"after '{_COMMAND_KEY_PREFIX}' in the running command's World entry key (the full key is "
    "accepted too). Cancelling a command that has already finished or never existed is a harmless "
    "no-op."
)


def _command_ack(call_id: str) -> str:
    """The stringified outcome carried by a Command's tool_result. It is a fixed *pointer*, not the
    result: a tool_result is pinned immediately after its tool_use (a tool_use can't be left
    dangling across turns), so it can only sit at the Command's dispatch site, not where the Command
    actually finished — and while a Command is in flight a tool_result present at all reads as "the
    call returned." So the real status/result is delivered by the Command's World entry
    (`agent:command:<call_id>`), rendered as an observation at the point it happens. See
    specs/commands.md, specs/agent.md ("Rendering to messages")."""
    return (
        f"Dispatched. Live status and result appear in the World state as entry "
        f"{_COMMAND_KEY_PREFIX}{call_id}."
    )


async def _noop_output_sink(text: str) -> None:
    return None


class Agent:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        system_prompt: str,
        world: World | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        coalesce_window: float = 0.2,
        output_sink: Callable[[str], Awaitable[None]] | None = None,
        on_prompt: Callable[[list[BaseMessage]], None] | None = None,
        on_trigger: Callable[[WorldEntry], None] | None = None,
        on_command: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._world = world if world is not None else get_world()
        # Trigger-coalescing window (seconds): a burst of triggers arriving within this window is
        # batched into a single step, rather than starting one step per trigger (and, under the
        # single-in-flight loop, dropping the rest). 0 disables it — each trigger fires immediately,
        # the pre-coalescing behavior. See specs/agent.md "Trigger coalescing".
        self._coalesce_window = coalesce_window
        self._output_sink = output_sink if output_sink is not None else _noop_output_sink
        # Optional debug/observability hooks — instrumentation only, never control flow: each is
        # fired via _fire_hook, which swallows+logs a raising hook so it can't abort a step. See
        # specs/agent.md "Instrumentation".
        #  - on_prompt(messages): the exact messages just before each model call.
        #  - on_trigger(entry):   the World entry that started a step (only for steps that run).
        #  - on_command(name, args): each Command the model issues, at dispatch time.
        self._on_prompt = on_prompt
        self._on_trigger = on_trigger
        self._on_command = on_command

        self._owns_loop = loop is None
        self._loop_thread: threading.Thread | None
        if loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        else:
            self._loop = loop
            self._loop_thread = None

        # name -> the LangChain tool backing each registered Command (see commands.md)
        self._commands: dict[str, BaseTool] = {}
        self._bound_model: Runnable[Any, AIMessage] = self.model
        self._history: list[HistoryRecord] = []
        self._busy = False
        # Coalescing-window state, touched only on the loop thread (like _running_tasks): the
        # triggers batched into the currently-open window, and the timer that will flush them.
        self._window_batch: list[WorldEntry] = []
        self._window_timer: asyncio.TimerHandle | None = None
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._command_keys: set[str] = set()

    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs: Any) -> Agent:
        model_kwargs = dict(config.model_kwargs)
        if config.api_key is not None:
            model_kwargs["api_key"] = config.api_key
        model = init_chat_model(config.model, model_provider=config.provider, **model_kwargs)
        return cls(model, system_prompt=config.system_prompt, **kwargs)

    def register_command(
        self,
        fn: Callable[..., Any] | BaseTool,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a Command. Commonly a plain function or an off-the-shelf LangChain
        tool — the tool is the under-the-hood primitive the model issues the Command
        through (see specs/commands.md)."""
        wrapped: BaseTool
        if isinstance(fn, BaseTool):
            wrapped = fn
        elif name is not None:
            wrapped = tool(name, description=description)(fn)
        else:
            wrapped = tool(fn, description=description)
        self._commands[wrapped.name] = wrapped
        self._bound_model = self.model.bind_tools(list(self._commands.values()))
        _logger.debug("registered command %r", wrapped.name)

    def start(self) -> None:
        # WICA-native control Command: let the model abort a Command it previously issued that is
        # still running. Auto-registered here (no app wiring) since it needs Agent internals;
        # registering at start (not construction) keeps __init__ inert — it never touches the model.
        # register_command is idempotent on the name, so a second start() is harmless.
        self.register_command(
            self._cancel_command_action,
            name=_CANCEL_COMMAND_NAME,
            description=_CANCEL_COMMAND_DESCRIPTION,
        )
        _logger.info("agent starting (%d command(s) registered)", len(self._commands))
        self._world.set_trigger_handler(self._on_world_trigger)
        if self._loop_thread is not None and not self._loop_thread.is_alive():
            self._loop_thread.start()

    def stop(self) -> None:
        _logger.info("agent stopping")
        self._world.set_trigger_handler(None)
        # Cancel any pending coalescing-window timer on the loop thread (where all window state
        # lives). Hygiene: for an owned loop it's about to stop anyway, but an injected loop keeps
        # running, so a stale timer must not fire a step after stop().
        try:
            self._loop.call_soon_threadsafe(self._cancel_window)
        except RuntimeError:
            pass  # loop already closed
        for key in list(self._running_tasks):
            call_id = key.removeprefix(_COMMAND_KEY_PREFIX)
            self.cancel_command(call_id)
        if self._owns_loop and self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()

    def _cancel_window(self) -> None:
        """Cancel a pending coalescing-window timer and drop its batch (loop thread only)."""
        if self._window_timer is not None:
            self._window_timer.cancel()
            self._window_timer = None
        self._window_batch.clear()

    def cancel_command(self, call_id: str) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            self._cancel_command_on_loop(call_id)
        else:
            self._loop.call_soon_threadsafe(self._cancel_command_on_loop, call_id)

    def _cancel_command_on_loop(self, call_id: str) -> bool:
        """Cancel a running command's task on the loop thread. Returns whether a live task was
        actually cancelled (False if it's unknown or already finished) — an id-guarded no-op,
        same pattern as the World's TTL `expected_id` guard."""
        key = f"{_COMMAND_KEY_PREFIX}{call_id}"
        task = self._running_tasks.get(key)
        if task is None or task.done():
            return False
        _logger.debug("cancelling command call_id=%s", call_id)
        task.cancel()
        return True

    async def _cancel_command_action(self, call_id: str) -> str:
        """Backing action for the model-issued `cancel_command` Command. Async so it runs on the
        Agent's loop thread (via `ainvoke`), keeping `_running_tasks` access race-free. Lenient:
        accepts either the bare call_id or the full `agent:command:<call_id>` entry key."""
        call_id = call_id.removeprefix(_COMMAND_KEY_PREFIX)
        if self._cancel_command_on_loop(call_id):
            return f"cancelling {call_id}"
        return f"{call_id} is not a running command (it already finished or never existed)"

    def _on_world_trigger(self, entry: WorldEntry) -> None:
        asyncio.run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)

    async def _handle_trigger(self, entry: WorldEntry) -> None:
        # Runs on the loop thread (scheduled by _on_world_trigger). All window state below is
        # therefore touched single-threaded, so open/join/flush are race-free — the same invariant
        # that makes cancellation race-free (see specs/agent.md "The Agent owns the event loop").
        _logger.debug("trigger received: %s", _describe_entry(entry))
        if self._busy:
            _logger.info(
                "dropping trigger (%s) — a call is already in flight", _describe_entry(entry)
            )
            # We don't run a step for a dropped trigger. A dropped *command completion* is left
            # in place on purpose — NOT retired here — so the terminal entry stays part of
            # current World state and gets rendered into history (then retired) by the next step
            # that observes it (see _append_observation). This keeps the invariant that a
            # completed Command is always in history or in current state, never silently lost.
            #
            # Coalescing only batches the arrival burst *before* a step starts; a trigger arriving
            # while one is in flight is still dropped (bypass_coalescing skips the wait, not this
            # drop). Collecting during-flight triggers is the deferred concurrency/queue question.
            return

        self._window_batch.append(entry)
        if entry.bypass_coalescing or self._coalesce_window <= 0:
            # Fire immediately: an urgent entry flushes the window early (carrying anything already
            # batched), and a zero window is simply a window that closes at once.
            self._flush_window()
        elif self._window_timer is None:
            # First trigger of a burst opens a fixed leading-edge window. Later triggers join the
            # batch above without rescheduling, so the window never extends (bounded latency).
            self._window_timer = self._loop.call_later(self._coalesce_window, self._flush_window)

    def _flush_window(self) -> None:
        """Close the coalescing window and start the single step for the batched triggers. Sync,
        runs on the loop thread (called directly for an immediate flush, or by the window timer)."""
        if self._window_timer is not None:
            self._window_timer.cancel()
            self._window_timer = None
        if not self._window_batch:
            return
        batch = self._window_batch
        self._window_batch = []
        # Set _busy before the step task runs so triggers arriving in the gap are dropped, not
        # folded into a second concurrent step (single-in-flight).
        self._busy = True
        self._loop.create_task(self._run_batch(batch))

    async def _run_batch(self, batch: list[WorldEntry]) -> None:
        try:
            await self._run_step(batch)
        finally:
            self._busy = False

    def _fire_hook(self, hook: Callable[..., None] | None, *args: Any) -> None:
        """Invoke an optional instrumentation hook, swallowing+logging any exception so a
        misbehaving hook can never abort a step."""
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:
            _logger.exception("agent instrumentation hook raised; ignoring")

    async def _run_step(self, batch: list[WorldEntry]) -> None:
        # A coalesced burst runs a single step, but on_trigger fires once per trigger that joined
        # the window (so a UI still shows every input). The Observation, on_prompt, and the model
        # call below happen once. See specs/agent.md "Trigger coalescing".
        representative = batch[-1]
        if len(batch) > 1:
            _logger.debug(
                "step starting (coalesced %d triggers, latest: %s)",
                len(batch),
                _describe_entry(representative),
            )
        else:
            _logger.debug("step starting (trigger: %s)", _describe_entry(representative))
        for triggered in batch:
            self._fire_hook(self._on_trigger, triggered)
        self._append_observation()
        messages = self._render_messages()
        self._fire_hook(self._on_prompt, messages)
        response = await self._bound_model.ainvoke(messages)

        text = response.text
        _logger.debug(
            "LLM output: text=%s, commands=%s",
            _truncate(text) if text else "(none)",
            [c["name"] for c in response.tool_calls] or "(none)",
        )
        if text:
            self._history.append(AssistantTextRecord(text))
            await self._output_sink(text)

        for call in response.tool_calls:
            call_id = call["id"] or uuid.uuid4().hex
            self._history.append(CommandRecord(call_id, call["name"], call["args"]))
            self._dispatch_command(call_id, call["name"], call["args"])
        _logger.debug("step complete (trigger: %s)", _describe_entry(representative))

    def _dispatch_command(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self._fire_hook(self._on_command, name, args)
        _logger.debug("dispatching command %s(%s) call_id=%s", name, _format_args(args), call_id)
        key = f"{_COMMAND_KEY_PREFIX}{call_id}"
        self._command_keys.add(key)
        self._world.register(
            key,
            CommandExecution,
            serialize_fn=_serialize_command_execution,
            include_in_prompt=True,
            triggers_llm_call=True,
            trigger_condition_fn=lambda old, new: new is not None and new.is_terminal(),
        )
        self._world.update(key, CommandExecution(name=name, args=args, state="running"))
        task = self._loop.create_task(self._run_command(key, call_id, name, args))
        self._running_tasks[key] = task

    async def _run_command(self, key: str, call_id: str, name: str, args: dict[str, Any]) -> None:
        _logger.debug("command %s(%s) [call_id=%s] executing", name, _format_args(args), call_id)
        try:
            command = self._commands[name]
            result = await command.ainvoke(args)
        except asyncio.CancelledError:
            _logger.debug("command %s [call_id=%s] cancelled", name, call_id)
            self._world.update(key, CommandExecution(name=name, args=args, state="cancelled"))
            raise
        except Exception as exc:
            _logger.warning("command %s [call_id=%s] failed: %s", name, call_id, exc)
            self._world.update(
                key, CommandExecution(name=name, args=args, state="failed", error=str(exc))
            )
        else:
            _logger.debug(
                "command %s [call_id=%s] complete → %s", name, call_id, _truncate(str(result))
            )
            self._world.update(
                key, CommandExecution(name=name, args=args, state="complete", result=str(result))
            )
        finally:
            self._running_tasks.pop(key, None)

    def _append_observation(self) -> None:
        observation = self._world.get_prompt_entries()
        self._history.append(ObservationRecord(observation))
        # This snapshot has now captured every terminal command entry's outcome (they render from
        # it in _render_messages), so retire them all here — not just the one that fired. This is
        # the *sole* retirement path: a completion whose own trigger was dropped by the
        # single-in-flight loop stays in the World until some step observes it here, so its outcome
        # always reaches history before the entry goes away (never silently lost). The captured
        # snapshot keeps re-rendering from history via render_entry's override after unregister.
        for world_entry in observation:
            if self._is_terminal_command(world_entry):
                _logger.debug("retiring completed command entry %r", world_entry.key)
                self._cleanup_command_entry(world_entry.key)

    def _is_terminal_command(self, entry: WorldEntry) -> bool:
        value = entry.current.value
        return (
            entry.key in self._command_keys
            and isinstance(value, CommandExecution)
            and value.is_terminal()
        )

    def _cleanup_command_entry(self, key: str) -> None:
        self._command_keys.discard(key)
        try:
            self._world.unregister(key)
        except KeyError:
            pass

    def _render_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        newest_observation_index: int | None = None
        for i, record in enumerate(self._history):
            if isinstance(record, ObservationRecord):
                newest_observation_index = i

        # A step's assistant output accumulates here until the next observation flushes it. We
        # render past Commands as the model's *native* tool_calls (not a "Calling …" text block,
        # which the model would otherwise imitate — emitting the description as text instead of a
        # real call), each paired with the tool_result carrying its outcome.
        pending_text: list[str] = []
        pending_calls: list[dict[str, Any]] = []

        def flush_assistant() -> None:
            if not pending_text and not pending_calls:
                return
            messages.append(
                AIMessage(content="\n".join(pending_text), tool_calls=list(pending_calls))
            )
            # The tool_result is a fixed ack pointing at the Command's World entry — never the
            # outcome. The outcome is delivered by that entry, rendered as an observation at the
            # step where the completion is observed (see _command_ack and the loop below).
            for call in pending_calls:
                messages.append(
                    ToolMessage(content=_command_ack(call["id"]), tool_call_id=call["id"])
                )
            pending_text.clear()
            pending_calls.clear()

        for i, record in enumerate(self._history):
            if isinstance(record, ObservationRecord):
                flush_assistant()
                archival = i != newest_observation_index
                blocks: list[str | dict[str, Any]] = []
                for world_entry in record.entries:
                    # Command entries carry the outcome (running → terminal). They render through
                    # the Agent's own serializer via render_entry's override, so the World stays
                    # command-agnostic and a retired entry still re-renders from this snapshot
                    # after its config was unregistered. Other entries use the registered fn.
                    if world_entry.key.startswith(_COMMAND_KEY_PREFIX):
                        rendered = self._world.render_entry(
                            world_entry, archival=archival, serialize_fn=_serialize_command_execution
                        )
                    else:
                        rendered = self._world.render_entry(world_entry, archival=archival)
                    blocks.extend(_content_to_message_blocks(rendered))
                messages.append(HumanMessage(content=blocks))
            elif isinstance(record, AssistantTextRecord):
                pending_text.append(record.text)
            elif isinstance(record, CommandRecord):
                pending_calls.append(
                    {"name": record.name, "args": record.args, "id": record.call_id}
                )

        flush_assistant()
        return messages
