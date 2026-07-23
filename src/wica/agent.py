from __future__ import annotations

import asyncio
import base64
import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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

from wica.content import Content, ImagePart, TextPart
from wica.world import World, WorldEntry, get_world

_logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    provider: str
    model: str
    system_prompt: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


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
        text = f"Calling {call}…"
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
        output_sink: Callable[[str], Awaitable[None]] | None = None,
        on_prompt: Callable[[list[BaseMessage]], None] | None = None,
        on_trigger: Callable[[WorldEntry], None] | None = None,
        on_command: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._world = world if world is not None else get_world()
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
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._command_keys: set[str] = set()
        # call_id -> stringified command result, used to render the tool_result that pairs with
        # each past command's native tool_call when re-rendering history (see _render_messages).
        self._command_results: dict[str, str] = {}

    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs: Any) -> Agent:
        model = init_chat_model(
            config.model, model_provider=config.provider, **config.model_kwargs
        )
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
        _logger.info("agent starting (%d command(s) registered)", len(self._commands))
        self._world.set_trigger_handler(self._on_world_trigger)
        if self._loop_thread is not None and not self._loop_thread.is_alive():
            self._loop_thread.start()

    def stop(self) -> None:
        _logger.info("agent stopping")
        self._world.set_trigger_handler(None)
        for key in list(self._running_tasks):
            call_id = key.removeprefix("agent:command:")
            self.cancel_command(call_id)
        if self._owns_loop and self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()

    def cancel_command(self, call_id: str) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            self._cancel_command_on_loop(call_id)
        else:
            self._loop.call_soon_threadsafe(self._cancel_command_on_loop, call_id)

    def _cancel_command_on_loop(self, call_id: str) -> None:
        key = f"agent:command:{call_id}"
        task = self._running_tasks.get(key)
        if task is not None:
            _logger.debug("cancelling command call_id=%s", call_id)
            task.cancel()

    def _on_world_trigger(self, entry: WorldEntry) -> None:
        asyncio.run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)

    async def _handle_trigger(self, entry: WorldEntry) -> None:
        _logger.debug("trigger received: %s", _describe_entry(entry))
        if self._busy:
            _logger.info(
                "dropping trigger (%s) — a call is already in flight", _describe_entry(entry)
            )
            # A dropped input trigger is simply not reacted to. But a dropped *command
            # completion* must still be cleaned up, or the terminal command entry would
            # linger in the World (and every future prompt) forever — the v1 single-in-flight
            # loop just won't run a step to react to it. Its effect already lives in whatever
            # other World entries the command wrote.
            if self._is_terminal_command(entry):
                _logger.debug("cleaning up dropped command completion %r", entry.key)
                self._cleanup_command_entry(entry.key)
            return
        self._busy = True
        try:
            await self._run_step(entry)
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

    async def _run_step(self, entry: WorldEntry) -> None:
        _logger.debug("step starting (trigger: %s)", _describe_entry(entry))
        self._fire_hook(self._on_trigger, entry)
        self._append_observation(entry)
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
        _logger.debug("step complete (trigger: %s)", _describe_entry(entry))

    def _dispatch_command(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self._fire_hook(self._on_command, name, args)
        _logger.debug("dispatching command %s(%s) call_id=%s", name, _format_args(args), call_id)
        key = f"agent:command:{call_id}"
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
            self._command_results[call_id] = "cancelled"
            self._world.update(key, CommandExecution(name=name, args=args, state="cancelled"))
            raise
        except Exception as exc:
            _logger.warning("command %s [call_id=%s] failed: %s", name, call_id, exc)
            self._command_results[call_id] = f"failed: {exc}"
            self._world.update(
                key, CommandExecution(name=name, args=args, state="failed", error=str(exc))
            )
        else:
            _logger.debug(
                "command %s [call_id=%s] complete → %s", name, call_id, _truncate(str(result))
            )
            self._command_results[call_id] = str(result)
            self._world.update(
                key, CommandExecution(name=name, args=args, state="complete", result=str(result))
            )
        finally:
            self._running_tasks.pop(key, None)

    def _append_observation(self, entry: WorldEntry) -> None:
        observation = self._world.get_prompt_entries()
        self._history.append(ObservationRecord(observation))
        # Now that this observation has captured their results, retire every command entry
        # that has reached a terminal state — not just the one that fired. Cleaning all of
        # them means a completion whose own trigger was dropped by the single-in-flight loop
        # (e.g. a second command finishing while we were busy reacting to the first) still
        # gets cleaned up on the next step that runs, instead of lingering forever.
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
            for call in pending_calls:
                result = self._command_results.get(call["id"], "(in progress)")
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            pending_text.clear()
            pending_calls.clear()

        for i, record in enumerate(self._history):
            if isinstance(record, ObservationRecord):
                flush_assistant()
                archival = i != newest_observation_index
                blocks: list[str | dict[str, Any]] = []
                for world_entry in record.entries:
                    # A command's outcome is carried by its tool_result above, so its World entry
                    # isn't re-rendered here (it would duplicate the result — and its serialize_fn
                    # may already be gone once the entry is retired).
                    if world_entry.key.startswith("agent:command:"):
                        continue
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
