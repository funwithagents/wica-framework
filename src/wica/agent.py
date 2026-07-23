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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self._world = world if world is not None else get_world()
        self._output_sink = output_sink if output_sink is not None else _noop_output_sink

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
        # Rendered (fresh, archival) Content for entries unregistered right after being
        # captured into history (see _append_observation) — World.render_entry needs the
        # entry's serialize_fn, which unregister() deletes, so a detached entry can no
        # longer be re-rendered from the World on a later call without this cache.
        self._detached_renders: dict[tuple[str, int], tuple[Content, Content]] = {}

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

    def start(self) -> None:
        self._world.set_trigger_handler(self._on_world_trigger)
        if self._loop_thread is not None and not self._loop_thread.is_alive():
            self._loop_thread.start()

    def stop(self) -> None:
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
            task.cancel()

    def _on_world_trigger(self, entry: WorldEntry) -> None:
        asyncio.run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)

    async def _handle_trigger(self, entry: WorldEntry) -> None:
        if self._busy:
            _logger.info(
                "dropping trigger for %r (id=%d) — a call is already in flight",
                entry.key,
                entry.current.id,
            )
            return
        self._busy = True
        try:
            await self._run_step(entry)
        finally:
            self._busy = False

    async def _run_step(self, entry: WorldEntry) -> None:
        self._append_observation(entry)
        messages = self._render_messages()
        response = await self._bound_model.ainvoke(messages)

        text = response.text
        if text:
            self._history.append(AssistantTextRecord(text))
            await self._output_sink(text)

        for call in response.tool_calls:
            call_id = call["id"] or uuid.uuid4().hex
            self._history.append(CommandRecord(call_id, call["name"], call["args"]))
            self._dispatch_command(call_id, call["name"], call["args"])

    def _dispatch_command(self, call_id: str, name: str, args: dict[str, Any]) -> None:
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
        try:
            command = self._commands[name]
            result = await command.ainvoke(args)
        except asyncio.CancelledError:
            self._world.update(key, CommandExecution(name=name, args=args, state="cancelled"))
            raise
        except Exception as exc:
            self._world.update(
                key, CommandExecution(name=name, args=args, state="failed", error=str(exc))
            )
        else:
            self._world.update(
                key, CommandExecution(name=name, args=args, state="complete", result=str(result))
            )
        finally:
            self._running_tasks.pop(key, None)

    def _append_observation(self, entry: WorldEntry) -> None:
        self._history.append(ObservationRecord(self._world.get_prompt_entries()))
        if entry.key in self._command_keys and entry.current.value.is_terminal():
            self._detached_renders[(entry.key, entry.current.id)] = (
                self._world.render_entry(entry, archival=False),
                self._world.render_entry(entry, archival=True),
            )
            self._world.unregister(entry.key)
            self._command_keys.discard(entry.key)

    def _render_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        newest_observation_index: int | None = None
        for i, record in enumerate(self._history):
            if isinstance(record, ObservationRecord):
                newest_observation_index = i

        pending_ai_blocks: list[str | dict[str, Any]] = []

        def flush_pending_ai() -> None:
            if pending_ai_blocks:
                messages.append(AIMessage(content=list(pending_ai_blocks)))
                pending_ai_blocks.clear()

        for i, record in enumerate(self._history):
            if isinstance(record, ObservationRecord):
                flush_pending_ai()
                archival = i != newest_observation_index
                blocks: list[str | dict[str, Any]] = []
                for world_entry in record.entries:
                    cached = self._detached_renders.get(
                        (world_entry.key, world_entry.current.id)
                    )
                    if cached is not None:
                        rendered = cached[1] if archival else cached[0]
                    else:
                        rendered = self._world.render_entry(world_entry, archival=archival)
                    blocks.extend(_content_to_message_blocks(rendered))
                messages.append(HumanMessage(content=blocks))
            elif isinstance(record, AssistantTextRecord):
                pending_ai_blocks.append({"type": "text", "text": record.text})
            elif isinstance(record, CommandRecord):
                pending_ai_blocks.append(
                    {
                        "type": "text",
                        "text": f"Calling {record.name}({_format_args(record.args)})…",
                    }
                )

        flush_pending_ai()
        return messages
