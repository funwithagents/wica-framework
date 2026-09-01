from __future__ import annotations

import asyncio
import base64
import copy
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

from wica.config import AgentConfig, resolve_api_key, resolve_system_prompt
from wica.content import Content, ImagePart, TextPart
from wica.events import Event
from wica.world import World, WorldEntry

_logger = logging.getLogger(__name__)

# Max seconds stop() waits for Agent-owned async tasks to unwind after cancellation. Genuinely
# async work settles instantly; the bound guards against an uncooperative task (see
# specs/commands.md, "Cancellation reaches the task, not always the work").
_STOP_DRAIN_TIMEOUT = 5.0


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


# The payload of the Agent's on_command instrumentation Event: a named, evolvable value for a
# Command the model issued, preferred over a bare (name, args) tuple. See specs/agent.md
# ("Instrumentation") and specs/wica.md.
@dataclass(frozen=True)
class CommandIssued:
    name: str
    args: dict[str, Any]


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


# WICA's provider value for the Hugging Face Hub's serverless Inference Providers. Deliberately
# *not* "huggingface" — that's init_chat_model's builtin, which builds a local transformers
# pipeline (heavy, and non-cancellable). See build_chat_model and specs/config.md ("Providers").
_HUGGINGFACE_HUB_PROVIDER = "huggingface-hub"

# WICA's provider value for the deterministic, network-free, key-less scripted test model. Built
# directly (like huggingface-hub, and before the init_chat_model fallthrough); its `model_kwargs`
# carry the fake's own config (script/default/loop/delay_s), not provider kwargs. See
# specs/fake-provider.md and src/wica/fake_model.py.
_FAKE_PROVIDER = "fake"


def build_chat_model(config: AgentConfig) -> BaseChatModel:
    """Construct the LangChain chat model for an AgentConfig — the single source of truth for
    provider construction, shared by the Agent constructor and the e2e helper (so tests exercise
    the exact model production builds).

    Most providers pass straight through to init_chat_model (switching between them is a config
    edit). `huggingface-hub` is the exception: it targets the Hugging Face Hub's serverless
    Inference Providers and is built directly as ChatHuggingFace(llm=HuggingFaceEndpoint(...)) —
    *not* via init_chat_model, whose builtin `huggingface` provider builds a local transformers
    pipeline, a blocking generation call the Agent's task.cancel() can't stop. See specs/agent.md
    ("Provider-agnostic model, from config") and specs/config.md ("Providers").

    Provider integration packages are imported lazily, inside their branch, so core wica needs
    none of them installed — an unselected provider fails here with a clear ImportError.

    The api key resolves *here*, at build: the config holds `api_key`/`api_key_env` verbatim, and
    resolve_api_key does the env read (raising MissingEnvError if the referenced var is unset). See
    specs/config.md ("API key").
    """
    if config.provider == _FAKE_PROVIDER:
        # The fake's construction payload lives in model_kwargs (script/default/loop/delay_s), which
        # is what keeps it JSON-expressible and on the ordinary config path. api_key/model are
        # ignored, so we don't resolve the key at all here.
        from wica.fake_model import FakeChatModel

        return FakeChatModel(**config.model_kwargs)

    api_key = resolve_api_key(config)

    if config.provider == _HUGGINGFACE_HUB_PROVIDER:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint_kwargs = dict(config.model_kwargs)
        if api_key is not None:
            # This branch builds the model directly, so it owns the kwarg name: the resolved
            # api_key/api_key_env is HF's token, not the generic `api_key` the init_chat_model
            # providers receive.
            endpoint_kwargs["huggingfacehub_api_token"] = api_key
        endpoint = HuggingFaceEndpoint(
            repo_id=config.model,
            provider=config.hf_provider,
            task="text-generation",
            **endpoint_kwargs,
        )
        return ChatHuggingFace(llm=endpoint)

    model_kwargs = dict(config.model_kwargs)
    if api_key is not None:
        model_kwargs["api_key"] = api_key
    return init_chat_model(config.model, model_provider=config.provider, **model_kwargs)


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        world: World,
        loop: asyncio.AbstractEventLoop,
        coalesce_window: float = 0.2,
        output_sink: Callable[[str], Awaitable[None]] | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        # Config-driven construction *is* the constructor — resolution happens here, at build. The
        # config holds system_prompt/system_prompt_file (and api_key/api_key_env, resolved inside
        # build_chat_model) verbatim; resolve_system_prompt reads the prompt file if that's the form
        # given, so an unreadable file or unset key env var surfaces here, not at config load. See
        # specs/config.md ("Flow into the Agent") and specs/agent.md.
        #
        # `model` is an optional override: config builds the model unless a bespoke BaseChatModel is
        # passed in (the raw-model injection seam — a caller supplying a model no config can express,
        # and the seam the Agent unit tests use to drive the loop over a fully-scripted fake). The
        # system prompt is always config-expressed. See specs/agent.md, specs/wica.md (open q. 3).
        self.system_prompt = resolve_system_prompt(config)
        self.model = model if model is not None else build_chat_model(config)
        self._world = world
        # The shared event loop, owned and run by the Wica facade in one daemon thread and injected
        # here. The Agent runs its tasks/timers/cancellation on it but never starts or stops it. See
        # specs/agent.md ("The shared event loop").
        self._loop = loop
        # Trigger-coalescing window (seconds): a burst of triggers arriving within this window is
        # batched into a single step, rather than starting one step per trigger (and, under the
        # single-in-flight loop, dropping the rest). 0 disables it — each trigger fires immediately,
        # the pre-coalescing behavior. See specs/agent.md "Trigger coalescing".
        self._coalesce_window = coalesce_window
        self._output_sink = (
            output_sink if output_sink is not None else _noop_output_sink
        )
        # Instrumentation Events — observability only, never control flow. The Agent emits on them
        # at the right points inside the loop; any number of consumers subscribe, and Event.emit's
        # per-subscriber isolation catches+logs a raising subscriber so it can neither abort a step
        # nor starve siblings (no _fire_hook guard needed). See specs/agent.md "Instrumentation".
        #  - on_trigger(entry):    once per trigger a run-to-completion step observes (filtered).
        #  - on_prompt(messages):  the exact rendered messages just before each model call.
        #  - on_command(command):  each Command the model issues, at dispatch time.
        self.on_trigger: Event[WorldEntry] = Event()
        self.on_prompt: Event[list[BaseMessage]] = Event()
        self.on_command: Event[CommandIssued] = Event()

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
        # Every task the Agent creates, including trigger shims, reasoning batches, and Commands.
        # stop() cancels this complete set. An off-loop caller also drains cancellation before an
        # owned loop thread stops; a same-loop caller cannot block and lets cancellation unwind on
        # subsequent loop turns. _running_tasks remains the call-id lookup used by cancel_command.
        self._owned_tasks: set[asyncio.Task[Any]] = set()
        self._command_keys: set[str] = set()
        # The sync shim subscribed to world.on_trigger while running (set in start()).
        self._trigger_sub: Callable[[WorldEntry], None] | None = None
        self._started = False

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
        if self._started:
            return
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

        # Subscribe to the World's raw trigger. The World emits on_trigger on the loop thread
        # (call_soon_threadsafe — see specs/world.md), so this sync shim runs there and create_task
        # is safe — no run_coroutine_threadsafe bridge. Wica owns and starts the loop; the Agent
        # only attaches to it here.
        def schedule_trigger(entry: WorldEntry) -> None:
            self._track_task(self._handle_trigger(entry))

        self._trigger_sub = schedule_trigger
        self._world.on_trigger.subscribe(schedule_trigger)
        self._started = True

    def stop(self) -> None:
        _logger.info("agent stopping")
        self._started = False
        if self._trigger_sub is not None:
            self._world.on_trigger.unsubscribe(self._trigger_sub)
            self._trigger_sub = None
        # Cancel the pending coalescing window and every Agent-owned task. An off-loop caller can
        # wait for cancellation to unwind while the World is still running; a caller already on the
        # shared loop cannot block it and returns after requesting cancellation. The Agent never
        # tears the loop down (Wica owns it).
        try:
            running_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            # Called from the loop thread itself — can't block on it; cancel without the drain.
            self._cancel_all_tasks_on_loop()
            return
        if not self._loop.is_running():
            # Loop never started (or already stopped): nothing was scheduled, so nothing to drain.
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._stop_on_loop(), self._loop)
            future.result(timeout=_STOP_DRAIN_TIMEOUT)
        except RuntimeError:
            pass  # loop not running / already closed — nothing to drain
        except FuturesTimeoutError:
            _logger.warning("agent stop: timed out draining Agent-owned tasks")

    async def _stop_on_loop(self) -> None:
        """Cancel the pending window and all Agent-owned tasks, then await them settling."""
        tasks = self._cancel_all_tasks_on_loop()
        current = asyncio.current_task()
        tasks = [task for task in tasks if task is not current]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._busy = False

    def _cancel_all_tasks_on_loop(self) -> list[asyncio.Task[Any]]:
        """Cancel all work belonging to the current running cycle (loop thread only)."""
        self._cancel_window()
        tasks = list(self._owned_tasks)
        for task in tasks:
            task.cancel()
        return tasks

    def _track_task[T](self, coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Create an Agent-owned loop task and forget it only after it has settled."""
        task = self._loop.create_task(coroutine)
        self._owned_tasks.add(task)
        task.add_done_callback(self._owned_tasks.discard)
        return task

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
        return (
            f"{call_id} is not a running command (it already finished or never existed)"
        )

    async def _handle_trigger(self, entry: WorldEntry) -> None:
        # Runs on the loop thread (the World emits on_trigger there, and the start() shim
        # create_tasks this coroutine on it). All window state below is therefore touched
        # single-threaded, so open/join/flush are race-free — the same invariant that makes
        # cancellation race-free (see specs/agent.md "The shared event loop").
        _logger.debug("trigger received: %s", _describe_entry(entry))
        if self._busy:
            _logger.info(
                "dropping trigger (%s) — a call is already in flight",
                _describe_entry(entry),
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
            self._window_timer = self._loop.call_later(
                self._coalesce_window, self._flush_window
            )

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
        self._track_task(self._run_batch(batch))

    async def _run_batch(self, batch: list[WorldEntry]) -> None:
        try:
            await self._run_step(batch)
        finally:
            self._busy = False

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
            _logger.debug(
                "step starting (trigger: %s)", _describe_entry(representative)
            )
        for triggered in batch:
            self.on_trigger.emit(triggered)
        self._append_observation()
        messages = self._render_messages()
        # Instrumentation is observation-only: subscribers receive a defensive copy, never the
        # message list subsequently handed to the model.
        self.on_prompt.emit(copy.deepcopy(messages))
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
            args = copy.deepcopy(call["args"])
            self._history.append(
                CommandRecord(call_id, call["name"], copy.deepcopy(args))
            )
            self._dispatch_command(call_id, call["name"], args)
        _logger.debug("step complete (trigger: %s)", _describe_entry(representative))

    def _dispatch_command(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        # A subscriber may annotate or otherwise mutate what it receives without changing the
        # arguments stored in history, shown in the World, or passed to the Command itself.
        self.on_command.emit(CommandIssued(name, copy.deepcopy(args)))
        _logger.debug(
            "dispatching command %s(%s) call_id=%s", name, _format_args(args), call_id
        )
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
        task = self._track_task(self._run_command(key, call_id, name, args))
        self._running_tasks[key] = task

    async def _run_command(
        self, key: str, call_id: str, name: str, args: dict[str, Any]
    ) -> None:
        _logger.debug(
            "command %s(%s) [call_id=%s] executing", name, _format_args(args), call_id
        )
        try:
            command = self._commands[name]
            result = await command.ainvoke(args)
        except asyncio.CancelledError:
            _logger.debug("command %s [call_id=%s] cancelled", name, call_id)
            try:
                self._world.update(
                    key, CommandExecution(name=name, args=args, state="cancelled")
                )
            except RuntimeError:
                # World already stopped — this cancel is part of Wica teardown (agent.stop() cancels
                # in-flight commands just before world.stop()). The terminal write is moot at
                # shutdown; swallow it so the task still unwinds cleanly. See specs/wica.md.
                pass
            raise
        except Exception as exc:
            _logger.warning("command %s [call_id=%s] failed: %s", name, call_id, exc)
            self._world.update(
                key,
                CommandExecution(name=name, args=args, state="failed", error=str(exc)),
            )
        else:
            _logger.debug(
                "command %s [call_id=%s] complete → %s",
                name,
                call_id,
                _truncate(str(result)),
            )
            self._world.update(
                key,
                CommandExecution(
                    name=name, args=args, state="complete", result=str(result)
                ),
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
                AIMessage(
                    content="\n".join(pending_text), tool_calls=list(pending_calls)
                )
            )
            # The tool_result is a fixed ack pointing at the Command's World entry — never the
            # outcome. The outcome is delivered by that entry, rendered as an observation at the
            # step where the completion is observed (see _command_ack and the loop below).
            for call in pending_calls:
                messages.append(
                    ToolMessage(
                        content=_command_ack(call["id"]), tool_call_id=call["id"]
                    )
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
                            world_entry,
                            archival=archival,
                            serialize_fn=_serialize_command_execution,
                        )
                    else:
                        rendered = self._world.render_entry(
                            world_entry, archival=archival
                        )
                    blocks.extend(_content_to_message_blocks(rendered))
                messages.append(HumanMessage(content=blocks))
            elif isinstance(record, AssistantTextRecord):
                pending_text.append(record.text)
            elif isinstance(record, CommandRecord):
                pending_calls.append(
                    {
                        "name": record.name,
                        "args": copy.deepcopy(record.args),
                        "id": record.call_id,
                    }
                )

        flush_assistant()
        return messages
