from __future__ import annotations

import asyncio
import base64
import contextvars
import copy
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
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
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from wica.command import Command
from wica.config import AgentConfig, resolve_api_key, resolve_system_prompt
from wica.content import Content, ImagePart, TextPart
from wica.events import Event
from wica.instrumentation import (
    CommandTrace,
    ReactionOutcome,
    ReactionTrace,
    TokenUsage,
    TriggerTrace,
    now,
    parent_trigger,
    tracer,
)
from wica.world import World, WorldEntry, validate_key

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
    # call_id is the WICA-owned id of this issuance: the agent:command:<call_id> World-entry key
    # suffix and the cancel_command target. For a dispatched Command the Event carrying this
    # payload fires once that entry is registered (before its running value is written), so a
    # subscriber can add_listener on it right away; noop has a call_id but no entry.
    name: str
    args: dict[str, Any]
    call_id: str


# History records — Observation captures the full include_in_prompt bundle at trigger
# time (not just the entry that fired), so passive include_in_prompt entries still reach
# the model. See specs/agent.md "History record shape".
@dataclass(frozen=True)
class ObservedEntry:
    """One entry of an Observation: the WorldEntry snapshot plus the serializers that governed it
    when observed. History renders through these, never through the World's live registration, so
    an observation renders identically after the key is unregistered or re-registered. See
    specs/agent.md ("History record shape")."""

    entry: WorldEntry
    serialize_fn: Callable[[Any, Any], Content]
    archival_serialize_fn: Callable[[Any, Any], Content]


@dataclass(frozen=True)
class ObservationRecord:
    entries: list[ObservedEntry]


@dataclass(frozen=True)
class AssistantTextRecord:
    text: str


@dataclass(frozen=True)
class CommandRecord:
    # call_id is the World-key suffix (agent:command:<call_id>) and the cancel_command target;
    # tool_call_id is the provider's id, used to reconstruct the native tool_call/tool_result pair.
    call_id: str
    tool_call_id: str
    name: str
    args: dict[str, Any]


# The model's explicit "I chose not to act" (a noop call). A dedicated record, not a CommandRecord:
# noop is not a World action, so it has no name/args to record and never becomes an Observation
# outcome. It exists to keep the provider message stream valid (a native tool_call needs a matching
# tool_result) and to show the declined reaction in context. See specs/agent.md, specs/commands.md.
@dataclass(frozen=True)
class NoReactionRecord:
    call_id: str
    tool_call_id: str


HistoryRecord = (
    ObservationRecord | AssistantTextRecord | CommandRecord | NoReactionRecord
)


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


@dataclass
class _PendingTrigger:
    """A trigger waiting in the coalescing window, with what the reaction record needs. See
    specs/instrumentation.md ("Layer 1", "Layer 2")."""

    entry: WorldEntry
    arrived_at: datetime
    context: contextvars.Context  # the writer's context, captured in _handle_trigger

    def trace(self) -> TriggerTrace:
        return TriggerTrace(
            key=self.entry.key,
            version_id=self.entry.current.id,
            written_at=self.entry.current.timestamp,
            arrived_at=self.arrived_at,
            is_command_completion=self.entry.key.startswith(_COMMAND_KEY_PREFIX),
        )


@dataclass
class _ReactionBuilder:
    """Mutable per-reaction scratch filled in as the phases pass; frozen into a ReactionTrace at
    the end (see _run_batch). See specs/instrumentation.md ("Layer 1")."""

    reaction_id: int
    triggers: tuple[TriggerTrace, ...]
    window_opened_at: datetime
    window_closed_at: datetime
    span: otel_trace.Span
    prompt_ready_at: datetime | None = None
    model_started_at: datetime | None = None
    model_ended_at: datetime | None = None
    outcome: ReactionOutcome = (
        "cancelled"  # overwritten by _run_step; stays "cancelled" if it
    )
    # never finishes (e.g. stop() cancels the reaction task before _run_step sets an outcome)
    error: str | None = None
    text_length: int = 0
    sink_duration: float | None = None
    command_call_ids: list[str] = field(default_factory=list)
    noop: bool = False
    usage: TokenUsage | None = None


def _span_ids(span: otel_trace.Span) -> tuple[str | None, str | None]:
    """The OpenTelemetry trace/span ids of a span, formatted as hex strings for a ReactionTrace —
    or (None, None) when no SDK is installed (a no-op span has an invalid context)."""
    sc = span.get_span_context()
    if not sc.is_valid:
        return None, None
    return otel_trace.format_trace_id(sc.trace_id), otel_trace.format_span_id(
        sc.span_id
    )


def _token_usage(response: AIMessage) -> TokenUsage | None:
    """TokenUsage from an AIMessage's usage_metadata, or None when the provider didn't report it.
    See specs/instrumentation.md ("Layer 1")."""
    usage = response.usage_metadata
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cache_read_tokens=usage.get("input_token_details", {}).get("cache_read"),
    )


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


# noop is a WICA-native, zero-arg control Command the model calls to declare *no reaction* — a
# reliable stand-in for "return empty text" (which models do poorly). It is the one Command that is
# not an action on the World: the loop intercepts it before dispatch, records a NoReactionRecord,
# and ends the step (no entry, no task, never triggers). See specs/commands.md ("noop").
_NOOP_COMMAND_NAME = "noop"
_NOOP_COMMAND_DESCRIPTION = (
    "Take no action this step. Call this when the current observation needs no response from you — "
    "you have nothing to say and no command to issue. This is how you explicitly choose to do "
    "nothing; do not try to reply with empty text."
)
# The tool_result paired with a rendered noop call — a plain acknowledgement, not the entry-pointer
# ack real Commands use (noop has no World entry to point at). See specs/agent.md ("History").
_NOOP_ACK = "Acknowledged — no action taken."

# The two Command names WICA owns. register_command rejects both (and any already-registered name);
# an output Command may not take either. See specs/commands.md ("Names are unique and two are
# reserved").
_RESERVED_COMMAND_NAMES = frozenset({_CANCEL_COMMAND_NAME, _NOOP_COMMAND_NAME})

# The *default* re-trigger flag for the output Command — applied when set_output_command is handed
# a bare callable (an explicit Command keeps its own triggers_on_completion). True lets the model
# chain utterances / self-continue (ended by noop), at ~2 LLM calls per utterance; False makes
# speaking go straight to idle. Flipping the framework-wide default is deliberately a one-line
# change. See specs/commands.md ("The output Command").
_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION = True

# The WICA runtime primer appended to the configured persona (see specs/agent.md, "System prompt
# composition"). Curated: only what changes how the model interprets the prompt or chooses actions —
# never runtime plumbing (coalescing, rendering, TTLs, the loop) it can't act on. Composed as:
# perception + acting (including that a response's tool calls run concurrently, so sequencing is
# the model's job)  →  an output-mode clause (default text OR the output Command)  →  the noop
# clause. The output-mode clause is conditional because *how you reply* differs by configuration:
# with no output Command, free text is the reply; with one, free text is private and the Command
# speaks. Getting this explicit matters — without a "how to reply" line, some models (gpt-4o
# observed) treat a plain request as an ignorable observation and noop it.
_RUNTIME_PRIMER = (
    "\n\n---\n"
    "How you operate (WICA runtime):\n"
    "- The messages you receive in the user role are observations of your World — your environment: "
    "inputs, sensor readings, or the status of actions you took. Respond to whatever is addressed "
    "to you or calls for your attention.\n"
    "- To act on the World, call a tool: a tool call is a command that runs asynchronously, and its "
    "immediate result only confirms it was dispatched. The actual outcome appears later as a World "
    "observation (the command's entry turning from running to complete or failed), not in that "
    "acknowledgement.\n"
    "- Several tool calls in one response all start at once and run concurrently, with no order "
    "guaranteed between them. When one action must finish before another starts, do not issue "
    "both together: issue the first, then issue the next in a later step, once you observe the "
    "first complete."
)
# Appended when there is *no* output Command: free text is the reply channel. Without this some
# models default to a tool (noop) instead of answering. See specs/agent.md ("System prompt
# composition").
_DEFAULT_OUTPUT_PROMPT = "\n- To reply to the user, write your answer as ordinary text — that is how you speak."
# Appended instead when an output Command is configured: the free-text-is-private contract the model
# can't infer from a tool schema. See specs/agent.md ("Output").
_OUTPUT_COMMAND_PROMPT = (
    "\n- Anything you want to communicate to the user must be said by calling {name}. Your "
    "free-text responses are private reasoning and are not shown to the user."
)
# Always appended last — references the reply mechanism established just above, and is deliberately
# conservative so it never suppresses an expected response. See specs/commands.md ("noop").
_NOOP_PROMPT = (
    "\n- You need not act on every observation. If one genuinely calls for nothing from you — "
    f"passive background state, say — call {_NOOP_COMMAND_NAME} to do nothing. But a request, a "
    f"question, or anything addressed to you should be answered; never use {_NOOP_COMMAND_NAME} to "
    "skip an expected response."
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
        # system prompt is always config-expressed. See specs/agent.md, specs/wica.md (open q. 2).
        #
        # Neither the output sink nor the optional output Command is a constructor argument: both
        # are application callables wired *after* construction through set_output_sink /
        # set_output_command, so the objects they belong to (a transcript, a UI, a TTS engine) can
        # be built against this Agent's World and Events first. Until wired: no output Command, and
        # a no-op sink. See specs/agent.md ("Output wiring").
        self._output_command: Command | None = None
        self._output_command_name: str | None = None
        # The two WICA-native control Commands, built once here so start() can re-attach the *same*
        # objects on every cycle (the identity check in _register_builtin then holds across a
        # restart). Building a Command wraps the callable into a tool; it does not touch the model,
        # so __init__ stays inert. See specs/commands.md ("Names are unique and two are reserved").
        self._cancel_command = Command(
            self._cancel_command_action,
            name=_CANCEL_COMMAND_NAME,
            description=_CANCEL_COMMAND_DESCRIPTION,
        )
        self._noop_command = Command(
            self._noop_action,
            name=_NOOP_COMMAND_NAME,
            description=_NOOP_COMMAND_DESCRIPTION,
        )
        # The system prompt the model receives is composed: the resolved persona (verbatim) followed
        # by the WICA runtime primer (+ the output clause when an output Command is set). The persona
        # is resolved once here (reading the prompt file if that's the form given) and kept, so
        # set_output_command can recompose the prompt — the only thing that changes its content —
        # without touching the file again. In the intended flow (wire, then start) the combined
        # string is fixed before the first step and stays in the cached deep prefix. See
        # specs/agent.md ("System prompt composition").
        self._persona = resolve_system_prompt(config)
        self.system_prompt = self._compose_system_prompt()
        self.model = model if model is not None else build_chat_model(config)
        # Read for the wica.agent.model span's GenAI attributes (see specs/instrumentation.md).
        self._provider = config.provider
        self._model_name = config.model
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
        self._output_sink: Callable[[str], Awaitable[None]] = _noop_output_sink
        # Instrumentation Events — observability only, never control flow. The Agent emits on them
        # at the right points inside the loop; any number of consumers subscribe, and Event.emit's
        # per-subscriber isolation catches+logs a raising subscriber so it can neither abort a step
        # nor starve siblings (no _fire_hook guard needed). See specs/agent.md "Instrumentation".
        #  - on_trigger(entry):    once per trigger a run-to-completion step observes (filtered).
        #  - on_prompt(messages):  the exact rendered messages just before each model call.
        #  - on_command(command):  each Command the model issues, when issued — including noop
        #                           (which is observable but not a World action; filter by name).
        #  - on_text(text):        the step's complete free text, right before the sink receives
        #                           it (observation; the sink is delivery).
        #  - on_reaction_ended(trace):  once at the end of every reaction that started, whatever
        #                           its outcome — see specs/instrumentation.md ("Layer 1").
        #  - on_trigger_dropped(entry): once per trigger dropped because a reaction was in flight.
        #  - on_command_ended(trace):   once per dispatched Command when its terminal state lands.
        self.on_trigger: Event[WorldEntry] = Event()
        self.on_prompt: Event[list[BaseMessage]] = Event()
        self.on_command: Event[CommandIssued] = Event()
        self.on_text: Event[str] = Event()
        self.on_reaction_ended: Event[ReactionTrace] = Event()
        self.on_trigger_dropped: Event[WorldEntry] = Event()
        self.on_command_ended: Event[CommandTrace] = Event()

        # name -> the registered Command (which holds the backing LangChain tool — see commands.md)
        self._commands: dict[str, Command] = {}
        self._bound_model: Runnable[Any, AIMessage] = self.model
        self._history: list[HistoryRecord] = []
        self._busy = False
        # Coalescing-window state, touched only on the loop thread (like _running_tasks): the
        # triggers batched into the currently-open window (with the reaction-record fields each
        # one needs), when the window opened, and the timer that will flush them.
        self._window_batch: list[_PendingTrigger] = []
        self._window_opened_at: datetime | None = None
        self._window_timer: asyncio.TimerHandle | None = None
        self._reaction_counter = 0
        # The reaction currently running _run_step, if any — read by _dispatch_command to stamp a
        # Command's reaction_id. None outside of _run_batch (loop thread only, so race-free).
        self._current_reaction: _ReactionBuilder | None = None
        # call_id -> (the running version's timestamp, the reaction that issued it), consumed by
        # _emit_command_ended when the terminal write lands. See specs/instrumentation.md.
        self._command_started: dict[str, tuple[datetime, int]] = {}
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

    @property
    def commands(self) -> Mapping[str, Command]:
        """Read-only view of the Commands the model is currently bound to, by name: application
        Commands plus, once start() has attached them, ``noop``/``cancel_command`` and the output
        Command. A consumer wanting only application actions filters those out by name (see
        ``output_command_name``). Live view, not a snapshot. See specs/agent.md ("Commands")."""
        return MappingProxyType(self._commands)

    @property
    def output_command_name(self) -> str | None:
        """The output Command's name, or None when no output Command is set."""
        return self._output_command_name

    def _compose_system_prompt(self) -> str:
        """The persona followed by the WICA runtime primer: perception + acting, then the
        output-mode clause (how you reply differs by whether an output Command is set), then the
        noop clause last. See specs/agent.md ("System prompt composition")."""
        primer = _RUNTIME_PRIMER
        if self._output_command_name is not None:
            primer += _OUTPUT_COMMAND_PROMPT.format(name=self._output_command_name)
        else:
            primer += _DEFAULT_OUTPUT_PROMPT
        primer += _NOOP_PROMPT
        return self._persona + primer

    def set_output_sink(self, sink: Callable[[str], Awaitable[None]] | None) -> None:
        """Set (or, with None, clear) the async sink that receives the model's complete free text
        each step. A single replaceable slot; read once per step at the delivery point, so a change
        takes effect at the next step. See specs/agent.md ("Output wiring")."""
        self._output_sink = sink if sink is not None else _noop_output_sink

    def set_output_command(self, fn: Callable[..., Any] | Command | None) -> None:
        """Set (or, with None, clear) the application's output Command — the user-facing output
        channel. When set, free text becomes the agent's private reasoning stream and this Command
        is how it speaks (see specs/commands.md "The output Command", specs/agent.md "Output").

        A callable is wrapped as ``Command(fn)``; a ``Command`` is used directly. Validation happens
        before any state changes: a reserved name (``noop``/``cancel_command``) or a name already
        registered as an application Command raises ``ValueError``. Then the previous output Command
        (if attached) is detached, the system prompt is recomposed so its output clause names the
        new one, and the new Command is attached immediately if the Agent is running — otherwise
        ``start()`` attaches it. An execution of a replaced Command that is still running finishes
        normally. See specs/agent.md ("Output wiring")."""
        command: Command | None
        if fn is None:
            command = None
        else:
            command = (
                fn
                if isinstance(fn, Command)
                else Command(
                    fn, triggers_on_completion=_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION
                )
            )
            if command.name in _RESERVED_COMMAND_NAMES:
                raise ValueError(
                    f"output command may not be named {command.name!r}: "
                    f"{sorted(_RESERVED_COMMAND_NAMES)} are reserved by WICA"
                )
            if (
                command.name in self._commands
                and command.name != self._output_command_name
            ):
                raise ValueError(f"command name {command.name!r} is already registered")
        previous = self._output_command
        if previous is not None and self._commands.get(previous.name) is previous:
            self._detach_command(previous.name)
        self._output_command = command
        self._output_command_name = None if command is None else command.name
        self.system_prompt = self._compose_system_prompt()
        if command is not None and self._started:
            self._register_builtin(command)
        _logger.debug("output command set to %r", self._output_command_name)

    def register_command(self, fn: Callable[..., Any] | Command) -> None:
        """Register a Command. Takes one argument: a plain callable (auto-wrapped — name from
        ``__name__``, description from the docstring) or a ``Command`` (used directly). To override
        name/description, or to wrap an off-the-shelf tool, pass ``Command(fn, name=…, …)`` /
        ``Command(tool)`` — there are no name/description kwargs here.

        Names are unique: registering a name already registered, a WICA-reserved name (``noop``,
        ``cancel_command``), or the output Command's name raises ``ValueError``. See
        specs/commands.md ("Names are unique and two are reserved")."""
        command = fn if isinstance(fn, Command) else Command(fn)
        if command.name in _RESERVED_COMMAND_NAMES:
            raise ValueError(f"command name {command.name!r} is reserved by WICA")
        if command.name == self._output_command_name:
            raise ValueError(
                f"command name {command.name!r} is the output Command's name"
            )
        if command.name in self._commands:
            raise ValueError(f"command {command.name!r} is already registered")
        self._attach_command(command)

    def _attach_command(self, command: Command) -> None:
        """Store the Command and rebind the model to the full current tool set."""
        self._commands[command.name] = command
        self._rebind_model()
        _logger.debug("registered command %r", command.name)

    def _detach_command(self, name: str) -> None:
        """Remove a Command from the bound set (used when an output Command is replaced or cleared)
        and rebind. A running execution of it is unaffected: it resolved its Command at dispatch."""
        del self._commands[name]
        self._rebind_model()
        _logger.debug("detached command %r", name)

    def _rebind_model(self) -> None:
        self._bound_model = (
            self.model.bind_tools([c.tool for c in self._commands.values()])
            if self._commands
            else self.model
        )

    def _register_builtin(self, command: Command) -> None:
        """Idempotent attach for WICA-native Commands and the output Command (used by start(), which
        may run again after stop()). Their names are reserved from application use, so this can
        never replace an application Command."""
        if self._commands.get(command.name) is command:
            return
        self._attach_command(command)

    def start(self) -> None:
        if self._started:
            return
        # WICA-native control Commands, auto-attached here (no app wiring) since they need Agent
        # internals. Attaching at start (not construction) keeps __init__ inert — it never touches
        # the model. _register_builtin is idempotent on object identity (the Command objects are
        # built once in __init__), so a second start() re-attaches the same ones harmlessly and
        # never replaces an application Command (their names are reserved from register_command).
        #  - cancel_command: abort a still-running Command previously issued (also reaches an output
        #    Command, enabling barge-in).
        #  - noop: declare no reaction (intercepted before dispatch — see _run_step).
        self._register_builtin(self._cancel_command)
        self._register_builtin(self._noop_command)
        # The optional application-supplied output Command (set via set_output_command, usually
        # before this start). An ordinary Command in every respect but its trigger-on-completion flag
        # (see _dispatch_command) and the prompt clause (see _compose_system_prompt).
        if self._output_command is not None:
            self._register_builtin(self._output_command)
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
        """Create an Agent-owned loop task and forget it only after it has settled. Every owned
        task also gets a logging done-callback: an exception that escapes a task is otherwise only
        reported by asyncio at garbage-collection time ("Task exception was never retrieved"),
        never under a wica.* logger. See specs/agent.md ("Instrumentation")."""
        task = self._loop.create_task(coroutine)
        self._owned_tasks.add(task)
        task.add_done_callback(self._owned_tasks.discard)
        task.add_done_callback(self._log_task_failure)
        return task

    def _log_task_failure(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        # Retrieving the exception also marks it handled, silencing asyncio's GC-time warning.
        exc = task.exception()
        if exc is not None:
            _logger.error("agent task %s raised", task.get_name(), exc_info=exc)

    def _cancel_window(self) -> None:
        """Cancel a pending coalescing-window timer and drop its batch (loop thread only)."""
        if self._window_timer is not None:
            self._window_timer.cancel()
            self._window_timer = None
        self._window_batch.clear()
        self._window_opened_at = None

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

    async def _noop_action(self) -> str:
        """Backing for the `noop` Command. Never actually invoked — `_run_step` intercepts a noop
        call before dispatch and records a NoReactionRecord — but a real tool is needed so the model
        can be bound to it and issue the call. Zero-arg (the model just names it)."""
        return _NOOP_ACK

    async def _handle_trigger(self, entry: WorldEntry) -> None:
        # Runs on the loop thread (the World emits on_trigger there, and the start() shim
        # create_tasks this coroutine on it). All window state below is therefore touched
        # single-threaded, so open/join/flush are race-free — the same invariant that makes
        # cancellation race-free (see specs/agent.md "The shared event loop"). The task itself runs
        # in the writer's context (the World scheduled on_trigger's emission with it, and
        # create_task inherits the current context), so capturing it here for the window batch is
        # all that's needed for propagation (see specs/instrumentation.md "Layer 2").
        arrived_at = now()
        _logger.debug("trigger received: %s", _describe_entry(entry))
        if self._busy:
            _logger.info(
                "dropping trigger (%s) — a call is already in flight",
                _describe_entry(entry),
            )
            self.on_trigger_dropped.emit(entry)
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

        pending = _PendingTrigger(entry, arrived_at, contextvars.copy_context())
        self._window_batch.append(pending)
        if len(self._window_batch) == 1:
            self._window_opened_at = arrived_at
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
        """Close the coalescing window, build this reaction's record and span, and start the single
        step for the batched triggers. Sync, runs on the loop thread (called directly for an
        immediate flush, or by the window timer). See specs/instrumentation.md ("Layer 1", "Layer
        2", "Which trigger is the reaction's parent")."""
        if self._window_timer is not None:
            self._window_timer.cancel()
            self._window_timer = None
        if not self._window_batch:
            return
        batch = self._window_batch
        self._window_batch = []

        window_closed_at = now()
        window_opened_at = self._window_opened_at or window_closed_at
        self._window_opened_at = None
        self._reaction_counter += 1
        traces = tuple(pending.trace() for pending in batch)
        parent = parent_trigger(traces)
        parent_pending = batch[traces.index(parent)]
        # The reaction span's parent is the context captured with the parent trigger (the trigger
        # a coalesced batch is attributed to — see parent_trigger); every other trigger in the
        # batch becomes a span link instead, so the trace still records it without splitting the
        # reaction's parentage.
        parent_ctx = parent_pending.context.run(otel_context.get_current)
        links = []
        for pending in batch:
            if pending is parent_pending:
                continue
            sc = pending.context.run(otel_trace.get_current_span).get_span_context()
            if sc.is_valid:
                links.append(otel_trace.Link(sc))
        span = tracer().start_span(
            "wica.agent.reaction",
            context=parent_ctx,
            links=links,
            attributes={
                "wica.reaction_id": self._reaction_counter,
                "wica.trigger_count": len(batch),
            },
        )
        builder = _ReactionBuilder(
            reaction_id=self._reaction_counter,
            triggers=traces,
            window_opened_at=window_opened_at,
            window_closed_at=window_closed_at,
            span=span,
        )
        # Set _busy before the step task runs so triggers arriving in the gap are dropped, not
        # folded into a second concurrent step (single-in-flight).
        self._busy = True
        self._track_task(self._run_batch([pending.entry for pending in batch], builder))

    async def _run_batch(
        self, batch: list[WorldEntry], builder: _ReactionBuilder
    ) -> None:
        self._current_reaction = builder
        try:
            with otel_trace.use_span(builder.span, end_on_exit=False):
                await self._run_step(batch, builder)
        finally:
            self._busy = False
            self._current_reaction = None
            ended_at = now()
            builder.span.set_attributes(
                {
                    "wica.outcome": builder.outcome,
                    "wica.text_length": builder.text_length,
                    "wica.command_count": len(builder.command_call_ids),
                    "wica.noop": builder.noop,
                }
            )
            builder.span.end()
            trace_id, span_id = _span_ids(builder.span)
            # Emitted after _busy is cleared, so a subscriber sees "the Agent is free again" and
            # the record at the same moment. Fires for every reaction that started, whatever its
            # outcome (ok/empty/model_error/cancelled) — see specs/instrumentation.md ("Layer 1").
            self.on_reaction_ended.emit(
                ReactionTrace(
                    reaction_id=builder.reaction_id,
                    triggers=builder.triggers,
                    window_opened_at=builder.window_opened_at,
                    window_closed_at=builder.window_closed_at,
                    prompt_ready_at=builder.prompt_ready_at,
                    model_started_at=builder.model_started_at,
                    model_ended_at=builder.model_ended_at,
                    outcome=builder.outcome,
                    error=builder.error,
                    text_length=builder.text_length,
                    sink_duration=builder.sink_duration,
                    command_call_ids=tuple(builder.command_call_ids),
                    noop=builder.noop,
                    usage=builder.usage,
                    ended_at=ended_at,
                    trace_id=trace_id,
                    span_id=span_id,
                )
            )

    async def _run_step(
        self, batch: list[WorldEntry], builder: _ReactionBuilder
    ) -> None:
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
        builder.prompt_ready_at = now()

        builder.model_started_at = now()
        try:
            with tracer().start_as_current_span(
                "wica.agent.model",
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": self._provider,
                    "gen_ai.request.model": self._model_name,
                },
            ) as model_span:
                response = await self._bound_model.ainvoke(messages)
                builder.usage = _token_usage(response)
                if builder.usage is not None:
                    model_span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": builder.usage.input_tokens,
                            "gen_ai.usage.output_tokens": builder.usage.output_tokens,
                        }
                    )
        except Exception as exc:
            # The observation is already in history (and terminal command entries are retired),
            # so the model never saw it this step; the renderer merges it into the next
            # observation's user message (see _render_messages). Nothing is dispatched. Ending the
            # step here frees the single-in-flight loop for the next trigger. CancelledError is a
            # BaseException and keeps propagating. See specs/agent.md ("Instrumentation").
            builder.model_ended_at = now()
            builder.outcome = "model_error"
            builder.error = str(exc)
            _logger.exception(
                "model call failed; step abandoned (trigger: %s)",
                _describe_entry(representative),
            )
            return
        builder.model_ended_at = now()

        text = response.text
        _logger.debug(
            "LLM output: text=%s, commands=%s",
            _truncate(text) if text else "(none)",
            [c["name"] for c in response.tool_calls] or "(none)",
        )
        builder.text_length = len(text) if text else 0
        builder.outcome = "empty" if not text and not response.tool_calls else "ok"
        if text:
            self._history.append(AssistantTextRecord(text))
            # Observation first, delivery second: the Event carries the same string the sink is
            # about to receive, so a watcher never has to occupy the single sink slot. A raising
            # sink below does not affect the emission. See specs/agent.md ("Instrumentation").
            self.on_text.emit(text)
            sink_started = now()
            with tracer().start_as_current_span("wica.agent.sink"):
                try:
                    await self._output_sink(text)
                except Exception:
                    # The sink is application code; its failure must not drop the Commands the
                    # model issued in the same response, nor escape the step. The text stays in
                    # history — the model did say it. See specs/agent.md ("Instrumentation").
                    _logger.exception(
                        "output sink raised; continuing with the step's commands"
                    )
            builder.sink_duration = (now() - sink_started).total_seconds()

        for call in response.tool_calls:
            # The provider's tool-call id reconstructs the native tool_call/tool_result pair; the
            # World-key suffix is that id when it's a safe, free key, else a generated one.
            tool_call_id = call["id"] or uuid.uuid4().hex
            call_id = self._world_call_id(tool_call_id)
            if call["name"] == _NOOP_COMMAND_NAME:
                # noop is the model declaring no reaction: record it (so context shows the choice
                # and the tool_call has a matching tool_result), but do not dispatch it — no World
                # entry, no task, no trigger; it is not a World *action*. It is still a command the
                # model issued, so on_command fires for it like every other (a consumer that wants
                # only real actions filters it out by name). See specs/commands.md, specs/agent.md.
                _logger.debug("noop issued (call_id=%s) — no action taken", call_id)
                builder.noop = True
                self.on_command.emit(CommandIssued(_NOOP_COMMAND_NAME, {}, call_id))
                self._history.append(NoReactionRecord(call_id, tool_call_id))
                continue
            args = copy.deepcopy(call["args"])
            self._history.append(
                CommandRecord(call_id, tool_call_id, call["name"], copy.deepcopy(args))
            )
            self._dispatch_command(call_id, call["name"], args)
            builder.command_call_ids.append(call_id)
        _logger.debug("step complete (trigger: %s)", _describe_entry(representative))

    def _world_call_id(self, provider_id: str | None) -> str:
        """The `<call_id>` for a new agent:command:<call_id> entry: the provider's tool-call id
        when it is a valid World key and that key is free, else a generated one. See
        specs/commands.md ("Command execution as a World entry")."""
        if provider_id:
            try:
                validate_key(provider_id)
            except ValueError:
                pass
            else:
                if not self._world.is_registered(f"{_COMMAND_KEY_PREFIX}{provider_id}"):
                    return provider_id
        return uuid.uuid4().hex

    def _dispatch_command(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        _logger.debug(
            "dispatching command %s(%s) call_id=%s", name, _format_args(args), call_id
        )
        key = f"{_COMMAND_KEY_PREFIX}{call_id}"
        self._command_keys.add(key)
        # Every Command re-triggers on completion by default (the uniform model). One registered
        # with triggers_on_completion=False (the output Command's default comes from the module
        # constant) completes without waking a step; its terminal entry is then observed and
        # retired by the next step that runs for any other reason. include_in_prompt stays True
        # either way, so a concurrent step can observe running work and cancel it (barge-in). A
        # name matching no registered Command registers with True and then fails in _run_command.
        # See specs/commands.md ("The `Command` object", "The output Command").
        registered = self._commands.get(name)
        triggers_llm_call = (
            registered.triggers_on_completion if registered is not None else True
        )
        self._world.register(
            key,
            CommandExecution,
            serialize_fn=_serialize_command_execution,
            include_in_prompt=True,
            triggers_llm_call=triggers_llm_call,
            trigger_condition_fn=lambda old, new: new is not None and new.is_terminal(),
        )
        # Emit once the entry exists and before its first value lands, so a subscriber can
        # add_listener(key) here and see running -> terminal. A subscriber may annotate or
        # otherwise mutate what it receives without changing the arguments stored in history,
        # shown in the World, or passed to the Command itself. See specs/agent.md.
        self.on_command.emit(CommandIssued(name, copy.deepcopy(args), call_id))
        self._world.update(key, CommandExecution(name=name, args=args, state="running"))
        started_at = self._world.get_entry(key).current.timestamp
        reaction_id = (
            self._current_reaction.reaction_id if self._current_reaction else 0
        )
        self._command_started[key] = (started_at, reaction_id)
        # No context= here: start_span parents to the current span, which is the reaction span
        # (this call happens inside _run_step, which _run_batch wraps in use_span). See
        # specs/instrumentation.md ("Layer 2").
        span = tracer().start_span(
            "wica.agent.command",
            attributes={"wica.command.name": name, "wica.call_id": call_id},
        )
        task = self._track_task(self._run_command(key, call_id, name, args, span))
        self._running_tasks[key] = task

    async def _run_command(
        self,
        key: str,
        call_id: str,
        name: str,
        args: dict[str, Any],
        span: otel_trace.Span,
    ) -> None:
        _logger.debug(
            "command %s(%s) [call_id=%s] executing", name, _format_args(args), call_id
        )
        # Current for the whole Command body (dispatch to terminal write), so any span an
        # application or third-party library opens inside it nests under this one with no
        # WICA-specific API, and the terminal world.update() below carries it — the completion
        # trigger's follow-up reaction then descends from the Command. See
        # specs/instrumentation.md ("Layer 2").
        with otel_trace.use_span(span, end_on_exit=True):
            try:
                command = self._commands.get(name)
                if command is None:
                    # The model named a tool the Agent never bound (or one unregistered since).
                    # Fail the entry with a readable error rather than the bare KeyError repr
                    # ("'ghost'").
                    raise LookupError(f"unknown command {name!r}")
                result = await command.tool.ainvoke(args)
            except asyncio.CancelledError:
                _logger.debug("command %s [call_id=%s] cancelled", name, call_id)
                span.set_attribute("wica.command.state", "cancelled")
                if self._write_terminal(
                    key,
                    call_id,
                    CommandExecution(name=name, args=args, state="cancelled"),
                ):
                    self._emit_command_ended(key, call_id, name, "cancelled")
                raise
            except Exception as exc:
                _logger.warning(
                    "command %s [call_id=%s] failed: %s", name, call_id, exc
                )
                span.set_attribute("wica.command.state", "failed")
                if self._write_terminal(
                    key,
                    call_id,
                    CommandExecution(
                        name=name, args=args, state="failed", error=str(exc)
                    ),
                ):
                    self._emit_command_ended(key, call_id, name, "failed")
            else:
                _logger.debug(
                    "command %s [call_id=%s] complete → %s",
                    name,
                    call_id,
                    _truncate(str(result)),
                )
                span.set_attribute("wica.command.state", "complete")
                if self._write_terminal(
                    key,
                    call_id,
                    CommandExecution(
                        name=name, args=args, state="complete", result=str(result)
                    ),
                ):
                    self._emit_command_ended(key, call_id, name, "complete")
            finally:
                self._running_tasks.pop(key, None)

    def _write_terminal(
        self, key: str, call_id: str, execution: CommandExecution
    ) -> bool:
        """Write a Command's terminal state to its World entry, tolerating a World that has
        already stopped. That happens during Wica teardown (agent.stop() cancels in-flight
        commands just before world.stop()) and for a Command that finishes after the World paused
        (an injected-loop shutdown, or one that swallows its cancellation). The write is moot then:
        log and drop it rather than raise into the task. Returns whether the write happened — the
        caller uses this to decide whether to emit on_command_ended: if the World is stopped, the
        entry's current version is still `running`, so there's no terminal timestamp to report.
        See specs/wica.md ("Lifecycle")."""
        try:
            self._world.update(key, execution)
        except RuntimeError:
            _logger.debug(
                "command %s [call_id=%s] reached %s after the World stopped; "
                "terminal state not recorded",
                execution.name,
                call_id,
                execution.state,
            )
            return False
        return True

    def _emit_command_ended(
        self,
        key: str,
        call_id: str,
        name: str,
        state: Literal["complete", "failed", "cancelled"],
    ) -> None:
        started = self._command_started.pop(key, None)
        if started is None:
            return
        try:
            ended_at = self._world.get_entry(key).current.timestamp
        except KeyError:
            return  # entry already gone (World stopped and the write was dropped)
        started_at, reaction_id = started
        self.on_command_ended.emit(
            CommandTrace(
                call_id=call_id,
                name=name,
                reaction_id=reaction_id,
                started_at=started_at,
                ended_at=ended_at,
                state=state,
            )
        )

    def _append_observation(self) -> None:
        snapshot = self._world.get_prompt_snapshot()
        observed = [
            ObservedEntry(
                entry=entry,
                serialize_fn=config.serialize_fn,
                archival_serialize_fn=config.archival_serialize_fn
                or config.serialize_fn,
            )
            for entry, config in snapshot
        ]
        self._history.append(ObservationRecord(observed))
        # This snapshot has now captured every terminal command entry's outcome (they render from
        # it in _render_messages), so retire them all here — not just the one that fired. This is
        # the *sole* retirement path: a completion whose own trigger was dropped by the
        # single-in-flight loop stays in the World until some step observes it here, so its outcome
        # always reaches history before the entry goes away (never silently lost). The captured
        # snapshot keeps re-rendering from history via its own captured serializer after unregister.
        for item in observed:
            if self._is_terminal_command(item.entry):
                _logger.debug("retiring completed command entry %r", item.entry.key)
                self._cleanup_command_entry(item.entry.key)

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
        # call_id -> the tool_result content for that call. A Command's is the fixed entry-pointer
        # ack (_command_ack); a noop's is the plain _NOOP_ACK — selected by record type below.
        pending_acks: dict[str, str] = {}

        def flush_assistant() -> None:
            if not pending_text and not pending_calls:
                return
            messages.append(
                AIMessage(
                    content="\n".join(pending_text), tool_calls=list(pending_calls)
                )
            )
            # A Command's tool_result is a fixed ack pointing at its World entry — never the outcome
            # (the outcome is delivered by that entry, rendered as an observation at the step where
            # the completion is observed). A noop's is a plain acknowledgement — it has no entry.
            for call in pending_calls:
                messages.append(
                    ToolMessage(
                        content=pending_acks[call["id"]], tool_call_id=call["id"]
                    )
                )
            pending_text.clear()
            pending_calls.clear()
            pending_acks.clear()

        for i, record in enumerate(self._history):
            if isinstance(record, ObservationRecord):
                flush_assistant()
                archival = i != newest_observation_index
                blocks: list[str | dict[str, Any]] = []
                for item in record.entries:
                    # Every observed entry renders through its own captured serializer — the fresh
                    # or archival one recorded at observation time. render_entry with an explicit
                    # serialize_fn ignores its archival flag and never touches the World's live
                    # config, so this cannot raise KeyError for a key that has since been
                    # unregistered, and a re-registered key never rewrites this older observation.
                    # Command entries are captured with the Agent's own command serializer (they
                    # are registered with it), so they render uniformly too — no key inspection.
                    serialize_fn = (
                        item.archival_serialize_fn if archival else item.serialize_fn
                    )
                    rendered = self._world.render_entry(
                        item.entry, serialize_fn=serialize_fn
                    )
                    blocks.extend(_content_to_message_blocks(rendered))
                previous = messages[-1]
                if isinstance(previous, HumanMessage):
                    # Consecutive observations (a failed/cancelled/empty step in between) merge
                    # into one user message — some providers reject back-to-back user turns, and
                    # the older observation may hold the only record of a retired Command's
                    # outcome. flush_assistant() above appended nothing, or the last message would
                    # be an AI/tool message. See specs/agent.md ("Rendering to messages").
                    merged = (
                        list(previous.content)
                        if isinstance(previous.content, list)
                        else [previous.content]
                    )
                    messages[-1] = HumanMessage(content=[*merged, *blocks])
                else:
                    messages.append(HumanMessage(content=blocks))
            elif isinstance(record, AssistantTextRecord):
                pending_text.append(record.text)
            elif isinstance(record, CommandRecord):
                # The native tool_call/tool_result pair uses the provider's tool_call_id; the ack
                # still points the model at the World entry keyed by call_id (equal in the common
                # case). See specs/commands.md ("Command execution as a World entry").
                pending_calls.append(
                    {
                        "name": record.name,
                        "args": copy.deepcopy(record.args),
                        "id": record.tool_call_id,
                    }
                )
                pending_acks[record.tool_call_id] = _command_ack(record.call_id)
            elif isinstance(record, NoReactionRecord):
                # Render as the model's native noop tool call (keeps the message stream valid) with
                # a plain-ack tool_result — selected here by record type. See specs/agent.md.
                pending_calls.append(
                    {"name": _NOOP_COMMAND_NAME, "args": {}, "id": record.tool_call_id}
                )
                pending_acks[record.tool_call_id] = _NOOP_ACK

        flush_assistant()
        return messages
