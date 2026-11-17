from __future__ import annotations

import asyncio
import base64
import contextvars
import copy
import inspect
import json
import math
import logging
import uuid
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call, tool_call
from langchain_core.runnables import Runnable
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from wica.command import Command
from wica.config import (
    AgentConfig,
    OPENAI_CHAT_COMPLETIONS_COMPATIBLE,
    ConfigError,
    resolve_api_key,
    resolve_system_prompt,
)
from wica.content import Content, ImagePart, TextPart
from wica.events import Event
from wica.guards import (
    DEFAULT_GUARDS,
    REJECTED_TOOL_RESULT,
    REMINDER_FRAME,
    SHOW_FRAME,
    MalformedCommand,
    ModelResponse,
    ProposedCommand,
    ResponseGuard,
    ResponseRejection,
    Retry,
    combine,
)
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
from wica.text_output import (
    NEXT_TAG,
    NOOP_TAG,
    OutputSignature,
    OutputStream,
    ParseResult,
    TextOutputParser,
    is_tag_name,
    output_signature,
)
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
    # call_id is the WICA-owned id of this issuance: the agent:command:<call_id> World-entry key
    # suffix and the cancel_command target. For a dispatched Command the Event carrying this
    # payload fires once that entry is registered (before its running value is written), so a
    # subscriber can add_listener on it right away. An output is reported once its body is
    # complete, when its entry is already running.
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


# The text of a response as the model wrote it — output tags, <noop/> and <next> included.
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


def _serialize_output_execution(
    value: CommandExecution | None, previous: CommandExecution | None
) -> Content:
    """An output's World entry: the tag the model wrote and how far its delivery got — never a
    call (the model is told its output Command is not a tool), and no verb: the output may be a
    voice, a chat bubble or a screen, and its tag name says which. See specs/agent.md ("Output:
    tags in the model's text")."""
    if value is None:
        return [TextPart("(no output)")]
    text = str(value.args.get("text", ""))
    attributes = "".join(f' {k}="{v}"' for k, v in value.args.items() if k != "text")
    head = f"<{value.name}{attributes}>" + (f' "{text}"' if text else "")
    if value.state == "running":
        body = f"{head} — still being delivered, not finished."
    elif value.state == "complete":
        # The output Command's own result, when it returned one ("Done." is the default).
        result = f": {value.result}" if value.result and value.result != "Done." else ""
        body = f"{head} — delivered{result}."
    elif value.state == "failed":
        body = f"{head} — failed: {value.error}"
    elif text:
        body = f"{head} — cut off there, the rest was not delivered."
    else:
        body = f"{head} — cut off before anything was delivered."
    return [TextPart(body)]


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

# The single text block an observation renders to when every one of its entries omitted itself
# (serializer returned []). Providers reject empty user content and the first message after the
# system prompt must be a user turn, so the observation cannot simply be skipped. See
# specs/agent.md ("Rendering to messages", omitted entries).
_EMPTY_OBSERVATION = "(no entries to observe)"
_EMPTY_OBSERVATION_BLOCK: dict[str, Any] = {"type": "text", "text": _EMPTY_OBSERVATION}


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
    first_output_at: datetime | None = None
    command_call_ids: list[str] = field(default_factory=list)
    noop: bool = False
    usage: TokenUsage | None = None  # summed over attempts
    # Response guards (see specs/agent.md, "Response guards"): model calls made so far, the
    # combined feedback of each rejected attempt, the seconds spent in guards, and the first
    # attempt's reported input tokens — the history budget's signal, since a retry's prompt is
    # larger than the history by its correction.
    attempts: int = 0
    rejections: list[str] = field(default_factory=list)
    guard_duration: float | None = None
    first_input_tokens: int | None = None
    # The current attempt's first chunk carrying text or a tool call (reset at each attempt).
    first_content_at: datetime | None = None


def _span_ids(span: otel_trace.Span) -> tuple[str | None, str | None]:
    """The OpenTelemetry trace/span ids of a span, formatted as hex strings for a ReactionTrace —
    or (None, None) when no SDK is installed (a no-op span has an invalid context)."""
    sc = span.get_span_context()
    if not sc.is_valid:
        return None, None
    return otel_trace.format_trace_id(sc.trace_id), otel_trace.format_span_id(
        sc.span_id
    )


_CACHE_BREAKPOINT: dict[str, str] = {"type": "ephemeral"}


def mark_cache_breakpoint(messages: list[BaseMessage]) -> list[BaseMessage]:
    """*messages* with an Anthropic `cache_control` breakpoint on the last block before the newest
    observation (the last user message) — the deepest block the next step renders byte-identically:
    the newest observation is re-rendered archival on the next step, so a breakpoint after it would
    be written and never read. On the first step that block is the system prompt (and the tools
    ahead of it). Returns a new list; the marked message is a copy, the others are shared."""
    newest = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage)
        ),
        None,
    )
    if newest is None or newest == 0:
        return messages
    target = messages[newest - 1]
    content = target.content
    blocks: list[str | dict[str, Any]] = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else [
            dict(b) if isinstance(b, dict) else {"type": "text", "text": b}
            for b in content
        ]
    )
    last = blocks[-1] if blocks else None
    if not isinstance(last, dict) or last.get("text") == "":
        return messages
    blocks[-1] = {**last, "cache_control": dict(_CACHE_BREAKPOINT)}
    marked = target.model_copy(update={"content": blocks})
    return [*messages[: newest - 1], marked, *messages[newest:]]


def _add_usage(total: TokenUsage | None, usage: TokenUsage | None) -> TokenUsage | None:
    """A reaction's usage over its attempts: summed; cache reads summed only when both report
    them."""
    if usage is None:
        return total
    if total is None:
        return usage
    cache_read = (
        None
        if total.cache_read_tokens is None or usage.cache_read_tokens is None
        else total.cache_read_tokens + usage.cache_read_tokens
    )
    return TokenUsage(
        input_tokens=total.input_tokens + usage.input_tokens,
        output_tokens=total.output_tokens + usage.output_tokens,
        cache_read_tokens=cache_read,
    )


def _retry_messages(
    messages: list[BaseMessage], response: AIMessage, feedback: str, *, show: bool
) -> list[BaseMessage]:
    """The prompt of a retry: the step's own messages plus the latest rejection's correction — never
    stored in history. Shown: the rejected response as returned, one fixed tool_result per tool call
    it holds (a tool_use must not be left without its result), then the feedback as a user message.
    A reminder: the feedback appended as a text block to the newest observation's user message (two
    user turns in a row are what the renderer merges for providers that reject them). The messages
    before the newest observation are untouched, so a cache breakpoint there still reads. See
    specs/agent.md ("Response guards")."""
    if show:
        calls = [
            {**call, "id": call["id"] or uuid.uuid4().hex}
            for call in response.tool_calls
        ]
        shown = (
            response
            if all(call["id"] for call in response.tool_calls)
            else response.model_copy(update={"tool_calls": calls})
        )
        return [
            *messages,
            shown,
            *(
                ToolMessage(content=REJECTED_TOOL_RESULT, tool_call_id=call["id"])
                for call in calls
            ),
            HumanMessage(content=SHOW_FRAME.format(feedback=feedback)),
        ]
    reminder: dict[str, Any] = {
        "type": "text",
        "text": REMINDER_FRAME.format(feedback=feedback),
    }
    last = messages[-1] if messages else None
    if isinstance(last, HumanMessage):
        content = (
            list(last.content) if isinstance(last.content, list) else [last.content]
        )
        return [
            *messages[:-1],
            last.model_copy(update={"content": [*content, reminder]}),
        ]
    return [*messages, HumanMessage(content=[reminder])]


def _strict_args(raw: str) -> tuple[dict[str, Any], None] | tuple[None, str]:
    """A streamed tool call's raw arguments, parsed strictly: (args, None), or (None, the error).
    Empty means no arguments; anything else must be one complete JSON object."""
    if not raw.strip():
        return {}, None
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"arguments are not complete JSON ({exc})"
    if not isinstance(args, dict):
        return None, f"arguments are a JSON {type(args).__name__}, not an object"
    return args, None


def _has_content(piece: AIMessage) -> bool:
    """Whether a streamed piece carries part of the answer: text or a tool call."""
    if isinstance(piece, AIMessageChunk):
        return bool(piece.text or piece.tool_call_chunks)
    return bool(piece.text or piece.tool_calls or piece.invalid_tool_calls)


def _model_response(response: AIMessage, bound: Mapping[str, Command]) -> ModelResponse:
    """The neutral view of a response the guards (and on_response_rejected) receive: its own
    copies, so a guard or a subscriber cannot rewrite what gets dispatched."""
    return ModelResponse(
        text=response.text,
        commands=tuple(
            ProposedCommand(call["name"], copy.deepcopy(call["args"]))
            for call in response.tool_calls
        ),
        bound=bound,
        malformed=tuple(
            MalformedCommand(
                name=call["name"],
                raw_args=call["args"] or "",
                error=call["error"] or "the arguments could not be read",
            )
            for call in response.invalid_tool_calls
        ),
        truncated=_was_truncated(response),
        output_tokens=(response.usage_metadata or {}).get("output_tokens"),
    )


# What providers report when they stop a response at its output cap, lower-cased: Anthropic's
# `stop_reason`, and the `finish_reason` of OpenAI, Mistral and the Hub (`length`) and of Gemini
# (`MAX_TOKENS`).
_CAP_STOP_REASONS = frozenset({"max_tokens", "length"})


def _was_truncated(response: AIMessage) -> bool:
    """Whether the provider stopped the response at its output cap, from its stop reason."""
    metadata = response.response_metadata
    reason = metadata.get("stop_reason") or metadata.get("finish_reason")
    return isinstance(reason, str) and reason.lower() in _CAP_STOP_REASONS


def _assemble(summed: AIMessageChunk) -> AIMessage:
    """The summed chunks of a streamed call as one AIMessage, its tool calls parsed strictly.

    LangChain assembles streamed arguments with a lenient partial-JSON parser that closes whatever
    is left open, so a call cut by the output cap would come out valid with truncated arguments.
    Each call's raw arguments are parsed strictly instead; a malformed call goes to
    `invalid_tool_calls`, where a non-streamed call with the same arguments lands, and is logged.
    Content, usage and metadata are kept as summed. See specs/agent.md ("Streaming the model
    call")."""
    calls: list[ToolCall] = []
    invalid: list[InvalidToolCall] = []
    for part in summed.tool_call_chunks:
        name, raw, call_id = part["name"], part["args"] or "", part["id"]
        args, error = _strict_args(raw)
        if args is not None and not name:
            args, error = None, "the call has no name"
        if args is None or not name:
            invalid.append(
                invalid_tool_call(name=name, args=raw, id=call_id, error=error)
            )
            _logger.warning("malformed tool call %r dropped: %s", name, error)
            continue
        calls.append(tool_call(name=name, args=args, id=call_id))
    return AIMessage(
        content=summed.content,
        additional_kwargs=summed.additional_kwargs,
        response_metadata=summed.response_metadata,
        usage_metadata=summed.usage_metadata,
        id=summed.id,
        name=summed.name,
        tool_calls=calls,
        invalid_tool_calls=invalid,
    )


async def _stream_response(
    bound_model: Runnable[Any, AIMessage],
    messages: list[BaseMessage],
    on_first_content: Callable[[], None] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> AIMessage:
    """One model call, streamed: the chunks summed and assembled into one AIMessage. A model with
    no streaming path yields its whole AIMessage once (LangChain's fallback), used as is.
    `on_first_content` is called once, when the first piece carrying text or a tool call arrives
    (a chunk of reasoning or metadata alone does not count). See specs/agent.md ("Streaming the
    model call")."""
    summed: AIMessageChunk | None = None
    whole: AIMessage | None = None
    async for piece in bound_model.astream(messages):
        if on_first_content is not None and _has_content(piece):
            on_first_content()
            on_first_content = None
        if on_text is not None:
            # The text, as it arrives: the output parser delivers an output before the response
            # is complete. See specs/agent.md ("Streaming the model call").
            text = str(piece.text)
            if text:
                on_text(text)
        if isinstance(piece, AIMessageChunk) and whole is None:
            summed = piece if summed is None else summed + piece
        elif isinstance(piece, AIMessage) and summed is None and whole is None:
            whole = piece
        else:
            raise TypeError(
                f"the model streamed an unexpected {type(piece).__name__} after "
                f"{'a whole message' if whole is not None else 'chunks'}"
            )
    if whole is not None:
        return whole
    if summed is None:
        raise ValueError("the model's stream yielded nothing")
    return _assemble(summed)


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


# The names WICA owns: its control Command, and the two tags of the model's text (see
# specs/agent.md, "Output: tags in the model's text"). register_command rejects them (and any
# already-registered name); an output Command may not take any of them. See specs/commands.md
# ("Names are unique and three are reserved").
_RESERVED_COMMAND_NAMES = frozenset({_CANCEL_COMMAND_NAME, NOOP_TAG, NEXT_TAG})

# The World entry holding the model's <next> note for the one step that follows it.
_NEXT_NOTE_KEY = "agent:next"


def _serialize_next_note(value: str | None, previous: str | None) -> Content:
    if not value:
        return []
    return [TextPart(f"You planned to do next: {value}")]


# The WICA runtime primer appended to the configured persona (see specs/agent.md, "System prompt
# composition"). Curated: only what changes how the model interprets the prompt or chooses actions —
# never runtime plumbing (coalescing, rendering, TTLs, the loop) it can't act on. Composed as:
# perception + acting (including that a response's tool calls run concurrently, so sequencing is
# the model's job)  →  the output clause (the output Command's tag, or that nobody reads the
# text)  →  the silence clause  →  the continuation clause. Saying how to reply matters —
# without it, some models (gpt-4o observed) treat a plain request as an ignorable observation.
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
# Appended only when AgentConfig.history_reactions is set: a bounded context changes how the model
# should read its prompt (a truncated history is not the whole conversation; a Command outcome may
# show up for a dispatch it no longer sees). Absent with unbounded history, so the primer stays
# fixed for a given config. See specs/agent.md ("History window", primer clause 2).
_HISTORY_WINDOW_PROMPT = (
    "\n- Your context holds only your most recent reactions — the latest observations and what "
    "you said or did in response — not the whole interaction: older exchanges are gone. The "
    "current observation is the authority on the present state of the World; anything that must "
    "persist comes from the entries it shows, not from memory of earlier turns. An entry may "
    "report the outcome of an action you no longer see yourself issuing."
)
# The output clause, appended when an output Command is set: how the model addresses its user,
# and what makes that unlike a chat reply (it runs in the background; speaking again replaces
# it). See specs/agent.md ("System prompt composition", "Output: tags in the model's text").
_OUTPUT_PROMPT = (
    "\n- To address the user, write inside a {name} tag in your response text, like this:\n"
    "<{name}>Hello there!</{name}>\n"
    "Only the text inside the tag reaches the user; any text outside it is private reasoning "
    "nobody sees. {name} is not a tool: never call it as a tool, write the tag in your text."
    "{description}{attributes}"
    " Delivery is not instant: what you write in a {name} tag is delivered in the background, "
    "like any command. It appears in your World as a command entry (agent:command:<id>), running "
    "until it is fully delivered, then finished. You can be called again while it is still being "
    "delivered. A new {name} tag replaces the one still being delivered: the earlier one is cut "
    f"off. To stop it without writing a new one, call {_CANCEL_COMMAND_NAME} with that entry's "
    "id, as for any other running command."
)
_OUTPUT_ATTRIBUTES_PROMPT = (
    " The {name} tag takes these optional attributes: {attributes}. Example:\n"
    "<{name} {example}>Hello there!</{name}>\n"
)
# Appended instead when no output Command is set.
_NO_OUTPUT_PROMPT = (
    "\n- Nobody reads the text you write: it is private. You act only by calling tools."
)
# Always appended — conservative, so it never suppresses an expected response. See specs/agent.md
# ("The agent may choose not to act").
_SILENCE_PROMPT = (
    "\n- You need not act on every observation. If one genuinely calls for nothing from you — "
    f"passive background state, for example — write <{NOOP_TAG}/> as your whole response to do nothing. "
    f"{NOOP_TAG} is not a tool: write the tag in your text. But a request, a question, or "
    f"anything addressed to you should be answered; never use <{NOOP_TAG}/> to skip an expected "
    "response."
)
# The continuation clause (an output Command and max_next_steps other than 0). Its example is
# deliberately neutral: one naming an application's action biases models toward that action.
_NEXT_PROMPT = (
    "\n- The end of the delivery of your own {name} tag does not call you again by itself; a "
    "command finishing does. If something remains for you to do once it has been delivered — "
    "an action that must wait until then, or more to add afterwards — write it as a note in a "
    "next tag, in the same response, like this:\n"
    "<{name}>Give me a moment.</{name}>\n"
    "<next>look it up, then tell them what I found</next>\n"
    "The note is private: nobody sees it. It appears in your World, and you are called again "
    "once your {name} tag has been delivered, so you can carry it out. Write a next note only "
    "when something really remains to be done; with no note you are simply called at the next "
    "event."
)
_NO_NEXT_PROMPT = (
    "\n- The end of the delivery of your own {name} tag does not call you again; a command "
    "finishing does."
)


@dataclass
class _OutputPart:
    """One output of a response (one tag): its stream and its Command execution."""

    index: int
    stream: OutputStream
    call_id: str
    key: str
    attributes: dict[str, str]
    started: bool = False  # delivery has begun (it was not still waiting for its turn)
    kept: str | None = None  # what history keeps after a cut; None = all of it


@dataclass
class _OutputSession:
    """One model response's text: its parser and the executions of its outputs."""

    command: Command | None
    signature: OutputSignature | None
    reaction_id: int
    parser: TextOutputParser | None = None
    parts: list[_OutputPart] = field(default_factory=list)
    parsed: ParseResult | None = None
    # The history record of what the model wrote (replaced when an output is cut short).
    record: AssistantTextRecord | None = None
    last_task: asyncio.Task[None] | None = None


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


# WICA's provider value for the Hugging Face Hub's serverless Inference Providers. Deliberately
# *not* "huggingface" — that's init_chat_model's builtin, which builds a local transformers
# pipeline (heavy, and non-cancellable). See build_chat_model and specs/config.md ("Providers").
_HUGGINGFACE_HUB_PROVIDER = "huggingface-hub"

# WICA's provider value for the deterministic, network-free, key-less scripted test model. Built
# directly (like huggingface-hub, and before the init_chat_model fallthrough); its `model_kwargs`
# carry the fake's own config (script/default/loop/delay_s), not provider kwargs. See
# specs/fake-provider.md and src/wica/fake_model.py.
_FAKE_PROVIDER = "fake"

# WICA's provider names whose LangChain `model_provider` id differs: a mapping, not a construction
# branch — these still pass through init_chat_model. See specs/config.md ("Provider names").
_LANGCHAIN_PROVIDER_IDS = {"gemini": "google_genai", "mistral": "mistralai"}

# The key sent to an openai-chat-completions-compatible server when none is configured: the OpenAI
# client refuses to start without one, and most local servers check none. Never OPENAI_API_KEY,
# which holds a key for OpenAI. See specs/config.md ("Providers").
_PLACEHOLDER_API_KEY = "not-needed"


def _resolve_history_budget(
    config: AgentConfig, model: BaseChatModel
) -> tuple[int | None, int | None]:
    """(high, low) token marks for the history budget, or (None, None) when it is off. The context
    window is `config.context_window`, else the model profile's `max_input_tokens` (a beta
    LangChain field loaded from the provider package's bundled data — absent for the Hub and for
    models newer than that data). A budget with no known window is a build-time ConfigError, never
    a silently inert budget. See specs/agent.md ("History budget")."""
    if config.history_budget is None:
        return None, None
    window = config.context_window
    if window is None:
        profile = model.profile
        window = profile.get("max_input_tokens") if profile is not None else None
    if window is None:
        hint = (
            " to the server's own context size (e.g. llama-server's -c)"
            if config.provider == OPENAI_CHAT_COMPLETIONS_COMPATIBLE
            else ""
        )
        raise ConfigError(
            "agent: 'history_budget' needs a known context window, and the model profile for "
            f"{config.provider}/{config.model} has no max_input_tokens — set 'context_window'"
            + hint
        )
    high = int(window * config.history_budget)
    return high, high // 2


def build_chat_model(config: AgentConfig) -> BaseChatModel:
    """Construct the LangChain chat model for an AgentConfig — the single source of truth for
    provider construction, shared by the Agent constructor and the e2e helper (so tests exercise
    the exact model production builds).

    Most providers pass straight through to init_chat_model (switching between them is a config
    edit), `gemini`/`mistral` under LangChain's own ids. `openai-chat-completions-compatible` is
    built as a ChatOpenAI pointed at the config's base_url. `huggingface-hub` is the other exception: it targets the Hugging Face Hub's serverless
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
        chat = ChatHuggingFace(llm=endpoint)
        if "max_new_tokens" not in config.model_kwargs:
            # HuggingFaceEndpoint caps output at 512 tokens by default and ChatHuggingFace copies
            # that cap into every request; a reasoning model is cut off before its tool call. Unset,
            # the request carries no cap and the backend's own limit applies, as with the other
            # providers. See specs/config.md ("Providers").
            chat.max_tokens = None
        return chat

    if config.provider == OPENAI_CHAT_COMPLETIONS_COMPATIBLE:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        # Built directly, not via init_chat_model, so this branch owns three things: the address
        # is the config's base_url, the key is never read from OPENAI_API_KEY (a placeholder
        # instead), and usage is reported on streamed calls (the client turns it off with a
        # base_url). See specs/agent.md ("Provider-agnostic model, from config").
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=SecretStr(api_key if api_key is not None else _PLACEHOLDER_API_KEY),
            stream_usage=True,
            **config.model_kwargs,
        )

    model_kwargs = dict(config.model_kwargs)
    if api_key is not None:
        model_kwargs["api_key"] = api_key
    return init_chat_model(
        config.model,
        model_provider=_LANGCHAIN_PROVIDER_IDS.get(config.provider, config.provider),
        **model_kwargs,
    )


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
        # The optional output Command is not a constructor argument: it is an application callable
        # wired *after* construction through set_output_command, so the objects it belongs to (a
        # transcript, a UI, a TTS engine) can be built against this Agent's World and Events
        # first. See specs/agent.md ("Output wiring").
        self._output_command: Command | None = None
        self._output_command_name: str | None = None
        self._output_signature: OutputSignature | None = None
        # How many steps in a row a <next> note may wake (0: the note is off; None: no cap), and
        # how many it has since something else triggered a step. See specs/agent.md
        # ("Continuing after an output").
        self._max_next_steps = config.max_next_steps
        self._next_chain = 0
        self._next_note: str | None = None
        self._next_note_reaction = 0
        # Outputs (see specs/agent.md, "Output: tags in the model's text"), by entry key: whether
        # the end of a registered one wakes a step; the same for every output ever opened (what
        # output_wakes reports); the response each running one belongs to; and the running or
        # waiting ones — one voice, so a later response that speaks cancels them.
        self._output_wake: dict[str, bool] = {}
        self._output_calls: dict[str, bool] = {}
        self._output_sessions: dict[str, _OutputSession] = {}
        self._live_outputs: dict[str, asyncio.Task[None]] = {}
        # Set while no reaction runs: an output that ends before the step that issued it waits
        # for it.
        self._step_over = asyncio.Event()
        self._step_over.set()
        # WICA's control Command, built once here so start() re-attaches the *same* object on
        # every cycle (the identity check in _register_builtin then holds across a restart).
        # Building a Command wraps the callable into a tool; it does not touch the model, so
        # __init__ stays inert.
        self._cancel_command = Command(
            self._cancel_command_action,
            name=_CANCEL_COMMAND_NAME,
            description=_CANCEL_COMMAND_DESCRIPTION,
        )
        # The system prompt the model receives is composed: the resolved persona (verbatim) followed
        # by the WICA runtime primer (+ the output clause when an output Command is set). The persona
        # is resolved once here (reading the prompt file if that's the form given) and kept, so
        # set_output_command can recompose the prompt — the only thing that changes its content —
        # without touching the file again. In the intended flow (wire, then start) the combined
        # string is fixed before the first step and stays in the cached deep prefix. See
        # specs/agent.md ("System prompt composition").
        self._persona = resolve_system_prompt(config)
        # None = unbounded. X keeps the last X reactions: history grows to 2X, then is cut back to
        # X at step start (hysteresis keeps the cached prefix stable). See specs/agent.md
        # ("History window").
        self._history_reactions = config.history_reactions
        self.model = model if model is not None else build_chat_model(config)
        # Token marks for the usage-driven cut (None = off): high = window × budget, low = half.
        # The window is resolved once here — config override, else the model profile — and a
        # budget without a known window is a build error. See specs/agent.md ("History budget").
        self._history_high, self._history_low = _resolve_history_budget(
            config, self.model
        )
        # The prompt's fixed part (system prompt + tool schemas), estimated as the smallest input
        # count any call has reported — the first call's, i.e. the fixed part plus one reaction.
        self._history_overhead: int | None = None
        self.system_prompt = self._compose_system_prompt()
        # Read for the wica.agent.model span's GenAI attributes (see specs/instrumentation.md).
        self._provider = config.provider
        self._model_name = config.model
        # Deadline on one whole model call (None = none); see specs/agent.md ("Model-call
        # deadline").
        self._model_timeout = config.model_timeout
        # Response guards, by name in run order: the defaults first, then the application's. A step
        # reads them once at its start, and retries a rejected response up to max_retries times.
        # See specs/agent.md ("Response guards").
        self._guards: dict[str, ResponseGuard] = dict(DEFAULT_GUARDS)
        self._max_retries = config.max_retries
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
        # Instrumentation Events — observability only, never control flow. The Agent emits on them
        # at the right points inside the loop; any number of consumers subscribe, and Event.emit's
        # per-subscriber isolation catches+logs a raising subscriber so it can neither abort a step
        # nor starve siblings (no _fire_hook guard needed). See specs/agent.md "Instrumentation".
        #  - on_trigger(entry):    once per trigger a run-to-completion step observes (filtered).
        #  - on_prompt(messages):  the exact rendered messages just before the reaction's first
        #                           model call (once per reaction; a retry's correction is
        #                           reported by on_response_rejected).
        #  - on_response_rejected(rejection): once per response attempt the guards reject.
        #  - on_command(command):  each Command the model issues, when issued; an output, once
        #                           its body is complete.
        #  - on_private_text(text): the text the model wrote outside its output tags in a step.
        #  - on_reaction_ended(trace):  once at the end of every reaction that started, whatever
        #                           its outcome — see specs/instrumentation.md ("Layer 1").
        #  - on_trigger_dropped(entry): once per trigger dropped because a reaction was in flight.
        #  - on_command_ended(trace):   once per dispatched Command when its terminal state lands.
        self.on_trigger: Event[WorldEntry] = Event()
        self.on_prompt: Event[list[BaseMessage]] = Event()
        self.on_response_rejected: Event[ResponseRejection] = Event()
        self.on_command: Event[CommandIssued] = Event()
        self.on_private_text: Event[str] = Event()
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
        # The last number _mint_call_id handed out; never reset, so a restart keeps ids unique.
        self._call_counter = 0
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
        """Read-only view of the Commands the model is bound to as tools, by name: application
        Commands plus, once start() has attached it, ``cancel_command``. The output Command is
        not among them (it is not a tool; see ``output_command_name``). Live view, not a
        snapshot. See specs/agent.md ("Commands")."""
        return MappingProxyType(self._commands)

    @property
    def output_command_name(self) -> str | None:
        """The output Command's name, or None when no output Command is set."""
        return self._output_command_name

    @property
    def response_guards(self) -> Mapping[str, ResponseGuard]:
        """Read-only view of the response guards by name, in the order they run: the defaults
        (``incomplete_response``, ``unknown_command``, ``invalid_args``) first, then the
        application's. See specs/agent.md ("Response guards")."""
        return MappingProxyType(self._guards)

    def add_response_guard(
        self, guard: ResponseGuard, *, name: str | None = None
    ) -> None:
        """**Experimental** — the shape may change without a deprecation period. Add a guard that
        checks every model response before it is used: it returns None to accept, or a
        ``wica.guards.Retry`` to reject (the model is called again, up to ``max_retries`` times).
        Sync or async; an async guard holds the reaction (triggers arriving meanwhile are
        dropped), and a sync one also blocks the loop while it runs, so keep it a fast rule.
        ``name`` defaults to the callable's ``__name__``; a name in use raises ``ValueError``.
        Takes effect at the next step. See specs/agent.md ("Response guards")."""
        key = name if name is not None else getattr(guard, "__name__", None)
        if not key:
            raise ValueError("a response guard needs a name: pass name=…")
        if key in self._guards:
            raise ValueError(f"response guard {key!r} is already added")
        self._guards[key] = guard

    def remove_response_guard(self, name: str) -> None:
        """Remove a response guard by name — a default one included. An unknown name raises
        ``ValueError``. Takes effect at the next step."""
        if name not in self._guards:
            raise ValueError(f"no response guard named {name!r}")
        del self._guards[name]

    def _compose_system_prompt(self) -> str:
        """The persona followed by the WICA runtime primer: perception + acting, the
        history-window clause (only with a bounded history), the output clause (the output
        Command's tag, or that nobody reads the text), the silence clause, and the continuation
        clause (an output Command and a <next> note that is not turned off). See specs/agent.md
        ("System prompt composition")."""
        primer = _RUNTIME_PRIMER
        if self._history_reactions is not None or self._history_high is not None:
            primer += _HISTORY_WINDOW_PROMPT
        command, signature = self._output_command, self._output_signature
        if command is None or signature is None:
            return self._persona + primer + _NO_OUTPUT_PROMPT + _SILENCE_PROMPT
        name = command.name
        attributes = ""
        if signature.attributes:
            described = "; ".join(
                f"{key} (one of: {', '.join(values)})" if values else key
                for key, values in signature.attributes.items()
            )
            example = " ".join(
                f'{key}="{signature.defaults[key]}"' for key in signature.attributes
            )
            attributes = _OUTPUT_ATTRIBUTES_PROMPT.format(
                name=name, attributes=described, example=example
            )
        description = command.description.strip()
        primer += _OUTPUT_PROMPT.format(
            name=name,
            description=f" What {name} does: {description}" if description else "",
            attributes=attributes,
        )
        primer += _SILENCE_PROMPT
        if self._max_next_steps == 0:
            primer += _NO_NEXT_PROMPT.format(name=name)
        else:
            primer += _NEXT_PROMPT.format(name=name)
        return self._persona + primer

    def set_output_command(self, fn: Callable[..., Any] | Command | None) -> None:
        """Set (or, with None, clear) the application's output Command — what the model says to
        its user. The model issues it by writing a tag of its name in its response text; the
        tag's body is delivered to the Command (as a stream or as a whole, by its first
        parameter's annotation) and its attributes are the Command's other parameters. See
        specs/commands.md ("The output Command"), specs/agent.md ("Output: tags in the model's
        text", "Output wiring").

        A callable is wrapped as ``Command(fn)``; a ``Command`` is used directly. Validation
        happens before any state changes: a reserved name (``noop``, ``next``,
        ``cancel_command``), a name already registered as an application Command, a name that is
        not a valid tag name, a Command wrapping an off-the-shelf tool, or a signature that
        cannot be an output's raises ``ValueError``. The system prompt is then recomposed. The
        output Command is not a tool: the model binding is untouched. An output of a replaced
        Command that is still running finishes normally."""
        command: Command | None = None
        signature: OutputSignature | None = None
        if fn is not None:
            command = fn if isinstance(fn, Command) else Command(fn)
            if command.name in _RESERVED_COMMAND_NAMES:
                raise ValueError(
                    f"output command may not be named {command.name!r}: "
                    f"{sorted(_RESERVED_COMMAND_NAMES)} are reserved by WICA"
                )
            if command.name in self._commands:
                raise ValueError(f"command name {command.name!r} is already registered")
            if not is_tag_name(command.name):
                raise ValueError(
                    f"output command name {command.name!r} is not a valid tag name (a letter, "
                    "then letters, digits, '_' or '-')"
                )
            if command.callable is None:
                raise ValueError(
                    "the output command must be built from a callable, not an existing tool"
                )
            signature = output_signature(command.callable)
            if signature.streams and not inspect.iscoroutinefunction(command.callable):
                raise ValueError(
                    "output Command: a function taking an OutputStream must be async"
                )
        self._output_command = command
        self._output_command_name = None if command is None else command.name
        self._output_signature = signature
        self.system_prompt = self._compose_system_prompt()
        _logger.debug("output command set to %r", self._output_command_name)

    def register_command(self, fn: Callable[..., Any] | Command) -> None:
        """Register a Command. Takes one argument: a plain callable (auto-wrapped — name from
        ``__name__``, description from the docstring) or a ``Command`` (used directly). To override
        name/description, or to wrap an off-the-shelf tool, pass ``Command(fn, name=…, …)`` /
        ``Command(tool)`` — there are no name/description kwargs here.

        Names are unique: registering a name already registered, a WICA-reserved name (``noop``,
        ``next``, ``cancel_command``), or the output Command's name raises ``ValueError``. See
        specs/commands.md ("Names are unique and three are reserved")."""
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

    def _rebind_model(self) -> None:
        """Bind the model to the current tool set. Every attach/detach comes through here. No
        tool choice is forced: the model speaks in its text (see specs/agent.md, "Output: tags
        in the model's text")."""
        if not self._commands:
            self._bound_model = self.model
            return
        self._bound_model = self.model.bind_tools(
            [c.tool for c in self._commands.values()]
        )

    def _register_builtin(self, command: Command) -> None:
        """Idempotent attach for WICA's own Command (used by start(), which may run again after
        stop()). Its name is reserved from application use, so this can never replace an
        application Command."""
        if self._commands.get(command.name) is command:
            return
        self._attach_command(command)

    def start(self) -> None:
        if self._started:
            return
        # WICA's control Command, auto-attached here (no app wiring) since it needs Agent
        # internals: cancel_command aborts a still-running Command previously issued — an output
        # included. Attaching at start (not construction) keeps __init__ inert. _register_builtin
        # is idempotent on object identity, so a second start() re-attaches the same one.
        self._register_builtin(self._cancel_command)
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
        self._step_over.clear()
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
            self._step_over.set()
            self._current_reaction = None
            ended_at = now()
            builder.span.set_attributes(
                {
                    "wica.outcome": builder.outcome,
                    "wica.text_length": builder.text_length,
                    "wica.command_count": len(builder.command_call_ids),
                    "wica.noop": builder.noop,
                    "wica.attempts": builder.attempts,
                }
            )
            builder.span.end()
            trace_id, span_id = _span_ids(builder.span)
            # Emitted after _busy is cleared, so a subscriber sees "the Agent is free again" and
            # the record at the same moment. Fires for every reaction that started, whatever its
            # outcome (ok/empty/model_error/model_timeout/rejected/cancelled) — see
            # specs/instrumentation.md ("Layer 1").
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
                    command_call_ids=tuple(builder.command_call_ids),
                    noop=builder.noop,
                    usage=builder.usage,
                    ended_at=ended_at,
                    trace_id=trace_id,
                    span_id=span_id,
                    attempts=builder.attempts,
                    rejections=tuple(builder.rejections),
                    guard_duration=builder.guard_duration,
                    first_content_at=builder.first_content_at,
                    first_output_at=builder.first_output_at,
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
        if any(entry.key not in self._output_calls for entry in batch):
            self._next_chain = (
                0  # something other than the Agent's own output woke this step
            )
        self._append_observation()
        if (
            self._next_note is not None
            and self._next_note_reaction < builder.reaction_id
        ):
            self._clear_next_note()  # this step's observation has shown it
        messages = self._render_messages()
        # Instrumentation is observation-only: subscribers receive a defensive copy, never the
        # message list subsequently handed to the model.
        self.on_prompt.emit(copy.deepcopy(messages))
        if self._provider == "anthropic":
            # Anthropic caches only at an explicit breakpoint. A transport detail, added after
            # on_prompt so observers keep the plain rendering. See specs/agent.md ("Prompt
            # caching").
            messages = mark_cache_breakpoint(messages)
        builder.prompt_ready_at = now()

        # The step's binding and guards, read once: every attempt runs against the same binding, and
        # a guard added or removed meanwhile applies from the next step. A rejected response is
        # retried over the same messages plus a correction, never stored in history; only the
        # accepted response goes on below. See specs/agent.md ("Response guards").
        bound_model = self._bound_model
        guards = list(self._guards.items())
        bound: Mapping[str, Command] = MappingProxyType(dict(self._commands))
        prompt = messages
        while True:
            builder.attempts += 1
            attempt = builder.attempts
            session = self._new_output_session(builder)
            parser = session.parser
            assert parser is not None
            response = await self._call_model(
                bound_model, prompt, builder, representative, parser.feed
            )
            parsed = parser.finish()
            if session.parts:
                # An output has started and cannot be taken back: the response is used as it
                # is, with no guard and no retry — and a model call that failed part-way keeps
                # what was said. See specs/agent.md ("Response guards").
                self._settle_text(session, parsed, builder)
                if response is None:
                    return
                break
            if response is None:
                return
            candidate = _model_response(response, bound)
            retry = await self._run_guards(guards, candidate, attempt, builder)
            if retry is None:
                break
            # An empty response cannot be shown (providers reject an empty assistant message), nor
            # can one holding a malformed tool call or cut at the output cap (a broken or cut
            # response cannot be replayed).
            show = (
                retry.show_response
                and bool(response.text or response.tool_calls)
                and not candidate.malformed
                and not candidate.truncated
            )
            will_retry = attempt <= self._max_retries
            builder.rejections.append(retry.feedback)
            self.on_response_rejected.emit(
                ResponseRejection(
                    attempt=attempt,
                    response=_model_response(response, bound),
                    feedback=retry.feedback,
                    show_response=show,
                    will_retry=will_retry,
                )
            )
            if not will_retry:
                # Out of retries: nothing recorded but the Observation (it renders merged into the
                # next one), nothing delivered or dispatched — a model error's ending.
                builder.outcome = "rejected"
                builder.error = retry.feedback
                _logger.error(
                    "response rejected after %d attempt(s); step abandoned (trigger: %s): %s",
                    attempt,
                    _describe_entry(representative),
                    retry.feedback,
                )
                return
            _logger.warning(
                "response rejected (attempt %d), retrying with %s: %s",
                attempt,
                "the response shown" if show else "a reminder",
                retry.feedback,
            )
            prompt = _retry_messages(messages, response, retry.feedback, show=show)

        if not session.parts:
            self._settle_text(session, parsed, builder)
        _logger.debug(
            "LLM output: text=%s, commands=%s",
            _truncate(parsed.raw) if parsed.raw else "(none)",
            [c["name"] for c in response.tool_calls] or "(none)",
        )
        said_or_did = bool(parsed.raw.strip() or response.tool_calls)
        builder.outcome = "ok" if said_or_did else "empty"
        # Doing nothing: a <noop/>, or no text at all, and no tool call. See specs/agent.md
        # ("The agent may choose not to act").
        builder.noop = parsed.noop and not parsed.private and not response.tool_calls
        if parsed.private:
            self.on_private_text.emit(parsed.private)

        for call in response.tool_calls:
            # The provider's tool-call id reconstructs the native tool_call/tool_result pair; the
            # model is pointed at a short id WICA mints instead (see _mint_call_id).
            tool_call_id = call["id"] or uuid.uuid4().hex
            call_id = self._mint_call_id()
            args = copy.deepcopy(call["args"])
            self._history.append(
                CommandRecord(call_id, tool_call_id, call["name"], copy.deepcopy(args))
            )
            self._dispatch_command(call_id, call["name"], args)
            builder.command_call_ids.append(call_id)
        if builder.first_input_tokens is not None:
            self._apply_history_budget(builder.first_input_tokens)
        _logger.debug("step complete (trigger: %s)", _describe_entry(representative))

    async def _call_model(
        self,
        bound_model: Runnable[Any, AIMessage],
        messages: list[BaseMessage],
        builder: _ReactionBuilder,
        representative: WorldEntry,
        on_text: Callable[[str], None] | None = None,
    ) -> AIMessage | None:
        """One attempt's model call, under its own deadline. None when the call raised or exceeded
        its deadline: the outcome is set and logged, and the step ends. See specs/agent.md
        ("Model-call deadline")."""
        if builder.model_started_at is None:
            builder.model_started_at = now()
        deadline = asyncio.timeout(self._model_timeout)
        usage: TokenUsage | None = None
        # The record keeps the last attempt's first content, like model_ended_at.
        builder.first_content_at = None
        try:
            with tracer().start_as_current_span(
                "wica.agent.model",
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": self._provider,
                    "gen_ai.request.model": self._model_name,
                    "wica.attempt": builder.attempts,
                },
            ) as model_span:

                def first_content() -> None:
                    builder.first_content_at = now()
                    model_span.add_event("wica.first_content")

                async with deadline:
                    response = await _stream_response(
                        bound_model, messages, first_content, on_text
                    )
                usage = _token_usage(response)
                if usage is not None:
                    model_span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": usage.input_tokens,
                            "gen_ai.usage.output_tokens": usage.output_tokens,
                        }
                    )
        except Exception as exc:
            # The observation is already in history (and terminal command entries are retired),
            # so the model never saw it this step; the renderer merges it into the next
            # observation's user message (see _render_messages). Nothing is dispatched. Ending the
            # step here frees the single-in-flight loop for the next trigger. CancelledError is a
            # BaseException and keeps propagating. See specs/agent.md ("Instrumentation").
            builder.model_ended_at = now()
            if isinstance(exc, TimeoutError) and deadline.expired():
                # Our deadline, not a provider's own TimeoutError: asyncio.timeout cancelled the
                # call. Same ending, its own outcome. A stop() cancellation is not caught here. See
                # specs/agent.md ("Model-call deadline").
                builder.outcome = "model_timeout"
                builder.error = (
                    f"model call exceeded its {self._model_timeout:g} s deadline"
                )
                _logger.error(
                    "%s; step abandoned (trigger: %s)",
                    builder.error,
                    _describe_entry(representative),
                )
                return None
            builder.outcome = "model_error"
            builder.error = str(exc)
            _logger.exception(
                "model call failed; step abandoned (trigger: %s)",
                _describe_entry(representative),
            )
            return None
        builder.model_ended_at = now()
        if usage is not None and builder.attempts == 1:
            builder.first_input_tokens = usage.input_tokens
        builder.usage = _add_usage(builder.usage, usage)
        return response

    async def _run_guards(
        self,
        guards: list[tuple[str, ResponseGuard]],
        response: ModelResponse,
        attempt: int,
        builder: _ReactionBuilder,
    ) -> Retry | None:
        """Run every guard on one attempt (so the feedback is complete) and combine the rejections;
        None when all accept. A guard that raises, or returns something other than a Retry or None,
        is logged at ERROR and counts as accepting: a buggy guard must not silence the agent. See
        specs/agent.md ("Response guards")."""
        if not guards:
            return None
        retries: list[Retry] = []
        started = now()
        with tracer().start_as_current_span(
            "wica.agent.guards", attributes={"wica.attempt": attempt}
        ) as span:
            for name, guard in guards:
                try:
                    result = guard(response)
                    if inspect.isawaitable(result):
                        result = await result
                except Exception:
                    _logger.exception(
                        "response guard %r raised; counted as accepting", name
                    )
                    continue
                if result is None:
                    continue
                if not isinstance(result, Retry):
                    _logger.error(
                        "response guard %r returned %r, not a Retry or None; counted as accepting",
                        name,
                        result,
                    )
                    continue
                retries.append(result)
            span.set_attribute("wica.rejected", bool(retries))
        elapsed = (now() - started).total_seconds()
        builder.guard_duration = (builder.guard_duration or 0.0) + elapsed
        return combine(retries) if retries else None

    # -- The model's text: outputs, the note, doing nothing -------------------------------------
    # See specs/agent.md ("Output: tags in the model's text", "Continuing after an output").

    def output_wakes(self, call_id: str) -> bool | None:
        """Whether the end of this output wakes a step; None for a call that is not an output.
        A harness waiting for the Agent to settle reads it: an output's completion is followed
        by a step only when it was granted one."""
        return self._output_calls.get(f"{_COMMAND_KEY_PREFIX}{call_id}")

    def _new_output_session(self, builder: _ReactionBuilder) -> _OutputSession:
        """The parser for one attempt's text, over the output Command the step started with."""
        command = self._output_command
        session = _OutputSession(
            command=command,
            signature=self._output_signature if command is not None else None,
            reaction_id=builder.reaction_id,
        )

        def on_open(attributes: dict[str, str]) -> None:
            self._open_output(session, attributes, builder)

        def on_delta(text: str) -> None:
            session.parts[-1].stream.push(text)

        def on_close() -> None:
            session.parts[-1].stream.close()

        session.parser = TextOutputParser(
            None if command is None else command.name,
            on_open=on_open,
            on_delta=on_delta,
            on_close=on_close,
        )
        return session

    def _open_output(
        self,
        session: _OutputSession,
        attributes: dict[str, str],
        builder: _ReactionBuilder,
    ) -> None:
        """An output's first words arrived: open its execution — a Command entry, running — and
        start delivering, before the response is complete. A response's outputs are delivered
        one after the other; its first one cuts off what an earlier response is still saying
        (one voice)."""
        command, signature = session.command, session.signature
        assert command is not None and signature is not None
        kept: dict[str, str] = {}
        for name, value in attributes.items():
            allowed = signature.attributes.get(name)
            if name in signature.attributes and (allowed is None or value in allowed):
                kept[name] = value
            else:
                _logger.warning(
                    "output %s: attribute %s=%r dropped (not declared, or not an allowed value)",
                    command.name,
                    name,
                    value,
                )
        if not session.parts:
            if builder.first_output_at is None:
                builder.first_output_at = now()
                otel_trace.get_current_span().add_event("wica.first_output")
            for old_key, old_task in list(self._live_outputs.items()):
                _logger.debug("output %s cut off by a new one", old_key)
                self._output_wake[old_key] = (
                    False  # cut off, not finished: wakes nothing
                )
                self._output_calls[old_key] = False
                old_task.cancel()
        call_id = self._mint_call_id()
        key = f"{_COMMAND_KEY_PREFIX}{call_id}"
        part = _OutputPart(len(session.parts), OutputStream(), call_id, key, kept)
        session.parts.append(part)
        self._command_keys.add(key)
        self._output_wake[key] = False
        self._output_calls[key] = False
        self._output_sessions[key] = session
        self._world.register(
            key,
            CommandExecution,
            serialize_fn=_serialize_output_execution,
            include_in_prompt=True,
            triggers_llm_call=True,
            trigger_condition_fn=lambda old, new: (
                new is not None
                and new.is_terminal()
                and self._output_wake.get(key, False)
            ),
        )
        self._world.update(
            key,
            CommandExecution(
                name=command.name, args={**kept, "text": ""}, state="running"
            ),
        )
        started_at = self._world.get_entry(key).current.timestamp
        self._command_started[key] = (started_at, session.reaction_id)
        # Parented to the reaction explicitly: an output opens during the model call, where the
        # current span is wica.agent.model, and its span is that one's sibling, not its child.
        # See specs/instrumentation.md ("Layer 2").
        span = tracer().start_span(
            "wica.agent.command",
            context=otel_trace.set_span_in_context(builder.span),
            attributes={"wica.command.name": command.name, "wica.call_id": call_id},
        )
        task = self._track_task(
            self._run_output(session, part, span, session.last_task)
        )
        session.last_task = task
        self._running_tasks[key] = task
        self._live_outputs[key] = task

    def _settle_text(
        self, session: _OutputSession, parsed: ParseResult, builder: _ReactionBuilder
    ) -> None:
        """A response's text is complete: record what the model wrote, publish each output as an
        issued Command, keep the note, and decide whether the end of the last output wakes a
        step. Synchronous, so an output cannot end before it is published."""
        session.parsed = parsed
        raw = parsed.raw.strip()
        if raw:
            session.record = AssistantTextRecord(raw)
            self._history.append(session.record)
        builder.text_length = len(raw)
        command = session.command
        notes = command is not None and self._max_next_steps != 0
        if notes and parsed.note is not None:
            self._set_next_note(parsed.note, builder.reaction_id)
        if command is None:
            return
        for part in session.parts:
            args = {**part.attributes, "text": part.stream.text.strip()}
            if part.key in self._running_tasks:
                self._world.update(
                    part.key,
                    CommandExecution(name=command.name, args=args, state="running"),
                )
            self.on_command.emit(
                CommandIssued(command.name, copy.deepcopy(args), part.call_id)
            )
            builder.command_call_ids.append(part.call_id)
        if session.parts and notes and parsed.note is not None:
            last = session.parts[-1]
            if last.key not in self._running_tasks:
                return  # already cut off
            cap = self._max_next_steps
            if cap is not None and self._next_chain >= cap:
                _logger.warning(
                    "output %s: its end wakes no step — %d step(s) in a row already woken by "
                    "a <next> note (max_next_steps)",
                    last.key,
                    self._next_chain,
                )
                return
            self._next_chain += 1
            self._output_wake[last.key] = True
            self._output_calls[last.key] = True

    def _set_next_note(self, note: str, reaction_id: int) -> None:
        self._next_note = note
        self._next_note_reaction = reaction_id
        if not self._world.is_registered(_NEXT_NOTE_KEY):
            self._world.register(_NEXT_NOTE_KEY, str, serialize_fn=_serialize_next_note)
        self._world.update(_NEXT_NOTE_KEY, note)

    def _clear_next_note(self) -> None:
        """A note is used once: the step that follows it has observed it."""
        self._next_note = None
        try:
            self._world.unregister(_NEXT_NOTE_KEY)
        except KeyError:
            pass

    async def _run_output(
        self,
        session: _OutputSession,
        part: _OutputPart,
        span: otel_trace.Span,
        after: asyncio.Task[None] | None,
    ) -> None:
        """One output's execution: wait for the response's previous output, hand the text to the
        output Command — the stream, or the whole text once the body is complete — and write the
        terminal state, like _run_command. The terminal state waits for the step that issued the
        output, so its completion is settled and never dropped by the single-in-flight loop."""
        command, signature = session.command, session.signature
        assert command is not None and signature is not None
        fn = command.callable
        assert fn is not None
        stream, key, call_id, name = part.stream, part.key, part.call_id, command.name

        def args() -> dict[str, Any]:
            return {**part.attributes, "text": stream.text.strip()}

        with otel_trace.use_span(span, end_on_exit=True):
            try:
                if after is not None:
                    await asyncio.wait({after})
                if signature.streams:
                    part.started = True
                    result = fn(stream, **part.attributes)
                    if inspect.isawaitable(result):
                        result = await result
                else:
                    text = (await stream.read()).strip()
                    part.started = True
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(text, **part.attributes)
                    else:
                        result = await asyncio.to_thread(fn, text, **part.attributes)
                await self._after_issuing_step(session)
            except asyncio.CancelledError:
                _logger.debug("output %s [call_id=%s] cancelled", name, call_id)
                span.set_attribute("wica.command.state", "cancelled")
                # A cancelled output wakes nothing, note or not — whoever cancelled it (a new
                # response, cancel_command, the application).
                self._output_wake[key] = False
                self._output_calls[key] = False
                heard = stream.heard if part.started else ""
                final = args()
                if heard is not None:
                    final["text"] = heard
                    part.kept = heard
                    self._cut_text_record(session)
                if self._write_terminal(
                    key,
                    call_id,
                    CommandExecution(name=name, args=final, state="cancelled"),
                ):
                    self._emit_command_ended(key, call_id, name, "cancelled")
                raise
            except Exception as exc:
                _logger.warning("output %s [call_id=%s] failed: %s", name, call_id, exc)
                span.set_attribute("wica.command.state", "failed")
                await self._after_issuing_step(session)
                if self._write_terminal(
                    key,
                    call_id,
                    CommandExecution(
                        name=name, args=args(), state="failed", error=str(exc)
                    ),
                ):
                    self._emit_command_ended(key, call_id, name, "failed")
            else:
                span.set_attribute("wica.command.state", "complete")
                if self._write_terminal(
                    key,
                    call_id,
                    CommandExecution(
                        name=name,
                        args=args(),
                        state="complete",
                        result="Done." if result is None else str(result),
                    ),
                ):
                    self._emit_command_ended(key, call_id, name, "complete")
            finally:
                self._running_tasks.pop(key, None)
                self._live_outputs.pop(key, None)
                self._output_sessions.pop(key, None)

    async def _after_issuing_step(self, session: _OutputSession) -> None:
        """Wait until the step that issued this output is over: an output's terminal state is
        written after it, so the output has been reported and its wake decided."""
        current = self._current_reaction
        if current is not None and current.reaction_id == session.reaction_id:
            await self._step_over.wait()

    def _cut_text_record(self, session: _OutputSession) -> None:
        """An output was cut short: the response's history record keeps only what was heard of
        it, and loses an output that never got its turn."""
        record, parsed, command = session.record, session.parsed, session.command
        if record is None or parsed is None or command is None:
            return
        kept = {p.index: p.kept for p in session.parts if p.kept is not None}
        text = parsed.rebuilt(command.name, kept)
        for index, item in enumerate(self._history):
            if item is record:
                if text:
                    session.record = AssistantTextRecord(text)
                    self._history[index] = session.record
                else:
                    del self._history[index]
                    session.record = None
                return

    def _mint_call_id(self) -> str:
        """The `<call_id>` for a new tool call: `c1`, `c2`, … over the Agent's lifetime, skipping a
        number whose agent:command:c<n> key is already registered. Never the provider's id: a
        model copying a long opaque id back into cancel_command garbles it (Claude Haiku dropped
        the `toolu_` prefix). See specs/commands.md ("Command execution as a World entry")."""
        while True:
            self._call_counter += 1
            call_id = f"c{self._call_counter}"
            if not self._world.is_registered(f"{_COMMAND_KEY_PREFIX}{call_id}"):
                return call_id

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
        self._apply_history_window()
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

    def _apply_history_window(self) -> None:
        """Cut the oldest reactions once history holds 2X of them, keeping the newest X (the
        current step's included). A reaction is counted by its ObservationRecord — every step
        appends exactly one, whatever its outcome. Cutting at an Observation boundary keeps the
        provider stream valid (first message a user turn; a step's tool_calls and tool_results
        travel together). Records are dropped from storage, not hidden — that is what bounds
        memory. Hysteresis (2X, not X+1) keeps the cached prefix byte-stable between cuts. See
        specs/agent.md ("History window")."""
        keep = self._history_reactions
        if keep is None or self._count_observations() < 2 * keep:
            return
        self._cut_history(keep, reason="history window")

    def _apply_history_budget(self, input_tokens: int) -> None:
        """The usage-driven cut: the call just made reported `input_tokens` for the prompt it read.
        Over the high-water mark, keep only as many newest reactions as fit under the low-water
        mark (at least the current one). The prompt is a fixed part (system prompt, tool schemas)
        plus one slice per reaction; the fixed part is taken as the smallest count ever reported
        (the first call's: fixed part + one reaction) and one reaction's cost as the rest of this
        report spread over its other reactions — so a large system prompt does not inflate the
        per-reaction estimate. See specs/agent.md ("History budget")."""
        if self._history_high is None or self._history_low is None:
            return
        if self._history_overhead is None or input_tokens < self._history_overhead:
            self._history_overhead = input_tokens
        if input_tokens <= self._history_high:
            return
        count = self._count_observations()
        if count < 2:
            _logger.warning(
                "history budget: prompt reported %d input tokens (mark: %d) but the single "
                "reaction in history cannot be cut",
                input_tokens,
                self._history_high,
            )
            return
        overhead = self._history_overhead
        per_reaction = max(1.0, (input_tokens - overhead) / (count - 1))
        keep = math.floor((self._history_low - overhead) / per_reaction)
        if keep < 1:
            _logger.warning(
                "history budget: the prompt's fixed part (~%d tokens: system prompt and tools) "
                "leaves no room for history under the low-water mark (%d); keeping one reaction",
                overhead,
                self._history_low,
            )
            keep = 1
        _logger.debug(
            "history budget: prompt reported %d input tokens (mark: %d) over %d reaction(s), "
            "fixed part ~%d, ~%d per reaction",
            input_tokens,
            self._history_high,
            count,
            overhead,
            per_reaction,
        )
        self._cut_history(keep, reason="history budget")

    def _count_observations(self) -> int:
        return sum(isinstance(r, ObservationRecord) for r in self._history)

    def _cut_history(self, keep: int, *, reason: str) -> None:
        """Drop every record before the `keep`-th newest Observation — whole reactions, from
        storage. The one place history is ever shortened; both cut policies land here."""
        observation_indexes = [
            i
            for i, record in enumerate(self._history)
            if isinstance(record, ObservationRecord)
        ]
        if keep >= len(observation_indexes):
            return
        cut = observation_indexes[-keep]
        _logger.debug(
            "%s: dropping %d reaction(s) (%d record(s)), keeping the last %d",
            reason,
            len(observation_indexes) - keep,
            cut,
            keep,
        )
        del self._history[:cut]

    def _is_terminal_command(self, entry: WorldEntry) -> bool:
        value = entry.current.value
        return (
            entry.key in self._command_keys
            and isinstance(value, CommandExecution)
            and value.is_terminal()
        )

    def _cleanup_command_entry(self, key: str) -> None:
        self._command_keys.discard(key)
        self._output_wake.pop(key, None)
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
        # call_id -> the tool_result content for that call: the fixed entry-pointer ack
        # (_command_ack).
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
            # the completion is observed).
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
                    # A placeholder left by an earlier all-omitted observation gives way to real
                    # blocks; it only stays if the merged observation is still empty.
                    combined = [
                        *(b for b in merged if b != _EMPTY_OBSERVATION_BLOCK),
                        *blocks,
                    ]
                    messages[-1] = HumanMessage(
                        content=combined or [_EMPTY_OBSERVATION_BLOCK]
                    )
                else:
                    messages.append(
                        HumanMessage(content=blocks or [_EMPTY_OBSERVATION_BLOCK])
                    )
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
        flush_assistant()
        return messages
