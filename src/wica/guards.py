"""Response guards: checks a model's response must pass before the Agent uses it. A rejected response
is retried within the same reaction, and only the accepted one enters history. See specs/agent.md
("Response guards").

The default guards (`malformed_command`, `unknown_command`, `invalid_args`,
`output_command_required`) are supported;
writing an application guard — `Agent.add_response_guard` / `Wica.add_response_guard` and the types
here — is **experimental**: the shape may change without a deprecation period.

LangChain-free on purpose: a guard sees a neutral `ModelResponse`, never an `AIMessage`, the same
way the command-definition surface speaks `Command` (see specs/commands.md).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from wica.command import Command

__all__ = [
    "DEFAULT_GUARDS",
    "MalformedCommand",
    "ModelResponse",
    "ProposedCommand",
    "ResponseGuard",
    "ResponseRejection",
    "Retry",
    "combine",
    "invalid_args",
    "malformed_command",
    "output_command_required",
    "unknown_command",
]

# The WICA-native "do nothing" Command, named in feedback. Duplicated from agent.py on purpose: this
# module sits below the Agent and must not import it.
_NOOP_COMMAND_NAME = "noop"


@dataclass(frozen=True)
class ProposedCommand:
    """One tool call of a response, before anything is dispatched (so no `call_id` yet)."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class MalformedCommand:
    """A tool call whose arguments could not be read — typically cut by the output cap. It is
    never dispatched. See specs/agent.md ("Streaming the model call")."""

    name: str | None  # the Command the call named, when it got that far
    raw_args: str  # the arguments as the model wrote them
    error: str  # why they could not be read


@dataclass(frozen=True)
class ModelResponse:
    """What a guard sees: the response's free text and tool calls, plus the binding they were made
    against."""

    text: str  # the free text, "" when none
    commands: tuple[ProposedCommand, ...]  # every well-formed tool call, in order
    bound: Mapping[str, Command]  # the Commands the model call was bound to
    output_command_name: str | None  # the bound output Command, if any
    # Tool calls whose arguments could not be read. Defaulted so a response built by hand stays
    # valid.
    malformed: tuple[MalformedCommand, ...] = ()


@dataclass(frozen=True)
class Retry:
    """A guard's rejection: what to tell the model, and whether to show it the rejected response
    (for a concrete, fixable mistake) or only remind it (when seeing the response would pull the
    model back toward it)."""

    feedback: str
    show_response: bool = True


@dataclass(frozen=True)
class ResponseRejection:
    """The payload of `on_response_rejected`: one rejected attempt."""

    attempt: int  # 1 = the reaction's first model call
    response: ModelResponse
    feedback: str  # the combined feedback of every guard that rejected it
    show_response: bool  # whether the retry shows the response (False for a reminder)
    will_retry: bool  # False when retries ran out and the reaction ends `rejected`


# A guard is sync or async: an awaitable result is awaited (like a World listener's).
type ResponseGuard = Callable[[ModelResponse], Retry | None | Awaitable[Retry | None]]


def combine(retries: Sequence[Retry]) -> Retry:
    """Several guards rejecting one attempt make one correction: one line of feedback each, shown
    with the response if any of them asked for it."""
    if len(retries) == 1:
        return retries[0]
    return Retry(
        feedback="\n".join(f"- {r.feedback}" for r in retries),
        show_response=any(r.show_response for r in retries),
    )


# --- The default guards ------------------------------------------------------------------------


def malformed_command(response: ModelResponse) -> Retry | None:
    """A tool call whose arguments could not be read: it was not executed, and the model is asked
    for it again. A reminder, never shown: the broken call cannot be replayed to a provider. The
    feedback says where the call stopped and gives a length to stay under — "be shorter" alone
    does not make a model shorten."""
    if not response.malformed:
        return None
    parts = []
    for call in response.malformed:
        name = f"`{call.name}`" if call.name else "a command"
        length = len(call.raw_args)
        parts.append(
            f"Your call to {name} was not executed: its arguments stopped after {length} "
            "characters and are not complete JSON, most likely because the response was too "
            f"long and was cut off. Call it again with much less in it: at most {max(length // 2, 1)} "
            "characters of arguments."
        )
    return Retry(" ".join(dict.fromkeys(parts)), show_response=False)


def unknown_command(response: ModelResponse) -> Retry | None:
    """A tool call naming a Command the model was not bound to."""
    unknown = [c.name for c in response.commands if c.name not in response.bound]
    if not unknown:
        return None
    names = ", ".join(repr(n) for n in dict.fromkeys(unknown))
    available = ", ".join(sorted(response.bound)) or "(none)"
    return Retry(
        f"There is no command named {names}. The commands you can call are: {available}."
    )


def invalid_args(response: ModelResponse) -> Retry | None:
    """A tool call whose arguments do not validate against its Command's argument schema — when
    that schema is a pydantic model; a tool declaring a raw JSON schema is not checked."""
    problems: list[str] = []
    for call in response.commands:
        command = response.bound.get(call.name)
        if command is None:
            continue  # unknown_command's case
        schema = command.tool.args_schema
        validate = getattr(schema, "model_validate", None)
        if not isinstance(schema, type) or validate is None:
            continue
        try:
            validate(call.args)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            problems.append(f"{call.name!r}: {_describe_validation_error(exc)}")
    if not problems:
        return None
    return Retry(
        "The arguments of your call are invalid — "
        + "; ".join(problems)
        + ". Call it again with valid arguments."
    )


def _describe_validation_error(exc: ValueError) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    listed = cast(list[dict[str, Any]], errors())
    parts = []
    for error in listed:
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid")
        parts.append(f"{location}: {message}" if location else message)
    return ", ".join(parts) or str(exc)


def output_command_required(response: ModelResponse) -> Retry | None:
    """An output Command is bound and the response calls no tool at all: its free text, if any, is
    private reasoning the user never hears. A response whose only call is malformed did call:
    that is malformed_command's case."""
    name = response.output_command_name
    if name is None or response.commands or response.malformed:
        return None
    return Retry(
        "Your response called no command, so the user heard nothing: your free text is private. "
        f"Call `{name}` to reply to the user, or `{_NOOP_COMMAND_NAME}` to do nothing.",
        # An empty response cannot be shown (providers reject an empty assistant message).
        show_response=bool(response.text),
    )


# Installed at Agent construction, in this order, before any application guard.
DEFAULT_GUARDS: tuple[tuple[str, ResponseGuard], ...] = (
    ("malformed_command", malformed_command),
    ("unknown_command", unknown_command),
    ("invalid_args", invalid_args),
    ("output_command_required", output_command_required),
)

# --- The correction frames ---------------------------------------------------------------------
#
# How a retry tells the model what went wrong. Module constants, like the runtime primer's text.

# The tool_result answering each tool call of a rejected response that is shown: a provider rejects
# a tool_use left without its result.
REJECTED_TOOL_RESULT = "Not executed: this response was rejected."

# The user message after a shown rejected response.
SHOW_FRAME = (
    "Your previous response was rejected and none of it was executed:\n"
    "{feedback}\n"
    "Respond again."
)

# The text block appended to the newest observation when the rejected response is not shown.
REMINDER_FRAME = (
    "A previous attempt at responding to this was rejected:\n"
    "{feedback}\n"
    "Make sure your response avoids this."
)
