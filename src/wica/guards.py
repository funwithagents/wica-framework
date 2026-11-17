"""Response guards: checks a model's response must pass before the Agent uses it. A rejected response
is retried within the same reaction, and only the accepted one enters history. See specs/agent.md
("Response guards").

The default guards (`incomplete_response`, `unknown_command`, `invalid_args`) are supported;
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
    "incomplete_response",
    "invalid_args",
    "unknown_command",
]


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
    """What a guard sees: the response's text and tool calls, plus the binding they were made
    against. Only a response that started no output is guarded (see specs/agent.md, "Response
    guards")."""

    text: str  # the text as the model wrote it, "" when none
    commands: tuple[ProposedCommand, ...]  # every well-formed tool call, in order
    bound: Mapping[str, Command]  # the Commands the model call was bound to
    # Tool calls whose arguments could not be read, and whether the provider stopped the response
    # at its output cap. Defaulted so a response built by hand stays valid.
    malformed: tuple[MalformedCommand, ...] = ()
    truncated: bool = False
    # The output tokens the provider reported for this response; None when it reported none. For a
    # truncated response it is the cap that was hit.
    output_tokens: int | None = None


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


def incomplete_response(response: ModelResponse) -> Retry | None:
    """A response that did not arrive whole where it matters: a tool call whose arguments could
    not be read, or a response the provider cut at its output cap while it held a tool call (the
    last one is incomplete, however well-formed it looks). Nothing of it is executed and the model
    is asked again. A reminder, never shown: a broken or cut response cannot be replayed. The
    feedback gives a length to stay under where one is known — "be shorter" alone does not make
    a model shorten. Text cut at the cap is accepted: an output is delivered as far as it got."""
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
    if parts:
        return Retry(" ".join(dict.fromkeys(parts)), show_response=False)
    if not response.truncated:
        return None
    # A length to stay under: a quarter of the cap that was hit, at about four characters per
    # token. Half was not enough: a tool call's own framing costs some thirty tokens (measured on
    # Anthropic, where a 120-character call still used 60).
    tokens = response.output_tokens
    limit = (
        f"The whole response must fit in about {tokens} tokens: keep it under "
        f"{tokens} characters."
        if tokens
        else "Make it at most half as long as what you were writing."
    )
    if response.commands:
        last = response.commands[-1].name
        return Retry(
            "Your response was cut off at the output length limit while you were writing the "
            f"call to `{last}`, so that call is incomplete and nothing was executed. Respond "
            f"again with much less in it. {limit}",
            show_response=False,
        )
    return None


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
    that schema is a pydantic model; a tool declaring a raw JSON schema is not checked. The last
    call of a response cut at the output cap is skipped: it is incomplete, not wrong, and
    incomplete_response says so."""
    problems: list[str] = []
    calls = response.commands[:-1] if response.truncated else response.commands
    for call in calls:
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


# Installed at Agent construction, in this order, before any application guard.
DEFAULT_GUARDS: tuple[tuple[str, ResponseGuard], ...] = (
    ("incomplete_response", incomplete_response),
    ("unknown_command", unknown_command),
    ("invalid_args", invalid_args),
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
