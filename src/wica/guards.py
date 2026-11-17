"""Response guards: checks a model's response must pass before the Agent uses it. A rejected response
is retried within the same reaction, and only the accepted one enters history. See specs/agent.md
("Response guards").

The default guards (`incomplete_response`, `unknown_command`, `invalid_args`,
`output_command_required` — active only when the output Command is a tool — and `garbled_text`)
are supported, and so is `find_garbled`, the check `garbled_text` makes (the Agent also runs it on
an output's text as it streams);
writing an application guard — `Agent.add_response_guard` / `Wica.add_response_guard` and the types
here — is **experimental**: the shape may change without a deprecation period.

LangChain-free on purpose: a guard sees a neutral `ModelResponse`, never an `AIMessage`, the same
way the command-definition surface speaks `Command` (see specs/commands.md).
"""

from __future__ import annotations

import re
import zlib
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
    "find_garbled",
    "garbled_text",
    "incomplete_response",
    "invalid_args",
    "output_command_required",
    "unknown_command",
]


# The `noop` tool of tool mode, named in feedback. Duplicated from wica.text_output on purpose:
# this module stays a leaf.
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
    """What a guard sees: the response's text and tool calls, plus the binding they were made
    against. A response whose output has started streaming is not guarded (see specs/agent.md,
    "Response guards")."""

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
    # The output Command's name when it is bound as a tool (tool mode), else None: the
    # response must then call a tool. See specs/agent.md ("The output Command as a tool call").
    output_command_name: str | None = None
    # What the response would say and the rest of its text: the text of each output in order (an
    # output tag's body, or in tool mode an output tool call's `text`), and the private text (outside
    # the tags; all of the text in tool mode or when no output Command is set).
    outputs: tuple[str, ...] = ()
    private_text: str = ""


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
    # True when an output of the rejected response had already started: it was cancelled, and the
    # response removed from history (only a garbled one is; see specs/agent.md, "Garbled text").
    output_started: bool = False


# A guard is sync or async: an awaitable result is awaited (like a World listener's).
type ResponseGuard = Callable[[ModelResponse], Retry | None | Awaitable[Retry | None]]


def combine(retries: Sequence[Retry]) -> Retry:
    """Several guards rejecting one attempt make one correction: one line of feedback each, shown
    with the response only if every one of them asked for it — a guard that hides the response
    (garbage, a cut response) does so whatever the others ask."""
    if len(retries) == 1:
        return retries[0]
    return Retry(
        feedback="\n".join(f"- {r.feedback}" for r in retries),
        show_response=all(r.show_response for r in retries),
    )


# --- The default guards ------------------------------------------------------------------------


def incomplete_response(response: ModelResponse) -> Retry | None:
    """A response that did not arrive whole where it matters: a tool call whose arguments could
    not be read, or a response the provider cut at its output cap while it held a tool call (the
    last one is incomplete, however well-formed it looks). Nothing of it is executed and the model
    is asked again. A reminder, never shown: a broken or cut response cannot be replayed. The
    feedback gives a length to stay under where one is known — "be shorter" alone does not make
    a model shorten. Text cut at the cap is accepted: an output is delivered as far as it got.
    In tool mode a response cut before it made any call is incomplete too: the call the output
    Command requires is missing."""
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
    if response.output_command_name is not None:
        return Retry(
            "Your response was cut off at the output length limit before you called any "
            "command, so the user heard nothing. Respond again and call "
            f"`{response.output_command_name}` (or `{_NOOP_COMMAND_NAME}`) right away. {limit}",
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


def output_command_required(response: ModelResponse) -> Retry | None:
    """The output Command is a tool and the response calls no tool at all: its free text, if
    any, is private reasoning the user never hears. A response whose only call is malformed, or
    that was cut at the output cap, is incomplete_response's case. Inactive in tag mode, where
    `output_command_name` is None."""
    name = response.output_command_name
    if name is None or response.commands or response.malformed or response.truncated:
        return None
    return Retry(
        "Your response called no command, so the user heard nothing: your free text is private. "
        f"Call `{name}` to reply to the user, or `{_NOOP_COMMAND_NAME}` to do nothing.",
        # An empty response cannot be shown (providers reject an empty assistant message).
        show_response=bool(response.text),
    )


# --- Garbled text ------------------------------------------------------------------------------
#
# A model sometimes degenerates: it writes dots, ellipses and blank space instead of words, or loops
# on one phrase until its output cap. The check is cheap, model-free and language-agnostic, and safe
# for any output (code, JSON, a table, emoji, a song the user asked for). See specs/agent.md
# ("Garbled text").

# A text is judged only once it holds this many characters: below it, nothing is decided. While a
# response streams, an output is held back until its text reaches it (or ends).
GARBLED_MIN_CHARS = 80
# The filler share is measured over the last this-many characters, so garbage after a clean start
# is still caught.
GARBLED_WINDOW_CHARS = 200
# Garbled when filler is more than this share of the window.
GARBLED_MAX_FILLER = 0.5
# A response cut at its output cap whose last this-many characters compress below the ratio is a
# loop.
GARBLED_LOOP_CHARS = 400
GARBLED_MIN_COMPRESSION = 0.2

# Filler: ellipses, runs of dots and `</` that starts no tag (a run of them counts whole, the space
# between them included), gaps of two or more spaces inside a line, and runs of blank lines.
# Indentation is removed before measuring: it is structure.
_FILLER_MARK = r"(?:…+|\.{2,}|</(?![A-Za-z/]))"
_FILLER = re.compile(
    rf"{_FILLER_MARK}(?:\s*{_FILLER_MARK})*|(?<=\S)[ \t]{{2,}}(?=\S)|\n{{3,}}"
)
_INDENTATION = re.compile(r"(?m)^[ \t]+")


def _filler_share(text: str) -> float:
    text = _INDENTATION.sub("", text)
    length = len(text.strip())
    if not length:
        return 0.0
    return sum(len(m) for m in _FILLER.findall(text)) / length


def find_garbled(
    text: str, *, truncated: bool = False, tail_only: bool = False
) -> str | None:
    """Why `text` is garbled, or None. Three signs: filler (ellipses, runs of dots, gaps inside a
    line, blank lines) making up most of any `GARBLED_WINDOW_CHARS` characters of it, once it holds
    `GARBLED_MIN_CHARS`; a complete text of any length with no letter or digit at all that is
    mostly filler (`……… ... ...</...>`); and, for a response cut at its output cap (`truncated`),
    an end that repeats itself — a text that repeats itself and ends on its own (a song) is not
    garbled. `tail_only` is for a text still being written, checked again each time it grows: the
    filler is measured over its last window only (the earlier ones were measured already), and a
    text without words is not judged (its words may be still to come)."""
    text = text.strip()
    if (
        not tail_only
        and text
        and not any(c.isalnum() for c in text)
        and _filler_share(text) > GARBLED_MAX_FILLER
    ):
        return "no words, only dots and ellipses"
    if len(text) < GARBLED_MIN_CHARS:
        return None
    ends = (
        [len(text)]
        if tail_only
        else [*range(GARBLED_MIN_CHARS, len(text), GARBLED_MIN_CHARS // 2), len(text)]
    )
    for end in ends:
        if (
            _filler_share(text[max(0, end - GARBLED_WINDOW_CHARS) : end])
            > GARBLED_MAX_FILLER
        ):
            return "mostly dots, ellipses and blank space"
    if truncated:
        tail = text[-GARBLED_LOOP_CHARS:].encode()
        if len(zlib.compress(tail)) / len(tail) < GARBLED_MIN_COMPRESSION:
            return "repeating itself until it was cut off"
    return None


GARBLED_FEEDBACK = (
    "Your previous response was garbled ({reason}) and was discarded. Answer again, briefly, in "
    "plain words."
)


def garbled_text(response: ModelResponse) -> Retry | None:
    """A response whose text is garbled: an output's text or the private text (see
    `find_garbled`). A reminder, never shown: shown its garbage, a model writes it again."""
    texts = [*response.outputs, response.private_text]
    if not response.outputs and not response.private_text:
        texts = [response.text]  # a response built by hand, with only its text
    for text in texts:
        reason = find_garbled(text, truncated=response.truncated)
        if reason is not None:
            return Retry(GARBLED_FEEDBACK.format(reason=reason), show_response=False)
    return None


# Installed at Agent construction, in this order, before any application guard.
DEFAULT_GUARDS: tuple[tuple[str, ResponseGuard], ...] = (
    ("incomplete_response", incomplete_response),
    ("unknown_command", unknown_command),
    ("invalid_args", invalid_args),
    ("output_command_required", output_command_required),
    ("garbled_text", garbled_text),
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
