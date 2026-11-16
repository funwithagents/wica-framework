"""The default response guards on their own, over hand-built responses. How the Agent runs them and
retries is in test_agent.py ("Response guards"). See specs/agent.md ("Response guards")."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from wica.command import Command
from wica.guards import (
    MalformedCommand,
    ModelResponse,
    ProposedCommand,
    Retry,
    combine,
    invalid_args,
    incomplete_response,
    output_command_required,
    unknown_command,
)


async def walk_to(room: str, speed: int) -> str:
    """Walk to a room."""
    return room


async def say(text: str) -> str:
    """Say something."""
    return text


async def noop() -> str:
    """Do nothing."""
    return ""


class _RawSchemaTool(BaseTool):
    """An off-the-shelf tool declaring its arguments as a raw JSON schema."""

    name: str = "raw"
    description: str = "A tool with a JSON-schema dict."
    args_schema: Any = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    def _run(self, **kwargs: Any) -> str:
        return "ok"


BOUND = {
    "walk_to": Command(walk_to),
    "say": Command(say),
    "noop": Command(noop),
    "raw": Command(_RawSchemaTool()),
}


def response(
    *calls: tuple[str, dict[str, Any]],
    text: str = "",
    output: str | None = None,
    malformed: tuple[MalformedCommand, ...] = (),
    truncated: bool = False,
    output_tokens: int | None = None,
) -> ModelResponse:
    return ModelResponse(
        text=text,
        commands=tuple(ProposedCommand(name, args) for name, args in calls),
        bound=BOUND,
        output_command_name=output,
        malformed=malformed,
        truncated=truncated,
        output_tokens=output_tokens,
    )


CUT_SAY = MalformedCommand(
    "say", '{"text": "Once upon', "arguments are not complete JSON"
)


# --- incomplete_response -------------------------------------------------------------------------


def test_incomplete_response_rejects_a_cut_call_as_a_reminder_with_a_length_to_stay_under() -> (
    None
):
    retry = incomplete_response(response(("noop", {}), malformed=(CUT_SAY,)))
    assert retry is not None
    assert retry.show_response is False
    # The cut call's 19 characters, and half of that as the limit to stay under.
    assert retry.feedback == (
        "Your call to `say` was not executed: its arguments stopped after 19 characters and are "
        "not complete JSON, most likely because the response was too long and was cut off. Call "
        "it again with much less in it: at most 9 characters of arguments."
    )


def test_incomplete_response_reports_each_distinct_call_once() -> None:
    nameless = MalformedCommand(None, "", "the call has no name")
    retry = incomplete_response(response(malformed=(CUT_SAY, CUT_SAY, nameless)))
    assert retry is not None
    assert retry.feedback.count("Your call to `say`") == 1
    assert retry.feedback.count("Your call to a command") == 1


def test_incomplete_response_accepts_a_whole_response() -> None:
    assert incomplete_response(response(("say", {"text": "hi"}))) is None
    assert incomplete_response(response(text="thinking")) is None


def test_incomplete_response_rejects_a_response_cut_while_calling_a_command() -> None:
    """Some providers return a cut call well-formed, with what arguments they had: the stop reason
    is the only sign. The last call is the cut one; the limit is a quarter of the cap."""
    retry = incomplete_response(
        response(("noop", {}), ("say", {}), truncated=True, output_tokens=60)
    )
    assert retry is not None
    assert retry.show_response is False
    assert retry.feedback == (
        "Your response was cut off at the output length limit while you were writing the call to "
        "`say`, so that call is incomplete and nothing was executed. Respond again with much "
        "less in it. The whole response must fit in about 60 tokens: keep it under 60 characters."
    )


def test_incomplete_response_rejects_a_response_cut_before_the_output_command() -> None:
    retry = incomplete_response(
        response(text="Let me think about", output="say", truncated=True)
    )
    assert retry is not None
    assert retry.show_response is False
    # No reported usage: no number to give.
    assert retry.feedback == (
        "Your response was cut off at the output length limit before you called any command, so "
        "the user heard nothing. Respond again and call `say` (or `noop`) right away. Make it at "
        "most half as long as what you were writing."
    )


def test_incomplete_response_accepts_cut_free_text_when_text_is_the_reply() -> None:
    assert incomplete_response(response(text="Once upon a", truncated=True)) is None


def test_a_malformed_call_takes_precedence_over_the_stop_reason() -> None:
    retry = incomplete_response(
        response(malformed=(CUT_SAY,), truncated=True, output_tokens=60, output="say")
    )
    assert retry is not None
    assert retry.feedback.startswith("Your call to `say` was not executed")
    assert "60 tokens" not in retry.feedback


def test_the_other_guards_leave_a_cut_response_to_incomplete_response() -> None:
    # The cut call's missing argument is not reported as invalid; an earlier call still is.
    assert invalid_args(response(("say", {}), truncated=True)) is None
    retry = invalid_args(response(("say", {}), ("noop", {}), truncated=True))
    assert retry is not None and "'say'" in retry.feedback
    assert invalid_args(response(("say", {}))) is not None
    assert (
        output_command_required(response(text="hm", output="say", truncated=True))
        is None
    )


def test_output_command_required_leaves_a_response_whose_only_call_is_malformed() -> (
    None
):
    assert output_command_required(response(output="say", malformed=(CUT_SAY,))) is None
    assert (
        output_command_required(response(text="hm", output="say", malformed=(CUT_SAY,)))
        is None
    )


# --- unknown_command ---------------------------------------------------------------------------


def test_unknown_command_names_the_call_and_lists_the_bound_commands() -> None:
    retry = unknown_command(response(("ghost", {}), ("say", {"text": "hi"})))
    assert retry is not None
    assert retry.show_response
    assert "'ghost'" in retry.feedback
    assert "noop, raw, say, walk_to" in retry.feedback


def test_unknown_command_accepts_bound_commands_and_no_call() -> None:
    assert unknown_command(response(("say", {"text": "hi"}), ("noop", {}))) is None
    assert unknown_command(response(text="thinking")) is None


# --- invalid_args ------------------------------------------------------------------------------


def test_invalid_args_rejects_a_missing_argument_with_the_validation_error() -> None:
    retry = invalid_args(response(("walk_to", {"room": "kitchen"})))
    assert retry is not None
    assert retry.show_response
    assert "'walk_to'" in retry.feedback
    assert "speed" in retry.feedback


def test_invalid_args_rejects_a_mistyped_argument() -> None:
    retry = invalid_args(response(("walk_to", {"room": "kitchen", "speed": "fast"})))
    assert retry is not None
    assert "speed" in retry.feedback


def test_invalid_args_accepts_valid_and_coercible_arguments() -> None:
    assert invalid_args(response(("walk_to", {"room": "hall", "speed": 2}))) is None
    # Lax validation, the same the tool itself runs on invoke: "2" is an int.
    assert invalid_args(response(("walk_to", {"room": "hall", "speed": "2"}))) is None


def test_invalid_args_skips_unknown_commands_and_raw_json_schemas() -> None:
    assert invalid_args(response(("ghost", {"anything": 1}))) is None
    assert invalid_args(response(("raw", {"x": "not an int"}))) is None


# --- output_command_required -------------------------------------------------------------------


def test_output_command_required_rejects_free_text_and_shows_it() -> None:
    retry = output_command_required(response(text="Hello there", output="say"))
    assert retry is not None
    assert retry.show_response
    assert "`say`" in retry.feedback and "`noop`" in retry.feedback


def test_output_command_required_rejects_an_empty_response_as_a_reminder() -> None:
    retry = output_command_required(response(output="say"))
    assert retry is not None
    assert not retry.show_response


def test_output_command_required_accepts_any_tool_call() -> None:
    assert output_command_required(response(("noop", {}), output="say")) is None
    assert (
        output_command_required(response(("walk_to", {}), text="hm", output="say"))
        is None
    )


def test_output_command_required_accepts_free_text_without_an_output_command() -> None:
    assert output_command_required(response(text="Hello there")) is None


# --- combine -----------------------------------------------------------------------------------


def test_combine_keeps_a_single_rejection_as_is() -> None:
    only = Retry("too long", show_response=False)
    assert combine([only]) == only


def test_combine_lists_every_feedback_and_shows_if_any_asks() -> None:
    combined = combine([Retry("a", show_response=False), Retry("b")])
    assert combined.feedback == "- a\n- b"
    assert combined.show_response
    assert not combine(
        [Retry("a", show_response=False), Retry("b", show_response=False)]
    ).show_response
