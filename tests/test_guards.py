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


async def write_note(text: str) -> str:
    """Write a note."""
    return text


async def wave() -> str:
    """Wave a hand."""
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
    "write_note": Command(write_note),
    "wave": Command(wave),
    "raw": Command(_RawSchemaTool()),
}


def response(
    *calls: tuple[str, dict[str, Any]],
    text: str = "",
    malformed: tuple[MalformedCommand, ...] = (),
    truncated: bool = False,
    output_tokens: int | None = None,
    output_command_name: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        text=text,
        commands=tuple(ProposedCommand(name, args) for name, args in calls),
        bound=BOUND,
        malformed=malformed,
        truncated=truncated,
        output_tokens=output_tokens,
        output_command_name=output_command_name,
    )


CUT_NOTE = MalformedCommand(
    "write_note", '{"text": "Once upon', "arguments are not complete JSON"
)


# --- incomplete_response -------------------------------------------------------------------------


def test_incomplete_response_rejects_a_cut_call_as_a_reminder_with_a_length_to_stay_under() -> (
    None
):
    retry = incomplete_response(response(("wave", {}), malformed=(CUT_NOTE,)))
    assert retry is not None
    assert retry.show_response is False
    # The cut call's 19 characters, and half of that as the limit to stay under.
    assert retry.feedback == (
        "Your call to `write_note` was not executed: its arguments stopped after 19 characters and are "
        "not complete JSON, most likely because the response was too long and was cut off. Call "
        "it again with much less in it: at most 9 characters of arguments."
    )


def test_incomplete_response_reports_each_distinct_call_once() -> None:
    nameless = MalformedCommand(None, "", "the call has no name")
    retry = incomplete_response(response(malformed=(CUT_NOTE, CUT_NOTE, nameless)))
    assert retry is not None
    assert retry.feedback.count("Your call to `write_note`") == 1
    assert retry.feedback.count("Your call to a command") == 1


def test_incomplete_response_accepts_a_whole_response() -> None:
    assert incomplete_response(response(("write_note", {"text": "hi"}))) is None
    assert incomplete_response(response(text="thinking")) is None


def test_incomplete_response_rejects_a_response_cut_while_calling_a_command() -> None:
    """Some providers return a cut call well-formed, with what arguments they had: the stop reason
    is the only sign. The last call is the cut one; the limit is a quarter of the cap."""
    retry = incomplete_response(
        response(("wave", {}), ("write_note", {}), truncated=True, output_tokens=60)
    )
    assert retry is not None
    assert retry.show_response is False
    assert retry.feedback == (
        "Your response was cut off at the output length limit while you were writing the call to "
        "`write_note`, so that call is incomplete and nothing was executed. Respond again with much "
        "less in it. The whole response must fit in about 60 tokens: keep it under 60 characters."
    )


def test_incomplete_response_accepts_text_cut_at_the_cap() -> None:
    assert incomplete_response(response(text="Once upon a", truncated=True)) is None


def test_a_malformed_call_takes_precedence_over_the_stop_reason() -> None:
    retry = incomplete_response(
        response(malformed=(CUT_NOTE,), truncated=True, output_tokens=60)
    )
    assert retry is not None
    assert retry.feedback.startswith("Your call to `write_note` was not executed")
    assert "60 tokens" not in retry.feedback


def test_the_other_guards_leave_a_cut_response_to_incomplete_response() -> None:
    # The cut call's missing argument is not reported as invalid; an earlier call still is.
    assert invalid_args(response(("write_note", {}), truncated=True)) is None
    retry = invalid_args(response(("write_note", {}), ("wave", {}), truncated=True))
    assert retry is not None and "'write_note'" in retry.feedback
    assert invalid_args(response(("write_note", {}))) is not None


# --- unknown_command ---------------------------------------------------------------------------


def test_unknown_command_names_the_call_and_lists_the_bound_commands() -> None:
    retry = unknown_command(response(("ghost", {}), ("write_note", {"text": "hi"})))
    assert retry is not None
    assert retry.show_response
    assert "'ghost'" in retry.feedback
    assert "raw, walk_to, wave, write_note" in retry.feedback


def test_unknown_command_accepts_bound_commands_and_no_call() -> None:
    assert (
        unknown_command(response(("write_note", {"text": "hi"}), ("wave", {}))) is None
    )
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


# --- output_command_required (tool mode only) ----------------------------------------------------


def test_output_command_required_is_inactive_in_tag_mode():
    assert output_command_required(response(text="Hello there!")) is None


def test_output_command_required_rejects_free_text_when_the_output_is_a_tool():
    retry = output_command_required(
        response(text="Hello there!", output_command_name="say")
    )
    assert retry is not None and retry.show_response
    assert "Call `say` to reply to the user, or `noop` to do nothing." in retry.feedback
    # An empty response cannot be shown: a reminder.
    empty = output_command_required(response(output_command_name="say"))
    assert empty is not None and not empty.show_response


def test_output_command_required_accepts_any_call_and_leaves_a_cut_response_alone():
    assert (
        output_command_required(response(("wave", {}), output_command_name="say"))
        is None
    )
    assert (
        output_command_required(response(truncated=True, output_command_name="say"))
        is None
    )
    cut = incomplete_response(
        response(truncated=True, output_tokens=40, output_command_name="say")
    )
    assert cut is not None and "call `say` (or `noop`) right away" in cut.feedback
    assert incomplete_response(response(truncated=True, output_tokens=40)) is None
