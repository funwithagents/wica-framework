"""The default response guards on their own, over hand-built responses. How the Agent runs them and
retries is in test_agent.py ("Response guards"). See specs/agent.md ("Response guards")."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from wica.command import Command
from wica.guards import (
    ModelResponse,
    ProposedCommand,
    Retry,
    combine,
    invalid_args,
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
) -> ModelResponse:
    return ModelResponse(
        text=text,
        commands=tuple(ProposedCommand(name, args) for name, args in calls),
        bound=BOUND,
        output_command_name=output,
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
