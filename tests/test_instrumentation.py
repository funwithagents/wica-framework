"""Tests for src/wica/instrumentation.py — see specs/instrumentation.md."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wica.instrumentation import (
    CommandTrace,
    ReactionTrace,
    TriggerTrace,
    now,
    parent_trigger,
    reaction_latency,
    reactions_per_input,
)

T0 = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _trigger(
    *, arrived: float, written: float | None = None, completion: bool = False
) -> TriggerTrace:
    return TriggerTrace(
        key="k",
        version_id=1,
        written_at=_t(written if written is not None else arrived),
        arrived_at=_t(arrived),
        is_command_completion=completion,
    )


def _reaction(
    *,
    triggers: tuple[TriggerTrace, ...],
    window_opened: float,
    window_closed: float,
    prompt_ready: float | None = None,
    model_started: float | None = None,
    model_ended: float | None = None,
    ended: float,
    command_call_ids: tuple[str, ...] = (),
    text_length: int = 0,
) -> ReactionTrace:
    return ReactionTrace(
        reaction_id=1,
        triggers=triggers,
        window_opened_at=_t(window_opened),
        window_closed_at=_t(window_closed),
        prompt_ready_at=_t(prompt_ready) if prompt_ready is not None else None,
        model_started_at=_t(model_started) if model_started is not None else None,
        model_ended_at=_t(model_ended) if model_ended is not None else None,
        outcome="ok",
        error=None,
        text_length=text_length,
        sink_duration=None,
        command_call_ids=command_call_ids,
        noop=False,
        usage=None,
        ended_at=_t(ended),
        trace_id=None,
        span_id=None,
    )


def test_now_is_timezone_aware_utc():
    assert now().tzinfo is timezone.utc


def test_reaction_properties_are_differences_in_seconds():
    reaction = _reaction(
        triggers=(_trigger(arrived=0.0),),
        window_opened=0.0,
        window_closed=0.2,
        prompt_ready=0.25,
        model_started=0.3,
        model_ended=1.3,
        ended=1.4,
    )
    assert reaction.coalescing_wait == pytest.approx(0.2)
    assert reaction.render_time == pytest.approx(0.05)
    assert reaction.model_latency == pytest.approx(1.0)
    assert reaction.busy_time == pytest.approx(1.2)


def test_parent_trigger_prefers_the_earliest_external_trigger():
    triggers = (
        _trigger(arrived=0.0, completion=True),
        _trigger(arrived=0.1, completion=False),
        _trigger(arrived=0.05, completion=False),
    )
    assert parent_trigger(triggers).arrived_at == _t(0.05)

    all_completions = (
        _trigger(arrived=0.1, completion=True),
        _trigger(arrived=0.0, completion=True),
    )
    assert parent_trigger(all_completions).arrived_at == _t(0.0)


def test_reaction_latency_counts_from_the_earliest_external_write():
    triggers = (
        _trigger(arrived=0.0, written=0.0),
        _trigger(arrived=0.1, written=0.1),
    )
    reaction = _reaction(
        triggers=triggers,
        window_opened=0.0,
        window_closed=0.1,
        ended=1.0,
        command_call_ids=("c1",),
    )
    assert reaction_latency(reaction) == pytest.approx(1.0)

    reaction_no_action = _reaction(
        triggers=triggers,
        window_opened=0.0,
        window_closed=0.1,
        ended=1.0,
        command_call_ids=(),
        text_length=0,
    )
    assert reaction_latency(reaction_no_action) is None

    only_completions = _reaction(
        triggers=(_trigger(arrived=0.0, completion=True),),
        window_opened=0.0,
        window_closed=0.1,
        ended=1.0,
        command_call_ids=("c1",),
    )
    assert reaction_latency(only_completions) is None


def test_reactions_per_input_counts_the_retrigger_chain():
    def reaction(kind: str) -> ReactionTrace:
        completion = kind == "completion"
        return _reaction(
            triggers=(_trigger(arrived=0.0, completion=completion),),
            window_opened=0.0,
            window_closed=0.0,
            ended=0.1,
            command_call_ids=("c",) if not completion else (),
            text_length=1 if completion else 0,
        )

    reactions = [
        reaction("external"),
        reaction("completion"),
        reaction("completion"),
        reaction("external"),
        reaction("completion"),
    ]
    assert reactions_per_input(reactions) == [3, 2]


def test_command_trace_duration():
    trace = CommandTrace(
        call_id="c1",
        name="add",
        reaction_id=1,
        started_at=_t(1.0),
        ended_at=_t(1.5),
        state="complete",
    )
    assert trace.duration == pytest.approx(0.5)
