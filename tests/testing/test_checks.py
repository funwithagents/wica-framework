"""The primer-level checks and measures (specs/benchmark.md, "Primer-level checks") over
hand-built timelines with real `ReactionTrace` / `TriggerTrace` records."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from wica.instrumentation import ReactionTrace, TokenUsage, TriggerTrace
from wica.testing import (
    CHECKS,
    Check,
    CheckContext,
    Report,
    Requires,
    Scenario,
    SutEvent,
    apply_checks,
    build_registry,
    measures,
    resolve_checks,
)
from wica.testing.checks import turns

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _ev(kind: Any, t: float, **fields: Any) -> SutEvent:
    return SutEvent(kind=kind, t=T0.timestamp() + t, **fields)


def _trigger(key: str, version: int, *, completion: bool = False) -> TriggerTrace:
    return TriggerTrace(
        key=key,
        version_id=version,
        written_at=T0,
        arrived_at=T0,
        is_command_completion=completion,
    )


def _reaction(
    reaction_id: int,
    triggers: list[TriggerTrace],
    *,
    outcome: Any = "ok",
    call_ids: tuple[str, ...] = (),
    noop: bool = False,
    error: str | None = None,
    usage: TokenUsage | None = None,
) -> ReactionTrace:
    return ReactionTrace(
        reaction_id=reaction_id,
        triggers=tuple(triggers),
        window_opened_at=_at(0),
        window_closed_at=_at(0.1),
        prompt_ready_at=_at(0.2),
        model_started_at=_at(0.2),
        model_ended_at=_at(0.8),
        outcome=outcome,
        error=error,
        text_length=0,
        sink_duration=None,
        command_call_ids=call_ids,
        noop=noop,
        usage=usage,
        ended_at=_at(1.0),
        trace_id=None,
        span_id=None,
    )


SCENARIO = Scenario(
    name="s", requires=Requires(inputs=("speech_input",), output_command=True)
)
CONTEXT = CheckContext(
    scenario=SCENARIO,
    output_command_name="say",
    inputs=("speech_input", "emotion"),
    triggering_inputs=("speech_input",),
    commands=("dance",),
)


def _answered_turn(version: int, t: float, reaction_id: int) -> list[SutEvent]:
    """A write to speech_input, observed, answered by `say` in one reaction."""
    call = f"c{reaction_id}"
    return [
        _ev("input_written", t, key="speech_input", version_id=version),
        _ev("trigger_observed", t + 0.01, key="speech_input", version_id=version),
        _ev("command_issued", t + 0.5, name="say", call_id=call, args={"text": "hi"}),
        _ev(
            "reaction_ended",
            t + 0.6,
            reaction=_reaction(
                reaction_id, [_trigger("speech_input", version)], call_ids=(call,)
            ),
        ),
    ]


def _check(name: str, timeline: list[SutEvent], context: CheckContext = CONTEXT):
    return CHECKS[name].fn(timeline, context)


# --- answers_each_input ----------------------------------------------------------------------


def test_answers_each_input_passes_when_every_turn_says_something():
    timeline = _answered_turn(1, 0.0, 1) + _answered_turn(2, 5.0, 2)
    finding = _check("answers_each_input", timeline)
    assert finding.ok and "2 turn(s)" in finding.detail


def test_answers_each_input_fails_on_a_turn_without_the_output_command():
    timeline = _answered_turn(1, 0.0, 1) + [
        _ev("input_written", 5.0, key="speech_input", version_id=2),
        _ev("trigger_observed", 5.01, key="speech_input", version_id=2),
        _ev("command_issued", 5.5, name="noop", call_id="n2", args={}),
        _ev(
            "reaction_ended",
            5.6,
            reaction=_reaction(2, [_trigger("speech_input", 2)], noop=True),
        ),
    ]
    finding = _check("answers_each_input", timeline)
    assert not finding.ok
    assert "t=5.000" in finding.detail and "'speech_input'" in finding.detail
    assert finding.event is timeline[4]  # the write that opened the unanswered turn


def test_a_dropped_write_opens_no_turn():
    timeline = _answered_turn(1, 0.0, 1)
    timeline[
        3:3
    ] = [  # a second write while the step runs: dropped, no reaction sees it
        _ev("input_written", 0.2, key="speech_input", version_id=2),
        _ev("trigger_dropped", 0.21, key="speech_input", version_id=2),
    ]
    assert len(turns(timeline, ("speech_input",))) == 1
    assert _check("answers_each_input", timeline).ok


def test_coalesced_writes_form_one_turn():
    timeline = [
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev("trigger_observed", 0.01, key="speech_input", version_id=1),
        _ev("input_written", 0.05, key="speech_input", version_id=2),
        _ev("trigger_observed", 0.06, key="speech_input", version_id=2),
        _ev("command_issued", 0.5, name="say", call_id="c1", args={"text": "hi"}),
        _ev(
            "reaction_ended",
            0.6,
            reaction=_reaction(
                1,
                [_trigger("speech_input", 1), _trigger("speech_input", 2)],
                call_ids=("c1",),
            ),
        ),
    ]
    only = turns(timeline, ("speech_input",))
    assert [(t.start, t.end) for t in only] == [(0, len(timeline))]
    assert _check("answers_each_input", timeline).ok


def test_turns_open_only_on_the_scenario_inputs_that_trigger():
    # `emotion` is a target input but neither triggering nor in the scenario's requires.
    timeline = _answered_turn(1, 0.0, 1) + [
        _ev("input_written", 3.0, key="emotion", version_id=1),
    ]
    assert CONTEXT.turn_inputs == ("speech_input",)
    assert _check("answers_each_input", timeline).ok


# --- no_noop_on_input -------------------------------------------------------------------------


def test_no_noop_on_input_fails_when_a_scenario_input_gets_noop_only():
    timeline = [
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev("command_issued", 0.5, name="noop", call_id="n1", args={}),
        _ev(
            "reaction_ended",
            0.6,
            reaction=_reaction(1, [_trigger("speech_input", 1)], noop=True),
        ),
    ]
    finding = _check("no_noop_on_input", timeline)
    assert not finding.ok and "noop only" in finding.detail
    assert finding.event is timeline[2]


def test_no_noop_on_input_allows_noop_on_a_completion_or_a_passive_entry():
    timeline = [
        _ev(
            "reaction_ended",
            1.0,
            reaction=_reaction(
                1, [_trigger("agent:command:c1", 1, completion=True)], noop=True
            ),
        ),
        _ev(
            "reaction_ended",
            2.0,
            reaction=_reaction(2, [_trigger("emotion", 1)], noop=True),
        ),
        # noop alongside a real Command is not "noop only"
        _ev(
            "reaction_ended",
            3.0,
            reaction=_reaction(
                3, [_trigger("speech_input", 1)], noop=True, call_ids=("c3",)
            ),
        ),
    ]
    assert _check("no_noop_on_input", timeline).ok


# --- no_failed_reactions ----------------------------------------------------------------------


def test_no_failed_reactions_cites_the_outcome_and_error():
    timeline = _answered_turn(1, 0.0, 1) + [
        _ev(
            "reaction_ended",
            2.0,
            reaction=_reaction(2, [], outcome="model_error", error="429 rate limited"),
        ),
    ]
    finding = _check("no_failed_reactions", timeline)
    assert not finding.ok
    assert "'model_error'" in finding.detail and "429 rate limited" in finding.detail
    assert _check("no_failed_reactions", _answered_turn(1, 0.0, 1)).ok


# --- no_colliding_cancel ----------------------------------------------------------------------


def _cancel_reaction(other: str | None) -> list[SutEvent]:
    calls = ["k1"] + (["o1"] if other else [])
    events = [
        _ev("command_issued", 0.5, name="cancel_command", call_id="k1", args={}),
    ]
    if other:
        events.append(_ev("command_issued", 0.5, name=other, call_id="o1", args={}))
    events.append(
        _ev(
            "reaction_ended",
            0.6,
            reaction=_reaction(1, [_trigger("speech_input", 1)], call_ids=tuple(calls)),
        )
    )
    return events


def test_no_colliding_cancel_fails_only_when_cancel_shares_its_reaction():
    collided = _check("no_colliding_cancel", _cancel_reaction("say"))
    assert not collided.ok and "['say']" in collided.detail
    assert _check("no_colliding_cancel", _cancel_reaction(None)).ok


# --- measures ---------------------------------------------------------------------------------


def test_measures_count_outputs_chained_replies_noops_failures_and_words():
    timeline = _answered_turn(1, 0.0, 1) + [
        # the chained reply: a reaction woken only by say's completion says again
        _ev(
            "command_issued",
            1.5,
            name="say",
            call_id="c2",
            args={"text": "and one more thing"},
        ),
        _ev(
            "reaction_ended",
            1.6,
            reaction=_reaction(
                2, [_trigger("agent:command:c1", 1, completion=True)], call_ids=("c2",)
            ),
        ),
        _ev(
            "reaction_ended",
            2.6,
            reaction=_reaction(
                3, [_trigger("agent:command:c2", 1, completion=True)], noop=True
            ),
        ),
        _ev("reaction_ended", 3.0, reaction=_reaction(4, [], outcome="empty")),
    ]
    assert measures(timeline, CONTEXT) == {
        "output_calls": 2.0,
        "unprompted_output": 1.0,
        "noop_steps": 1.0,
        "failed_reactions": 1.0,
        "output_words": 5.0,  # "hi" + "and one more thing"
        "turns": 1.0,
        "turns_answered": 1.0,
        "noops_on_input": 0.0,
    }


def test_turn_measures_count_unanswered_turns_and_noops_on_input():
    timeline = _answered_turn(1, 0.0, 1) + [
        _ev("input_written", 5.0, key="speech_input", version_id=2),
        _ev("trigger_observed", 5.01, key="speech_input", version_id=2),
        _ev("command_issued", 5.5, name="noop", call_id="n2", args={}),
        _ev(
            "reaction_ended",
            5.6,
            reaction=_reaction(2, [_trigger("speech_input", 2)], noop=True),
        ),
    ]
    m = measures(timeline, CONTEXT)
    assert (m["turns"], m["turns_answered"], m["noops_on_input"]) == (2.0, 1.0, 1.0)


# --- registry, resolution, application --------------------------------------------------------


def _always_ok(timeline: list[SutEvent], context: CheckContext):
    from wica.testing import Finding

    return Finding(expectation="always_ok", ok=True, detail="", t=0.0)


def test_build_registry_rejects_a_clash_with_a_built_in():
    with pytest.raises(ValueError, match="already registered"):
        build_registry([Check("answers_each_input", _always_ok)])
    registry = build_registry([Check("always_ok", _always_ok)])
    assert set(registry) == set(CHECKS) | {"always_ok"}


def test_resolve_checks_by_name_or_by_what_the_target_satisfies():
    assert [
        c.name for c in resolve_checks(["no_failed_reactions"], CHECKS, CONTEXT)
    ] == ["no_failed_reactions"]
    with pytest.raises(ValueError, match="unknown check 'nope'"):
        resolve_checks(["nope"], CHECKS, CONTEXT)
    # No output Command ⇒ answers_each_input (requires one) is left out of the default set.
    without_output = CheckContext(
        scenario=SCENARIO, output_command_name=None, inputs=("speech_input",)
    )
    assert {c.name for c in resolve_checks(None, CHECKS, without_output)} == set(
        CHECKS
    ) - {"answers_each_input"}


def test_apply_checks_appends_one_finding_per_check():
    timeline = _answered_turn(1, 0.0, 1)
    report = Report(
        scenario="s",
        sut_label="fake",
        tester_label=None,
        started_at="",
        finished_at="",
        timeline=timeline,
        metrics={},
        dropped_inputs=0,
        findings=[],
        issues=[],
        notes=[],
        verdict="pass",
        summary=None,
    )
    checked = apply_checks(report, ["answers_each_input", "no_noop_on_input"], CONTEXT)
    assert [f.expectation for f in checked.findings] == [
        "answers_each_input",
        "no_noop_on_input",
    ]
    assert checked.ok and report.findings == []  # the input report is untouched
