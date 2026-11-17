"""The checks and measures (specs/wica-tester.md, "Scenarios") over hand-built timelines with real
`ReactionTrace` / `TriggerTrace` records: each built-in kept and broken, the windows they judge,
and the registry. In these timelines an output is a `command_issued` whose `is_output` is set (named
`say`, the output Command, unless a test says otherwise), listed in its reaction's
`command_call_ids`, and a declined reaction a record whose `noop` flag is set — there is no `noop`
Command."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from wica.instrumentation import CommandTrace, ReactionTrace, TokenUsage, TriggerTrace
from wica.testing import (
    answer_latencies,
    CHECKS,
    Check,
    CheckContext,
    CheckSpec,
    Finding,
    SutEvent,
    build_registry,
    input_windows,
    measures,
    run_check,
)
from wica.testing.checks import command_arg, turns

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
    first_output_at: datetime | None = None,
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
        command_call_ids=call_ids,
        noop=noop,
        usage=usage,
        ended_at=_at(1.0),
        trace_id=None,
        span_id=None,
        first_output_at=first_output_at,
    )


CONTEXT = CheckContext(
    output_command_name="say",
    inputs=("speech_input", "emotion"),
    triggering_inputs=("speech_input",),
    commands=("dance",),
    scenario_inputs=("speech_input",),
    origin=T0.timestamp(),
)


def _set(t: float, key: str, value: Any) -> SutEvent:
    return _ev("action", t, name="set", args={"key": key, "value": value})


def _issued(t: float, name: str, cid: str, /, **args: Any) -> SutEvent:
    return _ev("command_issued", t, name=name, call_id=cid, args=args)


def _said(t: float, cid: str, /, **args: Any) -> SutEvent:
    """An output: a `command_issued` the framework reports with `is_output` set."""
    return _ev("command_issued", t, name="say", call_id=cid, args=args, is_output=True)


def _state(
    t: float, name: str, cid: str, state: str, *, output: bool = False
) -> SutEvent:
    return _ev(
        "command_state", t, name=name, call_id=cid, state=state, is_output=output
    )


def _answered_turn(version: int, t: float, reaction_id: int) -> list[SutEvent]:
    """The tester writes speech_input, the Agent observes it and answers through `say`."""
    call = f"c{reaction_id}"
    return [
        _set(t, "speech_input", f"utterance {version}"),
        _ev("input_written", t, key="speech_input", version_id=version),
        _ev("trigger_observed", t + 0.01, key="speech_input", version_id=version),
        _said(t + 0.5, call, text="hi"),
        _ev(
            "reaction_ended",
            t + 0.6,
            reaction=_reaction(
                reaction_id,
                [_trigger("speech_input", version)],
                call_ids=(call,),
            ),
        ),
    ]


def _check(name: str, window: list[SutEvent], arg: Any = None) -> Finding:
    return run_check(CHECKS[name], CheckSpec(name, arg), window, CONTEXT)


# --- answers_each_input ----------------------------------------------------------------------


def test_answers_each_input_kept_when_every_turn_says_something():
    timeline = _answered_turn(1, 0.0, 1) + _answered_turn(2, 5.0, 2)
    finding = _check("answers_each_input", timeline)
    assert finding.ok and "2 turn(s)" in finding.detail and finding.kind == "check"


def _unanswered_turn(*during: SutEvent, noop: bool = False) -> list[SutEvent]:
    """A second turn, opened at t=5, in which the SUT reports no output; *during* is what it did
    instead."""
    call_ids = tuple(
        e.call_id for e in during if e.kind == "command_issued" and e.call_id
    )
    return [
        _set(5.0, "speech_input", "hello?"),
        _ev("input_written", 5.0, key="speech_input", version_id=2),
        _ev("trigger_observed", 5.01, key="speech_input", version_id=2),
        *during,
        _ev(
            "reaction_ended",
            5.6,
            reaction=_reaction(
                2, [_trigger("speech_input", 2)], noop=noop, call_ids=call_ids
            ),
        ),
    ]


@pytest.mark.parametrize(
    "unanswered",
    [
        _unanswered_turn(noop=True),  # declined
        _unanswered_turn(_ev("private_text", 5.5, text="I should say hello.")),
        _unanswered_turn(_issued(5.5, "dance", "d2")),  # acted, said nothing
    ],
    ids=["declined", "private text only", "a Command only"],
)
def test_answers_each_input_broken_by_a_turn_without_an_output(
    unanswered: list[SutEvent],
):
    timeline = _answered_turn(1, 0.0, 1) + unanswered
    finding = _check("answers_each_input", timeline)
    assert not finding.ok
    assert "t=5.000" in finding.detail and "'speech_input'" in finding.detail
    assert finding.event is unanswered[1]  # the write that opened the unanswered turn


def test_a_dropped_write_opens_no_turn_and_coalesced_writes_open_one():
    timeline = _answered_turn(1, 0.0, 1)
    timeline[4:4] = [  # a second write while the step runs: dropped
        _ev("input_written", 0.2, key="speech_input", version_id=2),
        _ev("trigger_dropped", 0.21, key="speech_input", version_id=2),
    ]
    assert len(turns(timeline, ("speech_input",))) == 1
    assert _check("answers_each_input", timeline).ok
    coalesced = [
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev("input_written", 0.05, key="speech_input", version_id=2),
        _said(0.5, "c1", text="hi"),
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
    assert [(t.start, t.end) for t in turns(coalesced, ("speech_input",))] == [(0, 4)]


# --- no_noop_on_input, no_failed_reactions ------------------------------------------------------


def test_no_noop_on_input_reads_the_noop_flag_of_a_reaction_to_a_scenario_input():
    declined = _ev(
        "reaction_ended",
        0.6,
        reaction=_reaction(1, [_trigger("speech_input", 1)], noop=True),
    )
    finding = _check("no_noop_on_input", [declined])
    assert not finding.ok and finding.event is declined
    assert "reaction 1" in finding.detail and "'speech_input'" in finding.detail
    # An empty response declines just the same: the flag is set, whatever the outcome.
    empty = _ev(
        "reaction_ended",
        0.6,
        reaction=_reaction(
            1, [_trigger("speech_input", 1)], noop=True, outcome="empty"
        ),
    )
    assert not _check("no_noop_on_input", [empty]).ok
    allowed = [
        # declining a completion or a passive entry is fine; so is answering the input (the flag
        # is what is read, never a Command's name)
        _ev(
            "reaction_ended",
            1.0,
            reaction=_reaction(
                2, [_trigger("agent:command:c1", 1, completion=True)], noop=True
            ),
        ),
        _ev(
            "reaction_ended",
            2.0,
            reaction=_reaction(3, [_trigger("emotion", 1)], noop=True),
        ),
        _said(2.9, "c4", text="hi"),
        _ev(
            "reaction_ended",
            3.0,
            reaction=_reaction(4, [_trigger("speech_input", 2)], call_ids=("c4",)),
        ),
    ]
    assert _check("no_noop_on_input", allowed).ok


def test_no_failed_reactions_cites_the_outcome_and_error():
    failed = _answered_turn(1, 0.0, 1) + [
        _ev(
            "reaction_ended",
            2.0,
            reaction=_reaction(2, [], outcome="model_error", error="429 rate limited"),
        ),
    ]
    finding = _check("no_failed_reactions", failed)
    assert not finding.ok
    assert "'model_error'" in finding.detail and "429 rate limited" in finding.detail
    assert _check("no_failed_reactions", _answered_turn(1, 0.0, 1)).ok


def test_no_failed_reactions_passes_an_empty_response_a_decline_not_a_failure():
    empty = _ev(
        "reaction_ended",
        0.6,
        reaction=_reaction(
            1, [_trigger("speech_input", 1)], outcome="empty", noop=True
        ),
    )
    kept = _check("no_failed_reactions", [empty])
    assert kept.ok and "none failed" in kept.detail
    for outcome in ("model_timeout", "rejected", "cancelled"):
        broken = _ev("reaction_ended", 1.0, reaction=_reaction(2, [], outcome=outcome))
        finding = _check("no_failed_reactions", [empty, broken])
        assert not finding.ok and finding.event is broken


def test_no_noop_on_input_catches_a_decline_with_private_text():
    """`reasoning… <noop/>`: the model wrote private notes and declined. Its record's `noop` flag
    is set, so the check catches it whatever text came with it."""
    window = [
        _set(0.0, "speech_input", "Hello?"),
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev("private_text", 0.5, text="The user is probably not talking to me."),
        _ev(
            "reaction_ended",
            0.6,
            reaction=_reaction(1, [_trigger("speech_input", 1)], noop=True),
        ),
    ]
    finding = _check("no_noop_on_input", window)
    assert not finding.ok and finding.event is window[-1]
    assert measures(window, CONTEXT)["noops_on_input"] == 1.0


def test_an_output_is_read_from_is_output_not_from_its_name():
    """The checks read the field the framework reports: an output whose Command is named unlike
    the context's is still an answer, and a tool call that happens to share the name is not."""
    talk = dataclasses.replace(_said(0.5, "c1", text="hi"), name="talk")
    answered = _answered_turn(1, 0.0, 1)
    answered[3] = talk
    assert _check("answers_each_input", answered).ok
    assert not _check("quiet_on_background_change", answered[:1] + [talk]).ok
    tool_named_say = _answered_turn(1, 0.0, 1)
    tool_named_say[3] = _issued(0.5, "say", "c1", text="hi")
    assert not _check("answers_each_input", tool_named_say).ok
    assert _check("quiet_on_background_change", tool_named_say[3:4]).ok


# --- no_colliding_cancel ----------------------------------------------------------------------


def _cancel_reaction(
    target: str, *others: str, target_id: str = "d1"
) -> list[SutEvent]:
    """`target` running as d1, then one reaction cancelling *target_id* beside *others* (a `say`
    is an output)."""
    calls = ["k1"] + [f"o{i}" for i, _ in enumerate(others)]
    events = [
        _state(0.1, target, "d1", "running", output=target == "say"),
        _issued(0.5, "cancel_command", "k1", call_id=target_id),
    ]
    events += [
        _said(0.5, f"o{i}") if other == "say" else _issued(0.5, other, f"o{i}")
        for i, other in enumerate(others)
    ]
    outputs = tuple(f"o{i}" for i, other in enumerate(others) if other == "say")
    events.append(
        _ev(
            "reaction_ended",
            0.6,
            reaction=_reaction(
                1,
                [_trigger("speech_input", 1)],
                call_ids=(*outputs, *(c for c in calls if c not in outputs)),
            ),
        )
    )
    return events


def test_no_colliding_cancel_broken_only_by_a_new_call_of_the_cancelled_command():
    collided = _check("no_colliding_cancel", _cancel_reaction("dance", "dance"))
    assert (
        not collided.ok and "cancelled dance and issued dance again" in collided.detail
    )
    assert _check("no_colliding_cancel", _cancel_reaction("dance")).ok
    # A harmless action beside the cancel races with nothing; cutting off a speech to say "okay,
    # stopping" is what a person does: neither collides.
    assert _check("no_colliding_cancel", _cancel_reaction("dance", "set_emotion")).ok
    assert _check("no_colliding_cancel", _cancel_reaction("dance", "say")).ok
    # A cancel whose target is an output never collides: the check reads tool calls only, and a
    # response that speaks replaces the running output anyway.
    assert _check("no_colliding_cancel", _cancel_reaction("say", "say")).ok
    assert _check("no_colliding_cancel", _cancel_reaction("say", "say", "dance")).ok


def test_no_colliding_cancel_resolves_a_prefixed_target_and_skips_an_unknown_one():
    prefixed = _cancel_reaction("dance", "dance", target_id="agent:command:d1")
    assert not _check("no_colliding_cancel", prefixed).ok
    unknown = _cancel_reaction("dance", "dance", target_id="toolu_gone")
    assert _check("no_colliding_cancel", unknown).ok


# --- the step checks: judged over the window of their input -------------------------------------


def test_cancels_when_asked_kept_only_when_the_cancel_is_in_the_window():
    window = [_set(2.0, "speech_input", "Stop!"), _issued(2.5, "cancel_command", "k1")]
    kept = _check("cancels_when_asked", window)
    assert kept.ok and "t=2.500" in kept.detail
    broken = _check("cancels_when_asked", window[:1])
    assert not broken.ok and "never issued" in broken.detail


def test_cancels_when_asked_kept_by_speaking_over_the_running_output():
    """Speaking again cuts off the output that was running when the input was written, with no
    `cancel_command`: a correct way to stop talking."""
    before = [
        _state(0.4, "say", "s1", "running", output=True),
        _said(0.5, "s1", text="Once upon a time"),
    ]
    window = [
        _set(2.0, "speech_input", "Stop!"),
        _state(
            2.4, "say", "s1", "cancelled", output=True
        ),  # cut off by the new utterance
        _state(2.5, "say", "s2", "running", output=True),
        _said(2.6, "s2", text="Okay, stopping."),
    ]
    timeline = before + window
    assert input_windows(timeline) == [(2, 6)]
    kept = _check("cancels_when_asked", window)
    assert kept.ok and kept.event is window[1]
    assert "say (s1) cut off at t=2.400" in kept.detail


def test_cancels_when_asked_is_not_kept_by_what_only_looks_like_a_stop():
    stop = _set(2.0, "speech_input", "Stop!")
    # The running output finished by itself, and the SUT just spoke again.
    finished = [
        stop,
        _state(2.4, "say", "s1", "complete", output=True),
        _state(2.5, "say", "s2", "running", output=True),
        _said(2.6, "s2", text="And then…"),
    ]
    assert not _check("cancels_when_asked", finished).ok
    # An output said in the window and cut off in it was not running when the window opened.
    own = [
        stop,
        _state(2.5, "say", "s2", "running", output=True),
        _said(2.6, "s2", text="Okay"),
        _state(2.9, "say", "s2", "cancelled", output=True),
    ]
    assert not _check("cancels_when_asked", own).ok
    # For any other Command only cancel_command counts: a dance that ended `cancelled` by
    # itself (its own timeout) is not the SUT stopping it.
    dance = [stop, _state(2.4, "dance", "d1", "cancelled")]
    broken = _check("cancels_when_asked", dance)
    assert not broken.ok and "no running output was cut off" in broken.detail


def test_acts_when_asked_matches_the_command_and_its_args():
    window = [
        _set(3.0, "closest_user", "u2"),
        _issued(4.0, "switch_user_tracking", "s1", user_id="u2"),
    ]
    assert _check("acts_when_asked", window, "switch_user_tracking").ok
    assert _check(
        "acts_when_asked", window, {"switch_user_tracking": {"user_id": "u2"}}
    ).ok
    wrong = _check(
        "acts_when_asked", window, {"switch_user_tracking": {"user_id": "u9"}}
    )
    assert not wrong.ok and "switch_user_tracking({'user_id': 'u9'})" in wrong.detail
    assert not _check("acts_when_asked", window, "dance").ok


def test_command_arg_accepts_a_name_or_a_single_command_with_args():
    assert command_arg("dance") == ("dance", {})
    assert command_arg({"track": {"user_id": "u2"}}) == ("track", {"user_id": "u2"})
    for bad in (None, {"a": {}, "b": {}}, {"a": "u2"}, 3):
        with pytest.raises(ValueError):
            command_arg(bad)


def test_quiet_on_background_change_broken_when_the_robot_comments_on_it():
    quiet = [
        _set(1.5, "emotion", "happy"),
        _ev(
            "reaction_ended",
            3.0,
            reaction=_reaction(1, [_trigger("emotion", 1)], noop=True),
        ),
    ]
    assert _check("quiet_on_background_change", quiet).ok
    # Private text and a Command issued as a tool call are not speech.
    busy_but_silent = [
        _set(1.5, "emotion", "happy"),
        _ev("private_text", 2.9, text="The user looks happy; nothing to say."),
        _issued(3.0, "set_emotion", "e1", emotion="happy"),
    ]
    assert _check("quiet_on_background_change", busy_but_silent).ok
    chatty = [
        _set(1.5, "emotion", "happy"),
        _said(3.0, "c2", text="I feel happy!"),
    ]
    finding = _check("quiet_on_background_change", chatty)
    assert not finding.ok and "say output at t=3.000" in finding.detail
    assert finding.event is chatty[1]


def test_input_windows_run_from_one_input_to_the_next():
    timeline = [
        _set(0.0, "speech_input", "Hi"),
        _said(0.5, "c1", text="hello"),
        _set(2.0, "speech_input", "Stop!"),
        _issued(2.5, "cancel_command", "k1"),
    ]
    assert input_windows(timeline) == [(0, 2), (2, 4)]
    # The cancel belongs to "Stop!"'s window, not to the greeting's.
    first, second = (timeline[a:b] for a, b in input_windows(timeline))
    assert not _check("cancels_when_asked", first).ok
    assert _check("cancels_when_asked", second).ok


def test_run_check_prefixes_the_window_label_to_the_detail():
    window = [_set(2.0, "speech_input", "Stop!"), _issued(2.5, "cancel_command", "k1")]
    finding = run_check(
        CHECKS["cancels_when_asked"],
        CheckSpec("cancels_when_asked"),
        window,
        CONTEXT,
        label="on input 4 (speech_input='Stop!')",
    )
    assert finding.detail.startswith("on input 4 (speech_input='Stop!'): ")


# --- measures ---------------------------------------------------------------------------------


def test_measures_count_outputs_extra_replies_noops_failures_words_and_turns():
    timeline = _answered_turn(1, 0.0, 1) + [
        # the chained reply: a reaction woken only by say's completion says again (words are
        # counted in the text, not in the attributes)
        _said(1.5, "c2", text="and one more thing", emotion="happy"),
        _ev(
            "reaction_ended",
            1.6,
            reaction=_reaction(
                2,
                [_trigger("agent:command:c1", 1, completion=True)],
                call_ids=("c2",),
            ),
        ),
        _ev(
            "reaction_ended",
            2.6,
            reaction=_reaction(
                3, [_trigger("agent:command:c2", 1, completion=True)], noop=True
            ),
        ),
        # an empty response is a decline (noop), not a failure; a model error is one
        _ev(
            "reaction_ended",
            3.0,
            reaction=_reaction(4, [], outcome="empty", noop=True),
        ),
        _ev(
            "reaction_ended",
            3.5,
            reaction=_reaction(5, [], outcome="model_error", error="boom"),
        ),
    ]
    assert measures(timeline, CONTEXT) == {
        "output_calls": 2.0,
        "extra_output": 1.0,  # a second reply in the same turn
        "noop_steps": 2.0,
        "failed_reactions": 1.0,
        "output_words": 5.0,  # "hi" + "and one more thing"
        "turns": 1.0,
        "turns_answered": 1.0,
        "noops_on_input": 0.0,
    }


def test_measures_count_a_declined_input_from_the_noop_flag():
    timeline = _answered_turn(1, 0.0, 1) + _unanswered_turn(noop=True)
    m = measures(timeline, CONTEXT)
    assert m["noop_steps"] == 1.0 and m["noops_on_input"] == 1.0
    assert m["turns"] == 2.0 and m["turns_answered"] == 1.0 and m["output_calls"] == 1.0
    # A turn answered with private text only is unanswered, but it is not a noop.
    silent = _answered_turn(1, 0.0, 1) + _unanswered_turn(
        _ev("private_text", 5.5, text="hm")
    )
    m = measures(silent, CONTEXT)
    assert m["noop_steps"] == 0.0 and m["noops_on_input"] == 0.0
    assert m["turns_answered"] == 1.0


def _reply(
    t: float, reaction_id: int, trigger: TriggerTrace, *texts: str
) -> list[SutEvent]:
    """One reaction woken by *trigger*, speaking once per text: one reply, as many tags as
    texts."""
    calls = tuple(f"r{reaction_id}-{n}" for n in range(len(texts)))
    return [
        *(_said(t, call, text=text) for call, text in zip(calls, texts, strict=True)),
        _ev(
            "reaction_ended",
            t + 0.1,
            reaction=_reaction(
                reaction_id,
                [trigger],
                call_ids=calls,
            ),
        ),
    ]


def test_extra_output_counts_replies_not_tags():
    """Two tags in one response are one reply: the turn's answer, with nothing extra. A second
    reaction speaking again in the same turn is the extra reply, however many tags each holds."""
    opening = _answered_turn(1, 0.0, 1)[:3]
    two_tags = opening + _reply(
        0.5, 1, _trigger("speech_input", 1), "Good news!", "But there is a catch."
    )
    m = measures(two_tags, CONTEXT)
    assert m["output_calls"] == 2.0 and m["extra_output"] == 0.0
    assert m["turns_answered"] == 1.0
    woken = _trigger("agent:command:r1-1", 1, completion=True)  # a <next> note's wake
    two_replies = two_tags + _reply(1.5, 2, woken, "One more thing.", "And another.")
    m = measures(two_replies, CONTEXT)
    assert m["output_calls"] == 4.0 and m["extra_output"] == 1.0


def _split_answer(version: int, t: float) -> list[SutEvent]:
    """A turn answered in two steps: an emotion change first, `say` on the step its completion
    wakes."""
    return [
        _set(t, "speech_input", f"utterance {version}"),
        _ev("input_written", t, key="speech_input", version_id=version),
        _ev("trigger_observed", t + 0.01, key="speech_input", version_id=version),
        _issued(t + 0.5, "set_emotion", f"e{version}", emotion="happy"),
        _ev(
            "reaction_ended",
            t + 0.5,
            reaction=_reaction(
                version, [_trigger("speech_input", version)], call_ids=(f"e{version}",)
            ),
        ),
        _said(t + 1.2, f"s{version}", text="hello there"),
        _ev(
            "reaction_ended",
            t + 1.2,
            reaction=_reaction(
                version + 100,
                [_trigger(f"agent:command:e{version}", 1, completion=True)],
                call_ids=(f"s{version}",),
            ),
        ),
    ]


def test_an_answer_over_two_steps_is_one_answer_timed_to_its_first_word():
    timeline = _split_answer(1, 0.0) + _answered_turn(2, 5.0, 2)
    m = measures(timeline, CONTEXT)
    assert m["turns_answered"] == 2.0 and m["extra_output"] == 0.0
    assert _check("answers_each_input", timeline).ok
    assert _check("no_noop_on_input", timeline).ok
    # the split turn's first word is 1.2 s after the input, the plain turn's 0.5 s
    assert answer_latencies(timeline, CONTEXT) == pytest.approx([1.2, 0.5])


def _ended(t: float, cid: str, started: float, *, output: bool = True) -> SutEvent:
    """A Command's end record, carrying when it started (for an output: its first words)."""
    trace = CommandTrace(
        call_id=cid,
        name="say" if output else "dance",
        reaction_id=1,
        started_at=_at(started),
        ended_at=_at(t),
        state="complete",
        is_output=output,
    )
    return _ev("command_ended", t, name=trace.name, call_id=cid, command=trace)


def test_answer_latency_ends_at_the_first_words_of_the_output_not_at_its_report():
    """An output is reported once its body is complete; its first words were heard earlier, when
    it started — its own `CommandTrace.started_at`."""
    opened = T0.timestamp()
    timeline = [
        _set(0.0, "speech_input", "tell me a story"),
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev("trigger_observed", 0.01, key="speech_input", version_id=1),
        _said(3.0, "c1", text="Once upon a time, a very long story."),
        _ev(
            "reaction_ended",
            3.0,
            reaction=_reaction(1, [_trigger("speech_input", 1)], call_ids=("c1",)),
        ),
        _ended(9.0, "c1", started=0.7),
    ]
    assert timeline[1].t == opened
    assert answer_latencies(timeline, CONTEXT) == pytest.approx([0.7])


def test_answer_latency_reads_the_answering_outputs_own_start():
    """Whatever else the reaction issued, and in whatever order, the sample ends at the start of
    the turn's first output — not at a tool call's."""
    timeline = [
        _set(0.0, "speech_input", "dance and tell me"),
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev("trigger_observed", 0.01, key="speech_input", version_id=1),
        _issued(1.0, "dance", "d1"),
        _said(3.0, "c1", text="Dancing now!"),
        _ev(
            "reaction_ended",
            3.0,
            reaction=_reaction(1, [_trigger("speech_input", 1)], call_ids=("c1", "d1")),
        ),
        _ended(5.0, "d1", started=0.2, output=False),
        _ended(5.0, "c1", started=0.4),
    ]
    assert answer_latencies(timeline, CONTEXT) == pytest.approx([0.4])


def test_an_unanswered_turn_gives_no_answer_latency():
    timeline = [
        _set(0.0, "speech_input", "hi"),
        _ev("input_written", 0.0, key="speech_input", version_id=1),
        _ev(
            "reaction_ended",
            0.4,
            reaction=_reaction(1, [_trigger("speech_input", 1)], noop=True),
        ),
    ]
    assert answer_latencies(timeline, CONTEXT) == []


# --- registry ---------------------------------------------------------------------------------


def _always_kept(window: list[SutEvent], context: CheckContext) -> Finding:
    return Finding(expectation="always_kept", ok=True, detail="", t=0.0, kind="check")


def test_build_registry_adds_application_checks_and_rejects_a_clash():
    with pytest.raises(ValueError, match="already registered"):
        build_registry([Check("answers_each_input", _always_kept)])
    registry = build_registry([Check("always_kept", _always_kept)])
    assert set(registry) == set(CHECKS) | {"always_kept"}


def test_the_step_checks_are_marked_as_such():
    placements = {name: check.placement for name, check in CHECKS.items()}
    assert {n for n, p in placements.items() if p == "step"} == {
        "cancels_when_asked",
        "acts_when_asked",
        "quiet_on_background_change",
    }
    assert dataclasses.replace(CONTEXT, arg="dance").arg == "dance"
