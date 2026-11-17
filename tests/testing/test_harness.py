"""`TestRun` over a fake SUT (specs/wica-tester.md, "Harness", "The shared Command set")."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sut_support import ADD_THEN_TEXT, Voice, describe_timeline, wait_until

from wica import Command, Wica
from wica.instrumentation import CommandTrace
from wica.testing import SutEvent, TestRun
from wica.testing.harness import _busy_reasons


def test_set_expect_action_and_private_text_in_sequence(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut, scenario="add then speak", sut_label="fake")
    try:
        run.set("prompt", "Add 2 and 2.")
        issued = run.expect_action("add", where={"a": 2})
        assert issued.ok and issued.event is not None and issued.event.call_id
        done = run.expect_action("add", state="complete", result_contains="4")
        assert done.ok and done.event is not None
        assert done.event.call_id == issued.event.call_id
        text = run.expect_private_text(contains="sum is 4")
        assert text.ok, describe_timeline(run.timeline())
        # The order the watermark relies on: the Command's terminal state is recorded before the
        # text of the step its completion triggered (the Probe's listener is loop-ordered).
        assert done.event is not None and text.event is not None
        timeline = run.timeline()
        assert timeline.index(done.event) < timeline.index(text.event)
        wait_until(lambda: len([e for e in run.timeline() if e.reaction]) >= 2)
        report = run.report()
        assert report.ok and report.verdict is None
        metrics, dropped = run.metrics()
        assert metrics["command_duration"].count == 1
        assert (
            metrics["reaction_latency"].count == 1
        )  # the text reaction has no external trigger
        assert (
            metrics["reactions_per_input"].min == 2
        )  # the input's reaction + the re-trigger
        assert metrics["steps_per_turn"].min == 2
        assert dropped == 0
        assert [f.expectation for f in report.findings] == [
            "expect_action(name='add', where={'a': 2})",
            "expect_action(name='add', state='complete', result_contains='4')",
            "expect_private_text(contains='sum is 4')",
        ]
        assert [e.name for e in report.timeline if e.kind == "action"][0] == "set"
    finally:
        run.close()


def test_failed_expectation_is_a_finding_not_an_exception(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut)
    try:
        finding = run.expect_action("never", within=0.2)
        assert not finding.ok and "0.2" in finding.detail
        assert not run.report().ok
    finally:
        run.close()


def test_watermark_prevents_rematching(sut_factory: Callable[..., Wica]):
    sut = sut_factory(
        [
            {"tool_calls": [{"name": "add", "args": {"a": 1, "b": 1}}]},
            {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 2}}]},
            {"text": "done"},
        ]
    )
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Add twice.")
        first = run.expect_action("add")
        second = run.expect_action("add")
        assert first.ok and second.ok
        assert first.event is not None and second.event is not None
        assert first.event.call_id != second.event.call_id
        assert not run.expect_action("add", within=0.2).ok
    finally:
        run.close()


def test_expect_output_matches_an_output_text_whatever_the_command_name(
    sut_factory: Callable[..., Wica],
):
    said: list[tuple[str, str]] = []

    def voice(text: str, mood: str = "calm") -> str:
        """Speak to the user."""
        said.append((text, mood))
        return "spoken"

    sut = sut_factory(
        [
            {"text": 'a private thought <talk mood="happy">hello there</talk>'},
            {"text": '<talk mood="sad">hello again</talk>'},
        ],
        commands=(),
    )
    sut.set_output_command(Command(voice, name="talk"))
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Greet me.")
        # `contains` is matched against the output's text: not against an attribute, nor against
        # what the model wrote outside the tag.
        assert not run.expect_output(contains="happy", within=0.3).ok
        assert not run.expect_output(contains="private", within=0.3).ok
        heard = run.expect_output(contains="hello")
        assert heard.ok and heard.event is not None
        assert heard.event.args == {"mood": "happy", "text": "hello there"}
        # Matched once: the watermark is past it, so the same expectation waits for the next.
        assert not run.expect_output(contains="hello", within=0.3).ok
        run.set("prompt", "Again.")
        # The named-attribute form.
        assert not run.expect_output(contains="sad", arg="missing", within=0.3).ok
        assert run.expect_output(contains="sad", arg="mood").ok
        run.set(
            "prompt", "Once more."
        )  # the script is spent: the model answers nothing
        assert not run.expect_output(within=0.3).ok
        assert said == [("hello there", "happy"), ("hello again", "sad")]
    finally:
        run.close()


def test_expect_output_fails_at_once_without_an_output_command(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut)
    try:
        finding = run.expect_output(within=5.0)
        assert not finding.ok and "no output Command" in finding.detail
    finally:
        run.close()


def test_expect_no_action_and_expect_world(sut_factory: Callable[..., Wica]):
    sut = sut_factory(
        [
            {"text": "<noop/>"},
            {"tool_calls": [{"name": "add", "args": {"a": 1, "b": 1}}]},
        ],
        inputs=("prompt", "mood"),
    )
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Nothing to do.")
        assert run.expect_no_action("add", for_=0.3).ok  # the SUT declined
        run.set("prompt", "Add now.")
        broken = run.expect_no_action("add", for_=2.0)  # this time add *is* issued
        assert not broken.ok and broken.event is not None
        assert broken.event.kind == "command_issued" and broken.event.name == "add"
        sut.world.update("mood", "happy")
        assert run.expect_world("mood", equals="happy").ok
        sad = run.expect_world("mood", equals="sad", within=0.2)
        assert not sad.ok and "'happy'" in sad.detail
    finally:
        run.close()


def test_dropped_input_is_counted(sut_factory: Callable[..., Wica]):
    sut = sut_factory([{"text": "slow"}, {"text": "slow"}], delay_s=0.3)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "one")
        run.set(
            "prompt", "two"
        )  # lands while the first reaction's model call is in flight
        run.set("prompt", "three")
        assert run.expect_private_text(contains="slow").ok
        wait_until(lambda: run.metrics()[1] >= 1)
        metrics, dropped = run.metrics()
        assert dropped >= 1
        assert metrics["model_latency"].min >= 0.3
    finally:
        run.close()


def test_notes_issues_and_finish_land_in_the_report(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut, scenario="bookkeeping")
    try:
        run.note("starting")
        run.report_issue("low", "slow start", "reaction_latency p95 = 2.1")
        assert not run.finished.is_set()
        run.finish("fail", "too slow")
        assert run.finished.is_set()
        report = run.report(tester_label="python")
        assert report.notes == ["starting"]
        assert report.issues[0].what == "slow start"
        assert report.verdict == "fail" and report.summary == "too slow"
        assert not report.ok and report.tester_label == "python"
        assert [e.name for e in report.timeline if e.kind == "action"] == [
            "note",
            "report_issue",
            "finish",
        ]
    finally:
        run.close()


# --- wait_for --------------------------------------------------------------------------


def test_wait_for_idle_waits_for_the_step_a_completion_wakes(
    sut_factory: Callable[..., Wica],
):
    """`add` completes, its completion wakes a step, that step ends: only then idle — no timer."""
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "add 2 and 2")
        finding = run.wait_for("idle", within=5)
        assert finding.ok, finding.detail
        timeline = run.timeline()
        reactions = [e for e in timeline if e.kind == "reaction_ended"]
        assert len(reactions) == 2  # the add step and the one its completion woke
        assert any(
            e.kind == "command_state" and e.name == "add" and e.state == "complete"
            for e in timeline
        )
        assert timeline[-1].kind == "reaction_ended" or timeline[-1].kind == "action"
    finally:
        run.close()
        sut.close()


def test_wait_for_idle_reports_what_is_still_running(sut_factory: Callable[..., Wica]):
    async def slow() -> str:
        """Take a while."""
        await asyncio.sleep(1.0)
        return "done"

    sut = sut_factory(
        [{"tool_calls": [{"name": "slow", "args": {}}]}, {"text": ""}], commands=(slow,)
    )
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "go")
        run.expect_action("slow", within=2)
        finding = run.wait_for("idle", within=0.2)
        assert (
            not finding.ok
            and "slow (" in finding.detail
            and "running" in finding.detail
        )
        assert run.wait_for("idle", within=5).ok  # and then it is
    finally:
        run.close()
        sut.close()


def test_wait_for_ready_returns_over_a_running_command_so_the_next_input_lands(
    sut_factory: Callable[..., Wica],
):
    """ "ready" waits for the reaction (a slow model here), not for the Command it started: an
    input written then is reacted to, never dropped as busy."""

    async def slow() -> str:
        """Take a while."""
        await asyncio.sleep(1.5)
        return "done"

    sut = sut_factory(
        [{"tool_calls": [{"name": "slow", "args": {}}]}, {"text": "stopping"}],
        commands=(slow,),
        delay_s=0.3,
    )
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "go")
        assert run.wait_for("ready", within=5).ok
        timeline = run.timeline()
        assert any(e.kind == "reaction_ended" for e in timeline)  # the step ended…
        assert not any(  # …while slow still runs
            e.kind == "command_state" and e.name == "slow" and e.state == "complete"
            for e in timeline
        )
        assert not run.wait_for("idle", within=0.1).ok  # idle would wait for slow
        run.set("prompt", "stop")
        assert run.wait_for("idle", within=5).ok
        timeline = run.timeline()
        assert not any(e.kind == "trigger_dropped" for e in timeline)
        reacted = {
            t.version_id
            for e in timeline
            if e.reaction is not None
            for t in e.reaction.triggers
            if t.key == "prompt"
        }
        assert len(reacted) == 2  # both writes reached a reaction
    finally:
        run.close()
        sut.close()


def test_wait_for_idle_is_immediate_on_a_quiet_sut(sut_factory: Callable[..., Wica]):
    sut = sut_factory([{"text": ""}])
    sut.start()
    run = TestRun(sut)
    try:
        started = time.monotonic()
        assert run.wait_for("idle", within=5).ok
        assert time.monotonic() - started < 0.5
    finally:
        run.close()
        sut.close()


def test_wait_for_running_and_speaking_hold_once_started_since_the_last_input(
    sut_factory: Callable[..., Wica],
):
    async def slow() -> str:
        """Take a while."""
        await asyncio.sleep(0.5)
        return "done"

    sut = sut_factory(
        [{"tool_calls": [{"name": "slow", "args": {}}]}, {"text": ""}], commands=(slow,)
    )
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "go")
        assert run.wait_for({"running": "slow"}, within=5).ok
        timed_out = run.wait_for({"running": "ghost"}, within=0.2)
        assert not timed_out.ok and timed_out.kind == "wait"
        assert "ghost has not started since the last input" in timed_out.detail
        # "speaking" needs an output Command, which this SUT has none of.
        assert "no output Command" in run.wait_for("speaking", within=0.1).detail
        started = time.monotonic()
        assert run.wait_for({"seconds": 0.2}).ok
        # One tick of slack: Windows' monotonic clock moves in ~15.6 ms steps (specs/ci.md).
        assert time.monotonic() - started >= 0.2 - 0.016
    finally:
        run.close()
        sut.close()


# --- wait_for with outputs: an output's end wakes a step only when it was granted one -----------


def _reactions(run: TestRun) -> int:
    return len([e for e in run.timeline() if e.kind == "reaction_ended"])


def _output_states(run: TestRun) -> list[str | None]:
    return [e.state for e in run.timeline() if e.kind == "command_state"]


def _ends(run: TestRun) -> dict[str, bool]:
    """`call_id → triggers_step`, from the framework's record of each call's end."""
    return {
        e.command.call_id: e.command.triggers_step
        for e in run.timeline()
        if e.command is not None
    }


def test_wait_for_speaking_ready_and_idle_around_a_plain_spoken_reply(
    sut_factory: Callable[..., Wica],
):
    """Without a `<next>` note the end of the speech wakes nothing: "speaking" holds once the
    output is reported, "ready" while it is still being delivered, "idle" as soon as it is over
    — with no step awaited after it."""
    sut = sut_factory([{"text": "<say>Hello there.</say>"}], commands=())
    voice = Voice(delay=0.4)
    sut.set_output_command(voice.say)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Greet me.")
        assert run.wait_for("speaking", within=5).ok
        assert run.wait_for("ready", within=5).ok
        assert _output_states(run) == ["running"]  # ready over the running speech
        busy = run.wait_for("idle", within=0.1)
        assert not busy.ok and "say (" in busy.detail and "running" in busy.detail
        assert run.wait_for("idle", within=5).ok
        assert _output_states(run) == ["running", "complete"]
        time.sleep(0.3)
        assert _reactions(run) == 1  # idle was right: nothing was woken
        assert voice.said == ["Hello there."]
    finally:
        run.close()
        sut.close()


def test_wait_for_idle_does_not_wait_for_a_step_an_output_without_a_note_never_wakes(
    sut_factory: Callable[..., Wica],
):
    """The counterpart of the granted wake below, over the same slow coalescing window: with no
    note there is no step to wait for, so idle holds at the end of the speech."""
    sut = sut_factory(
        [{"text": "<say>Hello there.</say>"}], commands=(), coalesce_window=0.4
    )
    sut.set_output_command(Voice().say)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Greet me.")
        assert run.wait_for("idle", within=5).ok
        assert _reactions(run) == 1 and _output_states(run) == ["running", "complete"]
        call_id = [e.call_id for e in run.timeline() if e.kind == "command_issued"][0]
        assert _ends(run) == {
            call_id: False
        }  # the record idle read: no step to wait for
    finally:
        run.close()
        sut.close()


def test_wait_for_idle_waits_for_the_step_a_noted_output_wakes(
    sut_factory: Callable[..., Wica],
):
    """With a `<next>` note the end of the output wakes a step. The SUT's coalescing window keeps
    that wake pending for 0.4 s after the output ended — nothing running, no trigger observed
    yet — and idle must hold only once the woken step has run."""
    sut = sut_factory(
        [
            {"text": "<say>Give me a moment.</say><next>then report</next>"},
            {"text": "<say>Here it is.</say>"},
        ],
        commands=(),
        coalesce_window=0.4,
    )
    voice = Voice()
    sut.set_output_command(voice.say)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Look it up.")
        finding = run.wait_for("idle", within=10)
        assert finding.ok, finding.detail
        assert _reactions(run) == 2, describe_timeline(run.timeline())
        assert voice.said == ["Give me a moment.", "Here it is."]
        assert _output_states(run) == ["running", "complete", "running", "complete"]
        first, second = (
            e.call_id for e in run.timeline() if e.kind == "command_issued"
        )
        # What idle read: the framework's record of each output's end.
        assert _ends(run) == {first: True, second: False}  # no note: the chain ends
    finally:
        run.close()
        sut.close()


def test_wait_for_idle_names_the_wake_it_is_waiting_for(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(
        [
            {"text": "<say>Give me a moment.</say><next>then report</next>"},
            {"text": "<noop/>"},
        ],
        commands=(),
        coalesce_window=0.6,
    )
    sut.set_output_command(Voice().say)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Look it up.")
        wait_until(lambda: "complete" in _output_states(run))
        pending = run.wait_for("idle", within=0.2)
        assert not pending.ok and "its step not yet woken" in pending.detail
        assert run.wait_for("idle", within=10).ok
        assert _reactions(run) == 2
    finally:
        run.close()
        sut.close()


def test_a_cancelled_output_wakes_nothing_even_with_a_note(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(
        [
            {"text": "<say>Give me a moment.</say><next>then report</next>"},
            {"text": "<say>Here it is.</say>"},
        ],
        commands=(),
    )
    voice = Voice(delay=5.0)
    sut.set_output_command(voice.say)
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Look it up.")
        assert run.wait_for("speaking", within=5).ok
        assert run.wait_for("ready", within=5).ok
        call_id = [e.call_id for e in run.timeline() if e.kind == "command_issued"][0]
        assert call_id is not None
        sut.agent.cancel_command(call_id)
        assert run.wait_for("idle", within=5).ok
        assert _output_states(run) == ["running", "cancelled"]
        assert _ends(run) == {call_id: False}  # cancelled: its granted wake withdrawn
        time.sleep(0.3)
        assert _reactions(run) == 1 and voice.said == []  # no step was woken
    finally:
        run.close()
        sut.close()


# --- idle reads `triggers_step` alone --------------------------------------------------------


def _ended_call(
    call_id: str, *, is_output: bool, triggers_step: bool
) -> list[SutEvent]:
    """A call issued, its end recorded by the framework, then its terminal state."""
    t0 = datetime(2026, 11, 17, tzinfo=timezone.utc)
    trace = CommandTrace(
        call_id=call_id,
        name="say" if is_output else "dance",
        reaction_id=1,
        started_at=t0,
        ended_at=t0 + timedelta(seconds=1),
        state="complete",
        is_output=is_output,
        triggers_step=triggers_step,
    )
    return [
        SutEvent(
            kind="command_issued",
            t=1.0,
            name=trace.name,
            call_id=call_id,
            is_output=is_output,
        ),
        SutEvent(kind="command_ended", t=2.0, call_id=call_id, command=trace),
        SutEvent(
            kind="command_state",
            t=2.0,
            name=trace.name,
            call_id=call_id,
            state="complete",
            is_output=is_output,
        ),
    ]


def _woken(call_id: str) -> SutEvent:
    """The trigger the end produced, arrived at the Agent (dropped here, as when a reaction is
    in flight: what idle needs is that it arrived, not that it was reacted to)."""
    return SutEvent(
        kind="trigger_dropped", t=2.5, key=f"agent:command:{call_id}", version_id=3
    )


def _idle(timeline: list[SutEvent]) -> list[str]:
    return _busy_reasons(timeline, lambda key: False)


def test_idle_waits_for_a_step_exactly_when_the_end_record_says_it_triggers_one():
    """The rule the Harness applies, over hand-built timelines: a finished call is waited on past
    its end only when its `command_ended` record says `triggers_step` — for an output and a tool
    call alike, whatever their names — and only until that trigger arrives."""
    for is_output in (True, False):
        granted = _ended_call("c1", is_output=is_output, triggers_step=True)
        assert _idle(granted) == [
            f"{granted[0].name} (c1) completed, its step not yet woken"
        ]
        assert _idle([*granted, _woken("c1")]) == []
        # A trigger on another call's key is not this one's.
        assert _idle([*granted, _woken("c2")]) != []
        not_granted = _ended_call("c1", is_output=is_output, triggers_step=False)
        assert _idle(not_granted) == []


def test_idle_waits_for_the_end_record_of_a_call_seen_terminal():
    """Until the framework's record of the end has arrived, whether it wakes a step is unknown."""
    issued, _, terminal = _ended_call("c1", is_output=True, triggers_step=True)
    assert _idle([issued, terminal]) == ["say (c1) ended, its end not yet recorded"]
    assert _idle([issued]) == ["say (c1) running"]
