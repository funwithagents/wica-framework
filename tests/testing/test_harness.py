"""`TestRun` over a fake SUT (specs/wica-tester.md, "Harness", "The shared Command set")."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from sut_support import ADD_THEN_TEXT, describe_timeline, wait_until

from wica import Command, Wica
from wica.testing import TestRun


def test_set_expect_action_and_text_in_sequence(sut_factory: Callable[..., Wica]):
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
        text = run.expect_text(contains="sum is 4")
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
            "expect_text(contains='sum is 4')",
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


def test_expect_output_uses_the_output_command_whatever_its_name(
    sut_factory: Callable[..., Wica],
):
    def voice(utterance: str) -> str:
        """Speak to the user."""
        return "spoken"

    sut = sut_factory(
        [
            {"tool_calls": [{"name": "talk", "args": {"utterance": "hello there"}}]},
            {"tool_calls": [{"name": "noop", "args": {}}]},
        ],
        commands=(),
    )
    sut.set_output_command(Command(voice, name="talk"))
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Greet me.")
        assert run.expect_output(contains="hello").ok  # the arg is not named
        run.set("prompt", "Again.")  # the script is spent: default is empty text
        assert not run.expect_output(contains="hello", within=0.3).ok
    finally:
        run.close()

    # The named-arg form, on a fresh run of the same script.
    sut2 = sut_factory(
        [{"tool_calls": [{"name": "talk", "args": {"utterance": "hello again"}}]}],
        commands=(),
    )
    sut2.set_output_command(Command(voice, name="talk"))
    sut2.start()
    run2 = TestRun(sut2)
    try:
        run2.set("prompt", "Greet me.")
        assert run2.expect_output(contains="again", arg="utterance").ok
        assert not run2.expect_output(contains="again", arg="missing", within=0.2).ok
    finally:
        run2.close()


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
        [{"tool_calls": [{"name": "noop", "args": {}}]}], inputs=("prompt", "mood")
    )
    sut.start()
    run = TestRun(sut)
    try:
        run.set("prompt", "Nothing to do.")
        assert run.expect_no_action("add", for_=0.3).ok
        assert not run.expect_no_action("noop", for_=0.3).ok  # noop *was* issued
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
        assert run.expect_text(contains="slow").ok
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
        assert time.monotonic() - started >= 0.2
    finally:
        run.close()
        sut.close()
