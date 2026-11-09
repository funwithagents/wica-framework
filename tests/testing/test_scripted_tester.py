"""The scripted-fake tester `Wica` (specs/wica-tester.md, "Drivers (b)")."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sut_support import ADD_THEN_TEXT, describe_timeline, identity_serialize, wait_until

from wica import Wica
from wica.fake_model import FakeChatModel
from wica.testing import TestRun, build_tester, run_script


def _calls(command: str, /, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": command, "args": args}]}


def test_script_drives_the_sut_and_yields_a_passing_report(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut, scenario="scripted add", sut_label="fake")
    script = [
        _calls("set_prompt", value="Add 2 and 2."),
        _calls("expect_action", name="add", state="complete", result_contains="4"),
        _calls("expect_text", contains="sum"),
        _calls("finish", verdict="pass", summary="all good"),
    ]
    tester = build_tester(run, script)
    tester.start()
    try:
        tester.world.update("scenario", run.scenario)
        assert run.finished.wait(10)
        model = tester.agent.model
        assert isinstance(model, FakeChatModel)
        wait_until(lambda: len(model.calls) == 4)
    finally:
        tester.close()
        run.close()
    report = run.report(tester_label="fake")
    assert report.ok, describe_timeline(report.timeline) + "\n" + repr(report.findings)
    assert report.verdict == "pass" and report.summary == "all good"
    assert [f.ok for f in report.findings] == [True, True]
    assert report.metrics["command_duration"].count == 1


def test_only_settable_types_get_set_commands(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT, inputs=("prompt", "speech:input"))
    sut.world.register("blob", dict, serialize_fn=identity_serialize)
    sut.world.register("count", int, serialize_fn=identity_serialize)
    run = TestRun(sut)
    tester = build_tester(run, [])
    tester.start()
    try:
        names = set(tester.agent.commands)
        assert {"set_prompt", "set_speech_input", "set_count"} <= names
        assert "set_blob" not in names
        assert {"wait", "expect_action", "expect_output", "finish"} <= names
        count = tester.agent.commands["set_count"].tool
        assert count.args_schema is not None
        assert count.args["value"]["type"] == "integer"
    finally:
        tester.close()
        run.close()


def test_a_failed_expectation_lands_in_the_report_and_the_script_continues(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut)
    try:
        report = run_script(
            run,
            [
                _calls("expect_action", name="ghost", within=0.2),
                _calls("finish", verdict="fail"),
            ],
            timeout=10,
        )
    finally:
        run.close()
    assert report.verdict == "fail" and report.tester_label == "fake"
    assert len(report.findings) == 1
    assert report.findings[0].expectation.startswith("expect_action(name='ghost'")
    assert not report.findings[0].ok and not report.ok


def test_missing_finish_times_out_with_a_failed_finding(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut)
    try:
        report = run_script(
            run, [_calls("set_prompt", value="Add 2 and 2.")], timeout=1.0
        )
    finally:
        run.close()
    assert report.verdict is None
    assert not report.findings[-1].ok and "finish" in report.findings[-1].detail
    assert not report.ok


def test_mirror_entries_follow_the_sut(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT)
    sut.start()
    run = TestRun(sut)
    tester = build_tester(
        run,
        [
            _calls("set_prompt", value="Add 2 and 2."),
            _calls("expect_text", contains="sum"),
            _calls("finish", verdict="pass"),
        ],
    )
    tester.start()
    try:
        tester.world.update("scenario", "mirror")
        assert run.finished.wait(10)
        wait_until(lambda: "complete" in str(tester.world.get("sut:action")))
        action = tester.world.get("sut:action")
        assert "add" in action and "complete" in action
        assert tester.world.get("sut:reasoning") == "The sum is 4."
        assert tester.world.get("sut:output") is None  # the SUT has no output Command
    finally:
        tester.close()
        run.close()
