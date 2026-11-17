"""The report data model and `compute_metrics` (specs/wica-tester.md, "Harness")."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wica.instrumentation import CommandTrace, ReactionTrace, TokenUsage, TriggerTrace
from wica.testing import Finding, Report, SutEvent, compute_metrics, summarize

_T0 = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)


def _at(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def test_summarize_percentiles():
    summary = summarize([float(n) for n in range(1, 11)])
    assert summary is not None
    assert (summary.count, summary.min, summary.median, summary.p95, summary.max) == (
        10,
        1,
        5.5,
        10,
        10,
    )
    assert summarize([]) is None


def test_compute_metrics_from_a_hand_built_timeline():
    reaction = ReactionTrace(
        reaction_id=1,
        triggers=(
            TriggerTrace(
                key="prompt",
                version_id=2,
                written_at=_at(0.0),
                arrived_at=_at(0.01),
                is_command_completion=False,
            ),
        ),
        window_opened_at=_at(0.1),
        window_closed_at=_at(0.3),
        prompt_ready_at=_at(0.32),
        model_started_at=_at(0.35),
        model_ended_at=_at(1.0),
        first_content_at=_at(0.8),
        first_output_at=_at(0.9),
        outcome="ok",
        error=None,
        text_length=0,
        command_call_ids=("c1",),
        noop=False,
        usage=TokenUsage(input_tokens=120, output_tokens=8, cache_read_tokens=None),
        ended_at=_at(1.05),
        trace_id=None,
        span_id=None,
    )
    command = CommandTrace(
        call_id="c1",
        name="add",
        reaction_id=1,
        started_at=_at(1.05),
        ended_at=_at(1.55),
        state="complete",
    )
    t0 = _T0.timestamp()
    timeline = [
        SutEvent(kind="input_written", t=t0, key="prompt", version_id=2),
        SutEvent(kind="prompt", t=t0 + 0.32, message_count=2),
        SutEvent(kind="reaction_ended", t=t0 + 1.05, reaction=reaction),
        SutEvent(kind="command_ended", t=t0 + 1.55, name="add", command=command),
        SutEvent(kind="prompt", t=t0 + 1.6, message_count=3),
        SutEvent(kind="trigger_dropped", t=t0 + 1.7, key="prompt", version_id=3),
    ]
    metrics, dropped = compute_metrics(timeline)
    # The reaction opened an output: its latency ends at the first words, not at its own end.
    assert round(metrics["reaction_latency"].min, 6) == 0.9
    assert round(metrics["time_to_first_output"].min, 6) == 0.55
    assert round(metrics["model_latency"].min, 6) == 0.65
    assert round(metrics["time_to_first_content"].min, 6) == 0.45
    assert round(metrics["generation_time"].min, 6) == 0.2
    assert round(metrics["coalescing_wait"].min, 6) == 0.2
    assert metrics["command_duration"].min == 0.5
    assert metrics["input_tokens"].min == 120
    assert metrics["output_tokens"].max == 8
    assert metrics["steps_per_turn"].min == 2
    assert metrics["reactions_per_input"].min == 1
    assert dropped == 1


def _say(t: float, call_id: str, state: str) -> SutEvent:
    return SutEvent(
        kind="command_state",
        t=t,
        name="say",
        call_id=call_id,
        state=state,
        is_output=True,
    )


def test_barge_in_runs_from_the_input_over_an_output_to_that_output_cancelled():
    cancelled_by_command = [
        _say(0.0, "s1", "running"),
        SutEvent(kind="input_written", t=1.0, key="speech", version_id=2),
        SutEvent(kind="command_issued", t=1.4, name="cancel_command", call_id="x"),
        _say(1.5, "s1", "cancelled"),
    ]
    metrics, _ = compute_metrics(cancelled_by_command)
    assert round(metrics["barge_in"].min, 6) == 0.5 and metrics["barge_in"].count == 1
    # The output is the call the framework reports as one, not a Command named like it.
    not_an_output = [
        dataclasses.replace(e, is_output=False) for e in cancelled_by_command
    ]
    assert "barge_in" not in compute_metrics(not_an_output)[0]
    # A new utterance cutting the output off counts the same, with no cancel_command at all;
    # the sample starts at the first input written over the output.
    cut_off = [
        _say(0.0, "s1", "running"),
        SutEvent(kind="input_written", t=1.0, key="speech", version_id=2),
        SutEvent(kind="input_written", t=1.2, key="speech", version_id=3),
        _say(1.3, "s1", "cancelled"),
        _say(1.35, "s2", "running"),
        _say(2.0, "s2", "complete"),
    ]
    metrics, _ = compute_metrics(cut_off)
    assert round(metrics["barge_in"].min, 6) == 0.3 and metrics["barge_in"].count == 1


def test_barge_in_has_no_sample_when_the_output_was_not_talked_over_or_not_cut():
    finished = [  # talked over, but the output ran to its end
        _say(0.0, "s1", "running"),
        SutEvent(kind="input_written", t=1.0, key="speech", version_id=2),
        _say(1.5, "s1", "complete"),
    ]
    assert "barge_in" not in compute_metrics(finished)[0]
    unprompted = [  # cancelled with no input written while it ran
        SutEvent(kind="input_written", t=0.0, key="speech", version_id=1),
        _say(0.5, "s1", "running"),
        _say(1.5, "s1", "cancelled"),
    ]
    assert "barge_in" not in compute_metrics(unprompted)[0]
    another = [  # a Command that is not the output Command
        SutEvent(
            kind="command_state", t=0.0, name="dance", call_id="d1", state="running"
        ),
        SutEvent(kind="input_written", t=1.0, key="speech", version_id=2),
        SutEvent(
            kind="command_state", t=1.5, name="dance", call_id="d1", state="cancelled"
        ),
    ]
    assert "barge_in" not in compute_metrics(another)[0]


def test_report_roundtrips_to_json_and_renders_markdown(tmp_path: Path):
    timeline = [
        SutEvent(
            kind="action", t=10.0, name="set", args={"key": "prompt", "value": "hi"}
        ),
        SutEvent(
            kind="command_issued", t=10.5, name="add", call_id="c1", args={"a": 2}
        ),
    ]
    metrics, dropped = compute_metrics(timeline)
    report = Report(
        scenario="smoke",
        sut_label="fake",
        tester_label=None,
        started_at="2026-11-06T12:00:00+00:00",
        finished_at="2026-11-06T12:00:01+00:00",
        timeline=timeline,
        metrics={**metrics, "model_latency": summarize([0.5, 0.7]) or metrics["x"]},
        dropped_inputs=dropped,
        findings=[Finding("expect_action(name='add')", True, "issued", 10.5)],
        issues=[],
        notes=["one note"],
        verdict="pass",
        summary="fine",
    )
    assert report.ok
    parsed = json.loads(report.to_json())
    assert len(parsed["findings"]) == 1 and parsed["verdict"] == "pass"
    markdown = report.to_markdown()
    assert "model_latency" in markdown and "expect_action(name='add')" in markdown
    assert "one note" in markdown
    report.write(tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["scenario"] == "smoke"
    report.write(tmp_path / "r.md")
    assert (tmp_path / "r.md").read_text().startswith("# smoke")
    failed = Report(
        **{
            **report.to_dict(),
            "timeline": timeline,
            "metrics": {},
            "issues": [],
            "findings": [Finding("expect_private_text()", False, "timeout", 11.0)],
        }
    )
    assert not failed.ok
