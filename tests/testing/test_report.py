"""The report data model and `compute_metrics` (specs/wica-tester.md, "Harness")."""

from __future__ import annotations

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
        outcome="ok",
        error=None,
        text_length=0,
        sink_duration=None,
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
    metrics, dropped = compute_metrics(timeline, output_command_name=None)
    assert metrics["reaction_latency"].min == 1.05
    assert round(metrics["model_latency"].min, 6) == 0.65
    assert round(metrics["coalescing_wait"].min, 6) == 0.2
    assert metrics["command_duration"].min == 0.5
    assert metrics["input_tokens"].min == 120
    assert metrics["output_tokens"].max == 8
    assert metrics["steps_per_turn"].min == 2
    assert metrics["reactions_per_input"].min == 1
    assert dropped == 1


def test_barge_in_pairs_an_input_during_the_output_command_with_the_cancel():
    timeline = [
        SutEvent(
            kind="command_state", t=0.0, name="say", call_id="s1", state="running"
        ),
        SutEvent(kind="input_written", t=1.0, key="speech", version_id=2),
        SutEvent(kind="command_issued", t=1.4, name="cancel_command", call_id="x"),
        SutEvent(
            kind="command_state", t=1.5, name="say", call_id="s1", state="cancelled"
        ),
    ]
    metrics, _ = compute_metrics(timeline, output_command_name="say")
    assert round(metrics["barge_in"].min, 6) == 0.4
    assert "barge_in" not in compute_metrics(timeline, output_command_name=None)[0]


def test_report_roundtrips_to_json_and_renders_markdown(tmp_path: Path):
    timeline = [
        SutEvent(
            kind="action", t=10.0, name="set", args={"key": "prompt", "value": "hi"}
        ),
        SutEvent(
            kind="command_issued", t=10.5, name="add", call_id="c1", args={"a": 2}
        ),
    ]
    metrics, dropped = compute_metrics(timeline, output_command_name=None)
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
            "findings": [Finding("expect_text()", False, "timeout", 11.0)],
        }
    )
    assert not failed.ok
