"""`aggregate`, the summary's two renderings, batches and kept results (specs/benchmark.md,
"The summary") over hand-built runs."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from wica.benchmark import (
    Completed,
    Skipped,
    aggregate,
    git_commit,
    write_result,
    write_summary,
)
from wica.instrumentation import ReactionTrace, TokenUsage, TriggerTrace
from wica.testing import Finding, Report, SutEvent

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _reaction(cache_read: int | None) -> ReactionTrace:
    return ReactionTrace(
        reaction_id=1,
        triggers=(TriggerTrace("speech_input", 1, T0, T0, False),),
        window_opened_at=T0,
        window_closed_at=T0,
        prompt_ready_at=T0,
        model_started_at=T0,
        model_ended_at=T0,
        outcome="ok",
        error=None,
        text_length=0,
        sink_duration=None,
        command_call_ids=("c1",),
        noop=False,
        usage=TokenUsage(
            input_tokens=100, output_tokens=10, cache_read_tokens=cache_read
        ),
        ended_at=T0,
        trace_id=None,
        span_id=None,
    )


def _run(
    *,
    scenario: str = "greeting",
    label: str = "a",
    ok_findings: tuple[bool, ...] = (True,),
    verdict: str | None = "pass",
    latency: float = 1.0,
    dropped: int = 0,
    cache_read: int | None = None,
    repetition: int = 1,
) -> Completed:
    # finding 0 is the script's expectation; the rest are checks (named check_<i>)
    findings = [
        Finding(
            expectation="expect_output()" if i == 0 else f"check_{i}",
            ok=ok,
            detail="",
            t=0.0,
        )
        for i, ok in enumerate(ok_findings)
    ]
    report = Report(
        scenario=scenario,
        sut_label=label,
        tester_label="fake",
        started_at="",
        finished_at="",
        timeline=[
            SutEvent(kind="reaction_ended", t=0.0, reaction=_reaction(cache_read))
        ],
        metrics={},
        dropped_inputs=dropped,
        findings=findings,
        issues=[],
        notes=[],
        verdict=verdict,
        summary=None,
    )
    return Completed(
        report=report,
        measures={
            "output_calls": 1.0,
            "unprompted_output": 0.0,
            "output_words": 4.0,
            "turns": 2.0,
            "turns_answered": 1.0,
        },
        checks=tuple(f.expectation for f in findings[1:]),
        samples={
            "reaction_latency": [latency, latency + 0.2],
            "input_tokens": [100.0, 100.0],
            "output_tokens": [10.0, 10.0],
        },
        repetition=repetition,
    )


def _aggregate(runs: list[Completed], skips: list[Skipped] = [], *, k: int = 2):
    return aggregate(
        runs,
        skips,
        k=k,
        configs={
            "a": {"provider": "openai", "model": "gpt-4o"},
            "b": {"provider": "fake", "model": "x"},
        },
        scenarios=["greeting", "question"],
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:10:00",
        git_commit="abc123",
    )


def test_aggregate_pools_runs_into_cells_with_pass_rate_pass_k_and_failures():
    summary = _aggregate(
        [
            _run(latency=1.0, cache_read=50),
            _run(
                ok_findings=(True, False),
                verdict="fail",
                latency=3.0,
                dropped=1,
                repetition=2,
            ),
            _run(label="b", scenario="question"),
        ],
        [Skipped("b", "greeting", "key unset")],
    )
    a = summary.cell("greeting", "a")
    assert a is not None
    assert (a.runs, a.passed, a.pass_rate, a.pass_k) == (2, 1, 0.5, 0)
    assert a.failed == {"check_1": 1, "verdict: fail": 1}
    assert a.metrics["reaction_latency"].count == 4  # two samples per run, pooled
    assert a.metrics["reaction_latency"].max == 3.2
    assert a.per_run["input_tokens"].median == 200  # summed per run, then pooled
    assert a.per_run["dropped_inputs"].max == 1
    assert a.per_run["output_words"].median == 4
    assert a.cache_read_share == 50 / 100  # over the reactions that report cache reads
    assert a.per_run["answer_rate"].median == 0.5  # 1 of 2 turns, both runs
    assert a.per_run["input_tokens_per_turn"].median == 100  # 200 tokens over 2 turns
    assert a.per_run["words_per_turn"].median == 2
    assert a.per_run["compliance"].min == 0.0  # run 2's one check failed
    # The totals pool a configuration over every scenario.
    total_a = summary.cell("*", "a")
    assert total_a is not None and total_a.runs == 2 and total_a.scenario == "*"
    total_b = summary.cell("*", "b")
    assert total_b is not None and total_b.runs == 1 and total_b.pass_k == 1
    assert [c.label for c in summary.totals] == ["a", "b"]
    b = summary.cell("question", "b")
    assert b is not None and (b.runs, b.passed, b.pass_k) == (1, 1, 1)
    assert b.cache_read_share is None
    assert summary.cell("greeting", "b") is None
    assert [c.scenario for c in summary.cells] == ["greeting", "question"]


def test_compliance_is_checks_passed_over_checks_run():
    summary = _aggregate(
        [
            _run(ok_findings=(True, True, False), repetition=1),  # 1 of 2 checks
            _run(ok_findings=(True, True, True), repetition=2),  # 2 of 2
        ],
        k=2,
    )
    total = summary.cell("*", "a")
    assert total is not None
    assert total.per_run["compliance"].min == 0.5
    assert total.per_run["compliance"].max == 1.0
    assert total.failed == {"check_2": 1}


def test_markdown_opens_with_the_per_configuration_table():
    text = _aggregate(
        [_run(ok_findings=(True, True))], [Skipped("b", "*", "disabled")]
    ).to_markdown()
    head, _, rest = text.partition("## Configurations")
    assert "## Per configuration" in head and "## greeting" in rest
    row = next(line for line in head.splitlines() if line.startswith("| a |"))
    assert row.startswith("| a | 1 | 1 | 100% | 50% | 1.10 / 1.20 |")
    assert "| 100 / 10 | 2 |" in row  # tokens in / out per turn, words per turn
    assert "| b | skipped: disabled |" in head


def test_markdown_has_one_table_per_scenario_with_skips_in_place():
    full = _aggregate([_run()], [Skipped("b", "greeting", "key unset")]).to_markdown()
    assert "`abc123`" in full and "k = 2" in full
    text = full.partition("## Configurations")[2]  # past the per-configuration table
    assert "## greeting" in text and "## question" in text
    assert '- **a**: `{"model": "gpt-4o", "provider": "openai"}`' in text
    row = next(line for line in text.splitlines() if line.startswith("| a |"))
    assert row.startswith("| a | 1 | 1 | 100% | 1 | — | 1.10 / 1.20 |")
    assert "| b | skipped: key unset |" in text
    assert "- b / greeting: key unset" in text


def test_json_carries_the_same_numbers():
    document = json.loads(_aggregate([_run()]).to_json())
    assert document["git_commit"] == "abc123" and document["k"] == 2
    assert [t["label"] for t in document["totals"]] == ["a"]
    (cell,) = document["cells"]
    assert cell["pass_rate"] == 1.0 and cell["pass_k"] == 1
    assert cell["metrics"]["reaction_latency"]["count"] == 2
    assert cell["per_run"]["output_calls"]["median"] == 1.0


def test_write_summary_and_write_result(tmp_path: Path):
    summary = _aggregate([_run()], [Skipped("b", "greeting", "key unset")])
    batch = tmp_path / "runs" / "20260101-0000"
    write_summary(summary, batch)
    assert (batch / "summary.md").exists() and (batch / "summary.json").exists()

    result = write_result(
        summary, tmp_path / "results" / "20260101-0000_baseline", batch_dir=batch
    )
    assert sorted(p.name for p in result.iterdir()) == [
        "description.md",
        "findings.md",
        "runs.json",
        "summary.json",
        "summary.md",
    ]
    manifest = json.loads((result / "runs.json").read_text())
    assert manifest["batch_dir"] == str(batch)
    assert manifest["skips"] == [
        {"label": "b", "scenario": "greeting", "reason": "key unset"}
    ]
    description = (result / "description.md").read_text()
    assert description.startswith("# 20260101-0000_baseline\n")
    assert "- `a`: openai / gpt-4o" in description and "TODO" in description
    assert "# Findings — 20260101-0000_baseline" in (result / "findings.md").read_text()

    # Hand-finished files are never rewritten.
    (result / "description.md").write_text("# mine\n")
    (result / "findings.md").write_text("# read\n")
    write_result(summary, result, batch_dir=batch)
    assert (result / "description.md").read_text() == "# mine\n"
    assert (result / "findings.md").read_text() == "# read\n"


def test_git_commit_is_a_sha_in_a_repository_and_none_outside(tmp_path: Path):
    here = git_commit(Path(__file__).parent)
    assert here is None or re.fullmatch(r"[0-9a-f]{40}", here)
    assert git_commit(tmp_path) is None
