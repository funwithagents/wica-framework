"""`run_once` / `run_matrix` over a fake target (specs/benchmark.md, "The target", "The runner"):
the chain writes one report per run, a rule-breaking model breaks the checks, a chained reply is
a measure, a stopped run is counted and not scored, a `Skip` or an unmet requirement is a summary
skip."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench_support import NOOP, SAY_HI, FakeTarget

from wica.benchmark import RunResult, Skipped, run_matrix, run_once
from wica.testing import CheckSpec, ExpectStep, InputStep, Scenario, WaitStep

SESSION_CHECKS = tuple(
    CheckSpec(n)
    for n in ("answers_each_input", "no_noop_on_input", "no_failed_reactions")
)
GREETING = Scenario(
    name="greeting",
    timeout_s=15,
    checks=SESSION_CHECKS,
    steps=(InputStep("speech_input", "hello"), WaitStep("idle", within=5)),
)


def _step(command: str, /, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": command, "args": args}]}


def test_run_once_writes_both_files_and_returns_the_judged_report(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    target = fake_target()
    result = run_once(target, GREETING, repetition=1, out_dir=tmp_path)
    assert isinstance(result, RunResult) and result.scored
    assert result.report.ok, [f for f in result.report.findings if not f.ok]
    assert tuple(f.expectation for f in result.report.checks()) == GREETING.check_names
    assert result.measures["output_calls"] == 1 and result.measures["noop_steps"] == 1
    assert result.measures["turns"] == 1 and result.measures["turns_answered"] == 1
    assert result.samples["reaction_latency"]  # pooled by the summary
    json_path = tmp_path / "fake" / "greeting" / "1.json"
    assert json_path.exists() and (tmp_path / "fake" / "greeting" / "1.md").exists()
    document = json.loads(json_path.read_text())
    assert document["outcome"] == "completed"
    assert document["bench"]["repetition"] == 1
    assert document["bench"]["measures"]["output_calls"] == 1.0
    assert target.opened == 1 and target.closed == 1  # closed after the run


def test_a_model_that_noops_the_greeting_breaks_the_checks(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    result = run_once(fake_target([NOOP]), GREETING, repetition=1, out_dir=tmp_path)
    assert isinstance(result, RunResult) and result.scored  # it completed: it is scored
    broken = {f.expectation for f in result.report.checks() if not f.ok}
    assert broken == {"answers_each_input", "no_noop_on_input"}
    assert not result.report.ok


def test_a_stopped_run_is_counted_and_not_scored(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    needs_a_reply = Scenario(
        name="greeting",
        timeout_s=15,
        checks=SESSION_CHECKS,
        steps=(
            InputStep("speech_input", "hello"),
            ExpectStep("output", {}, within=0.5),
        ),
    )
    result = run_once(
        fake_target([NOOP]), needs_a_reply, repetition=1, out_dir=tmp_path
    )
    assert isinstance(result, RunResult) and not result.scored
    assert result.report.outcome == "failed"
    assert result.report.stopped_at == "step 1: expect_output()"
    assert (
        result.measures == {} and result.samples == {} and result.report.checks() == []
    )
    document = json.loads((tmp_path / "fake" / "greeting" / "1.json").read_text())
    assert document["bench"]["outcome"] == "failed"


def test_a_chained_reply_is_a_measure_not_a_failure(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    twice = SAY_HI + [_step("say", text="and one more thing")]
    result = run_once(fake_target(twice), GREETING, repetition=1, out_dir=tmp_path)
    assert isinstance(result, RunResult)
    assert result.report.ok
    assert result.measures["output_calls"] == 2
    assert result.measures["unprompted_output"] == 1
    assert result.measures["output_words"] == 6


def test_a_skip_from_the_target_is_a_skip_not_an_error(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    target = fake_target(skip="WICA_X_API_KEY is not set")
    outcome = run_once(target, GREETING, repetition=1, out_dir=tmp_path)
    assert outcome == Skipped("fake", "greeting", "WICA_X_API_KEY is not set")
    assert target.closed == 1 and not list(tmp_path.iterdir())


def test_an_unmet_requirement_skips_with_the_reason(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    needs_dance = Scenario(
        name="dance",
        steps=(InputStep("gaze", "left"), WaitStep({"running": "dance"})),
    )
    outcome = run_once(fake_target(), needs_dance, repetition=1, out_dir=tmp_path)
    assert isinstance(outcome, Skipped)
    assert outcome.reason == "the target lacks input 'gaze', command 'dance'"


def test_run_matrix_repeats_filters_and_does_not_retry_a_skip(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    good = fake_target(label="good")
    skipped = fake_target(label="nokey", skip="key unset")
    ignored = fake_target(label="ignored")
    other = Scenario(name="other", steps=GREETING.steps)
    seen: list[str] = []
    summary = run_matrix(
        [good, skipped, ignored],
        [GREETING, other],
        k=2,
        out_dir=tmp_path,
        only_configs=["good", "nokey"],
        only_scenarios=["greeting"],
        on_outcome=lambda o: seen.append(type(o).__name__),
        skipped=[Skipped("disabled", "*", "disabled")],
    )
    assert seen == ["RunResult", "RunResult", "Skipped"]
    assert good.opened == 2 and skipped.opened == 1 and ignored.opened == 0
    assert summary.k == 2 and summary.scenarios == ("greeting",)
    assert set(summary.configs) == {"good", "nokey"}
    cell = summary.cell("greeting", "good")
    assert cell is not None and (cell.runs, cell.passed, cell.pass_k) == (2, 2, 1)
    assert summary.cell("greeting", "nokey") is None
    assert summary.skips == (
        Skipped("disabled", "*", "disabled"),  # the caller's, first
        Skipped("nokey", "greeting", "key unset"),
    )
    assert summary.git_commit is None or len(summary.git_commit) == 40
    assert sorted(p.name for p in (tmp_path / "good" / "greeting").iterdir()) == [
        "1.json",
        "1.md",
        "2.json",
        "2.md",
    ]
