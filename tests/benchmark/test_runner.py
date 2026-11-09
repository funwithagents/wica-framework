"""`run_once` / `run_matrix` over a fake target (specs/benchmark.md, "The target", "The runner"):
the chain writes one report per run, a rule-breaking model fails the checks, a chained reply is a
measure, a `Skip` or an unmet requirement is a summary skip."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wica.benchmark import Completed, Skipped, run_matrix, run_once
from wica.testing import Requires, Scenario

from bench_support import NOOP, SAY_HI, FakeTarget


def _step(command: str, /, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": command, "args": args}]}


GREETING = Scenario(
    name="greeting",
    requires=Requires(inputs=("speech_input",), output_command=True),
    timeout_s=15,
    checks=("answers_each_input", "no_noop_on_input", "no_failed_reactions"),
    script=[
        _step("set_speech_input", value="hello"),
        _step("expect_output", within=3),
        _step("expect_no_action", name="say", for_=0.3),
        _step("finish", verdict="pass"),
    ],
)


def test_run_once_writes_both_files_and_returns_the_checked_report(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    target = fake_target()
    outcome = run_once(target, GREETING, repetition=1, out_dir=tmp_path)
    assert isinstance(outcome, Completed)
    assert outcome.report.ok, [f for f in outcome.report.findings if not f.ok]
    assert [f.expectation for f in outcome.report.findings][-3:] == list(
        GREETING.checks or ()
    )
    assert outcome.measures["output_calls"] == 1 and outcome.measures["noop_steps"] == 1
    assert outcome.samples["reaction_latency"]  # pooled by the summary
    assert outcome.checks == GREETING.checks
    assert outcome.measures["turns"] == 1 and outcome.measures["turns_answered"] == 1
    json_path = tmp_path / "fake" / "greeting" / "1.json"
    assert json_path.exists() and (tmp_path / "fake" / "greeting" / "1.md").exists()
    document = json.loads(json_path.read_text())
    assert document["bench"]["repetition"] == 1
    assert document["bench"]["measures"]["output_calls"] == 1.0
    assert document["bench"]["checks"] == list(GREETING.checks or ())
    assert target.opened == 1 and target.closed == 1  # closed after the run


def test_a_model_that_noops_the_greeting_fails_the_primer_check(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    outcome = run_once(fake_target([NOOP]), GREETING, repetition=1, out_dir=tmp_path)
    assert isinstance(outcome, Completed)
    failed = {f.expectation for f in outcome.report.findings if not f.ok}
    assert failed == {"expect_output()", "answers_each_input", "no_noop_on_input"}
    assert not outcome.report.ok


def test_a_chained_reply_is_a_measure_not_a_failure(
    tmp_path: Path, fake_target: Callable[..., FakeTarget]
):
    twice = SAY_HI + [_step("say", text="and one more thing")]
    scenario = Scenario(
        name="greeting",
        requires=GREETING.requires,
        timeout_s=15,
        checks=GREETING.checks,
        script=[GREETING.script[0], GREETING.script[1], GREETING.script[3]],
    )
    outcome = run_once(fake_target(twice), scenario, repetition=1, out_dir=tmp_path)
    assert isinstance(outcome, Completed)
    assert outcome.report.ok
    assert outcome.measures["output_calls"] == 2
    assert outcome.measures["unprompted_output"] == 1
    assert outcome.measures["output_words"] == 6


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
        requires=Requires(inputs=("speech_input", "gaze"), commands=("dance",)),
        script=GREETING.script,
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
    other = Scenario(name="other", requires=GREETING.requires, script=GREETING.script)
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
    assert seen == ["Completed", "Completed", "Skipped"]
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
