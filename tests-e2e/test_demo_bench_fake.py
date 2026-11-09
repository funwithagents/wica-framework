"""The demo benchmark over the matrix's `fake` fixture (specs/benchmark.md, "Testing"): key-less,
deterministic, **always runs** — the e2e-fake job's fourth file. The `conversation` session and the
greeting probe through the real `DemoTarget` (`build_system` + `wire` + `start`), the committed
matrix and workload load, and a batch becomes a result."""

from __future__ import annotations

from pathlib import Path

from benchmarks.bench import MATRIX, SCENARIOS_DIR, DemoTarget
from wica.benchmark import (
    Completed,
    load_matrix,
    run_matrix,
    run_once,
    write_result,
)
from wica.testing import load_scenario, load_scenarios

from support import greeting_probe

FAKE = load_matrix(MATRIX).entry("fake")


def test_the_committed_matrix_has_one_default_per_provider_and_the_fake_fixture():
    matrix = load_matrix(MATRIX)
    defaults = [e for e in matrix.entries if "default" in e.tags]
    assert sorted(e.agent["provider"] for e in defaults) == [
        "anthropic",
        "google_genai",
        "huggingface-hub",
        "mistralai",
        "openai",
    ]
    assert all(e.enabled and e.agent.get("api_key_env") for e in defaults)
    assert all(
        e.agent["system_prompt_file"].endswith("prompts/wica.md") for e in defaults
    )
    assert not FAKE.enabled and FAKE.skip_reason() is None


def test_the_workload_loads():
    assert [s.name for s in load_scenarios(SCENARIOS_DIR)] == [
        "commands",
        "conversation",
    ]


def test_the_conversation_session_passes_through_the_demo_target(tmp_path: Path):
    conversation = load_scenario(SCENARIOS_DIR / "conversation.json")
    outcome = run_once(DemoTarget(FAKE), conversation, repetition=1, out_dir=tmp_path)
    assert isinstance(outcome, Completed)
    assert outcome.report.ok, [f for f in outcome.report.findings if not f.ok]
    assert outcome.checks == conversation.checks
    m = outcome.measures
    # Seven user utterances, every one answered through `say`; the confirmation after the
    # cancel is issued on the completion step, so it counts as the one unprompted output.
    assert (m["turns"], m["turns_answered"], m["output_calls"]) == (7, 7, 7)
    assert m["unprompted_output"] == 1 and m["noops_on_input"] == 0
    assert (tmp_path / "fake" / "conversation" / "1.json").exists()
    assert (tmp_path / "fake" / "conversation" / "1.md").exists()


def test_the_greeting_probe_passes_through_the_demo_target(tmp_path: Path):
    greeting = greeting_probe()
    outcome = run_once(DemoTarget(FAKE), greeting, repetition=1, out_dir=tmp_path)
    assert isinstance(outcome, Completed)
    assert outcome.report.ok, [f for f in outcome.report.findings if not f.ok]
    assert (
        outcome.measures["output_calls"] == 1 and outcome.measures["output_words"] == 5
    )


def test_a_batch_becomes_a_result(tmp_path: Path):
    batch = tmp_path / "batch"
    summary = run_matrix([DemoTarget(FAKE)], [greeting_probe()], k=1, out_dir=batch)
    cell = summary.cell("greeting", "fake")
    total = summary.cell("*", "fake")
    assert cell is not None and cell.pass_k == 1 and not summary.skips
    assert total is not None and total.per_run["compliance"].median == 1.0
    result = write_result(summary, tmp_path / "results" / "x_smoke", batch_dir=batch)
    (result / "description.md").write_text("# mine\n")
    write_result(summary, result, batch_dir=batch)
    assert sorted(p.name for p in result.iterdir()) == [
        "description.md",
        "findings.md",
        "runs.json",
        "summary.json",
        "summary.md",
    ]
    assert (result / "description.md").read_text() == "# mine\n"
