"""A scripted-fake **tester** `Wica` driving a scripted-fake SUT through `wica.testing` — the
whole two-loop wiring of specs/wica-tester.md (phase 1, driver (b)), key-less and deterministic,
so it **always runs**; `pytest tests-e2e -k tester` selects it.

The SUT is the shape of an interaction app in miniature: a triggering `speech_input`, a passive
`closest_user`, and a `say` output Command. The tester's JSON script is the scenario.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wica import Wica
from wica.config import WicaConfig
from wica.testing import Report, TestRun, run_script

from support import identity_serialize


def _sut() -> Wica:
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "scripted",
                "system_prompt": "You are a test double.",
                "model_kwargs": {
                    "delay_s": 0,
                    "script": [
                        {"tool_calls": [{"name": "say", "args": {"text": "hi there"}}]},
                        {"tool_calls": [{"name": "noop", "args": {}}]},
                    ],
                    "default": {"text": ""},
                },
            }
        }
    )
    sut = Wica.init(config, coalesce_window=0.0)

    async def say(text: str) -> str:
        """Speak to the person."""
        return "OK"

    sut.set_output_command(say)
    sut.world.register(
        "speech_input", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    sut.world.register("closest_user", str, serialize_fn=identity_serialize)
    return sut


def _step(command: str, /, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": command, "args": args}]}


SCRIPT = [
    _step("set_speech_input", value="hello"),
    _step("expect_output", contains="hi", within=5),
    _step("set_closest_user", value="u2"),
    _step("expect_world", key="closest_user", equals="u2"),
    _step("expect_no_action", name="say", for_=0.5),
    _step("finish", verdict="pass", summary="greeted once"),
]


def _run() -> Report:
    sut = _sut()
    sut.start()
    run = TestRun(sut, scenario="greeting", sut_label="fake")
    try:
        return run_script(run, SCRIPT, timeout=15)
    finally:
        run.close()
        sut.close()


def test_fake_tester_drives_a_fake_sut():
    report = _run()
    assert report.ok, [f for f in report.findings if not f.ok]
    assert report.verdict == "pass" and report.tester_label == "fake"
    assert [f.ok for f in report.findings] == [True, True, True]
    assert (
        report.metrics["reaction_latency"].count == 1
    )  # the greeting; the noop step has none
    assert (
        report.metrics["reactions_per_input"].min == 2
    )  # say's completion woke the noop step
    assert report.dropped_inputs == 0
    assert "say" in report.to_markdown()


def test_report_is_written_when_a_directory_is_given(
    tmp_path: Path, write_report: Callable[[Report], None]
):
    report = _run()
    write_report(
        report
    )  # a no-op without --wica-report-dir; the explicit write below is the pin
    report.write(tmp_path / "greeting.json")
    parsed = json.loads((tmp_path / "greeting.json").read_text())
    assert parsed["verdict"] == "pass"
    assert {e["kind"] for e in parsed["timeline"]} >= {
        "input_written",
        "command_issued",
        "reaction_ended",
    }
