"""The live regression gate: the behaviours WICA promises, on every committed provider, against the
test robot (`robot_app.py`) — specs/testing.md, "The live set as a regression gate".

Each behaviour is one scenario file in `scenarios/` (the benchmark's format), played two ways:

- **focused** (`test_step`): one test per (provider, step), each on a fresh robot — the diagnostic
  that names the behaviour that broke;
- **conversation** (`TestConversation`): the steps in order as one test each, on one robot per
  provider with `history_reactions = 4`, so the history is cut several times; the last test checks
  a cut happened. Each step plays on its own `TestRun` (a scenario ends its run); before it, the
  previous step's run waits for the robot to settle and anything a failed step left running is
  cancelled. No step relies on an earlier one, so `-k` on one step runs it alone.

One run per model: a missed expect, a timed-out wait or a failed check fails the step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage

from wica import Wica
from wica import CommandExecution
from wica.testing import Report, TestRun, load_scenarios, run_scenario

from robot_app import SETTLE_WITHIN_S, build_robot
from support import E2E_DIR, PROVIDER_CONFIGS

SCENARIOS = {s.name: s for s in load_scenarios(E2E_DIR / "scenarios")}
STEPS = [
    "greeting",
    "act_on_request",
    "background",
    "long_answer",
    "interrupt",
    "memory",
    "goodbye",
]
CONVERSATION_HISTORY_REACTIONS = 4

# (config stem, step) pairs a committed model does not pass reliably, found while vetting a step
# (at least three runs by hand on every model): xfail(strict=False) with the reason, never dropped.
# Remove a pair once its model passes the step reliably. See specs/testing.md ("What fails").
KNOWN_FLAKY: dict[tuple[str, str], str] = {
    ("e2e.huggingface-hub.config", "background"): (
        "gpt-oss-120b speaks when a background sound wakes it (3 of 6 attempts, 2026-11-10): "
        "it checks in or follows up the previous turn instead of calling noop"
    ),
    ("e2e.mistralai.config", "background"): (
        "ministral-14b, woken by a background sound right after the dance step, dances again "
        "(1 of 6 attempts, 2026-11-10): it takes the wake as a cue to repeat the last request"
    ),
}

WriteReport = Callable[[Report], None]


def mark_known_flaky(request: pytest.FixtureRequest, label: str, step: str) -> None:
    reason = KNOWN_FLAKY.get((label, step))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))


def _failure(report: Report) -> str:
    failed = [f"{f.expectation}: {f.detail}" for f in report.findings if not f.ok]
    stopped = f"stopped at {report.stopped_at}; " if report.stopped_at else ""
    return stopped + "; ".join(failed)


def play(
    wica: Wica, step: str, sut_label: str, write_report: WriteReport
) -> tuple[TestRun, Report]:
    """Play *step*'s scenario on a new run over *wica* and write its report. Returns the run,
    still attached (the caller closes it), and the report the caller asserts on."""
    run = TestRun(wica, scenario=step, sut_label=sut_label)
    report = run_scenario(run, SCENARIOS[step])
    write_report(report)
    return run, report


def test_every_step_has_its_scenario_file() -> None:
    assert sorted(STEPS) == sorted(SCENARIOS)


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_step(
    config_path: Path,
    step: str,
    write_report: WriteReport,
    request: pytest.FixtureRequest,
) -> None:
    mark_known_flaky(request, config_path.stem, step)
    wica = build_robot(config_path)
    try:
        run, report = play(wica, step, config_path.stem, write_report)
        run.close()
    finally:
        wica.close()
    assert report.ok, _failure(report)


class Conversation:
    """One robot shared by a provider's conversation steps, plus what the steps leave behind: the
    last step's run (to settle on) and the size of every prompt (to see a history cut)."""

    def __init__(self, wica: Wica, label: str) -> None:
        self.wica = wica
        self.label = label
        self.last_run: TestRun | None = None
        self.prompt_sizes: list[int] = []
        self.steps_played = 0
        wica.on_agent_prompt.subscribe(self._on_prompt)

    def _on_prompt(self, messages: list[BaseMessage]) -> None:
        self.prompt_sizes.append(len(messages))

    def step(
        self, step: str, write_report: WriteReport, request: pytest.FixtureRequest
    ) -> None:
        mark_known_flaky(request, self.label, step)
        self._settle()
        self.steps_played += 1
        # Kept, passed or failed: the next step settles on it.
        self.last_run, report = play(self.wica, step, self.label, write_report)
        assert report.ok, _failure(report)

    def _settle(self) -> None:
        """Start the next step from a quiet robot, whatever the last one left: wait for it to
        settle, cancel what still runs (a failed step's speech or dance), and settle again."""
        run = self.last_run
        if run is None:
            return
        try:
            if run.wait_for("idle", within=SETTLE_WITHIN_S).ok:
                return
            for entry in self.wica.world.get_prompt_entries():
                value = entry.current.value
                if isinstance(value, CommandExecution) and value.state == "running":
                    self.wica.agent.cancel_command(
                        entry.key.removeprefix("agent:command:")
                    )
            run.wait_for("idle", within=SETTLE_WITHIN_S)
        finally:
            run.close()
            self.last_run = None

    def close(self) -> None:
        if self.last_run is not None:
            self.last_run.close()
        self.wica.on_agent_prompt.unsubscribe(self._on_prompt)
        self.wica.close()


@pytest.mark.parametrize(
    "config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem, scope="class"
)
class TestConversation:
    """The steps in order, on one robot per provider; one test per step."""

    @pytest.fixture(scope="class")
    def conversation(self, config_path: Path) -> Iterator[Conversation]:
        wica = build_robot(
            config_path, history_reactions=CONVERSATION_HISTORY_REACTIONS
        )
        conversation = Conversation(wica, config_path.stem)
        try:
            yield conversation
        finally:
            conversation.close()

    def test_greeting(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("greeting", write_report, request)

    def test_act_on_request(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("act_on_request", write_report, request)

    def test_background(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("background", write_report, request)

    def test_long_answer(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("long_answer", write_report, request)

    def test_interrupt(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("interrupt", write_report, request)

    def test_memory(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("memory", write_report, request)

    def test_goodbye(
        self,
        conversation: Conversation,
        write_report: WriteReport,
        request: pytest.FixtureRequest,
    ):
        conversation.step("goodbye", write_report, request)

    def test_history_was_cut(self, conversation: Conversation):
        """With `history_reactions = 4` the conversation outgrows the window several times: some
        prompt must hold fewer messages than the one before it. Checked without asking the model
        to show it; recall across a cut is not a promise (specs/agent.md, "History window")."""
        if conversation.steps_played < len(STEPS):
            pytest.skip("needs the whole conversation")
        sizes = conversation.prompt_sizes
        assert any(later < earlier for earlier, later in zip(sizes, sizes[1:])), (
            f"no history cut over {len(sizes)} prompts: {sizes}"
        )
