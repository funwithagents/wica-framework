"""Live-provider round trips through the real entrypoint (`Wica.init` over a committed config),
parametrized over `PROVIDER_CONFIGS` and skipping without the key — see specs/testing.md.

Each case drives its `Wica` through `wica.testing.TestRun` (specs/wica-tester.md, driver (a)):
the Harness records the timeline and the reaction / pure-model latencies as a by-product, and
`--wica-report-dir` keeps one JSON report per test per provider. These are the cases the test
robot does not cover (test_robot.py, specs/testing.md "The live set as a regression gate"): a
reply through a bare output Command taking the whole text, and `<noop/>` with no output Command.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from wica import ReactionTrace
from wica.testing import Report, TestRun

from support import PROVIDER_CONFIGS, identity_serialize, real_wica

WAIT_TIMEOUT = 15.0


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_spoken_reply_round_trip(
    config_path: Path, write_report: Callable[[Report], None]
):
    """A live provider, driven through the real entrypoint: Wica.init → set an output Command →
    register an Input → start → an update wakes the Agent, which reasons and replies by writing
    the output Command's tag in its text (taught by the runtime primer alone); the tag's body
    reaches the output Command."""
    wica = real_wica(config_path)
    spoken: list[str] = []

    async def say(text: str) -> str:
        """Speak out loud to the user."""
        spoken.append(text)
        return "Said it."

    wica.set_output_command(say)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    run = TestRun(wica, scenario="spoken reply", sut_label=config_path.stem)
    try:
        run.set("prompt", "Say hello in one short sentence.")
        assert run.expect_output(within=WAIT_TIMEOUT).ok
        assert run.expect_action("say", state="complete", within=WAIT_TIMEOUT).ok
        assert len(spoken) == 1 and spoken[0].strip()
        assert "<" not in spoken[0]  # the body alone, never the tag
        write_report(run.report())
    finally:
        run.close()
        wica.close()


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_noop_empty_reaction_live(
    config_path: Path, write_report: Callable[[Report], None]
):
    """A live provider *declining to act*: given an observation that plainly needs no response, and
    a persona telling it to do nothing then, the model should write `<noop/>` as its whole
    response rather than reply. This verifies models actually reach for it — explained only in
    the runtime primer, with no bespoke prompting here. The declined reaction issues no Command
    and makes no World entry: it is observable as the reaction record's `noop` flag."""
    persona = (
        "You are a silent monitoring agent. You act ONLY when an observation genuinely requires a "
        "response. When an observation requires nothing from you, you MUST do nothing. Never "
        "write anything else unless a response is actually needed."
    )
    wica = real_wica(config_path, system_prompt=persona)
    wica.world.register(
        "status", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    reactions: list[ReactionTrace] = []
    wica.on_agent_reaction_ended.subscribe(reactions.append)
    wica.start()
    run = TestRun(wica, scenario="noop", sut_label=config_path.stem)
    try:
        run.set(
            "status",
            "Routine heartbeat: all systems nominal, no user is present, nothing requires "
            "your attention.",
        )
        assert run.wait_for("idle", within=WAIT_TIMEOUT).ok
        # If the model instead wrote text, or called a tool, the flag is not set — which is the
        # real signal under test: whether the provider declines on demand.
        assert [(r.outcome, r.noop, r.command_call_ids) for r in reactions] == [
            ("ok", True, ())
        ]
        write_report(run.report())
    finally:
        run.close()
        wica.close()
