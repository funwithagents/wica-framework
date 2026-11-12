"""Live-provider round trips through the real entrypoint (`Wica.init` over a committed config),
parametrized over `PROVIDER_CONFIGS` and skipping without the key — see specs/testing.md.

Each case drives its `Wica` through `wica.testing.TestRun` (specs/wica-tester.md, driver (a)):
the Harness records the timeline and the reaction / pure-model latencies as a by-product, and
`--wica-report-dir` keeps one JSON report per test per provider. These are the cases the test
robot does not cover (test_robot.py, specs/testing.md "The live set as a regression gate"): a
reply in free text with no output Command, and `noop` with no output Command (not forced).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from wica.testing import Report, TestRun

from support import PROVIDER_CONFIGS, identity_serialize, real_wica

WAIT_TIMEOUT = 15.0


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_plain_text_round_trip(
    config_path: Path, write_report: Callable[[Report], None]
):
    """A live provider, driven through the real entrypoint: Wica.init → register an Input → start →
    an update wakes the Agent, which reasons and replies in free text (observed through the
    `on_agent_text` Event — no sink is needed for it to fire)."""
    wica = real_wica(config_path)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    run = TestRun(wica, scenario="plain text", sut_label=config_path.stem)
    try:
        run.set("prompt", "Say hello in one short sentence.")
        assert run.expect_text(within=WAIT_TIMEOUT).ok
        write_report(run.report())
    finally:
        run.close()
        wica.close()


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_noop_empty_reaction_live(
    config_path: Path, write_report: Callable[[Report], None]
):
    """A live provider *declining to act*: given an observation that plainly needs no response, and
    a persona telling it to do nothing then, the model should call the auto-registered `noop`
    Command rather than reply. This verifies models actually reach for `noop` — explained only in
    the runtime primer + the tool's own description, with no bespoke prompting here. The declined
    reaction is observable as a `command_issued` record named `noop` (noop makes no World entry
    and no output)."""
    persona = (
        "You are a silent monitoring agent. You act ONLY when an observation genuinely requires a "
        "response. When an observation requires nothing from you, you MUST call the noop tool to "
        "take no action. Never reply with text unless a response is actually needed."
    )
    wica = real_wica(config_path, system_prompt=persona)
    wica.world.register(
        "status", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    run = TestRun(wica, scenario="noop", sut_label=config_path.stem)
    try:
        run.set(
            "status",
            "Routine heartbeat: all systems nominal, no user is present, nothing requires "
            "your attention.",
        )
        # If the model instead replied with text, no noop is ever issued and the wait fails —
        # which is the real signal under test: whether the provider can produce an empty reaction
        # on demand.
        assert run.expect_action("noop", within=WAIT_TIMEOUT).ok
        write_report(run.report())
    finally:
        run.close()
        wica.close()
