"""Live-provider round trips through the real entrypoint (`Wica.init` over a committed config),
parametrized over `PROVIDER_CONFIGS` and skipping without the key — see specs/testing.md.

Each case drives its `Wica` through `wica.testing.TestRun` (specs/wica-tester.md, driver (a)):
the Harness records the timeline and the reaction / pure-model latencies as a by-product, and
`--wica-report-dir` keeps one JSON report per test per provider. The assertions stay behavioral
(a non-empty reply came, the Command was called), never a specific string and never a latency
threshold: one run is one sample per provider, a smoke test, not a benchmark.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from wica.content import TextPart
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
def test_real_tool_calling_round_trip(
    config_path: Path, write_report: Callable[[Report], None]
):
    """A live provider issuing a real tool call through Wica: the model calls the registered Command,
    whose execution is tracked as a World entry that goes running → complete."""
    wica = real_wica(config_path)

    async def add(a: int, b: int) -> int:
        """Add two integers and return their sum."""
        return a + b

    wica.register_command(add)
    wica.world.register(
        "prompt", str, serialize_fn=identity_serialize, triggers_llm_call=True
    )
    wica.start()
    run = TestRun(wica, scenario="tool calling", sut_label=config_path.stem)
    try:
        run.set("prompt", "What is 2 + 2? Use the add tool.")
        assert run.expect_action("add", within=WAIT_TIMEOUT).ok
        done = run.expect_action(
            "add", state="complete", result_contains="4", within=WAIT_TIMEOUT
        )
        assert done.ok, done.detail
        # Deliberately not asserting the model also speaks a follow-up text reply here: whether
        # free text follows a completed tool call is model-decided, not a WICA guarantee (see
        # agent.md open question #2) — asserting it would make this test flaky on real
        # non-determinism unrelated to the tool-calling round trip under test. The sink/output
        # path itself is already covered by test_plain_text_round_trip.
        write_report(run.report())
    finally:
        run.close()
        wica.close()


# A voice assistant's persona, the shape that exposed the plain-text failure: with it and the
# output Command unforced, ministral-3b answered a greeting in plain text on every run.
VOICE_PERSONA = """\
You are a friendly voice assistant. A person talks to you out loud and you talk back — a spoken \
conversation, not a text chat.

## How you speak

Everything the person hears from you goes through your `say` command — that is your voice. Free \
text you produce is only private thinking; it is never heard. So whenever you want to reply, \
**call `say` with what you want said.** If you have nothing to add, do not force a reply.

Keep replies short and natural to hear aloud: a sentence or two, plain spoken language, no lists \
or emoji.

## Turns

Each thing the person says arrives as one complete utterance; reply to it, then wait for the next.

## Dictation

If the person wants to dictate something longer — a message, a note — call `start_dictation`: \
their next speech then arrives as one longer passage. Use it only when they clearly want to \
dictate.
"""


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_output_command_live(config_path: Path, write_report: Callable[[Report], None]):
    """A live provider honoring the output Command contract: with an output Command set, the model
    must address the person by *calling* it — its free text is private reasoning, never heard —
    and end the follow-up step with `noop`. The binding forces a tool call (`force_tool_call`,
    default true — specs/agent.md, "Forcing a tool call"); this is the case that motivated it: a
    voice persona under which a weak model replies in plain text when not forced. A model that
    answers in plain text leaves `say` uncalled and its reply in the free-text stream — the
    timeline's `output_text` events are the evidence. One application action (`start_dictation`)
    sits beside `say`, as in a real voice app."""

    async def say(text: str) -> str:
        """Say something out loud to the person."""
        return "spoken"

    async def start_dictation() -> str:
        """Switch speech input to one long dictated passage."""
        return "dictation started; it ends on a pause"

    wica = real_wica(config_path, system_prompt=VOICE_PERSONA)
    wica.set_output_command(say)
    wica.register_command(start_dictation)
    wica.world.register(
        "user_speech",
        str,
        serialize_fn=lambda value, previous: [
            TextPart(f'The person said to you: "{value}"')
        ],
        triggers_llm_call=True,
    )
    wica.start()
    run = TestRun(wica, scenario="output command", sut_label=config_path.stem)
    try:
        run.set("user_speech", "Hello there. How are you?")
        # A text-only reply ends the reaction with nothing re-triggering, so a model that ignores
        # the contract never calls say and this wait times out.
        finding = run.expect_output(within=WAIT_TIMEOUT)
        free_text = [e.text for e in run.timeline() if e.kind == "output_text"]
        assert finding.ok, (
            "the model never called the output Command; free text it produced instead: "
            f"{free_text!r}"
        )
        assert finding.event is not None
        assert str((finding.event.args or {}).get("text", "")).strip()
        # say's completion re-triggers a step; with nothing more to say the model ends it with
        # noop — under the forced binding, its only way to do nothing.
        assert run.expect_action("noop", within=WAIT_TIMEOUT).ok
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
