"""The scripted-fake tester `Wica` — driver (b) of specs/wica-tester.md ("Drivers"): a second
`Wica` over `provider: "fake"` whose script *is* the scenario, an ordered list of tester Commands.

The tester's Command set is the Harness API (`TestRun`'s methods) registered as Commands:
`set_<key>` per settable SUT Input (read from the Probe), `wait`, the `expect_*` family, `note`,
`report_issue` and `finish`. Two rules make a run deterministic: the mirror entries (`sut:action`,
`sut:output`, `sut:reasoning`) are registered non-triggering, so only the tester's own Command
completions wake its next step — script step *k* runs exactly when Command *k−1* completes — and
the waiting lives in the Commands (each `expect_*` blocks off the tester's loop via
`asyncio.to_thread`), so the script never guesses the SUT's timing. `finish` is registered with
`triggers_on_completion=False`: its completion wakes no step, which ends the chain. The fake's
`default` is `noop`, so a script that forgets `finish` ends quietly and `run_script` times out.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import create_model

from wica.command import Command
from wica.config import WicaConfig
from wica.content import Content, TextPart
from wica.testing.harness import TestRun
from wica.testing.report import Finding, Report, SutEvent
from wica.wica import Wica

_SETTABLE_TYPES: tuple[type, ...] = (str, int, float, bool)

SET_PREFIX = "set_"

# The tester's fixed Command set (specs/wica-tester.md, "The shared Command set"); `set_<key>` is
# the one open-ended family, one per settable SUT Input. `tests/testing/test_scripted_tester.py`
# pins this set against what `build_tester` registers, so the two cannot drift.
TESTER_COMMANDS = frozenset(
    {
        "wait",
        "wait_for",
        "expect_action",
        "expect_output",
        "expect_text",
        "expect_no_action",
        "expect_world",
        "expect_latency",
        "note",
        "report_issue",
        "finish",
    }
)
_MIRROR_KEYS = ("sut:action", "sut:output", "sut:reasoning")


def _text(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


def set_command_name(key: str) -> str:
    """The tester Command that writes the SUT Input *key*: `set_<key>`, sanitized."""
    return SET_PREFIX + re.sub(r"[^A-Za-z0-9_]", "_", key)


def build_tester(
    run: TestRun,
    script: list[dict[str, Any]],
    *,
    persona: str = "You are a scripted tester.",
) -> Wica:
    """A tester `Wica` over the fake provider running *script* against *run*'s SUT: the mirror
    entries follow the SUT through the Probe, and the shared Command set is registered on it.
    Not started; `run_script` is the usual way to drive it."""
    config = WicaConfig.from_dict(
        {
            "agent": {
                "provider": "fake",
                "model": "tester",
                "system_prompt": persona,
                "model_kwargs": {
                    "script": script,
                    "delay_s": 0,
                    "default": {"tool_calls": [{"name": "noop", "args": {}}]},
                },
            }
        }
    )
    tester = Wica.init(config, coalesce_window=0.0)
    world = tester.world
    for key in _MIRROR_KEYS:
        world.register(
            key,
            str,
            serialize_fn=_text,
            include_in_prompt=True,
            triggers_llm_call=False,
        )
    # The kick-off entry: its first update fires script step 1.
    world.register("scenario", str, serialize_fn=_text, triggers_llm_call=True)

    def mirror(event: SutEvent) -> None:
        # Runs on the SUT's loop thread; tester.world.update hops to the tester's loop itself.
        # Nothing reads these in phase 1 (the fake tester ignores its prompt) — they are the live
        # tester's perception, registered now to prove the two-loop wiring.
        if not tester.is_running:
            return
        try:
            if event.kind in ("command_issued", "command_state"):
                world.update(
                    "sut:action",
                    f"{event.name}({event.args}) [{event.state or 'issued'}]",
                )
                if (
                    event.kind == "command_issued"
                    and event.name == run.probe.output_command_name
                ):
                    world.update("sut:output", f"{event.name}({event.args})")
            elif event.kind == "output_text":
                world.update("sut:reasoning", event.text or "")
        except RuntimeError:
            pass  # the tester stopped between the check and the write

    run.probe.subscribe(mirror)

    for key, entry_config in run.probe.inputs.items():
        if entry_config.type not in _SETTABLE_TYPES:
            continue  # reachable from Python only (run.set)
        tester.register_command(_set_command(run, key, entry_config.type))

    def ok_or_failed(finding: Finding) -> str:
        """A failed expect stops the run as failed, a timed-out wait as stalled: the run's
        `finished` is set, `run_script` returns, and any step still to come is refused."""
        if finding.ok:
            return "ok"
        run.stop("stalled" if finding.kind == "wait" else "failed", finding.expectation)
        return f"stopped: {finding.detail}"

    async def wait(seconds: float) -> str:
        """Let *seconds* pass before the next step."""
        await asyncio.to_thread(run.wait, seconds)
        return "ok"

    async def expect_action(
        name: str,
        where: dict[str, Any] | None = None,
        state: str | None = None,
        result_contains: str | None = None,
        within: float = 5.0,
    ) -> str:
        """Expect the SUT to issue the Command *name* (optionally with these args / in this state)."""
        finding = await asyncio.to_thread(
            run.expect_action,
            name,
            where=where,
            state=state,
            result_contains=result_contains,
            within=within,
        )
        return ok_or_failed(finding)

    async def expect_output(
        contains: str | None = None, arg: str | None = None, within: float = 5.0
    ) -> str:
        """Expect the SUT to call its output Command (its reply to the user)."""
        finding = await asyncio.to_thread(
            run.expect_output, contains=contains, arg=arg, within=within
        )
        return ok_or_failed(finding)

    async def expect_text(contains: str | None = None, within: float = 5.0) -> str:
        """Expect free text from the SUT (its output sink)."""
        finding = await asyncio.to_thread(
            run.expect_text, contains=contains, within=within
        )
        return ok_or_failed(finding)

    async def expect_no_action(name: str, for_: float = 2.0) -> str:
        """Expect the SUT *not* to issue the Command *name* for *for_* seconds."""
        finding = await asyncio.to_thread(run.expect_no_action, name, for_=for_)
        return ok_or_failed(finding)

    async def wait_for(condition: str | dict[str, Any], within: float = 60.0) -> str:
        """Pace the conversation: wait until *condition* holds — "idle", "speaking",
        {"running": name} or {"seconds": n}."""
        finding = await asyncio.to_thread(run.wait_for, condition, within=within)
        return ok_or_failed(finding)

    async def expect_world(key: str, equals: str, within: float = 5.0) -> str:
        """Expect the SUT's World entry *key* to equal *equals*."""
        finding = await asyncio.to_thread(
            run.expect_world, key, equals=equals, within=within
        )
        return ok_or_failed(finding)

    async def expect_latency(metric: str, p95_below: float) -> str:
        """Expect the p95 of *metric* so far to be below *p95_below* seconds."""
        finding = await asyncio.to_thread(
            run.expect_latency, metric, p95_below=p95_below
        )
        return ok_or_failed(finding)

    async def note(text: str) -> str:
        """Add a note to the report."""
        await asyncio.to_thread(run.note, text)
        return "ok"

    async def report_issue(severity: str, what: str, evidence: str) -> str:
        """Report a problem with the SUT, citing the observed evidence."""
        await asyncio.to_thread(run.report_issue, severity, what, evidence)
        return "ok"

    async def finish(verdict: str, summary: str = "") -> str:
        """End the run with a verdict ("pass" / "fail") and a summary."""
        await asyncio.to_thread(run.finish, verdict, summary)
        return "ok"

    for fn in (
        wait,
        expect_action,
        expect_output,
        expect_text,
        expect_no_action,
        wait_for,
        expect_world,
        expect_latency,
        note,
        report_issue,
    ):
        tester.register_command(fn)
    tester.register_command(Command(finish, triggers_on_completion=False))
    return tester


def _set_command(run: TestRun, key: str, entry_type: type) -> Command:
    name = set_command_name(key)
    fields: dict[str, Any] = {"value": (entry_type, ...)}
    args_schema = create_model(f"Set_{name}", **fields)

    async def fn(value: Any) -> str:
        await asyncio.to_thread(run.set, key, value)
        return "ok"

    tool = StructuredTool.from_function(
        coroutine=fn,
        name=name,
        description=f"Write the SUT's {key!r} input.",
        args_schema=args_schema,
    )
    return Command(tool)


def run_script(
    run: TestRun, script: list[dict[str, Any]], *, timeout: float = 30.0
) -> Report:
    """Drive *run* with a scripted-fake tester built from *script*: start it, fire step 1, wait
    for `finish` (up to *timeout*), close it and return the report (`tester_label="fake"`). A
    script that never calls `finish` yields a failed finding and `verdict=None`."""
    tester = build_tester(run, script)
    tester.start()
    try:
        tester.world.update("scenario", run.scenario or "scripted")
        if not run.finished.wait(timeout):
            run.add_finding(
                Finding(
                    expectation="run_script",
                    ok=False,
                    detail=f"script did not call finish within {timeout}s",
                    t=run.timeline()[-1].t if run.timeline() else 0.0,
                )
            )
    finally:
        tester.close()
    return run.report(tester_label="fake")
