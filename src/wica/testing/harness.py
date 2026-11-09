"""The Harness: `TestRun`, one run of a scenario against a `Wica` under test. Implements
specs/wica-tester.md, "Harness — timeline, metrics, expectations, report" and "The shared Command
set" (driver (a): a Python test calls these methods directly).

`TestRun` owns the **timeline** — every `SutEvent` the Probe forwards plus every action the tester
takes (`set`, `wait`, the expectations, `note`, `report_issue`, `finish`), stamped on arrival —
and derives the metrics from it (`report.compute_metrics`). The expectations **block** (they wait
on a `threading.Condition` the Probe's recorder notifies) and scan forward from a **watermark**, so
a sequence of them never re-matches an earlier event and repeating one means "the next one".

Threading, read twice: the Probe's recorder runs on the SUT's loop thread and only appends under
the lock and notifies. The expectation methods block and are called from the test's own thread —
or, as tester Commands, from a worker thread (`asyncio.to_thread`) — **never** from an Event
handler, a Command body on the SUT's loop, or a World listener.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from wica.instrumentation import now
from wica.testing.probe import Probe
from wica.testing.report import (
    Finding,
    Issue,
    MetricSummary,
    Report,
    SutEvent,
    compute_metrics,
)
from wica.wica import Wica

_POLL_S = 0.02


def _t() -> float:
    return now().timestamp()


def _args_match(args: dict[str, Any] | None, where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if args is None:
        return False
    return all(key in args and args[key] == value for key, value in where.items())


def _describe(method: str, /, **kwargs: Any) -> str:
    inner = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v is not None)
    return f"{method}({inner})"


class TestRun:
    """One run against *sut*: creates the Probe (attached at once), records the timeline, and
    exposes the shared Command set as blocking methods. `close()` detaches the Probe."""

    __test__ = False  # not a pytest test class, despite the name

    def __init__(
        self,
        sut: Wica,
        *,
        scenario: str = "",
        sut_label: str = "",
        record_prompt_text: bool = False,
    ) -> None:
        self._sut = sut
        self.scenario = scenario
        self.sut_label = sut_label
        self._cond = threading.Condition()
        self._timeline: list[SutEvent] = []
        self._watermark = 0
        self._findings: list[Finding] = []
        self._issues: list[Issue] = []
        self._notes: list[str] = []
        self._verdict: str | None = None
        self._summary: str | None = None
        self._started_at = now()
        self.finished = threading.Event()
        self.probe = Probe(sut, record_prompt_text=record_prompt_text)
        self.probe.subscribe(self._record)
        self.probe.attach()

    # --- the timeline ----------------------------------------------------------------------------

    def _record(self, event: SutEvent) -> None:
        with self._cond:
            self._timeline.append(event)
            self._cond.notify_all()

    def timeline(self) -> list[SutEvent]:
        """A snapshot copy of the timeline so far."""
        with self._cond:
            return list(self._timeline)

    def _action(self, method: str, /, **kwargs: Any) -> None:
        self._record(SutEvent(kind="action", t=_t(), name=method, args=dict(kwargs)))

    def _finding(
        self, expectation: str, ok: bool, detail: str, event: SutEvent | None = None
    ) -> Finding:
        finding = Finding(
            expectation=expectation, ok=ok, detail=detail, t=_t(), event=event
        )
        with self._cond:
            self._findings.append(finding)
        return finding

    def _wait_for(
        self, predicate: Callable[[SutEvent], bool], within: float
    ) -> SutEvent | None:
        """The first event at or past the watermark satisfying *predicate*, waiting up to *within*
        seconds for it to arrive; moves the watermark past it. None on timeout (watermark kept)."""
        deadline = time.monotonic() + within
        with self._cond:
            while True:
                for index in range(self._watermark, len(self._timeline)):
                    event = self._timeline[index]
                    if predicate(event):
                        self._watermark = index + 1
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    # --- the shared Command set (blocking) -------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Write the SUT's Input *key* and move the watermark to the end of the timeline as it
        stands *before* the write — the SUT may react before this returns."""
        self._action("set", key=key, value=value)
        with self._cond:
            self._watermark = len(self._timeline)
        self.probe.set(key, value)

    def wait(self, seconds: float) -> None:
        self._action("wait", seconds=seconds)
        time.sleep(seconds)

    def expect_action(
        self,
        name: str,
        *,
        where: dict[str, Any] | None = None,
        state: str | None = None,
        result_contains: str | None = None,
        within: float = 5.0,
    ) -> Finding:
        """With no *state*: the next `command_issued` named *name* whose args contain every *where*
        item. With *state*: the next `command_state` of that name in that state (and, when given,
        whose result contains *result_contains*)."""
        label = _describe(
            "expect_action",
            name=name,
            where=where,
            state=state,
            result_contains=result_contains,
        )
        self._action(
            "expect_action",
            name=name,
            where=where,
            state=state,
            result_contains=result_contains,
            within=within,
        )

        def matches(event: SutEvent) -> bool:
            if event.name != name or not _args_match(event.args, where):
                return False
            if state is None:
                return event.kind == "command_issued"
            if event.kind != "command_state" or event.state != state:
                return False
            return result_contains is None or result_contains in (event.result or "")

        event = self._wait_for(matches, within)
        if event is None:
            return self._finding(label, False, f"no match within {within}s")
        detail = f"{event.kind} call_id={event.call_id}"
        if event.result:
            detail += f" result={event.result!r}"
        return self._finding(label, True, detail, event)

    def expect_output(
        self,
        *,
        contains: str | None = None,
        arg: str | None = None,
        within: float = 5.0,
    ) -> Finding:
        """The next call of the SUT's output Command, whatever its name; *contains* is matched
        against `args[arg]` when *arg* is given, else against every string-valued arg. Fails at
        once when the SUT has no output Command."""
        label = _describe("expect_output", contains=contains, arg=arg)
        self._action("expect_output", contains=contains, arg=arg, within=within)
        name = self.probe.output_command_name
        if name is None:
            return self._finding(label, False, "the SUT has no output Command")

        def matches(event: SutEvent) -> bool:
            if event.kind != "command_issued" or event.name != name:
                return False
            if contains is None:
                return True
            args = event.args or {}
            if arg is not None:
                return contains in str(args.get(arg, ""))
            return any(
                contains in value for value in args.values() if isinstance(value, str)
            )

        event = self._wait_for(matches, within)
        if event is None:
            return self._finding(label, False, f"no {name} call within {within}s")
        return self._finding(label, True, f"{name}({event.args})", event)

    def expect_text(
        self, *, contains: str | None = None, within: float = 5.0
    ) -> Finding:
        """The next non-empty free text (the output sink's), containing *contains* when given."""
        label = _describe("expect_text", contains=contains)
        self._action("expect_text", contains=contains, within=within)

        def matches(event: SutEvent) -> bool:
            if event.kind != "output_text" or not (event.text or "").strip():
                return False
            return contains is None or contains in (event.text or "")

        event = self._wait_for(matches, within)
        if event is None:
            return self._finding(label, False, f"no matching text within {within}s")
        return self._finding(label, True, repr(event.text), event)

    def expect_no_action(self, name: str, *, for_: float = 2.0) -> Finding:
        """Passes only if no `command_issued` named *name* arrives at or past the watermark during
        the whole *for_* seconds. The watermark is not moved."""
        label = _describe("expect_no_action", name=name, for_=for_)
        self._action("expect_no_action", name=name, for_=for_)
        deadline = time.monotonic() + for_
        with self._cond:
            start = self._watermark
            while True:
                offending = [
                    e
                    for e in self._timeline[start:]
                    if e.kind == "command_issued" and e.name == name
                ]
                if offending:
                    return self._finding(
                        label, False, f"{name} was issued", offending[0]
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._finding(label, True, f"no {name} for {for_}s")
                self._cond.wait(remaining)

    def expect_world(self, key: str, *, equals: Any, within: float = 5.0) -> Finding:
        """Polls the SUT's World until `get(key) == equals`. No timeline event is matched and the
        watermark is unchanged."""
        label = _describe("expect_world", key=key, equals=equals)
        self._action("expect_world", key=key, equals=equals, within=within)
        deadline = time.monotonic() + within
        last: Any = None
        while True:
            try:
                last = self._sut.world.get(key)
            except KeyError:
                last = KeyError(key)
            if last == equals:
                return self._finding(label, True, f"{key} == {equals!r}")
            if time.monotonic() >= deadline:
                return self._finding(label, False, f"{key} is {last!r} after {within}s")
            time.sleep(_POLL_S)

    def expect_latency(self, metric: str, *, p95_below: float) -> Finding:
        """Computes the metrics now; passes when *metric* has samples and its p95 is below the
        threshold. The one expectation that is a threshold — never a built-in default."""
        label = _describe("expect_latency", metric=metric, p95_below=p95_below)
        self._action("expect_latency", metric=metric, p95_below=p95_below)
        summary = self.metrics()[0].get(metric)
        if summary is None:
            return self._finding(label, False, f"no samples for {metric}")
        ok = summary.p95 < p95_below
        return self._finding(
            label, ok, f"{metric} p95 = {summary.p95:.3f} over {summary.count}"
        )

    def note(self, text: str) -> None:
        self._action("note", text=text)
        with self._cond:
            self._notes.append(text)

    def report_issue(self, severity: str, what: str, evidence: str) -> None:
        self._action("report_issue", severity=severity, what=what, evidence=evidence)
        with self._cond:
            self._issues.append(
                Issue(severity=severity, what=what, evidence=evidence, t=_t())
            )

    def finish(self, verdict: str, summary: str = "") -> None:
        """Record the verdict and mark the run finished (`finished` is set)."""
        self._action("finish", verdict=verdict, summary=summary)
        with self._cond:
            self._verdict = verdict
            self._summary = summary or None
        self.finished.set()

    # --- results ---------------------------------------------------------------------------------

    def metrics(self) -> tuple[dict[str, MetricSummary], int]:
        return compute_metrics(
            self.timeline(), output_command_name=self.probe.output_command_name
        )

    def findings(self) -> list[Finding]:
        with self._cond:
            return list(self._findings)

    def add_finding(self, finding: Finding) -> None:
        """Append a finding made outside the expectation methods (a driver's own, e.g. a script
        that never called `finish`)."""
        with self._cond:
            self._findings.append(finding)

    def report(self, *, tester_label: str | None = None) -> Report:
        metrics, dropped = self.metrics()
        with self._cond:
            return Report(
                scenario=self.scenario,
                sut_label=self.sut_label,
                tester_label=tester_label,
                started_at=self._started_at.isoformat(),
                finished_at=now().isoformat(),
                timeline=list(self._timeline),
                metrics=metrics,
                dropped_inputs=dropped,
                findings=list(self._findings),
                issues=list(self._issues),
                notes=list(self._notes),
                verdict=self._verdict,
                summary=self._summary,
            )

    def close(self) -> None:
        """Detach the Probe from the SUT (idempotent)."""
        self.probe.detach()
