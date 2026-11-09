"""The WicaTester data model: the normalized `SutEvent`, findings, issues, metric summaries and the
`Report` — plus `compute_metrics`, the one place the Harness's numbers are defined. Implements
specs/wica-tester.md, "Harness — timeline, metrics, expectations, report".

Every stamp here is the framework's wall clock (`wica.instrumentation.now()`) as POSIX seconds, so
a Harness action and a framework record (a `ReactionTrace`, a `CommandTrace`) subtract. The
framework-stamped metrics come from the records through core's own definitions
(specs/instrumentation.md, "Metrics"); only the two that pair a *tester action* with the SUT's
reply (`steps_per_turn`, `barge_in`) are timeline heuristics.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from wica.instrumentation import (
    CommandTrace,
    ReactionTrace,
    reaction_latency,
    reactions_per_input,
)

SutEventKind = Literal[
    "input_written",
    "trigger_raw",
    "trigger_observed",
    "trigger_dropped",
    "prompt",
    "command_issued",
    "command_state",
    "command_ended",
    "output_text",
    "reaction_ended",
    "action",
]


@dataclass(frozen=True)
class SutEvent:
    """One record of the timeline: something the SUT did, or an action the tester took."""

    kind: SutEventKind
    t: float  # wall-clock POSIX seconds at arrival (wica.instrumentation.now())
    key: str | None = None  # World key (inputs, triggers, agent:command:<id>)
    version_id: int | None = None  # WorldEntry.current.id for input_written / trigger_*
    name: str | None = None  # Command name, or the tester action's name for "action"
    call_id: str | None = None
    args: dict[str, Any] | None = (
        None  # Command args, or the action's kwargs for "action"
    )
    state: str | None = None  # running / complete / failed / cancelled
    result: str | None = None
    error: str | None = None
    text: str | None = None  # output_text; the prompt text when recorded
    message_count: int | None = None  # prompt only
    reaction: ReactionTrace | None = (
        None  # reaction_ended: the framework's record, whole
    )
    command: CommandTrace | None = None  # command_ended: likewise


@dataclass(frozen=True)
class Finding:
    """The outcome of one expectation: a hard pass/fail with what matched or why it did not."""

    expectation: str  # e.g. "expect_text(contains='hi')"
    ok: bool
    detail: str
    t: float
    event: SutEvent | None = None  # the matching event, when there is one


@dataclass(frozen=True)
class Issue:
    """A soft, tester-reported problem with its evidence (never a pass/fail by itself)."""

    severity: (
        str  # "low" | "medium" | "high" — free text, validated by nobody in phase 1
    )
    what: str
    evidence: str
    t: float


@dataclass(frozen=True)
class MetricSummary:
    count: int
    min: float
    median: float
    p95: float
    max: float


def summarize(samples: list[float]) -> MetricSummary | None:
    """Distribution of *samples*; None when there is none. p95 is the nearest-rank percentile
    (`sorted[ceil(0.95 * n) - 1]`), the median the usual midpoint."""
    if not samples:
        return None
    ordered = sorted(samples)
    n = len(ordered)
    half = n // 2
    median = ordered[half] if n % 2 else (ordered[half - 1] + ordered[half]) / 2
    p95 = ordered[max(math.ceil(0.95 * n) - 1, 0)]
    return MetricSummary(
        count=n, min=ordered[0], median=median, p95=p95, max=ordered[-1]
    )


def compute_metrics(
    timeline: list[SutEvent], *, output_command_name: str | None
) -> tuple[dict[str, MetricSummary], int]:
    """The metrics of a timeline, as `(summaries by key, dropped-input count)`. A metric with no
    sample is omitted from the dict. See the module docstring for which come from the framework's
    records and which are timeline heuristics."""
    samples: dict[str, list[float]] = {}

    def add(key: str, value: float | None) -> None:
        if value is not None:
            samples.setdefault(key, []).append(float(value))

    reactions = [e.reaction for e in timeline if e.reaction is not None]
    for reaction in reactions:
        add("reaction_latency", reaction_latency(reaction))
        add("model_latency", reaction.model_latency)
        add("coalescing_wait", reaction.coalescing_wait)
        add("busy_time", reaction.busy_time)
        if reaction.usage is not None:
            add("input_tokens", reaction.usage.input_tokens)
            add("output_tokens", reaction.usage.output_tokens)
    for event in timeline:
        if event.command is not None:
            add("command_duration", event.command.duration)
    for count in reactions_per_input(reactions):
        add("reactions_per_input", count)

    # steps_per_turn: prompts between one input_written and the next (or the end).
    input_indexes = [i for i, e in enumerate(timeline) if e.kind == "input_written"]
    for n, start in enumerate(input_indexes):
        end = input_indexes[n + 1] if n + 1 < len(input_indexes) else len(timeline)
        add(
            "steps_per_turn",
            sum(1 for e in timeline[start:end] if e.kind == "prompt"),
        )

    # barge_in: an input written while the output Command runs → the next cancel_command.
    if output_command_name is not None:
        running_output: set[str] = set()
        pending_inputs: list[float] = []
        for event in timeline:
            if event.kind == "command_state" and event.name == output_command_name:
                if event.state == "running" and event.call_id is not None:
                    running_output.add(event.call_id)
                elif event.call_id is not None:
                    running_output.discard(event.call_id)
            elif event.kind == "input_written" and running_output:
                pending_inputs.append(event.t)
            elif (
                event.kind == "command_issued"
                and event.name == "cancel_command"
                and pending_inputs
            ):
                add("barge_in", event.t - pending_inputs.pop(0))

    dropped = sum(1 for e in timeline if e.kind == "trigger_dropped")
    metrics = {
        key: summary
        for key, values in samples.items()
        if (summary := summarize(values)) is not None
    }
    return metrics, dropped


@dataclass(frozen=True)
class Report:
    """One run's record: the timeline, the metrics, the findings (hard), the issues (soft) and the
    verdict. Serializable to JSON and Markdown; `ok` is what a pytest asserts."""

    scenario: str
    sut_label: str
    tester_label: str | None
    started_at: str  # ISO-8601 wall clock
    finished_at: str
    timeline: list[SutEvent]
    metrics: dict[str, MetricSummary]
    dropped_inputs: int
    findings: list[Finding]
    issues: list[Issue]
    notes: list[str]
    verdict: str | None  # "pass" | "fail" | None when finish() was never called
    summary: str | None

    @property
    def ok(self) -> bool:
        """Every finding passed and the verdict is not "fail"."""
        return all(f.ok for f in self.findings) and self.verdict != "fail"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.scenario or 'scenario'} — SUT `{self.sut_label or '?'}`"
            + (f", tester `{self.tester_label}`" if self.tester_label else ""),
            "",
            f"Verdict: **{self.verdict or 'none'}**"
            + (f" — {self.summary}" if self.summary else ""),
            f"Run: {self.started_at} → {self.finished_at}",
            "",
            "## Metrics",
            "",
            "| metric | count | min | median | p95 | max |",
            "|---|---|---|---|---|---|",
        ]
        for key, m in self.metrics.items():
            lines.append(
                f"| {key} | {m.count} | {m.min:.3f} | {m.median:.3f} | {m.p95:.3f} | {m.max:.3f} |"
            )
        lines += ["", f"Dropped inputs: {self.dropped_inputs}", "", "## Findings", ""]
        for f in self.findings:
            lines.append(f"- {'✅' if f.ok else '❌'} `{f.expectation}` — {f.detail}")
        if not self.findings:
            lines.append("- (none)")
        if self.issues:
            lines += ["", "## Issues", ""]
            for issue in self.issues:
                lines.append(
                    f"- **{issue.severity}** {issue.what} — evidence: {issue.evidence}"
                )
        if self.notes:
            lines += ["", "## Notes", ""]
            lines += [f"- {note}" for note in self.notes]
        lines += [
            "",
            "## Timeline",
            "",
            "| t | kind | key / name | state | text |",
            "|---|---|---|---|---|",
        ]
        origin = self.timeline[0].t if self.timeline else 0.0
        for e in self.timeline:
            what = e.name or e.key or ""
            text = e.text or e.result or e.error or ""
            if e.args:
                what += f" {_short(str(e.args), 60)}"
            lines.append(
                f"| {e.t - origin:.3f} | {e.kind} | {_short(what, 80)} | {e.state or ''} | "
                f"{_short(text, 60)} |"
            )
        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> None:
        """Write the report as `.json` or `.md` by suffix (JSON for anything else); creates the
        parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_markdown() if path.suffix == ".md" else self.to_json()
        path.write_text(text, encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _short(text: str, limit: int) -> str:
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"
