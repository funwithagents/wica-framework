"""The benchmark summary: one row per configuration pooled over every run (the global result),
then per (scenario, configuration) pass rate, pass^k, what failed and the pooled distributions —
and how a batch becomes a kept result. Implements specs/benchmark.md, "The summary".

No thresholds anywhere: the summary is a comparison, the reader draws the line. A **batch** (one
runner invocation, `write_summary` into its folder) is working data; a **result** (`write_result`)
is the batch's summaries plus a manifest and two hand-finished files, `description.md` and
`findings.md`, which the runner creates once and never rewrites.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wica.testing.report import MetricSummary, summarize

if TYPE_CHECKING:
    from wica.benchmark.runner import Completed, Skipped

# Per-reaction samples pooled as distributions; the token counts are summed per run instead.
_PER_RUN_TOTALS = ("input_tokens", "output_tokens")


@dataclass(frozen=True)
class Cell:
    """One (scenario, configuration) — or, with scenario `"*"`, one configuration over every
    scenario: its runs, how many passed, what failed, and the pooled numbers — `metrics` over
    every reaction of every run, `per_run` over one number per run (the measures, the token
    totals, and the derived `compliance`, `answer_rate`, `*_per_turn`)."""

    scenario: str
    label: str
    runs: int
    passed: int  # runs in which every rule held
    failed: dict[str, int] = field(default_factory=dict)  # rules broken, with counts
    script_missed: dict[str, int] = field(default_factory=dict)  # script expectations
    rules: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # rule → (held, runs)
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    per_run: dict[str, MetricSummary] = field(default_factory=dict)
    cache_read_share: float | None = None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.runs if self.runs else 0.0

    @property
    def pass_k(self) -> int:
        """1 when every run passed, else 0 — the "reliably follows the rules" bit."""
        return 1 if self.runs and self.passed == self.runs else 0


@dataclass(frozen=True)
class Summary:
    started_at: str
    finished_at: str
    git_commit: str | None
    k: int
    scenarios: tuple[str, ...]
    configs: dict[str, dict[str, Any]]  # label → the `agent` block
    cells: tuple[Cell, ...]  # per (scenario, configuration)
    skips: tuple[Skipped, ...]
    totals: tuple[Cell, ...] = ()  # per configuration, scenario "*": the global result

    def cell(self, scenario: str, label: str) -> Cell | None:
        """The (scenario, configuration) cell; scenario `"*"` is the configuration's total."""
        for cell in self.totals if scenario == "*" else self.cells:
            if cell.scenario == scenario and cell.label == label:
                return cell
        return None

    def _rule_names(self) -> list[str]:
        """Every rule any run applied, the built-ins in registry order first."""
        from wica.testing.checks import CHECKS

        seen = {rule for cell in self.totals for rule in cell.rules}
        ordered = [name for name in CHECKS if name in seen]
        return ordered + sorted(seen - set(ordered))

    def _skip_reason(self, label: str, scenario: str) -> str:
        skip = next(
            (
                s
                for s in self.skips
                if s.label == label and (s.scenario == "*" or s.scenario == scenario)
            ),
            None,
        )
        return skip.reason if skip else "not run"

    def _row_labels(self) -> list[str]:
        """The configurations, then the entries skipped whole (scenario `"*"`) that had no target."""
        labels = list(self.configs)
        labels += [
            s.label
            for s in self.skips
            if s.scenario == "*" and s.label not in self.configs
        ]
        return labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_commit": self.git_commit,
            "k": self.k,
            "scenarios": list(self.scenarios),
            "configs": self.configs,
            "totals": [_cell_dict(cell) for cell in self.totals],
            "cells": [_cell_dict(cell) for cell in self.cells],
            "skips": [dataclasses.asdict(skip) for skip in self.skips],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Benchmark summary",
            "",
            f"Run: {self.started_at} → {self.finished_at}",
            f"Commit: `{self.git_commit or 'unknown'}` · k = {self.k} · "
            f"scenarios: {', '.join(self.scenarios) or '(none)'}",
            "",
            "## Per configuration",
            "",
            "Every run of every scenario pooled: the row a reader compares models by.",
            "",
            "| config | runs | pass^k | compliance | answered | reaction p50 / p95 | "
            "model p50 / p95 | tokens in / out per turn | words per turn | unprompted | "
            "noops on input | dropped | rules broken | script missed |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for label in self._row_labels():
            total = self.cell("*", label)
            if total is None:
                lines.append(
                    f"| {label} | skipped: {_short(self._skip_reason(label, '*'), 60)} |"
                    + " |" * 12
                )
                continue
            lines.append(
                f"| {label} | {total.runs} | {total.pass_k} | "
                f"{_percent(total.per_run.get('compliance'))} | "
                f"{_percent(total.per_run.get('answer_rate'))} | "
                f"{_p50_p95(total.metrics.get('reaction_latency'))} | "
                f"{_p50_p95(total.metrics.get('model_latency'))} | "
                f"{_median(total.per_run.get('input_tokens_per_turn'))} / "
                f"{_median(total.per_run.get('output_tokens_per_turn'))} | "
                f"{_median(total.per_run.get('words_per_turn'))} | "
                f"{_median(total.per_run.get('unprompted_output'))} | "
                f"{_median(total.per_run.get('noops_on_input'))} | "
                f"{_median(total.per_run.get('dropped_inputs'))} | "
                f"{_failed(total.failed)} | {sum(total.script_missed.values())} |"
            )
        rule_names = self._rule_names()
        if rule_names:
            lines += [
                "",
                "## Rules",
                "",
                "Runs in which each rule held, over the runs that applied it.",
                "",
                "| config | " + " | ".join(rule_names) + " |",
                "|---|" + "---|" * len(rule_names),
            ]
            for label in self._row_labels():
                total = self.cell("*", label)
                if total is None:
                    continue
                cells = []
                for rule in rule_names:
                    held = total.rules.get(rule)
                    cells.append("—" if held is None else f"{held[0]}/{held[1]}")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += ["", "## Configurations", ""]
        for label, agent in self.configs.items():
            lines += [f"- **{label}**: `{json.dumps(agent, sort_keys=True)}`"]
        if not self.configs:
            lines.append("- (none)")
        for scenario in self.scenarios:
            lines += [
                "",
                f"## {scenario}",
                "",
                "| config | runs | passed | pass rate | pass^k | rules broken | reaction p50 / p95 | "
                "model p50 / p95 | reactions/input p50 | unprompted | dropped | "
                "tokens in / out per run | cache read | words per run |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for label in self._row_labels():
                cell = self.cell(scenario, label)
                if cell is None:
                    reason = self._skip_reason(label, scenario)
                    lines.append(
                        f"| {label} | skipped: {_short(reason, 60)} |" + " |" * 12
                    )
                    continue
                lines.append(
                    f"| {label} | {cell.runs} | {cell.passed} | {cell.pass_rate:.0%} | "
                    f"{cell.pass_k} | {_failed(cell.failed)} | "
                    f"{_p50_p95(cell.metrics.get('reaction_latency'))} | "
                    f"{_p50_p95(cell.metrics.get('model_latency'))} | "
                    f"{_median(cell.metrics.get('reactions_per_input'))} | "
                    f"{_median(cell.per_run.get('unprompted_output'))} | "
                    f"{_median(cell.per_run.get('dropped_inputs'))} | "
                    f"{_median(cell.per_run.get('input_tokens'))} / "
                    f"{_median(cell.per_run.get('output_tokens'))} | "
                    f"{_share(cell.cache_read_share)} | "
                    f"{_median(cell.per_run.get('output_words'))} |"
                )
        if self.skips:
            lines += ["", "## Skips", ""]
            lines += [
                f"- {s.label}"
                + (f" / {s.scenario}" if s.scenario != "*" else "")
                + f": {s.reason}"
                for s in self.skips
            ]
        return "\n".join(lines) + "\n"


def aggregate(
    completed: Iterable[Completed],
    skipped: Iterable[Skipped],
    *,
    k: int,
    configs: dict[str, dict[str, Any]],
    scenarios: Sequence[str],
    started_at: str,
    finished_at: str,
    git_commit: str | None,
) -> Summary:
    """Pool the runs into one `Cell` per (scenario, configuration), in the order of *scenarios*
    then *configs*, and one total per configuration over every scenario (`totals`)."""
    groups: dict[tuple[str, str], list[Completed]] = {}
    by_label: dict[str, list[Completed]] = {}
    for run in completed:
        groups.setdefault((run.scenario, run.label), []).append(run)
        by_label.setdefault(run.label, []).append(run)
    cells: list[Cell] = []
    for scenario in scenarios:
        for label in configs:
            runs = groups.get((scenario, label))
            if runs:
                cells.append(_cell(scenario, label, runs))
    totals = tuple(
        _cell("*", label, by_label[label]) for label in configs if label in by_label
    )
    return Summary(
        started_at=started_at,
        finished_at=finished_at,
        git_commit=git_commit,
        k=k,
        scenarios=tuple(scenarios),
        configs=dict(configs),
        cells=tuple(cells),
        skips=tuple(skipped),
        totals=totals,
    )


def _cell_dict(cell: Cell) -> dict[str, Any]:
    return {
        **dataclasses.asdict(cell),
        "pass_rate": cell.pass_rate,
        "pass_k": cell.pass_k,
    }


def _cell(scenario: str, label: str, runs: list[Completed]) -> Cell:
    failed: Counter[str] = Counter()
    pooled: dict[str, list[float]] = {}
    per_run: dict[str, list[float]] = {}
    cache_read = 0
    cache_input = 0
    script_missed: Counter[str] = Counter()
    rules: dict[str, list[int]] = {}
    for run in runs:
        report = run.report
        for finding in report.findings:
            if finding.expectation in run.checks:
                held = rules.setdefault(finding.expectation, [0, 0])
                held[1] += 1
                if finding.ok:
                    held[0] += 1
                else:
                    failed[finding.expectation] += 1
            elif not finding.ok:
                script_missed[finding.expectation] += 1
        for key, values in run.samples.items():
            if key in _PER_RUN_TOTALS:
                per_run.setdefault(key, []).append(float(sum(values)))
            else:
                pooled.setdefault(key, []).extend(values)
        for key, value in run.measures.items():
            per_run.setdefault(key, []).append(value)
        per_run.setdefault("dropped_inputs", []).append(float(report.dropped_inputs))
        for key, value in _derived(run).items():
            per_run.setdefault(key, []).append(value)
        for event in report.timeline:
            usage = event.reaction.usage if event.reaction is not None else None
            if usage is not None and usage.cache_read_tokens is not None:
                cache_read += usage.cache_read_tokens
                cache_input += usage.input_tokens
    return Cell(
        scenario=scenario,
        label=label,
        runs=len(runs),
        passed=sum(1 for run in runs if _rules_held(run)),
        failed=dict(failed),
        script_missed=dict(script_missed),
        rules={name: (held, total) for name, (held, total) in rules.items()},
        metrics=_summaries(pooled),
        per_run=_summaries(per_run),
        cache_read_share=(cache_read / cache_input) if cache_input else None,
    )


def _rules_held(run: Completed) -> bool:
    """A run passes when every rule held. The script's own expectations only drive the
    conversation; a missed one is reported apart (`script_missed`), never a failure."""
    return all(f.ok for f in run.report.findings if f.expectation in run.checks)


def _derived(run: Completed) -> dict[str, float]:
    """The per-run rates: `compliance` (checks passed over checks run), `answer_rate` (turns
    answered over turns), and the token / word counts per turn."""
    derived: dict[str, float] = {}
    checks = [f for f in run.report.findings if f.expectation in run.checks]
    if checks:
        derived["compliance"] = sum(1 for f in checks if f.ok) / len(checks)
    turns = run.measures.get("turns", 0.0)
    if turns:
        derived["answer_rate"] = run.measures.get("turns_answered", 0.0) / turns
        derived["input_tokens_per_turn"] = (
            sum(run.samples.get("input_tokens", [])) / turns
        )
        derived["output_tokens_per_turn"] = (
            sum(run.samples.get("output_tokens", [])) / turns
        )
        derived["words_per_turn"] = run.measures.get("output_words", 0.0) / turns
    return derived


def _percent(summary: MetricSummary | None) -> str:
    return "—" if summary is None else f"{summary.median:.0%}"


def _summaries(samples: dict[str, list[float]]) -> dict[str, MetricSummary]:
    return {
        key: summary
        for key, values in samples.items()
        if (summary := summarize(values)) is not None
    }


def _p50_p95(summary: MetricSummary | None) -> str:
    if summary is None:
        return "—"
    return f"{summary.median:.2f} / {summary.p95:.2f}"


def _median(summary: MetricSummary | None) -> str:
    if summary is None:
        return "—"
    value = summary.median
    return f"{value:.0f}" if value == int(value) else f"{value:.1f}"


def _share(share: float | None) -> str:
    return "—" if share is None else f"{share:.0%}"


def _failed(failed: dict[str, int]) -> str:
    if not failed:
        return "—"
    return ", ".join(f"{name} ×{count}" for name, count in sorted(failed.items()))


def _short(text: str, limit: int) -> str:
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- git, batches and results --------------------------------------------------------------------


def git_commit(cwd: Path | None = None) -> str | None:
    """The working tree's HEAD commit, or None outside a repository / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def write_summary(summary: Summary, out_dir: Path) -> None:
    """`summary.json` and `summary.md` into *out_dir* (created)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(summary.to_json(), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary.to_markdown(), encoding="utf-8")


def write_result(summary: Summary, result_dir: Path, *, batch_dir: Path) -> Path:
    """Keep a batch as a result: the two summaries, `runs.json` (the manifest) and, when absent,
    `description.md` and `findings.md` — created from a template once, then yours."""
    result_dir = Path(result_dir)
    write_summary(summary, result_dir)
    manifest = {
        "batch_dir": str(batch_dir),
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "git_commit": summary.git_commit,
        "k": summary.k,
        "scenarios": list(summary.scenarios),
        "configs": summary.configs,
        "skips": [dataclasses.asdict(skip) for skip in summary.skips],
    }
    (result_dir / "runs.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    description = result_dir / "description.md"
    if not description.exists():
        description.write_text(_description_template(summary, result_dir.name))
    findings = result_dir / "findings.md"
    if not findings.exists():
        findings.write_text(_findings_template(result_dir.name))
    return result_dir


def _description_template(summary: Summary, result_id: str) -> str:
    configs = "\n".join(
        f"- `{label}`: {agent.get('provider', '?')} / {agent.get('model', '?')}"
        for label, agent in summary.configs.items()
    )
    return (
        f"# {result_id}\n\n"
        f"- Date: {summary.started_at}\n"
        f"- Commit: `{summary.git_commit or 'unknown'}`\n"
        f"- k: {summary.k}\n"
        f"- Scenarios: {', '.join(summary.scenarios)}\n"
        f"- Configurations:\n{configs}\n\n"
        "## Setup\n\n"
        "TODO: the machine, the network, anything else that moves the numbers.\n\n"
        "## Why this batch\n\n"
        "TODO: the question it was run to answer.\n"
    )


def _findings_template(result_id: str) -> str:
    return (
        f"# Findings — {result_id}\n\n"
        "## What the numbers say\n\nTODO\n\n"
        "## Decisions fed\n\nTODO\n\n"
        "## What remains\n\nTODO\n"
    )
