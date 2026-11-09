"""The one argparse definition every benchmark entry point reuses. Implements specs/benchmark.md,
"The runner" (the CLI):

    <prog> [--matrix FILE] [--only GLOB ...] [--exclude GLOB ...] [--tags TAG ...]
           [--exclude-tags TAG ...] [--include-disabled] [--scenarios DIR] [--scenario NAME ...]
           [--out DIR] [--k N] [--result NAME]

The matrix is one file (`wica.benchmark.matrix`); the selection flags pick its entries by id glob
and tag, and a disabled entry runs only with `--include-disabled`. `--out` is the batch folder, by
default a dated one under the entry point's runs directory; `--result NAME` keeps the batch as a
result. Exit code 0 whatever the results: a benchmark compares, it does not gate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TextIO

from wica.benchmark.matrix import load_matrix, select_entries
from wica.benchmark.runner import (
    RunResult,
    Skipped,
    Target,
    TargetFactory,
    run_matrix,
)
from wica.benchmark.summary import write_result, write_summary
from wica.testing.checks import Check, build_registry
from wica.testing.scenario import load_scenario, load_scenarios

_STAMP = "%Y%m%d-%H%M"
DISABLED_REASON = "disabled (use --include-disabled)"


def cli(
    argv: Sequence[str] | None,
    *,
    target_factory: TargetFactory,
    default_matrix: str | Path,
    default_scenarios: str | Path,
    default_runs: str | Path,
    default_results: str | Path,
    prog: str,
    extra_checks: Iterable[Check] | None = None,
    out: TextIO | None = None,
) -> int:
    out = out or sys.stdout
    extra = list(extra_checks or ())
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a WICA benchmark: a matrix of configurations × a scenario set × k.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(default_matrix),
        help="the matrix file; default: %(default)s",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="run only the entries whose id matches one of these globs",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="drop the entries whose id matches one of these globs",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=[],
        metavar="TAG",
        help="run only the entries carrying any of these tags",
    )
    parser.add_argument(
        "--exclude-tags",
        nargs="+",
        default=[],
        metavar="TAG",
        help="drop the entries carrying any of these tags",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="run the entries marked enabled: false too (opt-in rows)",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(default_scenarios),
        help="the scenario directory (or one scenario file); default: %(default)s",
    )
    parser.add_argument(
        "--scenario", nargs="+", default=[], metavar="NAME", help="only these scenarios"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"the batch folder; default: {default_runs}/<{_STAMP.replace('%', '')}>",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="repetitions per (entry, scenario) (default 1)"
    )
    parser.add_argument(
        "--result",
        default=None,
        metavar="NAME",
        help=f"keep the batch as {default_results}/<stamp>_NAME",
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")

    stamp = datetime.now().strftime(_STAMP)
    if not args.matrix.is_file():
        parser.error(f"{args.matrix}: no such matrix file")
    matrix = load_matrix(args.matrix)
    entries = select_entries(
        matrix.entries,
        only=args.only,
        exclude=args.exclude,
        tags=args.tags,
        exclude_tags=args.exclude_tags,
    )
    if not entries:
        parser.error(f"no matrix entry matches the selection; ids: {matrix.ids}")
    registry = build_registry(extra)
    if args.scenarios.is_dir():
        scenarios = load_scenarios(args.scenarios, registry=registry)
    elif args.scenarios.is_file():
        scenarios = [load_scenario(args.scenarios, registry=registry)]
    else:
        parser.error(f"{args.scenarios}: no such scenario directory or file")
    if args.scenario:
        known_names = {s.name for s in scenarios}
        unknown = [name for name in args.scenario if name not in known_names]
        if unknown:
            parser.error(f"unknown scenario(s) {unknown}; known: {sorted(known_names)}")
    batch_dir: Path = args.out if args.out is not None else Path(default_runs) / stamp

    targets: list[Target] = []
    pre_skipped: list[Skipped] = []
    for entry in entries:
        if not entry.enabled and not args.include_disabled:
            pre_skipped.append(Skipped(entry.id, "*", DISABLED_REASON))
            print(f"{entry.id}: skipped ({DISABLED_REASON})", file=out)
            continue
        targets.append(target_factory(entry))

    def progress(outcome: RunResult | Skipped) -> None:
        if isinstance(outcome, Skipped):
            print(
                f"{outcome.label} / {outcome.scenario}: skipped ({outcome.reason})",
                file=out,
            )
            return
        report = outcome.report
        if report.outcome != "completed":
            verdict = f"{report.outcome.upper()} at {report.stopped_at}"
        else:
            broken = [f.expectation for f in report.checks() if not f.ok]
            verdict = "ok" if not broken else f"checks broken {broken}"
        print(
            f"{outcome.label} / {outcome.scenario} #{outcome.repetition}: {verdict}",
            file=out,
        )

    summary = run_matrix(
        targets,
        scenarios,
        k=args.k,
        out_dir=batch_dir,
        only_scenarios=args.scenario,
        extra_checks=extra,
        on_outcome=progress,
        skipped=pre_skipped,
    )
    write_summary(summary, batch_dir)
    print("", file=out)
    print(summary.to_markdown(), file=out)
    print(f"Batch written to {batch_dir}", file=out)
    if args.result:
        result_dir = write_result(
            summary,
            Path(default_results) / f"{stamp}_{args.result}",
            batch_dir=batch_dir,
        )
        print(
            f"Result kept at {result_dir} — finish its description.md and findings.md",
            file=out,
        )
    return 0
