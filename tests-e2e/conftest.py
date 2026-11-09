"""The e2e tier's pytest hooks and fixtures: the tier runs serially, and toolkit-driven tests can
keep one report per run (`--wica-report-dir`, specs/wica-tester.md "Relation to existing tests")."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from wica.testing import Report


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Run this tier serially, overriding the fast tier's parallel `addopts`: parallel live
    cases would only hit the providers' rate limits, and the scripted-fake flows take seconds.
    This conftest loads only when tests-e2e/ is collected, so the fast tier keeps its workers.
    See specs/testing.md ("Tooling")."""
    config.option.numprocesses = 0


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--wica-report-dir",
        default=None,
        help="directory where toolkit-driven e2e tests write one JSON report per test per "
        "provider (specs/wica-tester.md); nothing is written without it",
    )


@pytest.fixture
def report_dir(request: pytest.FixtureRequest) -> Path | None:
    """The `--wica-report-dir` option as a created directory, or None."""
    value = request.config.getoption("--wica-report-dir")
    if value is None:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def write_report(
    request: pytest.FixtureRequest, report_dir: Path | None
) -> Callable[[Report], None]:
    """Writes a report as `<report_dir>/<test node name>.json` — the node name carries the
    provider id, e.g. `test_plain_text_round_trip[e2e.openai.config]` — or does nothing when no
    directory was given."""

    def write(report: Report) -> None:
        if report_dir is None:
            return
        name = re.sub(r"[\[\]/]", "_", request.node.name).strip("_")
        report.write(report_dir / f"{name}.json")

    return write
