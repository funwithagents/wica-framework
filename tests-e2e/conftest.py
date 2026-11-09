"""The e2e tier's pytest hooks and fixtures: the tier runs serially, a provider's rate limit skips
a test instead of failing it, and toolkit-driven tests can keep one report per run
(`--wica-report-dir`, specs/wica-tester.md "Relation to existing tests")."""

from __future__ import annotations

import logging
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


def _rate_limited(exc: BaseException) -> BaseException | None:
    """The provider error in `exc`'s chain that reports HTTP 429 (a rate limit or an exhausted
    quota), if any. Duck-typed over the SDKs the providers raise through — `status_code` (OpenAI,
    Anthropic), `code` (google-genai's `ClientError`), `response.status_code` (httpx's
    `HTTPStatusError` from Mistral, huggingface_hub's `HfHubHTTPError`) — so no SDK is imported."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if 429 in (
            getattr(current, "status_code", None),
            getattr(current, "code", None),
            getattr(response, "status_code", None),
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


class _LoggedErrors(logging.Handler):
    """Keeps the exceptions logged under `wica` during a test: the Agent catches a failed model
    call and logs it (`model call failed; step abandoned`), so a provider's 429 reaches the test
    only as an expectation that timed out, never as a raised exception."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.errors: list[BaseException] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info and record.exc_info[1] is not None:
            self.errors.append(record.exc_info[1])


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """A provider that rate-limits the key (HTTP 429) says nothing about WICA: skip the failed
    test, as for a missing key, instead of failing the tier on the account's state
    (specs/testing.md, "Live tier"). The 429 is looked for in the raised exception's chain and in
    every exception logged under `wica` while the test ran. Any other failure still fails."""
    logged = _LoggedErrors()
    logger = logging.getLogger("wica")
    logger.addHandler(logged)
    try:
        return (yield)
    except Exception as exc:
        for candidate in (exc, *logged.errors):
            limited = _rate_limited(candidate)
            if limited is not None:
                pytest.skip(f"provider rate limit (429): {str(limited)[:160]}")
        raise
    finally:
        logger.removeHandler(logged)


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
