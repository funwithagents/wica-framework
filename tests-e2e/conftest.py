import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Run this tier serially, overriding the fast tier's parallel `addopts`: parallel live
    cases would only hit the providers' rate limits, and the scripted-fake flows take seconds.
    This conftest loads only when tests-e2e/ is collected, so the fast tier keeps its workers.
    See specs/testing.md ("Tooling")."""
    config.option.numprocesses = 0
