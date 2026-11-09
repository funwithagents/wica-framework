"""Fixtures for the `wica.benchmark` tests: the fake `Target` factory, every target closed on
teardown."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from bench_support import SAY_HI, FakeTarget


@pytest.fixture
def fake_target() -> Iterator[Callable[..., FakeTarget]]:
    created: list[FakeTarget] = []

    def make(script: list[dict[str, Any]] | None = None, **kwargs: Any) -> FakeTarget:
        target = FakeTarget(script if script is not None else SAY_HI, **kwargs)
        created.append(target)
        return target

    yield make
    for target in created:
        target.close()
