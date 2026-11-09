"""Fixtures for the `wica.testing` tests: a fake SUT built the way `tests/test_wica.py` builds one
(`Wica.init` over `provider: "fake"`, no coalescing, no model delay), closed on teardown."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sut_support import add, identity_serialize

from wica import Wica
from wica.config import WicaConfig


@pytest.fixture
def sut_factory() -> Iterator[Callable[..., Wica]]:
    """`make(script, *, delay_s=0, commands=(add,), inputs=("prompt",))` — an unstarted fake SUT
    with the `prompt` Input (str, triggering) and the `add` Command registered."""
    created: list[Wica] = []

    def make(
        script: list[dict[str, Any]],
        *,
        delay_s: float = 0,
        commands: tuple[Callable[..., Any], ...] = (add,),
        inputs: tuple[str, ...] = ("prompt",),
    ) -> Wica:
        config = WicaConfig.from_dict(
            {
                "agent": {
                    "provider": "fake",
                    "model": "scripted",
                    "system_prompt": "You are a test double.",
                    "model_kwargs": {
                        "delay_s": delay_s,
                        "script": script,
                        "default": {"text": ""},
                    },
                }
            }
        )
        sut = Wica.init(config, coalesce_window=0.0)
        for key in inputs:
            sut.world.register(
                key, str, serialize_fn=identity_serialize, triggers_llm_call=True
            )
        for command in commands:
            sut.register_command(command)
        created.append(sut)
        return sut

    yield make
    for sut in created:
        sut.close()
