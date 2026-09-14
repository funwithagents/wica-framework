"""Fixtures shared by the fast tier."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from wica import Command, Wica
from wica.config import WicaConfig


@pytest.fixture
def unstarted_wica() -> Iterator[Callable[[str | None], Wica]]:
    """A factory for an unstarted (no loop thread, no network) `Wica` over the fake provider, with
    an output Command of the given name set — what a presenter needs to read the voice's name
    live and to bind its per-Command World listeners. Closed on teardown."""
    created: list[Wica] = []

    def _make(output_command_name: str | None) -> Wica:
        config = WicaConfig.from_dict(
            {
                "agent": {
                    "provider": "fake",
                    "model": "scripted",
                    "system_prompt": "You are a test robot.",
                    "model_kwargs": {"delay_s": 0, "script": [{"text": ""}]},
                }
            }
        )
        wica = Wica.init(config)
        if output_command_name is not None:

            def voice(text: str) -> str:
                """The voice."""
                return text

            wica.set_output_command(Command(voice, name=output_command_name))
        created.append(wica)
        return wica

    yield _make
    for wica in created:
        wica.close()
