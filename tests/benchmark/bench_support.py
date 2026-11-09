"""Plain helpers for the `wica.benchmark` tests (the fixture is in conftest.py): a fake `Target` over
a scripted SUT of the demo's shape in miniature (a triggering `speech_input`, a `say` output
Command), opened fresh per run. Named `bench_support` so it shadows no other test module under
pytest's prepend import mode (no tests directory is a package)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wica import Wica
from wica.benchmark import Skip
from wica.config import WicaConfig
from wica.content import Content, TextPart
from wica.testing import Scenario

SAY_HI = [{"tool_calls": [{"name": "say", "args": {"text": "hi there"}}]}]
NOOP = {"tool_calls": [{"name": "noop", "args": {}}]}


def _text(value: Any, previous: Any) -> Content:
    return [TextPart(str(value))]


class FakeTarget:
    """A `Target` whose `open` builds a scripted-fake SUT; counts opens and closes."""

    def __init__(
        self,
        script: list[dict[str, Any]],
        *,
        label: str = "fake",
        skip: str | None = None,
        commands: tuple[Callable[..., Any], ...] = (),
    ) -> None:
        self.script = script
        self.label = label
        self.skip = skip
        self.commands = commands
        self.agent_config: dict[str, Any] = {"provider": "fake", "model": "scripted"}
        self.opened = 0
        self.closed = 0
        self.scenarios_seen: list[str] = []
        self._wica: Wica | None = None

    def open(self, scenario: Scenario) -> Wica:
        self.opened += 1
        self.scenarios_seen.append(scenario.name)
        if self.skip is not None:
            raise Skip(self.skip)
        config = WicaConfig.from_dict(
            {
                "agent": {
                    "provider": "fake",
                    "model": "scripted",
                    "system_prompt": "You are a test double.",
                    "model_kwargs": {
                        "delay_s": 0,
                        "script": self.script,
                        "default": NOOP,
                    },
                }
            }
        )
        sut = Wica.init(config, coalesce_window=0.0)

        async def say(text: str) -> str:
            """Speak to the person."""
            return "OK"

        sut.set_output_command(say)
        for command in self.commands:
            sut.register_command(command)
        sut.world.register(
            "speech_input", str, serialize_fn=_text, triggers_llm_call=True
        )
        sut.world.register("mood", str, serialize_fn=_text)
        sut.start()
        self._wica = sut
        return sut

    def close(self) -> None:
        self.closed += 1
        if self._wica is not None:
            self._wica.close()
            self._wica = None
