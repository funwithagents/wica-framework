"""The Probe over a fake SUT (specs/wica-tester.md, "Probe — the connection")."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from sut_support import ADD_THEN_TEXT, wait_until

from wica import Command, Wica
from wica.testing import Probe, SutEvent


class Collector:
    def __init__(self) -> None:
        self.events: list[SutEvent] = []
        self._lock = threading.Lock()

    def __call__(self, event: SutEvent) -> None:
        with self._lock:
            self.events.append(event)

    def kinds(self) -> list[str]:
        with self._lock:
            return [e.kind for e in self.events]

    def of(self, kind: str) -> list[SutEvent]:
        with self._lock:
            return [e for e in self.events if e.kind == kind]


def test_inputs_and_commands_read_the_sut_schema(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT)
    probe = Probe(sut)
    assert set(probe.inputs) == {"prompt"}
    assert probe.inputs["prompt"].type is str

    def voice(text: str) -> str:
        """Speak."""
        return text

    sut.set_output_command(Command(voice, name="talk"))
    sut.start()
    assert probe.output_command_name == "talk"
    assert set(probe.commands) == {"add", "talk"}  # noop / cancel_command filtered out
    assert set(sut.agent.commands) >= {"noop", "cancel_command"}


def test_stream_follows_one_step_end_to_end(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT)
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()

    probe.set("prompt", "Add 2 and 2.")
    wait_until(lambda: "output_text" in collector.kinds())
    wait_until(lambda: len(collector.of("reaction_ended")) >= 2)

    kinds = collector.kinds()
    expected_order = [
        "input_written",
        "trigger_raw",
        "trigger_observed",
        "prompt",
        "command_issued",
        "command_state",  # running
        "command_state",  # complete
        "output_text",
    ]
    positions = [kinds.index(k) for k in dict.fromkeys(expected_order)]
    assert positions == sorted(positions)
    assert kinds.count("command_state") == 2
    # Causal order, not just first occurrences: the terminal state is recorded before the text of
    # the step it triggered (the per-key listener is async, hence loop-ordered — a sync one ran on
    # the executor and could land after the text on a loaded machine).
    complete_at = max(i for i, k in enumerate(kinds) if k == "command_state")
    assert complete_at < kinds.index("output_text")
    assert (
        collector.of("input_written")[0].version_id
        == collector.of("trigger_raw")[0].version_id
    )
    states = collector.of("command_state")
    assert [s.state for s in states] == ["running", "complete"]
    assert states[1].name == "add" and "4" in (states[1].result or "")
    assert states[1].call_id == collector.of("command_issued")[0].call_id
    ended = collector.of("command_ended")[0]
    assert ended.command is not None and ended.command.state == "complete"
    reactions = collector.of("reaction_ended")
    assert reactions[0].reaction is not None and reactions[0].reaction.command_call_ids
    assert reactions[1].reaction is not None and reactions[1].reaction.text_length > 0
    assert collector.of("output_text")[0].text == "The sum is 4."


def test_noop_is_reported_as_a_command_issued_without_state(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory([{"tool_calls": [{"name": "noop", "args": {}}]}])
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()
    probe.set("prompt", "Nothing to do.")
    wait_until(lambda: collector.of("reaction_ended") != [])
    issued = collector.of("command_issued")
    assert [e.name for e in issued] == ["noop"]
    assert collector.of("command_state") == []
    reaction = collector.of("reaction_ended")[0].reaction
    assert reaction is not None and reaction.noop


def test_detach_stops_the_stream(sut_factory: Callable[..., Wica]):
    sut = sut_factory([{"text": "one"}, {"text": "two"}])
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()
    probe.set("prompt", "first")
    wait_until(lambda: collector.of("reaction_ended") != [])
    probe.detach()
    before = collector.kinds()
    probe.set("prompt", "second")  # still writes the World (and records the write) …
    wait_until(lambda: sut.world.get("prompt") == "second")
    time.sleep(0.3)  # … but the SUT's reaction to it is no longer forwarded
    assert collector.kinds() == [*before, "input_written"]
