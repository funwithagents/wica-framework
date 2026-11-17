"""The Probe over a fake SUT (specs/wica-tester.md, "Probe — the connection")."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from sut_support import ADD_THEN_TEXT, Voice, wait_until

from wica import Command, OutputStream, Wica
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

    assert probe.output_command is None  # no output Command yet
    sut.set_output_command(Command(voice, name="talk"))
    sut.start()
    # The output Command is reported apart: it is not a tool, so not one of the Commands.
    assert probe.output_command == "talk" and probe.output_command_name == "talk"
    assert set(probe.commands) == {"add"}  # cancel_command filtered out
    assert "cancel_command" in sut.agent.commands


def test_stream_follows_one_step_end_to_end(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT)
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()

    probe.set("prompt", "Add 2 and 2.")
    wait_until(lambda: "private_text" in collector.kinds())
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
        "private_text",
    ]
    positions = [kinds.index(k) for k in dict.fromkeys(expected_order)]
    assert positions == sorted(positions)
    assert kinds.count("command_state") == 2
    # Causal order, not just first occurrences: the terminal state is recorded before the text of
    # the step it triggered (the per-key listener is async, hence loop-ordered — a sync one ran on
    # the executor and could land after the text on a loaded machine).
    complete_at = max(i for i, k in enumerate(kinds) if k == "command_state")
    assert complete_at < kinds.index("private_text")
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
    assert collector.of("private_text")[0].text == "The sum is 4."


def test_a_declined_reaction_is_reported_by_its_noop_flag_alone(
    sut_factory: Callable[..., Wica],
):
    sut = sut_factory([{"text": "<noop/>"}])
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()
    probe.set("prompt", "Nothing to do.")
    wait_until(lambda: collector.of("reaction_ended") != [])
    # `<noop/>` issues no Command and is not private text: the reaction record is its only report.
    assert collector.of("command_issued") == []
    assert collector.of("command_state") == []
    assert collector.of("private_text") == []
    reaction = collector.of("reaction_ended")[0].reaction
    assert reaction is not None and reaction.noop and reaction.outcome == "ok"


def test_an_output_is_followed_from_its_start_to_its_end(
    sut_factory: Callable[..., Wica],
):
    """An output is followed from `on_agent_output_started`, at its first words: the Probe's
    listener records each state its entry goes through — `running` from the opening, with no
    text yet, then its outcome — each marked `is_output`, and `command_issued` comes once its
    body is complete, with the whole text. Nothing is reconstructed from an entry's earlier
    version."""
    sut = sut_factory(
        [{"text": 'thinking <say emotion="happy">Hello there.</say>'}], commands=()
    )
    voice = Voice(delay=0.2)
    sut.set_output_command(voice.say)
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()
    before = time.time()
    probe.set("prompt", "Greet me.")
    wait_until(
        lambda: [s.state for s in collector.of("command_state")][-1:] == ["complete"]
    )

    issued = collector.of("command_issued")
    assert [(e.name, e.args, e.is_output) for e in issued] == [
        ("say", {"emotion": "happy", "text": "Hello there."}, True)
    ]
    states = collector.of("command_state")
    assert [(s.state, s.call_id, s.args) for s in states] == [
        ("running", issued[0].call_id, {"emotion": "happy", "text": ""}),
        ("complete", issued[0].call_id, {"emotion": "happy", "text": "Hello there."}),
    ]
    kinds = collector.kinds()
    assert all(s.is_output for s in states)
    assert states[-1].result == "Said it."
    # Its terminal state is written once the step that issued it is over.
    assert kinds.index("reaction_ended") < len(kinds) - 1 - kinds[::-1].index(
        "command_state"
    )
    assert before <= states[0].t <= states[-1].t
    assert states[-1].t - states[0].t >= 0.2 - 0.016  # the delivery, not an instant
    ended = collector.of("command_ended")
    assert [(e.call_id, e.command.is_output) for e in ended if e.command] == [
        (issued[0].call_id, True)
    ]
    # Text outside the tag is private; the tag's body is not.
    assert [e.text for e in collector.of("private_text")] == ["thinking"]
    reaction = collector.of("reaction_ended")[0].reaction
    assert reaction is not None and not reaction.noop
    assert reaction.output_call_ids == (issued[0].call_id,)
    assert reaction.first_output_at is not None
    assert voice.said == ["Hello there."]


def test_an_output_whose_opening_the_probe_missed_is_followed_from_its_report(
    sut_factory: Callable[..., Wica],
):
    """A Probe attached while an output is already open (here, from inside that output's own
    `on_agent_output_started`, which it is too late to receive) follows it from its
    `command_issued` report: the state the entry is in then, and every later one — so `"idle"`
    never waits on an output whose end it could not see."""
    sut = sut_factory([{"text": "<say>Hello there.</say>"}], commands=())
    sut.set_output_command(Voice(delay=0.2).say)
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    sut.on_agent_output_started.subscribe(lambda issued: probe.attach())
    sut.start()
    sut.world.update("prompt", "Greet me.")
    wait_until(
        lambda: [s.state for s in collector.of("command_state")][-1:] == ["complete"]
    )
    followed = [
        e for e in collector.events if e.kind in ("command_issued", "command_state")
    ]
    assert [(e.kind, e.state) for e in followed] == [
        ("command_issued", None),
        ("command_state", "running"),  # the state it was in when reported
        ("command_state", "complete"),
    ]
    assert all(e.is_output for e in followed)


def test_a_tool_call_is_not_marked_as_an_output(sut_factory: Callable[..., Wica]):
    sut = sut_factory(ADD_THEN_TEXT)
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()
    probe.set("prompt", "Add 2 and 2.")
    wait_until(lambda: len(collector.of("command_state")) == 2)
    followed = collector.of("command_issued") + collector.of("command_state")
    assert [e.name for e in followed] == ["add", "add", "add"]
    assert not any(e.is_output for e in followed)


def test_an_output_that_failed_before_its_response_ended_is_still_followed(
    sut_factory: Callable[..., Wica],
):
    """An output Command that raises at the first words: the Probe, following the output since
    it opened, records each state its entry went through and the `command_issued` report."""
    sut = sut_factory(
        [{"text": "<say>One two three four five.</say> and a private tail"}],
        commands=(),
    )

    async def say(text: OutputStream) -> str:
        """Speak out loud."""
        raise RuntimeError("speaker unplugged")

    sut.set_output_command(say)
    probe = Probe(sut)
    collector = Collector()
    probe.subscribe(collector)
    probe.attach()
    sut.start()
    probe.set("prompt", "Greet me.")
    wait_until(lambda: collector.of("reaction_ended") != [])
    wait_until(lambda: any(e.state == "failed" for e in collector.of("command_state")))
    time.sleep(0.2)  # nothing more may follow
    # Each state the entry went through, once: opened, then failed.
    assert [(e.state, e.args) for e in collector.of("command_state")] == [
        ("running", {"text": ""}),
        ("failed", {"text": "One two three four five."}),
    ]
    assert len(collector.of("command_issued")) == 1
    failed = collector.of("command_state")[-1]
    assert failed.error == "speaker unplugged" and failed.is_output
    assert failed.call_id == collector.of("command_issued")[0].call_id
    assert collector.of("command_issued")[0].args == {
        "text": "One two three four five."
    }


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
