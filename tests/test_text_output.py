"""The text output parser and the output Command's signature — specs/agent.md ("Output: tags in
the model's text"), specs/commands.md ("The output Command")."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal

import pytest

from wica import Command, OutputStream, ReactionTrace, Wica
from wica.config import AgentConfig, WicaConfig
from wica.content import TextPart

from wica.text_output import (
    TextOutputParser,
    is_tag_name,
    output_signature,
)


def parse(pieces: list[str], name: str | None = "say") -> tuple[Any, list[Any]]:
    events: list[Any] = []
    parser = TextOutputParser(
        name,
        on_open=lambda attributes: events.append(("open", attributes)),
        on_delta=lambda text: events.append(("delta", text)),
        on_close=lambda: events.append(("close",)),
    )
    for piece in pieces:
        parser.feed(piece)
    return parser.finish(), events


def spoken(events: list[Any]) -> str:
    return "".join(e[1] for e in events if e[0] == "delta")


def test_a_tag_delivers_only_its_body_and_its_attributes() -> None:
    result, events = parse(
        list('thinking <say emotion="happy">Hello there!</say> done')
    )
    assert events[0] == ("open", {"emotion": "happy"})
    assert spoken(events) == "Hello there!"
    assert events[-1] == ("close",)
    assert [(o.attributes, o.text) for o in result.outputs] == [
        ({"emotion": "happy"}, "Hello there!")
    ]
    assert result.private == "thinking  done"
    assert not result.noop


def test_the_body_is_delivered_before_the_response_ends() -> None:
    events: list[str] = []
    parser = TextOutputParser(
        "say",
        on_open=lambda a: events.append("open"),
        on_delta=events.append,
        on_close=lambda: events.append("close"),
    )
    parser.feed("<say>Once upon")
    assert events == ["open", "Once upon"]
    parser.feed(" a time</s")  # what may be the closing tag is held back, not delivered
    assert events == ["open", "Once upon", " a time"]
    parser.feed("ay> and more")
    assert events[-1] == "close"  # closed as soon as the tag closes, not at the end
    parser.finish()
    assert "</s" not in "".join(events)


def test_each_tag_is_its_own_output_in_order() -> None:
    result, events = parse(
        list('<say emotion="happy">One.</say> <say emotion="sad">Two.</say>')
    )
    assert [(o.attributes, o.text) for o in result.outputs] == [
        ({"emotion": "happy"}, "One."),
        ({"emotion": "sad"}, "Two."),
    ]
    assert [e for e in events if e[0] != "delta"] == [
        ("open", {"emotion": "happy"}),
        ("close",),
        ("open", {"emotion": "sad"}),
        ("close",),
    ]


def test_text_with_no_tag_is_one_output_delivered_at_the_end() -> None:
    result, events = parse(["Hello ", "there"])
    assert events == [("open", {}), ("delta", "Hello there"), ("close",)]
    assert result.outputs[0].text == "Hello there" and not result.outputs[0].tagged
    assert result.private == ""


def test_noop_alone_is_doing_nothing() -> None:
    result, events = parse(list("  <noop/>\n"))
    assert events == [] and result.noop and result.outputs == []


def test_an_empty_response_is_doing_nothing() -> None:
    result, events = parse([])
    assert events == [] and result.noop and result.raw == ""


def test_noop_beside_an_output_is_ignored() -> None:
    result, _ = parse(["<say>Hi</say><noop/>"])
    assert result.outputs[0].text == "Hi" and not result.noop


def test_private_text_beside_noop_is_not_delivered() -> None:
    result, events = parse(["Nothing to do here. <noop/>"])
    assert events == [] and result.noop and result.private == "Nothing to do here."


def test_the_note_is_extracted_and_private() -> None:
    result, events = parse(list("<say>Watch.</say>\n<next>do it, then report</next>"))
    assert spoken(events) == "Watch." and result.note == "do it, then report"
    assert result.private == ""


def test_a_tag_left_open_is_closed_by_the_end() -> None:
    result, events = parse(["<say>Hello the", "re"])
    assert result.outputs[0].text == "Hello there" and events[-1] == ("close",)


def test_an_empty_tag_is_no_output() -> None:
    result, events = parse(["<say> </say>"])
    assert events == [] and result.noop


def test_other_angle_brackets_are_ordinary_text() -> None:
    result, _ = parse(list("<say>2 < 3 and <b>bold</b></say> a<b"))
    assert result.outputs[0].text == "2 < 3 and <b>bold</b>"
    assert result.private == "a<b"


def test_without_an_output_command_all_text_is_private() -> None:
    result, events = parse(["Some notes <say>not a tag here</say>"], name=None)
    assert events == [] and result.outputs == []
    assert result.private == "Some notes <say>not a tag here</say>"
    assert parse(["<noop/>"], name=None)[0].noop


def test_rebuilt_cuts_an_output_and_drops_one_that_never_started() -> None:
    result, _ = parse(
        [
            'aside <say emotion="happy">One two three.</say><say>Four.</say><next>later</next>'
        ]
    )
    assert (
        result.rebuilt("say", {0: "One two", 1: ""})
        == 'aside <say emotion="happy">One two</say><next>later</next>'
    )
    assert result.rebuilt("say", {}) == result.raw


def test_rebuilt_of_untagged_text_is_the_heard_text() -> None:
    result, _ = parse(["Once upon a time there was a robot."])
    assert result.rebuilt("say", {0: "Once upon"}) == "Once upon"


# -- the output Command's signature --------------------------------------------------------------


def test_stream_and_attributes_are_read_from_the_signature() -> None:
    async def say(
        text: OutputStream,
        emotion: Literal["happy", "sad"] = "happy",
        voice: str = "default",
    ) -> None: ...

    signature = output_signature(say)
    assert signature.streams and signature.text_parameter == "text"
    assert signature.attributes == {"emotion": ("happy", "sad"), "voice": None}
    assert signature.defaults == {"emotion": "happy", "voice": "default"}


def test_a_str_or_unannotated_text_parameter_takes_the_whole_text() -> None:
    def say(text: str) -> None: ...

    def speak(words) -> None: ...  # noqa: ANN001

    assert not output_signature(say).streams
    assert not output_signature(speak).streams


@pytest.mark.parametrize(
    "source",
    [
        "def f(): ...",
        "def f(text: int): ...",
        "def f(text: str, emotion: str): ...",
        "def f(text: str, level: int = 1): ...",
        "def f(text: str, **extra): ...",
    ],
)
def test_a_signature_that_cannot_be_an_output_is_refused(source: str) -> None:
    scope: dict[str, Any] = {}
    exec(source, scope)  # noqa: S102
    with pytest.raises(ValueError, match="output Command"):
        output_signature(scope["f"])


def test_tag_names() -> None:
    assert is_tag_name("say") and is_tag_name("speak-out_2")
    assert not is_tag_name("noop") and not is_tag_name("next")
    assert (
        not is_tag_name("2say") and not is_tag_name("say hello") and not is_tag_name("")
    )


def test_output_stream_yields_pieces_then_ends() -> None:
    async def run() -> tuple[list[str], str]:
        stream = OutputStream()
        stream.push("Hello ")
        stream.push("there")
        stream.close()
        return [piece async for piece in stream], await stream.read()

    assert asyncio.run(run()) == (["Hello ", "there"], "Hello there")


# -- the Agent loop over a scripted model ---------------------------------------------------------


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


class Rig:
    """A Wica over a scripted fake model, with a recording output Command."""

    def __init__(
        self,
        script: list[dict[str, Any]],
        *,
        output: str | None = "stream",
        max_next_steps: int | None = 1,
        commands: tuple[Callable[..., Any], ...] = (),
    ) -> None:
        config = WicaConfig(
            agent=AgentConfig(
                provider="fake",
                model="fake",
                system_prompt="You are a robot.",
                model_kwargs={"script": script, "delay_s": 0.0},
                max_next_steps=max_next_steps,
            )
        )
        self.wica = Wica.init(config, coalesce_window=0.0)
        self.said: list[tuple[str, str]] = []
        self.private: list[str] = []
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.reactions: list[ReactionTrace] = []
        self.delay = 0.0
        self.wica.world.register(
            "speech_input",
            str,
            serialize_fn=lambda v, p: [TextPart(f"User said: {v}")],
            triggers_llm_call=True,
        )
        rig = self

        async def say(
            text: OutputStream, emotion: Literal["happy", "sad", "neutral"] = "neutral"
        ) -> str:
            """Speak out loud."""
            words = await text.read()
            try:
                await asyncio.sleep(rig.delay)
            except asyncio.CancelledError:
                text.heard = words[:5]
                raise
            rig.said.append((words, emotion))
            return "Said it."

        def say_whole(text: str, emotion: str = "neutral") -> None:
            """Speak out loud."""
            rig.said.append((text, emotion))

        if output == "stream":
            self.wica.set_output_command(say)
        elif output == "whole":
            self.wica.set_output_command(Command(say_whole, name="say"))
        for command in commands:
            self.wica.register_command(command)
        self.wica.on_agent_command.subscribe(
            lambda c: self.commands.append((c.name, dict(c.args)))
        )
        self.wica.on_agent_private_text.subscribe(self.private.append)
        self.wica.on_agent_reaction_ended.subscribe(self.reactions.append)
        self.wica.start()

    def hear(self, text: str, reactions: int) -> None:
        self.wica.world.update("speech_input", text)
        wait_until(lambda: len(self.reactions) >= reactions)

    def assistant(self) -> list[Any]:
        return [m.content for m in self.wica.agent._render_messages() if m.type == "ai"]

    def observations(self) -> list[str]:
        return [
            m.content
            if isinstance(m.content, str)
            else "".join(b.get("text", "") for b in m.content if isinstance(b, dict))
            for m in self.wica.agent._render_messages()
            if m.type == "human"
        ]


@pytest.fixture
def rig() -> Iterator[Callable[..., Rig]]:
    made: list[Rig] = []

    def make(script: list[dict[str, Any]], **options: Any) -> Rig:
        made.append(Rig(script, **options))
        return made[-1]

    yield make
    for r in made:
        r.wica.close()


def test_a_tag_is_delivered_with_its_attribute_and_text_outside_is_private(
    rig: Callable[..., Rig],
) -> None:
    r = rig([{"text": 'hmm <say emotion="happy">Hello!</say>'}])
    r.hear("hi", 1)
    wait_until(lambda: r.said == [("Hello!", "happy")])
    assert r.commands == [("say", {"emotion": "happy", "text": "Hello!"})]
    assert r.private == ["hmm"]
    assert r.reactions[0].first_output_at is not None and not r.reactions[0].noop
    # History keeps what the model wrote, as text: no tool call stands for the speech.
    assert r.assistant() == ['hmm <say emotion="happy">Hello!</say>']
    assert (
        "say" not in r.wica.agent.commands and r.wica.agent.output_command_name == "say"
    )
    time.sleep(0.2)
    assert len(r.reactions) == 1  # the end of the output woke nothing


def test_an_attribute_outside_its_allowed_values_falls_back_to_the_default(
    rig: Callable[..., Rig],
) -> None:
    r = rig([{"text": '<say emotion="hapy" volume="11">Hello!</say>'}])
    r.hear("hi", 1)
    wait_until(lambda: r.said == [("Hello!", "neutral")])


def test_two_tags_are_delivered_in_order_each_with_its_attributes(
    rig: Callable[..., Rig],
) -> None:
    r = rig(
        [
            {
                "text": '<say emotion="happy">Good news!</say><say emotion="sad">But wait.</say>'
            }
        ]
    )
    r.delay = 0.1
    r.hear("hi", 1)
    wait_until(lambda: r.said == [("Good news!", "happy"), ("But wait.", "sad")])
    assert [name for name, _ in r.commands] == ["say", "say"]


def test_a_str_output_command_gets_the_whole_text_once(rig: Callable[..., Rig]) -> None:
    r = rig([{"text": "<say>Hello there, friend.</say>"}], output="whole")
    r.hear("hi", 1)
    wait_until(lambda: r.said == [("Hello there, friend.", "neutral")])


def test_text_with_no_tag_is_still_delivered(rig: Callable[..., Rig]) -> None:
    r = rig([{"text": "Hello there."}])
    r.hear("hi", 1)
    wait_until(lambda: r.said == [("Hello there.", "neutral")])
    assert r.private == []


def test_noop_does_nothing_and_is_not_a_tool(rig: Callable[..., Rig]) -> None:
    r = rig([{"text": "<noop/>"}, {"text": ""}])
    r.hear("a door closes", 1)
    assert r.reactions[0].noop and r.reactions[0].outcome == "ok"
    assert r.said == [] and r.commands == [] and r.private == []
    assert r.assistant() == ["<noop/>"]  # the choice stays visible in history
    assert set(r.wica.agent.commands) == {"cancel_command"}
    r.hear("another", 2)
    assert r.reactions[1].noop and r.reactions[1].outcome == "empty"


def test_an_agent_without_an_output_command_keeps_its_text_private(
    rig: Callable[..., Rig],
) -> None:
    r = rig([{"text": "Just thinking <say>aloud</say>."}], output=None)
    r.hear("hi", 1)
    assert r.private == ["Just thinking <say>aloud</say>."]
    assert r.commands == [] and not r.reactions[0].noop
    assert "<say>" not in r.wica.agent.system_prompt
    assert "Nobody reads the text you write" in r.wica.agent.system_prompt


def test_a_note_wakes_one_step_that_sees_it_once(rig: Callable[..., Rig]) -> None:
    r = rig(
        [
            {"text": "<say>Give me a moment.</say><next>then report</next>"},
            {"text": "<say>Done.</say>"},
            {"text": "<noop/>"},
        ]
    )
    r.hear("go", 2)
    wait_until(lambda: [s for s, _ in r.said] == ["Give me a moment.", "Done."])
    time.sleep(0.2)
    assert len(r.reactions) == 2  # the second output carried no note
    assert "You planned to do next: then report" in r.observations()[1]
    r.hear("again", 3)
    assert "You planned to do next" not in r.observations()[2]


def test_a_short_output_that_ends_before_its_response_still_wakes(
    rig: Callable[..., Rig],
) -> None:
    long_note = " ".join(["then"] * 40)
    r = rig(
        [
            {"text": f"<say>Hi.</say><next>{long_note}</next>"},
            {"text": "<noop/>"},
        ]
    )
    r.hear("go", 2)
    assert r.reactions[1].noop


def test_chained_notes_stop_at_the_cap(rig: Callable[..., Rig]) -> None:
    step = {"text": "<say>More.</say><next>keep going</next>"}
    r = rig([step] * 10, max_next_steps=2)
    r.hear("go", 3)
    time.sleep(0.3)
    assert len(r.reactions) == 3  # the first reply, then two chained wakes
    r.hear("something else", 4)  # an outside trigger resets the count
    wait_until(lambda: len(r.reactions) >= 6)


def test_the_note_can_be_turned_off(rig: Callable[..., Rig]) -> None:
    r = rig([{"text": "<say>More.</say><next>keep going</next>"}] * 3, max_next_steps=0)
    r.hear("go", 1)
    wait_until(lambda: len(r.said) == 1)
    time.sleep(0.2)
    assert len(r.reactions) == 1
    assert "<next>" not in r.wica.agent.system_prompt


def test_speaking_again_cuts_off_the_running_output_and_history_keeps_what_was_heard(
    rig: Callable[..., Rig],
) -> None:
    r = rig(
        [
            {"text": "<say>Once upon a time there was a robot.</say>"},
            {"text": "<say>Okay, stopping.</say>"},
        ]
    )
    r.delay = 5.0
    r.hear("story", 1)
    r.delay = 0.0
    r.hear("stop", 2)
    wait_until(lambda: [s for s, _ in r.said] == ["Okay, stopping."])
    assert r.assistant() == ["<say>Once </say>", "<say>Okay, stopping.</say>"]


def test_speaking_does_not_cancel_a_running_command(rig: Callable[..., Rig]) -> None:
    finished: list[str] = []

    async def dance() -> str:
        """Dance."""
        await asyncio.sleep(0.4)
        finished.append("dance")
        return "Danced."

    made = Rig(
        [
            {"text": "", "tool_calls": [{"name": "dance", "args": {}}]},
            {"text": "<say>Still dancing!</say>"},
            {"text": "<noop/>"},
        ],
    )
    try:
        made.wica.register_command(dance)
        made.hear("dance", 1)
        made.hear("how is it going", 2)
        wait_until(lambda: finished == ["dance"])
    finally:
        made.wica.close()


def test_a_cancelled_output_keeps_only_what_was_heard(rig: Callable[..., Rig]) -> None:
    r = rig([{"text": "aside <say>Once upon a time.</say><say>Never said.</say>"}])
    r.delay = 5.0
    r.hear("story", 1)
    for call_id in r.reactions[0].command_call_ids:
        r.wica.agent.cancel_command(call_id)
    wait_until(lambda: r.assistant() == ["aside <say>Once </say>"])


def test_a_response_that_spoke_is_not_retried_but_a_silent_one_is(
    rig: Callable[..., Rig],
) -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    bad_call = {"name": "add", "args": {"a": "not a number", "b": 2}}
    r = rig(
        [
            {"text": "<say>Sure!</say>", "tool_calls": [bad_call]},
            {"text": "", "tool_calls": [bad_call]},
            {"text": "<say>Sorry.</say>"},
        ],
        commands=(add,),
    )
    r.hear("hi", 1)
    assert r.reactions[0].attempts == 1 and r.reactions[0].outcome == "ok"
    r.hear("again", 2)
    assert (
        r.reactions[1].attempts == 2
    )  # no output had started: the guards ran and retried
    wait_until(lambda: "Sorry." in [s for s, _ in r.said])


def test_the_output_command_is_validated_when_set() -> None:
    wica = Wica.init(
        WicaConfig(agent=AgentConfig(provider="fake", model="fake", system_prompt="x"))
    )
    try:

        def noop(text: str) -> None:
            """x"""

        def stream_sync(text: OutputStream) -> None:
            """x"""

        def needs(text: str, emotion: str) -> None:
            """x"""

        for bad in (noop, stream_sync, needs):
            with pytest.raises(ValueError):
                wica.set_output_command(bad)
        assert wica.agent.output_command_name is None
    finally:
        wica.close()


def test_an_output_entry_reads_as_speech_not_as_a_call(rig: Callable[..., Rig]) -> None:
    r = rig(
        [
            {"text": '<say emotion="happy">Once upon a time.</say>'},
            {"text": "<say>Okay.</say>"},
            {"text": "<noop/>"},
        ]
    )
    r.delay = 5.0
    r.hear("story", 1)
    r.delay = 0.0
    r.hear("stop", 2)
    running = r.observations()[1]
    assert "Saying (emotion='happy') \"Once upon a time.\"… (still speaking" in running
    assert "Calling say(" not in running
    r.hear("thanks", 3)
    assert "Said (emotion='happy') \"Once \" → cut off there" in r.observations()[2]
    assert 'Said "Okay." → finished: Said it.' in r.observations()[2]
