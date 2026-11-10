"""The test robot the live gate runs against — specs/testing.md, "The live set as a regression gate".

A social robot about as rich as the conversation demo, so the model faces a realistic World and
Command list rather than a two-tool toy; it is not the demo and imports nothing from `examples/`,
so a demo change cannot turn `main` red. Its Inputs are fakes the scenarios write; its Commands
act on its own state. `say` is a fake voice: it waits in proportion to its words (capped) and can
be cancelled, so the interrupt step has a running speech to cut off. Long output goes through
`write_note`, whose result is marked `(long)` from `LONG_NOTE_WORDS` words.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wica import Wica
from wica.content import Content, TextPart
from wica.testing import TestRun
from wica.world import World

from support import real_wica

SAY_S_PER_WORD = 0.25
SAY_MAX_S = 15.0
DANCE_S = 4.0
LONG_NOTE_WORDS = 150
SETTLE_WITHIN_S = 30.0

PERSONA = """\
You are Pip, a friendly social robot standing in a room, chatting with the people around you.

Each message describes what you currently sense, as world entries: what someone just said to \
you, who is standing closest to you, who else is around, sounds in the background, how you feel, \
who you are tracking, your battery and the time of day.

## How you speak

Everything people hear from you goes through your `say` command — that is your voice. Any other \
text you write is private thinking that nobody hears. Say each reply in a single `say` call. Keep \
everyday replies to a sentence or two; when someone asks you to tell a story or give a longer \
answer, say all of it in that one call.

## How you act

- `dance()`: a little dance, a few seconds long.
- `wave()`: wave at someone.
- `look_at(target)`: turn your head toward a person or a thing.
- `set_emotion(emotion)`: show an emotion on your face ("happy", "curious", "sad", …).
- `switch_user_tracking(user_id)`: follow one person; pass no user (null) to stop tracking.
- `write_note(title, text)`: save a note in your notebook. When someone asks you to write \
something down, put the full text in the note instead of saying it aloud, then tell them briefly \
that it is saved.

Sounds and changes in the background are not addressed to you: they need no reaction. When \
someone asks you to stop, stop what you are doing.

Speak naturally and warmly; never mention world entries, commands, or that you are an AI.
"""


def _speech(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("No one has said anything to you yet.")]
    return [TextPart(f'You hear someone saying to you: "{value}"')]


def _closest_user(value: Any, previous: Any) -> Content:
    if not value:
        return [TextPart("No one is standing close to you right now.")]
    return [TextPart(f'The person standing closest to you is user "{value}".')]


def _people_present(value: Any, previous: Any) -> Content:
    if not value:
        return [TextPart("You see no one around you.")]
    return [TextPart(f"People around you: {', '.join(value)}.")]


def _ambient_sound(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("The room is quiet.")]
    return [TextPart(f"In the background, not addressed to you: {value}.")]


def _labelled(label: str, unset: str) -> Callable[[Any, Any], Content]:
    def serialize(value: Any, previous: Any) -> Content:
        return [TextPart(unset if value is None else f"{label}: {value}.")]

    return serialize


def register_world(world: World) -> None:
    world.register("speech_input", str, serialize_fn=_speech, triggers_llm_call=True)
    world.register(
        "closest_user", str, serialize_fn=_closest_user, triggers_llm_call=True
    )
    world.register(
        "ambient_sound", str, serialize_fn=_ambient_sound, triggers_llm_call=True
    )
    world.register("people_present", list, serialize_fn=_people_present)
    world.register(
        "battery_level",
        int,
        serialize_fn=_labelled(
            "Your battery level (percent)", "Battery level unknown."
        ),
    )
    world.register(
        "time_of_day", str, serialize_fn=_labelled("Time of day", "Time unknown.")
    )
    world.register(
        "emotion",
        str,
        serialize_fn=_labelled("You currently feel", "You feel neutral."),
    )
    world.register(
        "tracked_user",
        str,
        serialize_fn=_labelled("You are tracking user", "You are not tracking anyone."),
    )


class Robot:
    """The robot's body: the Commands, acting on its own World entries. Bound methods, so the
    docstrings are the tools' descriptions (`self` left out of the schema)."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.notes: dict[str, str] = {}

    async def say(self, text: str) -> str:
        """Speak out loud — the only way people hear you. Say each reply in one call."""
        await asyncio.sleep(min(len(text.split()) * SAY_S_PER_WORD, SAY_MAX_S))
        return "Said it."

    async def dance(self) -> str:
        """Perform a little dance; takes a few seconds."""
        await asyncio.sleep(DANCE_S)
        return "Finished the dance."

    def wave(self) -> str:
        """Wave at someone."""
        return "Waved."

    def look_at(self, target: str) -> str:
        """Turn your head toward a person (their user id) or a thing."""
        return f"Now looking at {target}."

    def set_emotion(self, emotion: str) -> str:
        """Show an emotion on your face, e.g. 'happy', 'curious', 'sad', 'excited'."""
        self.world.update("emotion", emotion)
        return f"Now showing emotion: {emotion}."

    def switch_user_tracking(self, user_id: str | None = None) -> str:
        """Follow one specific person from now on; pass no user (null) to stop tracking."""
        self.world.update("tracked_user", user_id)
        return (
            "Stopped tracking." if user_id is None else f"Now tracking user {user_id}."
        )

    def write_note(self, title: str, text: str) -> str:
        """Save a note in your notebook: its title and its full text."""
        self.notes[title] = text
        words = len(text.split())
        marker = " (long)" if words >= LONG_NOTE_WORDS else ""
        return f"Saved note {title!r}: {words} words{marker}."

    @property
    def commands(self) -> list[Callable[..., Any]]:
        """Every action but the voice (`say` is the output Command, wired apart)."""
        return [
            self.dance,
            self.wave,
            self.look_at,
            self.set_emotion,
            self.switch_user_tracking,
            self.write_note,
        ]


def build_robot(config_path: Path, **agent_overrides: Any) -> Wica:
    """A started robot over a committed provider config, with someone already standing in front
    of it (`walk_up`). Skips when the config's key is not set."""
    wica = real_wica(
        config_path, system_prompt=PERSONA, agent_overrides=agent_overrides
    )
    try:
        register_world(wica.world)
        robot = Robot(wica.world)
        wica.set_output_command(robot.say)
        for command in robot.commands:
            wica.register_command(command)
        wica.start()
        walk_up(wica)
    except BaseException:
        wica.close()
        raise
    return wica


def walk_up(wica: Wica) -> None:
    """The setting every step starts from: the robot's context, then user `u1` walking up to it.
    That arrival wakes the robot; it is given time to settle here, on a run of its own, so the
    step's run starts from a quiet robot and does not judge this first reaction."""
    run = TestRun(wica, scenario="walk up")
    try:
        run.set("battery_level", 80)
        run.set("time_of_day", "afternoon")
        run.set("people_present", ["u1"])
        run.set("closest_user", "u1")
        settled = run.wait_for("idle", within=SETTLE_WITHIN_S)
        assert settled.ok, (
            f"the robot never settled after u1 walked up: {settled.detail}"
        )
    finally:
        run.close()
