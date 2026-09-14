"""WICA conversation demo — talk to a simulated social robot in the browser.

Run with:

    uv run --group demo python -m examples.conversation_demo.app

Configuration comes from agent.config.json (next to this file), which reads its API key from the
WICA-namespaced WICA_ANTHROPIC_API_KEY env var (see agent.config.json's "api_key_env") — set it
before running:

    export WICA_ANTHROPIC_API_KEY=sk-...

To use a different provider/model, or a literal key, edit agent.config.json directly (see
specs/config.md) — e.g. copy it to a `*.local.json` file (git-ignored) with a literal "api_key".

Logging is the application's concern, not the framework's: this demo configures the wica.* loggers
below (raise the level to DEBUG to see the full World+Agent lifecycle trace — registrations, every
update and whether it triggered a call, LLM output, command start/end).

This is the first runnable example (see specs/conversation-demo.md). It drives WICA purely through
its public API: speech and sensor events enter the World; the Agent reasons over the World, issues
Commands, and replies; the UI shows the live World state and the exact prompt sent to the model.

This file is the **app**: the World entries (and how each shows to the model and to the person),
the `Robot` whose methods are the Commands, and the composition — in this order, which is what the
framework's output wiring exists to allow (specs/wica.md, "Output wiring is delegated"):

    1. build_system(config)         → the Wica (or a World-only fallback), entries registered
    2. build_ui(wica, world, …)     → the Gradio page, which owns its presenters (app_ui.py)
    3. wire(wica, transcript, slot) → the Robot over the World + the UI's Speaking slot; output
                                      sink + `say` output Command set on the Wica; Commands registered
    4. wica.start(); blocks.launch()

Nothing is bound late: every object is built after the ones it needs, and everything is wired
before `start()`. Steps 1 and 3 are Gradio-free, so the tests run the same wiring against a
transcript and slot they build themselves (tests-e2e/test_example_flow.py).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wica import Content, TextPart, Wica, World, WorldEntry
from wica.agent import CommandExecution
from wica.config import MissingEnvError, WicaConfig

from examples.conversation_demo.speaking import SpeakingSlot
from examples.conversation_demo.transcript import EntryDisplay, TranscriptLog

# Importing this module has no side effects: it only defines the World entries, the Robot, and the
# standup functions. Logging config and the Gradio UI import live in `main()`, so tests can import
# the wiring and stand it up against any config (e.g. a fake provider) without pulling in the
# demo-only Gradio dependency or configuring the root logger.

# --- Configuration -------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "agent.config.json"

# --- World entries the demo owns ----------------------------------------------------
#
# All are include_in_prompt=True so they reach the model and show up in the World panel.
# Sensor inputs (speech, closest user) trigger the agent; the robot's own state (emotion,
# tracking) does not — otherwise the agent's own actions would re-trigger it in a loop.


def _speech(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("The person hasn't said anything to you yet.")]
    return [TextPart(f'The closest person just said to you: "{value}"')]


def _closest_user(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("No one is standing close to you right now.")]
    return [TextPart(f'The person standing closest to you is user "{value}".')]


def _emotion(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("You currently feel neutral.")]
    return [TextPart(f"You currently feel {value}.")]


def _tracked_user(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("You are not tracking anyone.")]
    return [TextPart(f'You are tracking user "{value}".')]


def register_world(world: World) -> None:
    world.register("speech_input", str, serialize_fn=_speech, triggers_llm_call=True)
    world.register(
        "closest_user", str, serialize_fn=_closest_user, triggers_llm_call=True
    )
    world.register("emotion", str, serialize_fn=_emotion)
    world.register("tracked_user", str, serialize_fn=_tracked_user)


def display_entry(entry: WorldEntry) -> EntryDisplay | None:
    """How the demo's entries show in the transcript — the person-facing counterpart of the
    serialize_fns above (the same per-entry knowledge, rendered for a human instead of the model).
    This is the one hook the generic transcript takes; anything it doesn't recognise returns None
    and gets the generic default (`⚡ key = value`, `🦾 name(args)` for a Command). See
    specs/conversation-demo.md ("A reusable transcript")."""
    value = entry.current.value
    if entry.key == "speech_input":
        return EntryDisplay(f'🗣️ "{value}"')
    if entry.key == "closest_user":
        return EntryDisplay(
            "👤 Closest user gone"
            if value is None
            else f"👤 Closest user detected: {value}"
        )
    if isinstance(value, CommandExecution) and value.name == "say":
        # The voice: title it by channel and show what the robot set out to say as the body.
        return EntryDisplay("🗣️ say", str(value.args.get("text", "")))
    return None


# --- The robot: its Commands (fake actions) -----------------------------------------

# Simulated per-word speaking pace. Because `say` is a real Command, taking time here means it stays
# "running" (and cancellable — barge-in) in the World for its whole duration, and the Speaking panel
# shows the words as they're "spoken". This is a demo simulation, not framework token streaming
# (which is post-v1 — see specs/agent.md "Future improvements").
_SAY_WORD_DELAY_S = 0.3


class Robot:
    """The simulated robot: its Commands are bound methods over the World it acts on and the
    Speaking slot its voice drives. Built at wiring time, after both exist — so nothing here is
    bound late. A bound method is a valid Command (name from the method, description from its
    docstring, `self` left out of the schema)."""

    def __init__(self, world: World, speaking: SpeakingSlot) -> None:
        self.world = world
        self.speaking = speaking

    async def say(self, text: str) -> str:
        """Speak out loud to the person in front of you — this is the only way they hear you. Use it
        for anything you want to say; keep it to a sentence or two."""
        words = text.split()
        if not words:
            return "Said it."
        # `say` only drives the Speaking panel: the transcript shows this utterance as a Command
        # item (full text, live state) through the generic log, like any other Command. Simulating
        # speech as a slow async loop keeps the Command "running" (cancellable) while it speaks; a
        # cancellation lands at the sleep, and the slot records it before the CancelledError
        # propagates.
        token = self.speaking.start(text)
        try:
            for _ in words:
                await asyncio.sleep(_SAY_WORD_DELAY_S)
                self.speaking.advance(token)
        except asyncio.CancelledError:
            self.speaking.cancelled(token)
            raise
        self.speaking.complete(token)
        return "Said it."

    async def dance(self) -> str:
        """Perform a fun little dance. Takes about 10 seconds to complete."""
        await asyncio.sleep(10)
        return "Finished the dance."

    def set_emotion(self, emotion: str) -> str:
        """Show an emotion on your face, e.g. 'happy', 'curious', 'sad', 'excited'."""
        self.world.update("emotion", emotion)
        return f"Now showing emotion: {emotion}."

    def switch_user_tracking(self, user_id: str | None = None) -> str:
        """Follow one specific person, tracking them from now on. Pass no user (null) to stop
        tracking anyone. You follow at most one person at a time."""
        self.world.update("tracked_user", user_id)
        return (
            "Stopped tracking." if user_id is None else f"Now tracking user {user_id}."
        )

    @property
    def commands(self) -> list[Callable[..., Any]]:
        """The robot's actions other than its voice (`say` is the output Command, wired apart)."""
        return [self.dance, self.set_emotion, self.switch_user_tracking]


# --- Composition ---------------------------------------------------------------------


def build_system(config: WicaConfig) -> tuple[Wica | None, World, str | None]:
    """Step 1: build the Wica through the single entry point (Wica.init) and register the demo's
    World entries. Returns `(wica, world, config_error)`. Falls back to explore-only if the config's
    api_key_env isn't set — `wica` is None, `world` is a World-only system on its own loop, and
    `config_error` says what to set — so the app still opens and is explorable without credentials.

    The api key resolves at Agent build inside Wica.init, so that's what we guard: an unset
    api_key_env raises MissingEnvError there (see specs/config.md, specs/wica.md)."""
    try:
        wica = Wica.init(config)
    except MissingEnvError as exc:
        # Explore-only: a World-only system (no Agent) on its own loop, so the panel and sensor
        # inputs still work while nothing reasons over them.
        explore_loop = asyncio.new_event_loop()
        threading.Thread(target=explore_loop.run_forever, daemon=True).start()
        world = World(explore_loop)
        world.start()
        register_world(world)
        return None, world, f"environment variable {exc.env_var!r} is not set"
    register_world(wica.world)
    return wica, wica.world, None


def wire(wica: Wica, transcript: TranscriptLog, speaking: SpeakingSlot) -> Robot:
    """Step 3: wire the robot to the Wica and to the UI's presenters, before `start()`. The Robot is
    built over the Wica's World and the Speaking slot; the transcript's `output_sink` receives the
    model's free text (private reasoning, since `say` is the output Command); `say` is the output
    Command (the voice); the other Commands are registered. Gradio-free: the tests call this with a
    transcript and slot of their own.

    Wica owns the single event loop in a daemon thread and the World is thread-safe, so the Gradio
    side stays fully synchronous: sensor events call world.update(...) directly, and the Agent's
    instrumentation Events (emitted on the loop) reach the transcript, which pushes chat items onto
    its thread-safe queue that the UI's timer drains."""
    robot = Robot(wica.world, speaking)
    wica.set_output_sink(transcript.output_sink)
    wica.set_output_command(robot.say)
    for command in robot.commands:
        wica.register_command(command)
    return robot


def main() -> None:
    # WICA is a library and configures no logging itself; the application decides what is shown. Keep
    # third-party logs quiet and show wica.* at INFO (raise to logging.DEBUG for the full trace).
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("wica").setLevel(logging.INFO)

    # Import the Gradio UI lazily so importing this module needs only core deps (see module docstring).
    from examples.conversation_demo.app_ui import build_ui

    wica, world, config_error = build_system(WicaConfig.from_json_file(CONFIG_PATH))
    ui = build_ui(wica, world, display_entry, config_error)
    if wica is not None:
        wire(wica, ui.transcript, ui.speaking)
        wica.start()
    try:
        ui.blocks.launch()
    finally:
        if wica is not None:
            wica.close()


if __name__ == "__main__":
    main()
