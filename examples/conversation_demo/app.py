"""WICA conversation demo — talk to a simulated social robot in the browser.

Run with:

    uv run --group demo python -m examples.conversation_demo.app [--folder PATH]
        [--host ADDRESS] [--port N] [--start | --no-start]

The page has three tabs: the demo's own **Conversation** (Start, Stop and the robot's panels),
then app-setup's **Setup** (choose a provider, configure it, install its extra, test its key)
and **Secrets** (the key variables, saved to the folder's `.env`). The app's folder (its
`config.json`, its `.env`, its `launch.json`) is `~/.config/wica-conversation-demo` unless
`--folder` names another. The default config (`default_config.json`) runs Anthropic's
claude-haiku-4-5 with the persona copied into the folder (`wica.md`, edit it there), its key read
from ANTHROPIC_API_KEY (the shell, or the Secrets tab). The session starts at launch unless the
Setup panel's checkbox or `--no-start` says otherwise; without a key, the Conversation tab says so
instead (see specs/conversation-demo.md).

Logging is the application's concern, not the framework's: this demo configures the wica.* loggers
in `main()` (raise the level to DEBUG to see the full World+Agent lifecycle trace — registrations,
every update and whether it triggered a call, LLM output, command start/end).

It drives WICA purely through its public API: speech and sensor events enter the World; the Agent
reasons over the World, issues Commands, and replies; the UI shows the live World state and the
exact prompt sent to the model.

This file is the **app**: the World entries (and how each shows to the model and to the person),
the `Robot` whose methods are the Commands, and the composition of one session — in this order,
which is what the framework's output wiring exists to allow (specs/wica.md, "Output wiring is
delegated"):

    1. build_system(config)         → the Wica, entries registered
    2. the presenters               → the contrib's TranscriptLog + ReactionLog (subscribed to the
                                      Wica's Events as they are built) and the Speaking slot
    3. wire(wica, slot)             → the Robot over the World + the Speaking slot; `say` set as
                                      the output Command on the Wica; Commands registered
    4. wica.start()

`open_session` runs the four steps; the page's Start runs it through `Conversation`
(session.py) with the saved config. Steps 1 and 3 build no Gradio page, so the tests and the
benchmark run the same wiring (tests-e2e/test_example_flow.py, benchmarks/bench.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app_setup.app import App, AppModel, Section
from app_setup.autostart import add_start_arguments, starts_at_launch
from app_setup.folder import ConfigFolder, default_folder

import wica.setup
from examples.conversation_demo.app_ui import build_page
from examples.conversation_demo.session import Conversation, Running
from wica import (
    CommandExecution,
    Content,
    OutputStream,
    TextPart,
    Wica,
    World,
    WorldEntry,
)
from wica.config import WicaConfig
from wica.contrib.gradio import EntryDisplay, ReactionLog, TranscriptLog

from examples.conversation_demo.speaking import SpeakingSlot

# Importing this module has no side effects: it only defines the World entries, the Robot, and the
# standup functions. Logging config and the page build live in `main()`, so tests can import the
# wiring and stand it up against any config (e.g. a fake provider) without building a Gradio page
# or configuring the root logger. (The `wica.contrib.gradio` import above does pull Gradio, which
# the `dev` group provides — see specs/gradio-contrib.md "Packaging".)

# --- Configuration -------------------------------------------------------------------

# The persona the example configs read, and the one copied into the app's folder.
PERSONA_FILE = Path(__file__).parent / "prompts" / "wica.md"

# The demo's default config, committed next to this file: Anthropic with the persona the app's
# folder holds (`wica.md`, relative, so it resolves against the folder), and no key: the key is read
# from ANTHROPIC_API_KEY, the provider's default variable. app-setup writes it into an empty
# folder at the first launch.
DEFAULT_CONFIG_FILE = Path(__file__).parent / "default_config.json"
DEFAULT_CONFIG: dict[str, Any] = json.loads(
    DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")
)

APP = App(
    name="wica-conversation-demo",
    title="WICA — talk to a social robot",
    host="wica",
    sections=(Section("wica", wica.setup, title="Agent"),),
    default_config=DEFAULT_CONFIG,
    autostart=True,
)


def ensure_persona(folder: Path) -> Path:
    """The persona in the app's folder, copied there from the bundled one when the folder has
    none: the default config names it (`wica.md`), and the user edits it there. Never
    overwritten."""
    persona = folder / PERSONA_FILE.name
    if not persona.exists():
        persona.write_text(PERSONA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return persona


_demo_logger = logging.getLogger(__name__)

# Opt-in OpenTelemetry export, off by default (see specs/instrumentation.md "Consumers" —
# exporters are an application concern; WICA itself never imports the SDK). Set
# WICA_DEMO_TRACES=console to print every span (wica.world.update, wica.agent.reaction/model/
# guards/command, plus anything a Command opens) to stdout via the SDK's ConsoleSpanExporter.
_DEMO_TRACES_ENV = "WICA_DEMO_TRACES"


def _maybe_enable_console_traces() -> None:
    if os.environ.get(_DEMO_TRACES_ENV) != "console":
        return
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _demo_logger.info(
        "%s=console: exporting OpenTelemetry spans to the console", _DEMO_TRACES_ENV
    )


# --- World entries the demo owns ----------------------------------------------------
#
# All are include_in_prompt=True so they reach the model and show up in the World panel.
# Sensor inputs (speech, closest user) trigger the agent; the robot's own state (emotion,
# tracking) does not — otherwise the agent's own actions would re-trigger it in a loop.


def _speech(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("You haven't heard anyone say anything to you yet.")]
    # Who spoke is not known — speech arrives without a speaker, so this must not claim it came
    # from the closest person (which contradicts an empty closest_user and makes models wait).
    return [TextPart(f'You hear someone saying to you: "{value}"')]


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
    specs/conversation-demo.md ("Reused from the Gradio contrib")."""
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

# Simulated per-word speaking pace. Because `say` is a real Command execution, taking time here
# means it stays "running" (and cancellable — barge-in) in the World for its whole duration, and
# the Speaking panel shows the words as they're "spoken". The streaming is the framework's (the
# words reach `say` as the model writes them); the pace is the demo's simulated TTS.
_SAY_WORD_DELAY_S = 0.3


class Robot:
    """The simulated robot: its Commands are bound methods over the World it acts on and the
    Speaking slot its voice drives. Built at wiring time, after both exist — so nothing here is
    bound late. A bound method is a valid Command (name from the method, description from its
    docstring, `self` left out of the schema)."""

    def __init__(self, world: World, speaking: SpeakingSlot) -> None:
        self.world = world
        self.speaking = speaking

    async def say(self, text: OutputStream) -> None:
        """Speak out loud to the person in front of you — this is the only way they hear you. Use it
        for anything you want to say; keep it to a sentence or two."""
        # The output Command: the model writes a <say> tag and its words arrive here as they are
        # written, so the robot starts speaking before the response is complete. `say` only
        # drives the Speaking panel: the transcript shows this utterance as a Command item (full
        # text, live state) through the generic log, like any other Command. Simulating speech
        # as a slow async loop keeps the execution "running" (cancellable) while it speaks; a
        # cancellation lands at a sleep (or while waiting for more words), and `say` then
        # reports what was heard and marks the slot before the CancelledError propagates.
        token: int | None = None
        spoken = 0
        words: list[str] = []
        try:
            while True:
                # Everything written so far; the last word may still be growing unless the
                # body is over or a space followed it.
                over = text.closed
                words = text.text.split()
                whole = len(words)
                if not over and words and not text.text[-1].isspace():
                    whole -= 1
                if words:
                    if token is None:
                        token = self.speaking.start(text.text)
                    else:
                        self.speaking.extend(token, text.text)
                if spoken < whole:
                    await asyncio.sleep(_SAY_WORD_DELAY_S)
                    spoken += 1
                    if token is not None:
                        self.speaking.advance(token)
                elif over:
                    break
                else:
                    # Caught up with the model: wait for more of the sentence (or its end).
                    try:
                        await anext(text)
                    except StopAsyncIteration:
                        pass
        except asyncio.CancelledError:
            # Cut off: history keeps the words the person heard, not the whole sentence.
            text.heard = " ".join(words[:spoken])
            if token is not None:
                self.speaking.cancelled(token)
            raise
        if token is not None:
            self.speaking.complete(token)

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


def build_system(config: WicaConfig) -> Wica:
    """Step 1: build the Wica through the single entry point (Wica.init) and register the demo's
    World entries. The key and the prompt file resolve here: MissingSecretError, ConfigError and
    ImportError surface from this call (see specs/config.md, specs/wica.md)."""
    wica = Wica.init(config)
    register_world(wica.world)
    return wica


def wire(wica: Wica, speaking: SpeakingSlot) -> Robot:
    """Step 3: wire the robot to the Wica and to the UI's Speaking slot, before `start()`. The Robot
    is built over the Wica's World and the slot; `say` is the output Command (the voice: what the
    model writes inside a <say> tag); the other Commands are registered. The transcript needs no
    wiring — it subscribed to the Wica's Events when it was built. Gradio-free: the tests call
    this with a slot of their own.

    Wica owns the single event loop in a daemon thread and the World is thread-safe, so the Gradio
    side stays fully synchronous: sensor events call world.update(...) directly, and the Agent's
    instrumentation Events (emitted on the loop) reach the transcript, which pushes chat items onto
    its thread-safe queue that the UI's timer drains."""
    robot = Robot(wica.world, speaking)
    wica.set_output_command(robot.say)
    for command in robot.commands:
        wica.register_command(command)
    return robot


def open_session(config: WicaConfig) -> Running:
    """One session, in order: build, the presenters (subscribed before anything runs), wire,
    start. The Wica is closed again when a later step fails."""
    wica = build_system(config)
    try:
        running = Running(
            config=config,
            wica=wica,
            transcript=TranscriptLog(wica, display_entry),
            reactions=ReactionLog(wica, display_entry),
            speaking=SpeakingSlot(),
        )
        wire(wica, running.speaking)
        wica.start()
    except BaseException:
        wica.close()
        raise
    return running


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    folder = default_folder(APP.name)
    parser = argparse.ArgumentParser(description=APP.title)
    parser.add_argument(
        "--folder",
        type=Path,
        default=folder,
        help=f"the demo's folder: its config and its keys (default: {folder})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="the address to serve on")
    parser.add_argument("--port", type=int, default=None, help="the port to serve on")
    add_start_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    options = _arguments(argv)
    # WICA is a library and configures no logging itself; the application decides what is shown. Keep
    # third-party logs quiet and show wica.* at INFO (raise to logging.DEBUG for the full trace).
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("wica").setLevel(logging.INFO)
    _demo_logger.setLevel(logging.INFO)
    _maybe_enable_console_traces()

    conversation = Conversation(open_session)
    # Building the model makes the folder ready and loads its .env; the config is locked while a
    # session runs.
    model = AppModel(
        APP,
        ConfigFolder(options.folder, APP.default_config),
        locked=conversation.locked,
    )
    _demo_logger.info("folder: %s", model.path)
    ensure_persona(model.path)
    # A relative path in the config (the default's `wica.md`) is the folder's: the session builds
    # with the folder as `base_dir`, and app-setup passes the setup checks the same folder.
    starts = starts_at_launch(model, options.start)
    if starts:
        # With blockers (no key yet, at the first launch), they show on the Conversation tab.
        for reason in conversation.start(model):
            _demo_logger.warning("not started at launch: %s", reason)
    try:
        build_page(
            model, conversation, open_on="conversation" if starts else "setup"
        ).launch(server_name=options.host, server_port=options.port)
    finally:
        conversation.stop()


if __name__ == "__main__":
    main()
