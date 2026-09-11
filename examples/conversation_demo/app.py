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

This file is the **app**: the World entries and robot Commands, plus the standup that wires WICA to
the presenter and UI. The transcript/prompt state the agent callbacks and UI share lives in
`app_state.DemoState`; the Gradio surfaces live in `app_ui.build_ui`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wica import Content, TextPart, Wica, World
from wica.config import MissingEnvError, WicaConfig

from examples.conversation_demo.app_state import DemoState

# Importing this module has no side effects: it only defines the World entries, Commands, and the
# `build_app` standup. Logging config and the Gradio UI import live in `main()`, so tests can import
# the wiring and stand it up against any config (e.g. a fake provider) without pulling in the
# demo-only Gradio dependency or configuring the root logger. See tests-e2e/test_example_flow.py.

# --- Configuration -------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "agent.config.json"

# --- World entries the demo owns ----------------------------------------------------
#
# All are include_in_prompt=True so they reach the model and show up in the World panel.
# Sensor inputs (speech, closest user) trigger the agent; the robot's own state (emotion,
# tracking) does not — otherwise the agent's own actions would re-trigger it in a loop.
#
# `world` is bound once the system is stood up below (from wica.world, or a World-only fallback in
# explore-only mode). The serialize_fns / commands reference it at call time, which is always after
# that binding.
world: World


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


def register_world() -> None:
    world.register("speech_input", str, serialize_fn=_speech, triggers_llm_call=True)
    world.register(
        "closest_user", str, serialize_fn=_closest_user, triggers_llm_call=True
    )
    world.register("emotion", str, serialize_fn=_emotion)
    world.register("tracked_user", str, serialize_fn=_tracked_user)


# --- Commands (the robot's fake actions) --------------------------------------------

# The presenter holds the transcript/prompt state the agent callbacks and UI share; `say` grows its
# spoken bubble through it. Bound by build_app() (like `world`), so a test can stand the app up with
# a fresh presenter; the commands reference it at call time, always after that binding.
state: DemoState

# Simulated per-word speaking pace. Because `say` is a real Command, taking time here means it stays
# "running" (and cancellable — barge-in) in the World for its whole duration, and the demo streams
# the words into the transcript as they're "spoken". This is a demo simulation, not framework token
# streaming (which is post-v1 — see specs/agent.md "Future improvements").
_SAY_WORD_DELAY_S = 0.3


async def say(text: str) -> str:
    """Speak out loud to the person in front of you — this is the only way they hear you. Use it for
    anything you want to say; keep it to a sentence or two."""
    words = text.split()
    if not words:
        return "Said it."
    # Capture the reaction this utterance belongs to *before* the first await: `say` streams in the
    # background, so a later step (e.g. a fast command completing in the same reaction) can open a
    # newer reaction group while we sleep. Binding the parent now keeps the spoken words in their
    # originating group rather than adopting the newer one. See specs/_fixes.md.
    reaction_id = state.current_reaction_id()
    # Open the say bubble once (the presenter enqueues it), then grow its content word by word:
    # tick appends this exact dict to the transcript, so mutating its content in place streams the
    # words into the UI. Simulating speech as a slow async loop also keeps the Command "running"
    # (cancellable) while it speaks.
    message: dict[str, Any] | None = None
    spoken: list[str] = []
    for word in words:
        await asyncio.sleep(_SAY_WORD_DELAY_S)
        spoken.append(word)
        if message is None:
            message = state.open_say_bubble(word, reaction_id)
        else:
            message["content"] = " ".join(spoken)
    return "Said it."


async def dance() -> str:
    """Perform a fun little dance. Takes about 10 seconds to complete."""
    await asyncio.sleep(10)
    return "Finished the dance."


def set_emotion(emotion: str) -> str:
    """Show an emotion on your face, e.g. 'happy', 'curious', 'sad', 'excited'."""
    world.update("emotion", emotion)
    return f"Now showing emotion: {emotion}."


def switch_user_tracking(user_id: str | None = None) -> str:
    """Follow one specific person, tracking them from now on. Pass no user (null) to stop
    tracking anyone. You follow at most one person at a time."""
    world.update("tracked_user", user_id)
    return "Stopped tracking." if user_id is None else f"Now tracking user {user_id}."


COMMANDS = [
    dance,
    set_emotion,
    switch_user_tracking,
]

# --- Standup -------------------------------------------------------------------------


@dataclass
class AppHandle:
    """What build_app() returns: the running (or explore-only) system the UI and tests drive.

    `wica` is None in explore-only mode (no key), where `world` is a World-only fallback and
    `config_error` explains what to set; otherwise the Agent is running and `config_error` is None."""

    wica: Wica | None
    world: World
    state: DemoState
    config_error: str | None


def build_app(config: WicaConfig) -> AppHandle:
    """Stand up the whole system through the single entry point (Wica.init); fall back to
    explore-only if the config's api_key_env isn't set, so the app still opens and is explorable
    without credentials. Binds the module globals `world` and `state` the Commands close over, and
    returns the handle the UI (and tests) drive.

    Wica owns the single event loop in a daemon thread and the World is thread-safe, so the Gradio
    side stays fully synchronous: sensor events call world.update(...) directly, and the Agent's
    instrumentation Events (emitted on the loop) reach the presenter's callbacks, which push chat
    events onto its thread-safe queue that the UI's timer drains into the transcript. The demo's
    panels subscribe to the surfaced instrumentation Events (multi-consumer): the prompt panel to
    on_agent_prompt, the input side to on_agent_trigger (what the robot actually observed), the
    assistant side to on_agent_command.

    The api key resolves at Agent build inside Wica.init, so that's what we guard: an unset
    api_key_env raises MissingEnvError there, and we degrade to a World-only system. See
    specs/config.md, specs/wica.md."""
    global world, state
    state = DemoState()
    try:
        wica = Wica.init(config, output_sink=state.output_sink, output_command=say)
    except MissingEnvError as exc:
        # Explore-only: a World-only system (no Agent) on its own loop, so the panel and sensor
        # inputs still work while nothing reasons over them.
        explore_loop = asyncio.new_event_loop()
        threading.Thread(target=explore_loop.run_forever, daemon=True).start()
        world = World(explore_loop)
        world.start()
        register_world()
        config_error = f"environment variable {exc.env_var!r} is not set"
        return AppHandle(wica=None, world=world, state=state, config_error=config_error)

    world = wica.world
    register_world()
    for command in COMMANDS:
        wica.register_command(command)
    wica.on_agent_prompt.subscribe(state.on_prompt)
    wica.on_agent_trigger.subscribe(state.on_trigger)
    wica.on_agent_command.subscribe(state.on_command)
    wica.start()
    return AppHandle(wica=wica, world=world, state=state, config_error=None)


def main() -> None:
    # WICA is a library and configures no logging itself; the application decides what is shown. Keep
    # third-party logs quiet and show wica.* at INFO (raise to logging.DEBUG for the full trace).
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("wica").setLevel(logging.INFO)

    # Import the Gradio UI lazily so importing this module needs only core deps (see module docstring).
    from examples.conversation_demo.app_ui import build_ui

    app = build_app(WicaConfig.from_json_file(CONFIG_PATH))
    try:
        build_ui(app.state, app.world, app.config_error).launch()
    finally:
        if app.wica is not None:
            app.wica.close()


if __name__ == "__main__":
    main()
