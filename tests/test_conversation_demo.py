"""Fast, deterministic unit tests for the conversation demo's own pieces.

These pin what the demo contributes on top of the `wica.contrib.gradio` components: its
`display_entry` hook (`app.py`), the Speaking panel's model (`speaking.py`) and the robot's `say`
output Command driving it from the words the model writes — without any network or a browser. The generic transcript, prompt history and
World table are the contrib's and are tested in `tests/contrib/`; the full end-to-end wiring
(World entries + Commands + presenters, driven by a scripted fake model) is covered separately in
tests-e2e/test_example_flow.py.

See specs/conversation-demo.md ("Reused from the Gradio contrib").
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from examples.conversation_demo import app
from examples.conversation_demo.app import Robot, display_entry
from examples.conversation_demo.speaking import SpeakingSlot
from tests.support import command_entry, make_entry
from wica import OutputStream
from wica.contrib.gradio import EntryDisplay
from wica.text_output import output_signature

# --- the demo's display_entry hook ------------------------------------------------------


def test_demo_display_entry_labels_the_robot_entries():
    assert display_entry(make_entry("speech_input", "hi")) == EntryDisplay('🗣️ "hi"')
    assert display_entry(make_entry("closest_user", "alice")) == EntryDisplay(
        "👤 Closest user detected: alice"
    )
    assert display_entry(make_entry("closest_user", None)) == EntryDisplay(
        "👤 Closest user gone"
    )
    # The voice: channel title, spoken text as body.
    assert display_entry(
        command_entry("c1", "say", {"text": "hi there"}, "running")
    ) == EntryDisplay("🗣️ say", "hi there")
    # Anything else is left to the generic default.
    assert display_entry(make_entry("emotion", "happy")) is None
    assert display_entry(command_entry("c2", "dance", {}, "running")) is None


# --- SpeakingSlot and the say Command -------------------------------------------------


def test_speaking_slot_tracks_words_and_ending():
    slot = SpeakingSlot()
    assert slot.read() is None

    token = slot.start("hello big world")
    slot.advance(token)
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("hello", "big", "world")
    assert speaking.spoken == 1
    assert speaking.state == "speaking"
    assert speaking.text == "hello big world"

    slot.advance(token)
    slot.cancelled(token)
    speaking = slot.read()
    assert speaking is not None
    assert (speaking.spoken, speaking.state) == (2, "cancelled")


def test_speaking_slot_ignores_a_stale_utterance():
    """An older `say` ending after a newer one started must not stamp the newer one."""
    slot = SpeakingSlot()
    old = slot.start("one two")
    new = slot.start("three four")
    slot.advance(old)
    slot.extend(old, "one two and more")
    slot.complete(old)
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("three", "four")
    assert (speaking.spoken, speaking.state) == (0, "speaking")
    slot.complete(new)
    assert slot.read().state == "complete"  # type: ignore[union-attr]


def test_speaking_slot_grows_as_the_words_arrive():
    """The model is still writing the sentence: the words to come grow, the spoken count and
    the state are kept."""
    slot = SpeakingSlot()
    token = slot.start("hello big")
    slot.advance(token)
    slot.extend(token, "hello big wide world")
    speaking = slot.read()
    assert speaking is not None
    assert speaking.words == ("hello", "big", "wide", "world")
    assert (speaking.spoken, speaking.state) == (1, "speaking")


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch, unstarted_wica) -> Robot:
    """The robot over an unstarted Wica's World and a fresh Speaking slot — `say` touches only the
    slot, so nothing needs to run."""
    monkeypatch.setattr(app, "_SAY_WORD_DELAY_S", 0.01)
    return Robot(unstarted_wica(None).world, SpeakingSlot())


def stream_of(*pieces: str, close: bool = True) -> OutputStream:
    """An output's text as the framework hands it to `say`: pieces pushed, then the end."""
    stream = OutputStream()
    for piece in pieces:
        stream.push(piece)
    if close:
        stream.close()
    return stream


def test_say_is_a_streaming_output_command_without_attributes(robot: Robot):
    """`say` takes the words as they are written — what makes the robot start speaking before
    the response is complete — and declares no tag attribute."""
    signature = output_signature(robot.say)
    assert signature.streams
    assert signature.attributes == {}


def test_say_drives_the_speaking_slot_to_complete(robot: Robot):
    async def run() -> None:
        # Pieces cut mid-word, as a model's chunks are.
        return await robot.say(stream_of("hi th", "ere fri", "end"))

    assert asyncio.run(run()) is None
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.text == "hi there friend"
    assert (speaking.spoken, speaking.state) == (3, "complete")


def test_say_starts_on_the_first_words_and_waits_for_the_rest(robot: Robot):
    """Streaming: the robot speaks the words it has while the model is still writing — never a
    word that may still be growing — then waits for more and finishes once the text ends."""

    async def run() -> None:
        stream = stream_of("one two thr", close=False)
        task = asyncio.create_task(robot.say(stream))
        await asyncio.sleep(0.1)  # far longer than the three words would take
        speaking = robot.speaking.read()
        assert speaking is not None
        # "thr" is not a whole word yet: two words spoken, the sentence still in progress.
        assert (speaking.spoken, speaking.state) == (2, "speaking")
        assert not task.done()

        stream.push("ee four")
        stream.close()
        assert await task is None

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.words == ("one", "two", "three", "four")
    assert (speaking.spoken, speaking.state) == (4, "complete")


def test_say_with_no_words_leaves_the_panel_alone(robot: Robot):
    token = robot.speaking.start("before")
    robot.speaking.complete(token)

    async def run() -> None:
        return await robot.say(stream_of("  "))

    assert asyncio.run(run()) is None
    speaking = robot.speaking.read()
    assert speaking is not None and speaking.text == "before"


def test_say_marks_the_slot_cancelled_and_reports_what_was_heard(robot: Robot):
    """Barge-in: cancelling the say task mid-sentence leaves the words spoken so far, marks the
    utterance cancelled and sets `heard` to exactly those words (what the robot's history
    keeps) — and the cancellation still propagates to the caller."""
    stream = stream_of("one two three four five six")

    async def run() -> None:
        task = asyncio.create_task(robot.say(stream))
        await asyncio.sleep(0.025)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert speaking.state == "cancelled"
    assert 0 < speaking.spoken < 6
    assert stream.heard == " ".join(speaking.words[: speaking.spoken])


def test_say_cut_off_while_waiting_for_more_words_reports_what_was_heard(robot: Robot):
    stream = stream_of("one two ", close=False)

    async def run() -> None:
        task = asyncio.create_task(robot.say(stream))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    speaking = robot.speaking.read()
    assert speaking is not None
    assert (speaking.spoken, speaking.state) == (2, "cancelled")
    assert stream.heard == "one two"


def test_persona_describes_the_say_tag_and_no_removed_mechanism():
    """The persona under test (the benchmark's too) must teach the tag, not a `say` call, and
    must not tell the model to call `noop`."""
    persona = (Path(app.__file__).parent / "prompts" / "wica.md").read_text()
    assert "<say>" in persona and "</say>" in persona
    assert "say(" not in persona
    assert "noop" not in persona


# --- The app and its session over app-setup's app model (session.py) ---------------

import json  # noqa: E402

from app_setup.app import AppModel  # noqa: E402
from app_setup.folder import ConfigFolder  # noqa: E402

from examples.conversation_demo.app_ui import button_states, status_text  # noqa: E402
from examples.conversation_demo.session import Conversation, Running  # noqa: E402
from wica import setup as wica_setup  # noqa: E402
from wica.config import WicaConfig  # noqa: E402

_ROOT = Path(app.__file__).resolve().parents[2]


def _model(tmp_path: Path, conversation: Conversation) -> AppModel:
    return AppModel(
        app.APP,
        ConfigFolder(tmp_path / "demo", app.APP.default_config),
        locked=conversation.locked,
    )


def _fake(model: AppModel) -> None:
    """Save the scripted fake (the example config) as the app's config."""
    block = json.loads((_ROOT / "examples" / "configs" / "fake.json").read_text())
    block["agent"]["system_prompt_file"] = str(app.PERSONA_FILE)
    block["agent"]["model_kwargs"]["delay_s"] = 0
    model.save({"wica": block})


def test_the_committed_default_config_is_the_app_s_default(tmp_path: Path):
    committed = json.loads(
        (Path(app.__file__).parent / "default_config.json").read_text()
    )
    model = _model(tmp_path, Conversation(app.open_session))

    assert app.APP.default_config == committed
    assert model.saved_config() == committed  # written into the empty folder
    assert committed == {
        "wica": {
            "agent": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "system_prompt_file": "wica.md",
            }
        }
    }
    assert wica_setup.check_config(committed["wica"]) == []
    assert app.APP.autostart is True


def test_the_persona_is_copied_into_the_folder_once_and_never_overwritten(
    tmp_path: Path,
):
    persona = app.ensure_persona(tmp_path)

    assert persona == tmp_path / "wica.md"
    assert persona.read_text() == app.PERSONA_FILE.read_text()
    persona.write_text("You are a grumpy robot.")
    app.ensure_persona(tmp_path)
    assert persona.read_text() == "You are a grumpy robot."


def test_with_no_key_the_default_config_says_so_instead_of_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conversation = Conversation(lambda config: _never())
    model = _model(tmp_path, conversation)
    app.ensure_persona(model.path)
    monkeypatch.chdir(model.path)  # as main() does

    reasons = conversation.start(model)

    assert reasons and all("ANTHROPIC_API_KEY" in reason for reason in reasons)
    assert conversation.reasons == reasons and conversation.status == "stopped"
    assert not conversation.locked()
    assert "ANTHROPIC_API_KEY" in status_text(conversation)
    assert button_states(conversation) == (True, False)


def test_start_runs_the_saved_config_and_locks_it_until_stopped(tmp_path: Path):
    conversation = Conversation(app.open_session)
    model = _model(tmp_path, conversation)
    _fake(model)

    assert conversation.start(model) == []
    running = conversation.current
    assert running is not None and running.wica.is_running
    assert running.config.agent.provider == "fake"
    assert conversation.status == "running"
    assert conversation.locked() and model.locked()
    assert button_states(conversation) == (False, True)
    with pytest.raises(RuntimeError):
        model.save(model.saved_config())
    version = conversation.version
    assert conversation.start(model) == []  # already running: nothing more
    assert conversation.current is running and conversation.version == version
    assert "Running" in status_text(conversation)

    conversation.stop()

    assert conversation.current is None and not conversation.locked()
    assert conversation.status == "stopped"
    assert button_states(conversation) == (True, False)
    assert not running.wica.is_running
    assert conversation.version == version + 1
    model.save(model.saved_config())  # unlocked again


def test_both_buttons_are_greyed_while_starting_and_stopping(tmp_path: Path):
    seen: list[tuple[str, tuple[bool, bool], str]] = []
    conversation: Conversation

    def observed(config: WicaConfig) -> Running:
        seen.append(
            (
                conversation.status,
                button_states(conversation),
                status_text(conversation),
            )
        )
        return app.open_session(config)

    conversation = Conversation(observed)
    model = _model(tmp_path, conversation)
    _fake(model)
    assert conversation.start(model) == []
    assert seen == [("starting", (False, False), "⏳ **Starting…**")]

    running = conversation.current
    assert running is not None
    closing: list[tuple[str, tuple[bool, bool]]] = []
    real_close = running.wica.close

    def close() -> None:
        closing.append((conversation.status, button_states(conversation)))
        real_close()

    running.wica.close = close  # type: ignore[method-assign]
    conversation.stop()
    assert closing == [("stopping", (False, False))]


def _never() -> Running:
    raise AssertionError("nothing should be built")


def test_a_failed_build_is_said_and_leaves_the_config_unlocked(tmp_path: Path):
    def failing(config: WicaConfig) -> Running:
        raise RuntimeError("the robot is unplugged")

    conversation = Conversation(failing)
    model = _model(tmp_path, conversation)
    _fake(model)

    assert conversation.start(model) == [
        "The agent could not start: the robot is unplugged"
    ]
    assert conversation.current is None and not conversation.locked()
    assert "unplugged" in status_text(conversation)


def test_a_relative_prompt_file_is_read_from_the_app_s_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: list[WicaConfig] = []

    def recording(config: WicaConfig) -> Running:
        seen.append(config)
        return app.open_session(config)

    conversation = Conversation(recording)
    model = _model(tmp_path, conversation)
    _fake(model)
    app.ensure_persona(model.path)
    agent = dict(model.saved_config()["wica"]["agent"])
    agent["system_prompt_file"] = "wica.md"
    model.save({"wica": {"agent": agent}})
    # The setup checks look a relative path up against the working directory (specs/setup.md,
    # Open questions), which main() makes the folder: from elsewhere, they block it.
    monkeypatch.chdir(tmp_path)
    assert any("wica.md" in reason for reason in conversation.start(model))
    monkeypatch.chdir(model.path)

    assert conversation.start(model) == []
    conversation.stop()
    assert seen[0].agent.system_prompt_file == str((model.path / "wica.md").resolve())


@pytest.mark.parametrize(
    ("argv", "start"), [([], None), (["--start"], True), (["--no-start"], False)]
)
def test_the_command_line_overrides_the_saved_launch_choice(
    argv: list[str], start: bool | None, tmp_path: Path
):
    options = app._arguments([*argv, "--folder", str(tmp_path)])  # pyright: ignore[reportPrivateUsage]
    assert (options.start, options.folder, options.host) == (
        start,
        tmp_path,
        "127.0.0.1",
    )
