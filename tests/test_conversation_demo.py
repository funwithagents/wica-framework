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
import os
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

from examples.conversation_demo.app_ui import (  # noqa: E402
    button_states,
    follow_blockers,
    status_text,
)
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
    # The app is bare: the folder's config.json is the WICA document itself, wrapped for the
    # panels only.
    written = json.loads((tmp_path / "demo" / "config.json").read_text())
    assert written == committed  # written into the empty folder
    assert model.saved_config() == {"wica": committed}
    assert committed == {
        "agent": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "system_prompt_file": "wica.md",
        }
    }
    assert wica_setup.check_config(committed) == []
    assert app.APP.autostart is True and app.APP.bare is True


def test_a_save_writes_the_wica_document_alone_to_the_folder(tmp_path: Path):
    model = _model(tmp_path, Conversation(app.open_session))
    _fake(model)

    written = json.loads((tmp_path / "demo" / "config.json").read_text())
    assert written["agent"]["provider"] == "fake" and "wica" not in written
    assert WicaConfig.from_json_file(tmp_path / "demo" / "config.json").agent.model == (
        "scripted"
    )


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
    monkeypatch.chdir(tmp_path)  # anywhere: the folder is the setup checks' base_dir

    reasons = conversation.start(model)

    assert reasons and all("ANTHROPIC_API_KEY" in reason for reason in reasons)
    assert conversation.error is None and conversation.status == "stopped"
    assert not conversation.locked()
    assert "ANTHROPIC_API_KEY" in status_text(conversation, reasons)
    assert button_states(conversation) == (True, False)
    assert not any("wica.md" in reason for reason in reasons)  # the persona is found


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
    # Kept while the blockers come and go (it is not one); the next Start clears it.
    blockers, _ = follow_blockers(model, conversation, [], None)
    assert blockers == [] and "unplugged" in status_text(conversation, blockers)
    model.save({"wica": {"agent": {"provider": "fake", "model": "m"}}})
    blockers, _ = follow_blockers(model, conversation, [], None, force=True)
    assert blockers and "unplugged" in status_text(conversation, blockers)
    assert conversation.start(model) == blockers
    assert conversation.error is None
    assert "unplugged" not in status_text(conversation, blockers)


def test_a_key_saved_in_the_store_clears_the_blockers_without_a_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "restored after the test")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    conversation = Conversation(lambda config: _never())
    model = _model(tmp_path, conversation)
    app.ensure_persona(model.path)
    assert conversation.start(model)  # the key is missing: not started

    blockers, seen = follow_blockers(model, conversation, [], None)
    assert any("ANTHROPIC_API_KEY" in b for b in blockers)
    assert "Not started" in status_text(conversation, blockers)

    model.secrets.set(
        "ANTHROPIC_API_KEY", "sk-test"
    )  # the Secrets panel, or a line's Save
    blockers, seen = follow_blockers(model, conversation, blockers, seen)

    assert blockers == []
    assert "ready to start" in status_text(conversation, blockers)


def test_the_blockers_are_not_checked_again_on_a_tick_with_no_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    conversation = Conversation(lambda config: _never())
    model = _model(tmp_path, conversation)
    blockers, seen = follow_blockers(model, conversation, [], None)
    calls: list[object] = []
    monkeypatch.setattr(model, "blockers", lambda *a, **k: calls.append(a) or ["x"])

    assert follow_blockers(model, conversation, blockers, seen) == (blockers, seen)
    assert calls == []
    # Selecting the tab, or a save or reset in the Setup panel, forces it.
    assert follow_blockers(model, conversation, blockers, seen, force=True)[0] == ["x"]


def test_a_saved_config_naming_a_set_variable_clears_the_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MY_ANTHROPIC_KEY", "sk-test")
    conversation = Conversation(lambda config: _never())
    model = _model(tmp_path, conversation)
    app.ensure_persona(model.path)
    blockers, seen = follow_blockers(model, conversation, [], None)
    assert blockers

    saved = model.saved_config()
    saved["wica"]["agent"]["api_key_env"] = "MY_ANTHROPIC_KEY"
    model.save(saved)  # the Setup panel's Save: `saved.change` forces a check
    blockers, _ = follow_blockers(model, conversation, blockers, seen, force=True)

    assert blockers == []
    assert "ready to start" in status_text(conversation, blockers)


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
    # Run from elsewhere: the setup checks and the session both read the folder's.
    monkeypatch.chdir(tmp_path)

    assert conversation.start(model) == []
    conversation.stop()
    assert seen[0].agent.system_prompt_file == str((model.path / "wica.md").resolve())


def test_launching_leaves_the_working_directory_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    launched: list[str] = []

    class Page:
        def launch(self, **kwargs: object) -> None:
            launched.append(os.getcwd())

    monkeypatch.setattr(app, "build_page", lambda *a, **k: Page())
    monkeypatch.setattr(app.logging, "basicConfig", lambda **k: None)

    app.main(["--folder", str(tmp_path / "demo"), "--no-start"])

    assert len(launched) == 1 and os.path.samefile(launched[0], elsewhere)
    assert (tmp_path / "demo" / "wica.md").is_file()


@pytest.mark.parametrize(
    ("argv", "start"), [([], None), (["--start"], True), (["--no-start"], False)]
)
def test_the_command_line_overrides_the_saved_launch_choice(
    argv: list[str], start: bool | None, tmp_path: Path
):
    parser = app._parser()  # pyright: ignore[reportPrivateUsage]
    options = parser.parse_args([*argv, "--folder", str(tmp_path)])
    assert (options.start, options.folder, options.config, options.host) == (
        start,
        tmp_path,
        None,
        "127.0.0.1",
    )


def test_folder_and_config_are_one_or_the_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    parser = app._parser()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(SystemExit):
        parser.parse_args(["--folder", str(tmp_path), "--config", "x.json"])
    assert "not allowed with" in capsys.readouterr().err


# --- Running from a config file, and the config app ---------------------------------

from app_setup.autostart import starts_at_launch  # noqa: E402
from app_setup.config_file import ConfigFile  # noqa: E402
from app_setup.launch import open_source  # noqa: E402

from examples.config_app import __main__ as config_app  # noqa: E402

# A variable no shell sets, so the tests never read nor write a real key.
_TEST_KEY = "WICA_DEMO_TEST_ONLY_KEY"


def _config_file(folder: Path, agent: dict[str, object]) -> Path:
    """A WICA document in *folder*, its persona next to it (`persona.md`, relative)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "persona.md").write_text("You are a test robot.", encoding="utf-8")
    path = folder / "robot.json"
    path.write_text(
        json.dumps({"agent": {**agent, "system_prompt_file": "persona.md"}}, indent=2),
        encoding="utf-8",
    )
    return path


class _Launched:
    """`build_page` stood in for: what `main` hands the page, seen while it is served."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, model: AppModel, conversation: Conversation, open_on: str
    ) -> object:
        launched = self

        class Page:
            def launch(self, **kwargs: object) -> None:
                running = conversation.current
                launched.calls.append(
                    {
                        "model": model,
                        "open_on": open_on,
                        "status": conversation.status,
                        "config": None if running is None else running.config,
                    }
                )

        return Page()


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> _Launched:
    page = _Launched()
    monkeypatch.setattr(app, "build_page", page)
    monkeypatch.setattr(app.logging, "basicConfig", lambda **k: None)
    return page


def test_the_demo_runs_a_config_file_as_it_is_and_starts_at_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: _Launched
):
    path = _config_file(
        tmp_path / "configs",
        {
            "provider": "fake",
            "model": "scripted",
            "model_kwargs": {"default": {"text": "<say>Hi.</say>"}, "delay_s": 0},
        },
    )
    before = path.read_bytes()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the persona is found against the file, not here

    app.main(["--config", str(path)])

    [call] = launched.calls
    model = call["model"]
    assert isinstance(model, AppModel) and isinstance(model.source, ConfigFile)
    # Started at launch, from the file's own config and persona, and the page opens on it.
    assert (call["status"], call["open_on"]) == ("running", "conversation")
    config = call["config"]
    assert isinstance(config, WicaConfig)
    assert config.agent.provider == "fake"
    assert config.agent.system_prompt_file == str(
        (path.parent / "persona.md").resolve()
    )
    # Read-only: never written, nothing copied next to it, no folder made.
    assert not model.editable
    with pytest.raises(RuntimeError):
        model.save(model.saved_config())
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["persona.md", "robot.json"]


def test_from_a_file_no_start_leaves_the_session_stopped(
    tmp_path: Path, launched: _Launched
):
    path = _config_file(tmp_path, {"provider": "fake", "model": "scripted"})

    app.main(["--config", str(path), "--no-start"])

    [call] = launched.calls
    assert (call["status"], call["open_on"]) == ("stopped", "setup")


def test_from_a_file_the_keys_come_from_the_env_next_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: _Launched
):
    monkeypatch.setenv(_TEST_KEY, "restored after the test")
    monkeypatch.delenv(_TEST_KEY)
    path = _config_file(
        tmp_path,
        {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "api_key_env": _TEST_KEY,
        },
    )
    (path.parent / ".env").write_text(f"{_TEST_KEY}=sk-test\n", encoding="utf-8")

    app.main(["--config", str(path), "--no-start"])

    model = launched.calls[0]["model"]
    assert isinstance(model, AppModel)
    assert model.source.env_path == path.parent / ".env"
    assert model.blockers(model.saved_config()) == []


def test_a_missing_config_file_stops_the_demo_at_launch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], launched: _Launched
):
    missing = tmp_path / "nowhere.json"

    with pytest.raises(SystemExit) as exited:
        app.main(["--config", str(missing)])

    assert exited.value.code == 2
    assert str(missing) in capsys.readouterr().err
    assert launched.calls == [] and not missing.exists()


def test_every_example_config_runs_as_a_config_file(tmp_path: Path):
    options = app._parser().parse_args([])  # pyright: ignore[reportPrivateUsage]
    examples = sorted((_ROOT / "examples" / "configs").glob("*.json"))
    assert examples
    for path in examples:
        options.config = path
        model = AppModel(app.APP, open_source(options, app.APP))
        saved = model.saved_config()
        assert saved == {"wica": json.loads(path.read_text())}, path.name
        assert model.check_config(saved) == [], path.name
        # The persona, relative to the file, is found against its directory.
        prompt = [
            item
            for item in model.check_setup(saved)
            if item.id == "file:system_prompt_file"
        ]
        assert [item.status for item in prompt] == ["ready"], path.name
        assert starts_at_launch(model, None) and not model.editable


def _config_app_model(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> AppModel:
    """The config app's `main`, its page stood in for: the model it would serve."""
    models: list[AppModel] = []

    class Page:
        def launch(self, **kwargs: object) -> None:
            pass

    def page(model: AppModel) -> Page:
        models.append(model)
        return Page()

    monkeypatch.setattr(config_app, "config_page", page)
    config_app.main(argv)
    [model] = models
    return model


def test_the_config_app_saves_a_wica_document_to_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _config_file(tmp_path, {"provider": "fake", "model": "scripted"})

    model = _config_app_model(monkeypatch, [str(path)])
    saved = model.saved_config()
    saved["wica"]["agent"]["model"] = "another"
    model.save(saved)

    written = json.loads(path.read_text())
    assert written == {
        "agent": {
            "provider": "fake",
            "model": "another",
            "system_prompt_file": "persona.md",
        }
    }
    assert WicaConfig.from_json_file(path).agent.model == "another"
    assert model.source.env_path == tmp_path / ".env"


def test_the_config_app_takes_another_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _config_file(tmp_path / "configs", {"provider": "fake", "model": "m"})
    env = tmp_path / "keys.env"

    model = _config_app_model(monkeypatch, [str(path), "--env", str(env)])

    assert model.editable and model.source.env_path == env


def test_the_config_app_refuses_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    missing = tmp_path / "nowhere.json"
    with pytest.raises(SystemExit) as exited:
        _config_app_model(monkeypatch, [str(missing)])
    assert exited.value.code == 2
    assert str(missing) in capsys.readouterr().err
    assert not missing.exists()
