"""The demo's conversation session, with no Gradio page (specs/conversation-demo.md, "Session").

A `Conversation` holds at most one running session: the `Wica` built from the app's saved
config, the presenters the page reads (transcript, reaction history, Speaking slot) and the
robot wired to it. Start checks the saved config first (`AppModel.blockers`: its problems, what
the machine is missing) and says why instead of starting; Stop closes the Wica. The config is
locked (`locked()`) from Start until the close has finished, so the Setup panel cannot save a
config out from under a running session.

The session is built by a factory the app passes in (`open_session` in `app.py`), so this module
knows nothing of the robot.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app_setup.app import AppModel

from examples.conversation_demo.speaking import SpeakingSlot
from wica import Wica, WicaConfig
from wica.contrib.gradio import ReactionLog, TranscriptLog

__all__ = ["Conversation", "Running", "Status"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Running:
    """A started session: the Wica and what the page shows of it."""

    config: WicaConfig
    wica: Wica
    transcript: TranscriptLog
    reactions: ReactionLog
    speaking: SpeakingSlot


# What a Conversation is doing: the page greys Start and Stop by it.
Status = Literal["stopped", "starting", "running", "stopping"]


class Conversation:
    """At most one running session, started from the app's saved config. Thread-safe: the
    page's handlers run on Gradio's worker threads, from any browser tab."""

    def __init__(self, open_session: Callable[[WicaConfig], Running]) -> None:
        self._open_session = open_session
        self._lock = threading.Lock()
        self._running: Running | None = None
        self._status: Status = "stopped"
        # Why the last Start did not start (its blockers, or the build's error); cleared by a
        # Start that runs, so every browser tab shows it until then.
        self._reasons: list[str] = []
        # Bumped at every start and stop, so the page redraws its panels for the new session.
        self._version = 0

    @property
    def current(self) -> Running | None:
        return self._running

    @property
    def status(self) -> Status:
        return self._status

    @property
    def reasons(self) -> list[str]:
        """Why the last Start did not start; empty once one ran."""
        return list(self._reasons)

    @property
    def version(self) -> int:
        return self._version

    def locked(self) -> bool:
        """Whether the saved config is in use: from Start until the close has finished."""
        return self._status != "stopped"

    def start(self, model: AppModel) -> list[str]:
        """Start a session from the saved config. Returns why it cannot, in words (its
        blockers, or the error the build raised); empty once it runs. Nothing to do unless
        stopped."""
        with self._lock:
            if self._status != "stopped":
                return []
            saved = model.saved_config()
            blockers = model.blockers(saved)
            if blockers:
                self._reasons = blockers
                return blockers
            self._status = "starting"
        running: Running | None = None
        reasons: list[str] = []
        try:
            config = WicaConfig.from_dict(
                model.used_config(saved)["wica"], base_dir=model.path
            )
            running = self._open_session(config)
        except Exception as exc:  # noqa: BLE001 - said on the page
            _log.warning("the session could not start: %s", exc)
            reasons = [f"The agent could not start: {exc}"]
        finally:
            with self._lock:
                self._reasons = reasons
                if running is not None:
                    self._running = running
                    self._status = "running"
                    self._version += 1
                else:
                    self._status = "stopped"
        return reasons

    def stop(self) -> None:
        """Stop the running session and close its Wica; nothing when none runs."""
        with self._lock:
            running = self._running
            if running is None or self._status != "running":
                return
            self._running = None
            self._status = "stopping"
            self._version += 1
        try:
            running.wica.close()
        finally:
            with self._lock:
                self._status = "stopped"
