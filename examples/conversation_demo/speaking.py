"""The Speaking panel's model: the simulated TTS's word-by-word progress.

`SpeakingSlot` is the one piece of presenter state that *is* specific to this robot's voice. The
UI owns it (it is created in `build_ui`, next to the transcript) and hands it to the robot at
wiring time, so `say` (in `app.py`) drives it from the agent loop while the UI reads it on its
own thread — hence the lock. Gradio-free, so the default test tier can exercise it. See
specs/conversation-demo.md ("What the user sees" 2., "Composition").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

SpeakingState = Literal["speaking", "complete", "cancelled"]


@dataclass(frozen=True)
class Speaking:
    """A read of the current (or last) utterance: its words, how many have been "spoken" so far,
    and whether it is still in progress, finished, or was cut off."""

    words: tuple[str, ...]
    spoken: int
    state: SpeakingState

    @property
    def text(self) -> str:
        return " ".join(self.words)


class SpeakingSlot:
    """The Speaking panel's state: the utterance the simulated TTS is currently producing.

    `start(text)` opens a new utterance and returns a token; `advance`/`complete`/`cancelled`
    take that token and are ignored if a newer utterance has started since — so an overlapping
    `say` (the model can chain one while another still speaks) never has the older one's ending
    stamped onto the newer one. The panel keeps showing the last utterance, complete or
    cancelled, until the next one starts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = 0
        self._current: Speaking | None = None

    def start(self, text: str) -> int:
        with self._lock:
            self._token += 1
            self._current = Speaking(tuple(text.split()), 0, "speaking")
            return self._token

    def advance(self, token: int) -> None:
        """One more word spoken."""
        self._set(token, lambda s: Speaking(s.words, s.spoken + 1, s.state))

    def complete(self, token: int) -> None:
        self._set(token, lambda s: Speaking(s.words, s.spoken, "complete"))

    def cancelled(self, token: int) -> None:
        self._set(token, lambda s: Speaking(s.words, s.spoken, "cancelled"))

    def _set(self, token: int, change) -> None:  # noqa: ANN001 — a tiny local updater
        with self._lock:
            if token == self._token and self._current is not None:
                self._current = change(self._current)

    def read(self) -> Speaking | None:
        """The current/last utterance, or None if the robot hasn't spoken yet. UI thread."""
        with self._lock:
            return self._current
