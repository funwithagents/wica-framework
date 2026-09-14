"""Demo-specific presenter state: the generic transcript, plus the Speaking panel's slot.

`DemoState` is what the app stands up and the UI and tests read. It composes two things:

  - `transcript` — the **generic** `TranscriptLog` (`transcript.py`), built with the demo's
    `display_entry` hook (defined in `app.py`, next to the entries' `serialize_fn`s); it knows
    nothing about the robot beyond what that hook tells it;
  - `speaking` — the `SpeakingSlot`, the one piece of state that *is* specific to this robot's
    voice: the simulated TTS's word-by-word progress that the Speaking panel renders. `say` (in
    `app.py`) drives it from the agent loop; the UI reads it on its own thread.

Both are the seam between the agent loop and the Gradio thread, so everything they share sits
behind a lock or a thread-safe queue. See specs/conversation-demo.md ("What the user sees" 2.,
"A reusable transcript").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from examples.conversation_demo.transcript import DisplayEntry, TranscriptLog

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


class DemoState:
    """What the app hands the UI and the tests: the generic transcript (with the demo's display
    hook) and the Speaking slot."""

    def __init__(self, display_entry: DisplayEntry | None = None) -> None:
        self.transcript = TranscriptLog(display_entry)
        self.speaking = SpeakingSlot()
