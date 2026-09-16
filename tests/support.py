"""Plain helpers shared by the fast tier's presenter tests (fixtures live in conftest.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from wica import CommandExecution, ReactionTrace, WorldEntry, WorldEntryVersion


def make_entry(key: str, value: Any) -> WorldEntry:
    """A minimal WorldEntry snapshot as a presenter's callbacks receive it."""
    version = WorldEntryVersion(id=1, value=value, timestamp=datetime.now(timezone.utc))
    return WorldEntry(
        key=key,
        type=type(value) if value is not None else str,
        bypass_coalescing=False,
        current=version,
        previous=None,
    )


def command_entry(
    call_id: str, name: str, args: dict[str, Any], state: str, **kw: Any
) -> WorldEntry:
    """The agent:command:<call_id> entry snapshot a listener receives at a given state."""
    execution = CommandExecution(name=name, args=args, state=state, **kw)  # type: ignore[arg-type]
    return make_entry(f"agent:command:{call_id}", execution)


def titles(conversation: list[dict[str, Any]]) -> list[str | None]:
    return [m.get("metadata", {}).get("title") for m in conversation]


def item_titled(conversation: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        m
        for m in conversation
        if str(m.get("metadata", {}).get("title", "")).startswith(prefix)
    ]
    assert len(matches) == 1, titles(conversation)
    return matches[0]


def make_reaction_trace(**overrides: Any) -> ReactionTrace:
    """A hand-built ReactionTrace with sensible defaults (an ordinary, fast, textful reaction), any
    field overridable by keyword — shared by the reaction-history and (future) other contrib tests
    that need a fully-formed trace rather than one driven through a live Agent."""
    t0 = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
    defaults: dict[str, Any] = {
        "reaction_id": 1,
        "triggers": (),
        "window_opened_at": t0,
        "window_closed_at": t0,
        "prompt_ready_at": t0 + timedelta(seconds=0.05),
        "model_started_at": t0 + timedelta(seconds=0.05),
        "model_ended_at": t0 + timedelta(seconds=0.5),
        "outcome": "ok",
        "error": None,
        "text_length": 5,
        "sink_duration": 0.01,
        "command_call_ids": (),
        "noop": False,
        "usage": None,
        "ended_at": t0 + timedelta(seconds=0.6),
        "trace_id": None,
        "span_id": None,
    }
    return ReactionTrace(**{**defaults, **overrides})
