"""Plain helpers shared by the fast tier's presenter tests (fixtures live in conftest.py)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from wica import CommandExecution, WorldEntry, WorldEntryVersion


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
