"""The `display_entry` hook: how a World entry shows to a *person*, shared by the transcript and
the prompt history.

This is the one place an application injects persona knowledge into the generic surfaces of
`wica.contrib.gradio`. Every transcript item and prompt-history label that comes from a World
entry is rendered through the hook; returning `None` falls back to the generic default here.
Because a Command's execution *is* a World entry (value: `CommandExecution`), the same hook
customises both sides of the transcript. See specs/gradio-contrib.md ("The `display_entry` hook").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gradio as gr  # noqa: F401 — the contrib requires the extra; fail at import, not later

from wica import CommandExecution, WorldEntry


@dataclass(frozen=True)
class EntryDisplay:
    """How one World entry shows in a surface. `label` is the one-line text with its icon: the
    input-side message for a triggering entry, the item title for a Command execution. `detail`
    is a Command item's body (ignored for the input side)."""

    label: str
    detail: str | None = None


DisplayEntry = Callable[[WorldEntry], "EntryDisplay | None"]


def format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def display_entry_or_default(
    entry: WorldEntry,
    hook: DisplayEntry | None,
    output_command_name: str | None,
) -> EntryDisplay:
    """Render an entry through the application hook, falling back to the generic default:
    `⚡ key = value` for any entry; for a Command execution `🦾 name` (🗣️ when it is the
    configured output Command) with `name(args)` as detail."""
    if hook is not None:
        custom = hook(entry)
        if custom is not None:
            return custom
    value = entry.current.value
    if isinstance(value, CommandExecution):
        icon = "🗣️" if value.name == output_command_name else "🦾"
        return EntryDisplay(
            f"{icon} {value.name}", f"{value.name}({format_args(value.args)})"
        )
    return EntryDisplay(f"⚡ {entry.key} = {value!r}")
