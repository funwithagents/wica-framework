"""Reusable Gradio components for WICA's three framework-observability surfaces — the live
World-state table, the reaction history (prompt + instrumentation) and the conversation
transcript — behind the `wica[gradio]` extra. Each surface is a thread-safe presenter built over a
`Wica` (subscribing to its Events in the constructor) plus a panel function called inside a
`gr.Blocks` that creates the components and the timer keeping them live. See
specs/gradio-contrib.md.
"""

from wica.contrib.gradio.display import (
    DisplayEntry,
    EntryDisplay,
    display_entry_or_default,
    format_args,
)
from wica.contrib.gradio.reactions import (
    ReactionLog,
    ReactionSnapshot,
    reaction_panel,
    render_messages,
    render_reaction_trace,
)
from wica.contrib.gradio.transcript import (
    TranscriptLog,
    TranscriptSnapshot,
    conversation_panel,
)
from wica.contrib.gradio.world_state import world_html, world_rows, world_state_panel

__all__ = [
    "DisplayEntry",
    "EntryDisplay",
    "ReactionLog",
    "ReactionSnapshot",
    "TranscriptLog",
    "TranscriptSnapshot",
    "conversation_panel",
    "display_entry_or_default",
    "format_args",
    "reaction_panel",
    "render_messages",
    "render_reaction_trace",
    "world_html",
    "world_rows",
    "world_state_panel",
]
