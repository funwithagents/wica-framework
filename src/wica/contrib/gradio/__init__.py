"""Reusable Gradio components for WICA's three framework-observability surfaces — the live
World-state table, the prompt history and the conversation transcript — behind the `wica[gradio]`
extra. Each surface is a thread-safe presenter built over a `Wica` (subscribing to its Events in
the constructor) plus a panel function called inside a `gr.Blocks` that creates the components
and the timer keeping them live. See specs/gradio-contrib.md.
"""

from wica.contrib.gradio.display import (
    DisplayEntry,
    EntryDisplay,
    display_entry_or_default,
    format_args,
)
from wica.contrib.gradio.prompts import (
    PromptLog,
    PromptSnapshot,
    prompt_panel,
    render_messages,
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
    "PromptLog",
    "PromptSnapshot",
    "TranscriptLog",
    "TranscriptSnapshot",
    "conversation_panel",
    "display_entry_or_default",
    "format_args",
    "prompt_panel",
    "render_messages",
    "world_html",
    "world_rows",
    "world_state_panel",
]
