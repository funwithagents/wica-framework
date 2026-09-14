"""The live World-state panel: every prompt entry of a `World` as a self-refreshing table.

The table is rendered as HTML rather than a `gr.Dataframe`. A Dataframe fed from a timer diffs its
rows on the frontend, and a *row appearing then disappearing* — exactly what a short-lived
Command entry does (e.g. a `say` that is only "running" for the second it takes to speak, then is
retired) — was rendered unreliably, so those transient rows often never showed. `gr.HTML`
re-renders its whole value each tick, so whatever `world_rows()` captures is what's displayed,
with no row-diffing to drop a fleeting entry. Command rows are highlighted so they stand out during
their brief life. See specs/gradio-contrib.md ("Component 1").
"""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from wica import CommandExecution, World, WorldEntry
from wica.contrib.gradio.display import format_args

_WORLD_TABLE_HEADERS = ["Key", "Value", "Updated"]

# Fixed column widths (percent, summing to 100) so the table never reflows when a long value
# lands. Paired with `table-layout:fixed` + `word-break` below, a wide cell (e.g. a command with
# a long name/args) wraps within its column instead of stretching it and shoving the others around.
_WORLD_COL_WIDTHS = [34, 44, 22]


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, CommandExecution):
        return f"{value.name}({format_args(value.args)}) [{value.state}]"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "[]"
    return str(value)


def _rows(world: World) -> list[tuple[list[str], bool]]:
    """One `([key, value, updated], is_command)` per prompt entry — a snapshot taken under the
    World lock. A Command row is recognised by its value type, not by its key."""
    rows: list[tuple[list[str], bool]] = []
    entries: list[WorldEntry] = world.get_prompt_entries()
    for entry in entries:
        version = entry.current
        stamp = version.timestamp.astimezone().strftime("%H:%M:%S")
        row = [entry.key, _format_value(version.value), stamp]
        rows.append((row, isinstance(version.value, CommandExecution)))
    return rows


def world_rows(world: World) -> list[list[str]]:
    """`[key, value, updated]` per prompt entry, values formatted for a person: a running Command
    as `name(args) [state]`, `None` as `—`, a list as `[a, b]`."""
    return [row for row, _ in _rows(world)]


def world_html(world: World) -> str:
    """The whole table as self-styled HTML (Gradio theme variables), Command rows marked
    `class="cmd"`. Every cell is escaped."""
    cols = "".join(f'<col style="width:{w}%">' for w in _WORLD_COL_WIDTHS)
    header_cells = "".join(f"<th>{html.escape(h)}</th>" for h in _WORLD_TABLE_HEADERS)
    body_rows: list[str] = []
    for row, is_command in _rows(world):
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        cls = ' class="cmd"' if is_command else ""
        body_rows.append(f"<tr{cls}>{cells}</tr>")
    return (
        "<div class='world-table'><style>"
        ".world-table table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px}"
        ".world-table th,.world-table td{border:1px solid var(--border-color-primary,#ccc);"
        "padding:4px 8px;text-align:left;vertical-align:top;"
        "overflow-wrap:anywhere;word-break:break-word}"
        ".world-table th{font-weight:600}"
        ".world-table tr.cmd td{background:var(--color-accent-soft,#fff4e5)}"
        f"</style><table><colgroup>{cols}</colgroup><thead><tr>"
        f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def world_state_panel(world: World, *, refresh_s: float = 0.2) -> gr.HTML:
    """Create the live World table and the timer that refreshes it. Call inside a `gr.Blocks`
    context; the panel owns no framework state, it re-reads `world` every tick."""
    view = gr.HTML(value=world_html(world))
    timer = gr.Timer(refresh_s)
    timer.tick(lambda: world_html(world), outputs=view)
    return view
