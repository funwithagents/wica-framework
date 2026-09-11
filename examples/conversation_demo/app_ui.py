"""The Gradio UI for the WICA conversation demo.

`build_ui(state, world, config_error)` assembles the four surfaces (Conversation / World state /
Prompt / Sensor inputs) and wires their handlers and refresh timers. It reads live World state
directly (via `world`) and the transcript/prompt history through the `DemoState` presenter — it
never owns framework state of its own beyond a little tick-local bookkeeping. The app file
(`app.py`) stands the system up and calls this.
"""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from wica import World, WorldEntry
from wica.agent import CommandExecution

from examples.conversation_demo.app_state import DemoState

_NO_PROMPT_YET = "(no prompt sent to the model yet)"

# --- World state table ---------------------------------------------------------------
#
# The World state is rendered as an HTML table rather than a gr.Dataframe. A Dataframe fed from a
# timer diffs its rows on the frontend, and a *row appearing then disappearing* — exactly what a
# short-lived command entry does (e.g. `say`, which is only "running" for the second or so it takes
# to speak, then is retired) — was rendered unreliably, so those transient command rows often never
# showed. gr.HTML re-renders its whole value each tick, so whatever _world_rows() captures is what's
# displayed, with no row-diffing to drop a fleeting entry. Command rows are highlighted so they
# stand out during their brief life. (Long actions like `dance` linger for their whole duration.)
_WORLD_TABLE_HEADERS = ["Key", "Value", "Updated"]

# Fixed column widths (percent, summing to 100) so the table never reflows when a long value
# lands. Paired with `table-layout:fixed` + `word-break` below, a wide cell (e.g. a command with
# a long name/args) wraps within its column instead of stretching it and shoving the others around.
_WORLD_COL_WIDTHS = [34, 44, 22]


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, CommandExecution):
        args = ", ".join(f"{k}={v!r}" for k, v in value.args.items())
        return f"{value.name}({args}) [{value.state}]"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "[]"
    return str(value)


def _world_rows(world: World) -> list[list[str]]:
    rows: list[list[str]] = []
    entries: list[WorldEntry] = world.get_prompt_entries()
    for entry in entries:
        version = entry.current
        stamp = version.timestamp.astimezone().strftime("%H:%M:%S")
        rows.append([entry.key, _format_value(version.value), stamp])
    return rows


def _world_html(world: World) -> str:
    cols = "".join(f'<col style="width:{w}%">' for w in _WORLD_COL_WIDTHS)
    header_cells = "".join(f"<th>{html.escape(h)}</th>" for h in _WORLD_TABLE_HEADERS)
    body_rows: list[str] = []
    for row in _world_rows(world):
        is_command = row[0].startswith("agent:command:")
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


# --- Layout --------------------------------------------------------------------------


def build_ui(state: DemoState, world: World, config_error: str | None) -> gr.Blocks:
    """Assemble the demo UI. Input handlers only touch the World; the transcript is fed by the
    agent's callbacks via `state` (a trigger shows up on the right once it actually starts a step —
    a trigger dropped by the busy single-in-flight loop simply won't appear, and no reply follows
    it)."""

    # Tick-local UI bookkeeping (not framework state): how many prompts the view has shown, and the
    # signature of the transcript last pushed to the chatbot. Held as closure state so tick can
    # update them across timer fires without module globals.
    last_shown_count = 0
    # The conversation timer fires every 0.2s, but re-sending an unchanged transcript re-triggers
    # gr.Chatbot's autoscroll and keeps yanking the view to the bottom while the user tries to scroll
    # up. So tick pushes a new chatbot value only when this signature changes (new/edited messages,
    # incl. a `say` bubble growing word by word) and sends a no-op update otherwise — autoscroll then
    # follows genuinely new content but leaves reading alone.
    last_conv_sig: tuple[Any, ...] | None = None

    def on_send(user_text: str) -> str:
        text = user_text.strip()
        if text:
            world.update("speech_input", text)
        return ""  # clear the textbox; the transcript updates via the tick timer

    def on_detect(user_id: str) -> str:
        uid = user_id.strip()
        if uid:
            world.update("closest_user", uid)
        return ""

    def on_user_gone() -> None:
        world.update("closest_user", None)

    def on_select_prompt(index: int | None) -> Any:
        """User picked a prompt from the dropdown — show its exact text."""
        text = state.prompt_text(index)
        return gr.update() if text is None else text

    def tick() -> tuple[Any, Any, Any]:
        nonlocal last_shown_count, last_conv_sig
        snap = state.snapshot()

        # Push the transcript only when it actually changed; otherwise send a no-op update so
        # gr.Chatbot's autoscroll isn't re-triggered every 0.2s (which would keep dragging the view
        # to the bottom and fight the user scrolling up). A genuine change still updates and
        # autoscroll follows it.
        if snap.conv_sig != last_conv_sig:
            last_conv_sig = snap.conv_sig
            chatbot_update: Any = snap.conversation
        else:
            chatbot_update = gr.update()

        # Snap the view to the newest prompt only when a new step has appeared; between steps leave
        # the dropdown and textbox untouched (bare gr.update()) so the user can browse older prompts.
        if snap.prompt_count > last_shown_count:
            last_shown_count = snap.prompt_count
            newest = snap.prompt_count - 1
            selector_update = gr.update(choices=snap.prompt_choices, value=newest)
            prompt_update = gr.update(value=snap.newest_prompt_text)
        else:
            selector_update = gr.update()
            prompt_update = gr.update()
        return chatbot_update, selector_update, prompt_update

    def tick_world() -> str:
        """Refresh only the World state table, on its own timer. Split off the main tick so the live
        World view isn't coupled to the transcript: the main tick re-sends the whole (growing) chat
        history every fire, and driving the World table from the same handler made a new World entry
        appear only as fast as that heavier payload could round-trip. This reads live World state
        directly (get_prompt_entries, under the World lock) and renders it as a full HTML table (see
        _world_html), so it stays cheap and current regardless of transcript size, and a fleeting
        command entry isn't lost to Dataframe row-diffing."""
        return _world_html(world)

    with gr.Blocks(title="WICA — social robot demo") as demo:
        gr.Markdown("# WICA — talk to a social robot")
        if config_error is not None:
            gr.Markdown(
                f"> ⚠️ **Agent not started: {config_error}.** Set it and restart to enable the "
                "robot's reasoning. You can still explore the World panel: sensor inputs below "
                "update the World, but nothing will reason over it yet."
            )
        gr.Markdown(
            "The robot handles **one reasoning call at a time** (v1): while the model is thinking, "
            "a new input is *dropped*, not queued. Long actions are different: a 10s dance keeps "
            "running in the background, so a later input can start a new step that sees or cancels "
            "it. Inputs appear on the **right**; on the **left**, each **reasoning step is one "
            "collapsible group** (`💬 reaction N`), and inside it every item is labelled by the "
            "framework channel it came from: **🗣️ say** (the robot's spoken reply — its `say` output "
            "Command, what the person hears), **💭 output sink** (the model's free text, now private "
            "reasoning), **🦾** command calls, and **🚫 noop** (the robot explicitly choosing not to "
            "react — often its own re-triggered step, since speaking wakes the agent again)."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Say something to the robot…",
                        show_label=False,
                        scale=5,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)

                gr.Markdown("### Sensor inputs")
                with gr.Row():
                    user_id = gr.Textbox(
                        placeholder="user id, e.g. alice",
                        show_label=False,
                        scale=3,
                    )
                    detect = gr.Button("Closest user detected", scale=2)
                    gone = gr.Button("Closest user gone", scale=2)

            with gr.Column(scale=2):
                gr.Markdown("### World state (live)")
                # Rendered as HTML (not gr.Dataframe) so short-lived command rows render reliably —
                # see _world_html. Fed by its own timer (tick_world) below.
                world_view = gr.HTML(value=_world_html(world))
                prompt_selector = gr.Dropdown(
                    label="Prompt sent to the model (newest shown automatically)",
                    choices=[],
                    interactive=True,
                )
                prompt_view = gr.Textbox(
                    show_label=False,
                    lines=16,
                    interactive=False,
                    value=_NO_PROMPT_YET,
                )

        send.click(on_send, inputs=msg, outputs=msg)
        msg.submit(on_send, inputs=msg, outputs=msg)
        detect.click(on_detect, inputs=user_id, outputs=user_id)
        gone.click(on_user_gone)
        prompt_selector.change(
            on_select_prompt, inputs=prompt_selector, outputs=prompt_view
        )

        # Two independent timers so the panels don't share one round-trip. The conversation timer
        # refreshes faster than the per-word speaking pace (_SAY_WORD_DELAY_S) so `say` streams into
        # the transcript smoothly, roughly one word at a time; it also drives the prompt panel.
        timer = gr.Timer(0.2)
        timer.tick(tick, outputs=[chatbot, prompt_selector, prompt_view])

        # The World table gets its own timer (tick_world) so its liveness isn't bottlenecked by the
        # growing transcript payload the conversation tick re-sends each fire.
        world_timer = gr.Timer(0.2)
        world_timer.tick(tick_world, outputs=world_view)

    return demo
