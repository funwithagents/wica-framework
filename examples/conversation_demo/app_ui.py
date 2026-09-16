"""The Gradio UI for the WICA conversation demo.

`build_ui(wica, world, display_entry, config_error)` assembles the five surfaces (Conversation /
Speaking / World state / Reaction history / Sensor inputs). Three of them are the package's
reusable components (`wica.contrib.gradio`: `conversation_panel`, `world_state_panel`,
`reaction_panel` — see specs/gradio-contrib.md); the UI **owns their presenters**: it creates the
`TranscriptLog` and `ReactionLog` (each subscribed to the Wica's Events in its constructor, both
given the demo's `display_entry` hook) and the Speaking panel's `SpeakingSlot`, and returns the
transcript and the slot on the `DemoUi` handle so the app can wire them into the robot — the
transcript's `output_sink` onto the Wica, the slot into `say`. The Speaking panel and the sensor
inputs are the demo's own. The app file (`app.py`) builds the Wica first, calls this, then wires
and starts (see specs/conversation-demo.md, "Composition").
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import gradio as gr

from wica import Wica, World
from wica.contrib.gradio import (
    DisplayEntry,
    ReactionLog,
    TranscriptLog,
    conversation_panel,
    reaction_panel,
    world_state_panel,
)

from examples.conversation_demo.speaking import Speaking, SpeakingSlot

# --- Speaking panel ------------------------------------------------------------------
#
# The simulated TTS's state, rendered on its own under the conversation: the words already
# "spoken" set apart from the ones still to come, a spoken/total count and the utterance's state.
# The transcript shows the same `say` only as a Command item (full text + live state), so this is
# the one place the word-by-word progress is visible — and the one surface specific to this
# robot's voice rather than to the framework. See specs/conversation-demo.md ("What the user
# sees" 2.).
_SPEAKING_BADGES = {
    "speaking": "🔊 speaking…",
    "complete": "✅ complete",
    "cancelled": "⏹ cancelled",
}


def _speaking_html(speaking: Speaking | None) -> str:
    style = (
        "<style>"
        ".speaking{border:1px solid var(--border-color-primary,#ccc);border-radius:6px;"
        "padding:6px 10px;font-size:14px;min-height:2.6em}"
        ".speaking .badge{font-size:12px;color:var(--body-text-color-subdued,#666);"
        "margin-bottom:2px}"
        ".speaking .spoken{font-weight:600}"
        ".speaking .remaining{color:var(--body-text-color-subdued,#999)}"
        ".speaking .idle{color:var(--body-text-color-subdued,#999);font-style:italic}"
        "</style>"
    )
    if speaking is None:
        return (
            f"<div class='speaking'>{style}"
            "<div class='badge'>🗣️ say</div>"
            "<div class='idle'>(the robot hasn't spoken yet)</div></div>"
        )
    spoken = " ".join(html.escape(w) for w in speaking.words[: speaking.spoken])
    remaining = " ".join(html.escape(w) for w in speaking.words[speaking.spoken :])
    badge = _SPEAKING_BADGES[speaking.state]
    return (
        f"<div class='speaking'>{style}"
        f"<div class='badge'>🗣️ say · {badge} · {speaking.spoken}/{len(speaking.words)} words"
        "</div><div class='words'>"
        f"<span class='spoken'>{spoken}</span> <span class='remaining'>{remaining}</span>"
        "</div></div>"
    )


# --- Layout --------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoUi:
    """What build_ui() returns: the page, plus the two presenters the app wires into the robot
    (`transcript.output_sink` → `wica.set_output_sink`, `speaking` → `say`)."""

    blocks: gr.Blocks
    transcript: TranscriptLog
    speaking: SpeakingSlot


def build_ui(
    wica: Wica | None,
    world: World,
    display_entry: DisplayEntry | None,
    config_error: str | None,
) -> DemoUi:
    """Assemble the demo UI over an already-built (not yet started) Wica. Input handlers only touch
    the World; the transcript is fed by the agent's Events (a trigger shows up on the right once it
    actually starts a step — a trigger dropped by the busy single-in-flight loop simply won't
    appear, and no reply follows it), and the Speaking panel by `say` via the slot. `wica` is None
    in explore-only mode (no key): the panels still render, nothing reasons."""
    transcript = TranscriptLog(wica, display_entry)
    reactions = ReactionLog(wica, display_entry)
    speaking = SpeakingSlot()

    def on_send(user_text: str) -> str:
        text = user_text.strip()
        if text:
            world.update("speech_input", text)
        return ""  # clear the textbox; the transcript updates via its own timer

    def on_detect(user_id: str) -> str:
        uid = user_id.strip()
        if uid:
            world.update("closest_user", uid)
        return ""

    def on_user_gone() -> None:
        world.update("closest_user", None)

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
            "react — often its own re-triggered step, since speaking wakes the agent again). "
            "Every Command item shows its **execution state live**: a spinner while it runs, then "
            "✅ complete / ❌ failed / ⏹ cancelled with its result and duration. The **Speaking** "
            "panel under the conversation shows the current `say` word by word."
        )

        with gr.Row():
            with gr.Column(scale=3):
                conversation_panel(transcript)
                speaking_view = gr.HTML(value=_speaking_html(None))
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
                world_state_panel(world)
                gr.Markdown("### Reaction history")
                reaction_panel(reactions)

        send.click(on_send, inputs=msg, outputs=msg)
        msg.submit(on_send, inputs=msg, outputs=msg)
        detect.click(on_detect, inputs=user_id, outputs=user_id)
        gone.click(on_user_gone)

        # The Speaking panel is cheap to re-render whole each tick (like the World table), and its
        # words advance faster than a diff would be worth; its timer refreshes faster than the
        # per-word speaking pace (_SAY_WORD_DELAY_S) so it advances roughly one word at a time.
        speaking_timer = gr.Timer(0.2)
        speaking_timer.tick(
            lambda: _speaking_html(speaking.read()), outputs=speaking_view
        )

    return DemoUi(blocks=demo, transcript=transcript, speaking=speaking)
