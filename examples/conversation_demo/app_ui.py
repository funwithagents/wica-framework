"""The Gradio page of the WICA conversation demo.

`build_page(model, conversation, notice)` is the page: the demo's **Conversation** tab first, then
app-setup's **Setup** and **Secrets** panels; it opens on Setup. The Conversation tab has Start
and Stop over the `Conversation` (session.py) and, while a session runs, its five surfaces
(Conversation / Speaking / World state / Reaction history / Sensor inputs). Three of them are the
package's reusable components (`wica.contrib.gradio`: `conversation_panel`, `world_state_panel`,
`reaction_panel` — see specs/gradio-contrib.md) over the session's presenters; the Speaking panel
and the sensor inputs are the demo's own. Each session is a new Wica with new presenters, so the
surfaces are drawn in a `gr.render` that runs again whenever a session starts or stops (see
specs/conversation-demo.md, "The page").
"""

from __future__ import annotations

import html

import gradio as gr
from app_setup.app import AppModel
from app_setup.ui import secrets_panel, setup_panel

from examples.conversation_demo.session import Conversation, Running
from examples.conversation_demo.speaking import Speaking
from wica.contrib.gradio import conversation_panel, reaction_panel, world_state_panel

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

_HOW_TO_READ = (
    "The robot handles **one reasoning call at a time** (v1): while the model is thinking, "
    "a new input is *dropped*, not queued. Long actions are different: a 10s dance keeps "
    "running in the background, so a later input can start a new step that sees or cancels "
    "it. Inputs appear on the **right**; on the **left**, each **reasoning step is one "
    "collapsible group** (`💬 reaction N`), and inside it every item is labelled by the "
    "framework channel it came from: **🗣️ say** (the robot's spoken reply — what the model wrote "
    "inside a `<say>` tag, delivered to its `say` output Command: what the person hears), "
    "**💭 private text** (what the model wrote outside the tag, which nobody hears), **🦾** "
    "command calls, and **🚫 noop** (the robot explicitly choosing not to react, typically "
    "to a change it was not asked about). "
    "Every Command item shows its **execution state live**: a spinner while it runs, then "
    "✅ complete / ❌ failed / ⏹ cancelled with its result and duration. The **Speaking** "
    "panel under the conversation shows the current `say` word by word."
)


def status_text(conversation: Conversation, reasons: list[str] | None = None) -> str:
    """The line above Start and Stop: running (which provider and model), or not, and why."""
    running = conversation.current
    if running is not None:
        agent = running.config.agent
        return (
            f"🟢 **Running** — `{agent.provider}` / `{agent.model}`. Stop it to "
            "change the config in the Setup tab."
        )
    if reasons:
        listed = "\n".join(f"- {reason}" for reason in reasons)
        return (
            "⚠️ **Not started.** Fix this in the Setup and Secrets tabs, then Start:\n\n"
            + listed
        )
    return "⚪ **Stopped.** Start runs the agent from the saved config (the Setup tab)."


def _session_panels(running: Running) -> None:
    """The five surfaces of one running session. Input handlers only touch its World; the
    transcript is fed by the agent's Events (a trigger shows up on the right once it actually
    starts a step — a trigger dropped by the busy single-in-flight loop simply won't appear, and
    no reply follows it), and the Speaking panel by `say` via the slot."""
    world = running.wica.world
    speaking = running.speaking

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

    with gr.Row():
        with gr.Column(scale=3):
            conversation_panel(running.transcript)
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
            reaction_panel(running.reactions)

    send.click(on_send, inputs=msg, outputs=msg)
    msg.submit(on_send, inputs=msg, outputs=msg)
    detect.click(on_detect, inputs=user_id, outputs=user_id)
    gone.click(on_user_gone)

    # The Speaking panel is cheap to re-render whole each tick (like the World table), and its
    # words advance faster than a diff would be worth; its timer refreshes faster than the
    # per-word speaking pace (_SAY_WORD_DELAY_S) so it advances roughly one word at a time.
    speaking_timer = gr.Timer(0.2)
    speaking_timer.tick(lambda: _speaking_html(speaking.read()), outputs=speaking_view)


def _conversation_tab(
    model: AppModel, conversation: Conversation, notice: list[str]
) -> None:
    """Start, Stop, the status line, and the running session's surfaces."""
    status = gr.Markdown(status_text(conversation, notice))
    with gr.Row():
        start = gr.Button("Start", variant="primary", scale=0)
        stop = gr.Button("Stop", scale=0)
    gr.Markdown(_HOW_TO_READ)
    # The session's version, polled: a start or a stop from any browser tab (or at launch) redraws
    # the surfaces below.
    version = gr.State(conversation.version)

    def on_start() -> tuple[str, int]:
        reasons = conversation.start(model)
        return status_text(conversation, reasons), conversation.version

    def on_stop() -> tuple[str, int]:
        conversation.stop()
        return status_text(conversation), conversation.version

    start.click(on_start, outputs=[status, version])
    stop.click(on_stop, outputs=[status, version])
    gr.Timer(1.0).tick(lambda: conversation.version, outputs=version)

    @gr.render(inputs=version)
    def _surfaces(_version: int) -> None:
        running = conversation.current
        if running is None:
            gr.Markdown("*No session is running.*")
            return
        _session_panels(running)


def build_page(
    model: AppModel, conversation: Conversation, notice: list[str] | None = None
) -> gr.Blocks:
    """The demo's page: its Conversation tab, then app-setup's Setup and Secrets panels. It
    opens on Setup. *notice* is why the session did not start at launch, shown until Start."""
    with gr.Blocks(title=model.title) as page:
        gr.Markdown(f"# {model.title}")
        with gr.Tabs(selected="setup"):
            with gr.Tab("Conversation", id="conversation"):
                _conversation_tab(model, conversation, notice or [])
            with gr.Tab("Setup", id="setup"):
                setup = setup_panel(model)
            with gr.Tab("Secrets", id="secrets"):
                secrets_panel(model, setup.draft)
    return page
