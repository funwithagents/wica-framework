"""The Gradio page of the WICA conversation demo.

`build_page(model, conversation, open_on=...)` is the page: the demo's **Conversation** tab first, then
app-setup's **Setup** and **Secrets** panels; it opens on Conversation when the session starts
at launch, else on Setup. The Conversation tab has Start
and Stop over the `Conversation` (session.py), a status line with the saved config's current
blockers (checked again when the tab is selected, the config saved or a key changed), a notice
while the Setup tab has unsaved changes, and, while a session runs, its five surfaces
(Conversation / Speaking / World state / Reaction history / Sensor inputs). Three of them are the
package's reusable components (`wica.contrib.gradio`: `conversation_panel`, `world_state_panel`,
`reaction_panel` — see specs/gradio-contrib.md) over the session's presenters; the Speaking panel
and the sensor inputs are the demo's own. Each session is a new Wica with new presenters, so the
surfaces are drawn in a `gr.render` that runs again whenever a session starts or stops (see
specs/conversation-demo.md, "The page").
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gradio as gr
from app_setup.app import AppModel
from app_setup.ui import secrets_panel, setup_panel, unsaved_notice

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


def status_text(conversation: Conversation, blockers: Sequence[str] = ()) -> str:
    """The line above Start and Stop: running (which provider and model), starting, stopping, or
    stopped, with the last Start's build error and the saved config's current *blockers*, or
    ready to start when there are none."""
    status = conversation.status
    running = conversation.current
    if status == "running" and running is not None:
        agent = running.config.agent
        return (
            f"🟢 **Running** — `{agent.provider}` / `{agent.model}`. Stop it to "
            "change the config in the Setup tab."
        )
    if status == "starting":
        return "⏳ **Starting…**"
    if status == "stopping":
        return "⏳ **Stopping…**"
    reasons = [*([conversation.error] if conversation.error else []), *blockers]
    if reasons:
        listed = "\n".join(f"- {reason}" for reason in reasons)
        return (
            "⚠️ **Not started.** Fix this in the Setup and Secrets tabs, then Start:\n\n"
            + listed
        )
    return (
        "⚪ **Stopped — ready to start.** Start runs the agent from the saved config (the "
        "Setup tab)."
    )


# What the status line's blockers were computed against: the store's key revision, the
# session's version and status. A change of any recomputes them; nothing else does.
Watched = tuple[int, int, str]


def current_blockers(model: AppModel, conversation: Conversation) -> list[str]:
    """Why the saved config cannot start now (`AppModel.blockers`); none unless stopped."""
    if conversation.status != "stopped":
        return []
    return model.blockers(model.saved_config())


def follow_blockers(
    model: AppModel,
    conversation: Conversation,
    blockers: list[str],
    seen: Watched | None,
    *,
    force: bool = False,
) -> tuple[list[str], Watched]:
    """The status line's blockers, and what they were computed against. Recomputed when
    *force*d (the tab selected, the config saved or reset), or when a key changed in the store
    (`SecretStore.revision`) or the session started or stopped since *seen*; otherwise
    *blockers* are kept, so the 1 s tick costs no check."""
    now: Watched = (model.secrets.revision, conversation.version, conversation.status)
    if force or now != seen:
        return current_blockers(model, conversation), now
    return blockers, now


def button_states(conversation: Conversation) -> tuple[bool, bool]:
    """Whether Start and Stop can be pressed: Start only while stopped, Stop only while
    running (both greyed while starting or stopping)."""
    status = conversation.status
    return status == "stopped", status == "running"


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


@dataclass(frozen=True)
class _SessionView:
    """The Conversation tab's components the page wires to the Setup panel."""

    notice: gr.Markdown
    outputs: list[Any]
    recheck: Callable[[], tuple[Any, ...]]


def _conversation_tab(model: AppModel, conversation: Conversation) -> _SessionView:
    """Start, Stop, the status line, the unsaved-changes notice, and the running session's
    surfaces."""
    can_start, can_stop = button_states(conversation)
    status = gr.Markdown(status_text(conversation))
    with gr.Row():
        start = gr.Button("Start", variant="primary", scale=0, interactive=can_start)
        stop = gr.Button("Stop", scale=0, interactive=can_stop)
        # "The Setup tab has unsaved changes: Start uses the saved config.", or nothing.
        notice = gr.Markdown()
    gr.Markdown(_HOW_TO_READ)
    # The session's version: a start or a stop from any browser tab (or at launch) redraws the
    # surfaces below.
    version = gr.State(conversation.version)
    # This browser tab's blockers and what they were computed against (None: not yet).
    blockers = gr.State([])
    seen = gr.State(None)

    def view(found: list[str], watched: Watched) -> tuple[Any, ...]:
        can_start, can_stop = button_states(conversation)
        return (
            status_text(conversation, found),
            gr.update(interactive=can_start),
            gr.update(interactive=can_stop),
            conversation.version,
            found,
            watched,
        )

    def refresh(found: list[str], watched: Watched | None) -> tuple[Any, ...]:
        return view(*follow_blockers(model, conversation, found, watched))

    def recheck() -> tuple[Any, ...]:
        return view(*follow_blockers(model, conversation, [], None, force=True))

    def on_start() -> tuple[Any, ...]:
        conversation.start(model)
        return recheck()

    def on_stop() -> tuple[Any, ...]:
        conversation.stop()
        return recheck()

    outputs = [status, start, stop, version, blockers, seen]
    # Greyed at once on a click, before the (possibly slow) start or stop runs.
    start.click(
        lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[start, stop],
    ).then(on_start, outputs=outputs)
    stop.click(
        lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[start, stop],
    ).then(on_stop, outputs=outputs)
    # On the page's load, then polled: a key saved, or a start or stop from another browser
    # tab or at launch, recomputes the blockers; a tick with no change only redraws.
    gr.on(None, recheck, outputs=outputs, show_progress="hidden")
    gr.Timer(1.0).tick(
        refresh, inputs=[blockers, seen], outputs=outputs, show_progress="hidden"
    )

    @gr.render(inputs=version)
    def _surfaces(_version: int) -> None:
        running = conversation.current
        if running is None:
            gr.Markdown("*No session is running.*")
            return
        _session_panels(running)

    return _SessionView(notice=notice, outputs=outputs, recheck=recheck)


def build_page(
    model: AppModel, conversation: Conversation, *, open_on: str = "setup"
) -> gr.Blocks:
    """The demo's page: its Conversation tab, then app-setup's Setup and Secrets panels. It
    opens on *open_on* (`"conversation"` when the session starts at launch, else `"setup"`)."""
    with gr.Blocks(title=model.title) as page:
        gr.Markdown(f"# {model.title}")
        with gr.Tabs(selected=open_on):
            with gr.Tab("Conversation", id="conversation") as session_tab:
                session = _conversation_tab(model, conversation)
            with gr.Tab("Setup", id="setup"):
                setup = setup_panel(model)
            with gr.Tab("Secrets", id="secrets"):
                secrets_panel(model, setup.draft)
        # The status line's blockers are the saved config's now: checked again when the tab is
        # selected and when the Setup panel saves or resets (a key change is caught by the tick).
        session_tab.select(
            session.recheck, outputs=session.outputs, show_progress="hidden"
        )
        setup.saved.change(
            session.recheck, outputs=session.outputs, show_progress="hidden"
        )
        # Start uses the saved config: said next to it while the Setup tab has unsaved edits.
        for changed in (setup.draft.change, setup.saved.change):
            changed(
                lambda draft: unsaved_notice(model, draft),
                inputs=[setup.draft],
                outputs=[session.notice],
                show_progress="hidden",
            )
    return page
