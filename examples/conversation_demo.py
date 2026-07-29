"""WICA conversation demo — talk to a simulated social robot in the browser.

Run with:

    uv run --group demo python examples/conversation_demo.py

Configuration comes from agent.config.json (next to this file), which reads its API key from the
WICA-namespaced WICA_ANTHROPIC_API_KEY env var (see agent.config.json's "api_key_env") — set it
before running:

    export WICA_ANTHROPIC_API_KEY=sk-...

To use a different provider/model, or a literal key, edit agent.config.json directly (see
specs/config.md) — e.g. copy it to a `*.local.json` file (git-ignored) with a literal "api_key".
Its "logging" field controls the wica.* loggers; DEBUG shows the full World+Agent lifecycle trace
(registrations, every update and whether it triggered a call, LLM output, command start/end).

This is the first runnable example (see specs/conversation-demo.md). It drives WICA
purely through its public API: speech and sensor events enter the World; the Agent reasons
over the World, issues Commands, and replies; the UI shows the live World state and the exact
prompt sent to the model.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from wica import Content, TextPart, WorldEntry, get_world
from wica.agent import Agent, CommandExecution
from wica.config import MissingEnvError, WicaConfig, apply_logging

# Keep third-party logs quiet; the config's "logging" field sets the wica.* level once the Agent
# is built. Until then (or if it fails to build), default to INFO.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("wica").setLevel(logging.INFO)

# --- Configuration -------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "agent.config.json"

# --- World entries the demo owns ----------------------------------------------------
#
# All are include_in_prompt=True so they reach the model and show up in the World panel.
# Sensor inputs (speech, closest user) trigger the agent; the robot's own state (emotion,
# tracking) does not — otherwise the agent's own actions would re-trigger it in a loop.

world = get_world()


def _speech(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("The person hasn't said anything to you yet.")]
    return [TextPart(f'The closest person just said to you: "{value}"')]


def _closest_user(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("No one is standing close to you right now.")]
    return [TextPart(f'The person standing closest to you is user "{value}".')]


def _emotion(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("You currently feel neutral.")]
    return [TextPart(f"You currently feel {value}.")]


def _tracked_user(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("You are not tracking anyone.")]
    return [TextPart(f'You are tracking user "{value}".')]


def register_world() -> None:
    world.register("speech_input", str, serialize_fn=_speech, triggers_llm_call=True)
    world.register("closest_user", str, serialize_fn=_closest_user, triggers_llm_call=True)
    world.register("emotion", str, serialize_fn=_emotion)
    world.register("tracked_user", str, serialize_fn=_tracked_user)


# --- Commands (the robot's fake actions) --------------------------------------------


async def dance() -> str:
    """Perform a fun little dance. Takes about 10 seconds to complete."""
    await asyncio.sleep(10)
    return "Finished the dance."


def set_emotion(emotion: str) -> str:
    """Show an emotion on your face, e.g. 'happy', 'curious', 'sad', 'excited'."""
    world.update("emotion", emotion)
    return f"Now showing emotion: {emotion}."


def switch_user_tracking(user_id: str | None = None) -> str:
    """Follow one specific person, tracking them from now on. Pass no user (null) to stop
    tracking anyone. You follow at most one person at a time."""
    world.update("tracked_user", user_id)
    return "Stopped tracking." if user_id is None else f"Now tracking user {user_id}."


COMMANDS = [
    dance,
    set_emotion,
    switch_user_tracking,
]

# --- Agent ↔ UI bridges --------------------------------------------------------------
#
# The Agent owns its own event loop in a daemon thread and the World is thread-safe, so the
# Gradio side stays fully synchronous: sensor events call world.update(...) directly, and the
# agent's async callbacks (fired on the loop) push chat events onto one thread-safe queue that a
# timer drains into the transcript. A single ordered queue keeps triggers, the robot's spoken
# replies, and its command calls interleaved in the exact order they happened.
#
# Roles map to the two sides of the conversation:
#  - "user"      → right side: the World entry that triggered a call (on_trigger).
#  - "assistant" → left side:  the robot's spoken reply (output_sink) and its command calls
#                              (on_command).

_events: queue.Queue[dict[str, str]] = queue.Queue()

_NO_PROMPT_YET = "(no prompt sent to the model yet)"

_state_lock = threading.Lock()
_conversation: list[dict[str, str]] = []
# Every reasoning step's prompt, append-only, each {"label": ..., "text": ...} — never mutated in
# place, so readers (tick/on_select_prompt) can pull `text` outside the lock once they've read len.
_prompts: list[dict[str, str]] = []
# The trigger that started the current step. on_trigger fires just before on_prompt on the agent
# loop (see agent.py _run_step), so on_prompt reads this to label each captured prompt.
_last_trigger_label = "start"
# How many prompts the UI has shown; tick snaps the view to the newest whenever this trails len().
_last_shown_count = 0


async def output_sink(text: str) -> None:
    """Agent speech out — one complete utterance per step (v1). Runs on the agent loop."""
    if text:
        _events.put({"role": "assistant", "content": text})


def _describe_trigger(entry: WorldEntry) -> str:
    key = entry.key
    value = entry.current.value
    if key == "speech_input":
        return f'🗣️ "{value}"'
    if key == "closest_user":
        return "👤 Closest user gone" if value is None else f"👤 Closest user detected: {value}"
    if key.startswith("agent:command:"):
        name = value.name if isinstance(value, CommandExecution) else key
        return f"⚡ command finished: {name}"
    return f"⚡ {key} = {value!r}"


def on_trigger(entry: WorldEntry) -> None:
    """A World entry triggered a step — show it on the input (right) side. Skip the robot's own
    command-completion re-triggers on the transcript (they're already shown as command calls on the
    left), but always record the label so on_prompt can tag this step's prompt correctly."""
    global _last_trigger_label
    label = _describe_trigger(entry)
    _last_trigger_label = label
    if entry.key.startswith("agent:command:"):
        return
    _events.put({"role": "user", "content": label})


def on_command(name: str, args: dict[str, Any]) -> None:
    """The robot issued a command — show it on the assistant (left) side, italicised to set it
    apart from spoken replies."""
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    _events.put({"role": "assistant", "content": f"_🦾 {name}({rendered})_"})


def _flatten_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "image":
                parts.append(f"[image {block.get('mime_type', '')}]")
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    # Concatenate (don't newline-join): each World-rendered block already carries its own
    # newlines (an `<entry ...>\n` opener, a `\n...</entry>\n` closer), so this shows the exact
    # text the model receives when the provider merges adjacent blocks — no phantom blank lines.
    return "".join(parts)


def _render_message(m: BaseMessage) -> str:
    """Render one LangChain message to text for the prompt panel. An assistant message that only
    issues a command has empty .content — the call lives in the structured .tool_calls field — so
    we render those (and the tool_call id on a tool result) explicitly, or the AI block looks blank."""
    lines = [f"### {m.type.upper()}"]
    if isinstance(m, ToolMessage):
        lines.append(f"🔧 result for [id={m.tool_call_id}] => {_flatten_content(m.content)}")
        return "\n".join(lines)
    body = _flatten_content(m.content)
    if body:
        lines.append(body)
    if isinstance(m, AIMessage):
        for call in m.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            lines.append(f"🔧 tool_call {call['name']}({args})  [id={call['id']}]")
    return "\n".join(lines)


def on_prompt(messages: list[BaseMessage]) -> None:
    """Debug hook — capture the exact messages sent to the model, appending to the prompt history
    labelled by time + the trigger that caused this step. Runs on the agent loop."""
    rendered = "\n\n".join(_render_message(m) for m in messages)
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    with _state_lock:
        label = f"{stamp} — {_last_trigger_label}"
        _prompts.append({"label": label, "text": rendered})


# Build the agent from the committed config; fall back to explore-only if its api_key_env isn't
# set, so the app still opens and is explorable without credentials.
register_world()
agent: Agent | None = None
config_error: str | None = None
try:
    wica_config = WicaConfig.from_json(CONFIG_PATH)
except MissingEnvError as exc:
    config_error = f"environment variable {exc.env_var!r} is not set"
else:
    apply_logging(wica_config.logging)
    agent = Agent.from_config(
        wica_config.agent,
        output_sink=output_sink,
        on_prompt=on_prompt,
        on_trigger=on_trigger,
        on_command=on_command,
    )
    for command in COMMANDS:
        agent.register_command(command)
    agent.start()


# --- UI handlers ---------------------------------------------------------------------


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, CommandExecution):
        args = ", ".join(f"{k}={v!r}" for k, v in value.args.items())
        return f"{value.name}({args}) [{value.state}]"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]" if value else "[]"
    return str(value)


def _world_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    entries: list[WorldEntry] = world.get_prompt_entries()
    for entry in entries:
        version = entry.current
        stamp = version.timestamp.astimezone().strftime("%H:%M:%S")
        rows.append([entry.key, str(version.id), _format_value(version.value), stamp])
    return rows


# Input handlers only touch the World; the transcript is fed by the agent's callbacks (a trigger
# shows up on the right once it actually starts a step — a trigger dropped by the busy single-in-
# flight loop simply won't appear, and no reply follows it).


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
    if index is None:
        return gr.update()
    with _state_lock:
        if 0 <= index < len(_prompts):
            return _prompts[index]["text"]
    return gr.update()


def tick() -> tuple[list[dict[str, str]], list[list[str]], Any, Any]:
    global _last_shown_count
    with _state_lock:
        while True:
            try:
                event = _events.get_nowait()
            except queue.Empty:
                break
            _conversation.append(event)
        conversation = list(_conversation)
        count = len(_prompts)
        choices = [(p["label"], i) for i, p in enumerate(_prompts)]
        newest_text = _prompts[-1]["text"] if _prompts else None

    # Snap the view to the newest prompt only when a new step has appeared; between steps leave the
    # dropdown and textbox untouched (bare gr.update()) so the user can browse older prompts.
    if count > _last_shown_count:
        _last_shown_count = count
        newest = count - 1
        selector_update = gr.update(choices=choices, value=newest)
        prompt_update = gr.update(value=newest_text)
    else:
        selector_update = gr.update()
        prompt_update = gr.update()
    return conversation, _world_rows(), selector_update, prompt_update


# --- Layout --------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WICA — social robot demo") as demo:
        gr.Markdown("# WICA — talk to a social robot")
        if config_error is not None:
            gr.Markdown(
                f"> ⚠️ **Agent not started: {config_error}.** Set it and restart to enable the "
                "robot's reasoning. You can still explore the World panel: sensor inputs below "
                "update the World, but nothing will reason over it yet."
            )
        gr.Markdown(
            "The robot handles **one thought at a time** (v1): while it's thinking or in the "
            "middle of a long action (like a 10s dance), a new input is *dropped*, not queued — "
            "so it won't appear in the transcript and gets no reply. That's expected, not a bug. "
            "Inputs (right) and the robot's replies + 🦾 command calls (left) appear as they "
            "actually happen."
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
                world_view = gr.Dataframe(
                    headers=["Key", "Version", "Value", "Updated"],
                    datatype=["str", "str", "str", "str"],
                    label="World state (live)",
                    interactive=False,
                    wrap=True,
                )
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
        prompt_selector.change(on_select_prompt, inputs=prompt_selector, outputs=prompt_view)

        timer = gr.Timer(0.4)
        timer.tick(tick, outputs=[chatbot, world_view, prompt_selector, prompt_view])

    return demo


def main() -> None:
    try:
        build_ui().launch()
    finally:
        if agent is not None:
            agent.stop()


if __name__ == "__main__":
    main()
