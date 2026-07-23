"""WICA conversation demo — talk to a simulated social robot in the browser.

Run with:

    uv run --group demo python examples/conversation_demo.py

Set your provider key first (WICA-namespaced, default provider is Anthropic):

    export WICA_ANTHROPIC_API_KEY=sk-...

Override the model/provider with WICA_PROVIDER / WICA_MODEL (the key var follows:
WICA_<PROVIDER>_API_KEY, e.g. WICA_OPENAI_API_KEY).

This is the first runnable example (see specs/gradio-conversation-demo.md). It drives WICA
purely through its public API: speech and sensor events enter the World; the Agent reasons
over the World, issues Commands, and replies; the UI shows the live World state and the exact
prompt sent to the model.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import Any

import gradio as gr
from langchain_core.messages import BaseMessage

from wica import AgentConfig, Content, TextPart, WorldEntry, get_world
from wica.agent import Agent, CommandExecution

# --- Configuration -------------------------------------------------------------------

PROVIDER = os.environ.get("WICA_PROVIDER", "anthropic")
MODEL = os.environ.get("WICA_MODEL", "claude-sonnet-5")

# WICA reads its *own* namespaced key (e.g. WICA_ANTHROPIC_API_KEY) so it never collides with a
# provider key another tool in your environment already uses. We route it into the provider's
# standard env var (ANTHROPIC_API_KEY, …) for this process only, so LangChain picks it up normally.
_STANDARD_KEY_ENV_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
}
WICA_KEY_ENV = f"WICA_{PROVIDER.upper()}_API_KEY"
_standard_key_env = _STANDARD_KEY_ENV_BY_PROVIDER.get(PROVIDER)
_wica_key = os.environ.get(WICA_KEY_ENV)
HAS_KEY = bool(_wica_key)
if _wica_key and _standard_key_env:
    os.environ[_standard_key_env] = _wica_key

SYSTEM_PROMPT = """\
You are Wica, a friendly social robot standing in a room, greeting and chatting with people.

Your senses and body are described to you as a set of world entries in each message: what the
closest person said to you, who is standing closest to you, how you currently feel, and who you
are tracking. React naturally to what changes — you may respond to someone walking up to you,
not only to what they say.

You can act on the world with these commands:
- dance(): do a little dance (takes about 10 seconds).
- set_emotion(emotion): show an emotion on your face (e.g. "happy", "curious", "sad").
- start_user_tracking(user_id) / stop_user_tracking(user_id): begin or end following a person.
- switch_user_tracking(user_id): focus your attention on one specific person.

Keep spoken replies short and warm — one or two sentences. Set an emotion when your mood shifts,
and use tracking when it makes sense to follow someone. Speak naturally; never mention "world
entries", "commands", or that you are an AI.
"""

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


def _tracked_users(value: Any, previous: Any) -> Content:
    if not value:
        return [TextPart("You are not tracking anyone.")]
    joined = ", ".join(str(u) for u in value)
    return [TextPart(f"You are tracking these users: {joined}.")]


def _active_user(value: Any, previous: Any) -> Content:
    if value is None:
        return [TextPart("You are not focused on anyone in particular.")]
    return [TextPart(f'You are focused on user "{value}".')]


# Keys the World panel reads back, in display order.
DEMO_KEYS = ["speech_input", "closest_user", "emotion", "tracked_users", "active_tracked_user"]


def register_world() -> None:
    world.register("speech_input", str, serialize_fn=_speech, triggers_llm_call=True)
    world.register("closest_user", str, serialize_fn=_closest_user, triggers_llm_call=True)
    world.register("emotion", str, serialize_fn=_emotion)
    world.register("tracked_users", list, serialize_fn=_tracked_users)
    world.register("active_tracked_user", str, serialize_fn=_active_user)


# --- Commands (the robot's fake actions) --------------------------------------------


async def dance() -> str:
    """Perform a fun little dance. Takes about 10 seconds to complete."""
    await asyncio.sleep(10)
    return "Finished the dance."


def set_emotion(emotion: str) -> str:
    """Show an emotion on your face, e.g. 'happy', 'curious', 'sad', 'excited'."""
    world.update("emotion", emotion)
    return f"Now showing emotion: {emotion}."


def start_user_tracking(user_id: str) -> str:
    """Begin visually tracking the user with this id, following them over time."""
    tracked = list(world.get("tracked_users") or [])
    if user_id not in tracked:
        tracked.append(user_id)
        world.update("tracked_users", tracked)
    return f"Started tracking user {user_id}."


def stop_user_tracking(user_id: str) -> str:
    """Stop visually tracking the user with this id."""
    tracked = list(world.get("tracked_users") or [])
    if user_id in tracked:
        tracked.remove(user_id)
        world.update("tracked_users", tracked)
    if world.get("active_tracked_user") == user_id:
        world.update("active_tracked_user", None)
    return f"Stopped tracking user {user_id}."


def switch_user_tracking(user_id: str) -> str:
    """Focus your attention on this user specifically, tracking them from now on."""
    tracked = list(world.get("tracked_users") or [])
    if user_id not in tracked:
        tracked.append(user_id)
        world.update("tracked_users", tracked)
    world.update("active_tracked_user", user_id)
    return f"Now focused on user {user_id}."


COMMANDS = [
    dance,
    set_emotion,
    start_user_tracking,
    stop_user_tracking,
    switch_user_tracking,
]

# --- Agent ↔ UI bridges --------------------------------------------------------------
#
# The Agent owns its own event loop in a daemon thread and the World is thread-safe, so the
# Gradio side stays fully synchronous: sensor events call world.update(...) directly, and the
# agent's async output / prompt hook hand data back through thread-safe holders that a timer
# polls into the panels.

_reply_queue: queue.Queue[str] = queue.Queue()

_state_lock = threading.Lock()
_conversation: list[dict[str, str]] = []
_latest_prompt = "(no prompt sent to the model yet)"


async def output_sink(text: str) -> None:
    """Agent speech out — one complete utterance per step (v1). Runs on the agent loop."""
    if text:
        _reply_queue.put(text)


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
    return "\n".join(parts)


def on_prompt(messages: list[BaseMessage]) -> None:
    """Debug hook — capture the exact messages sent to the model. Runs on the agent loop."""
    global _latest_prompt
    rendered = "\n\n".join(
        f"### {m.type.upper()}\n{_flatten_content(m.content)}" for m in messages
    )
    with _state_lock:
        _latest_prompt = rendered


# Build the agent only when a key is present, so the app opens and is explorable without one.
agent: Agent | None = None
if HAS_KEY:
    register_world()
    agent = Agent.from_config(
        AgentConfig(provider=PROVIDER, model=MODEL, system_prompt=SYSTEM_PROMPT),
        output_sink=output_sink,
        on_prompt=on_prompt,
    )
    for command in COMMANDS:
        agent.register_command(command)
    agent.start()
else:
    register_world()


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


def on_send(user_text: str) -> tuple[list[dict[str, str]], str]:
    text = user_text.strip()
    if not text:
        with _state_lock:
            return list(_conversation), ""
    with _state_lock:
        _conversation.append({"role": "user", "content": text})
        snapshot = list(_conversation)
    world.update("speech_input", text)
    return snapshot, ""


def on_detect(user_id: str) -> str:
    uid = user_id.strip()
    if uid:
        world.update("closest_user", uid)
    return ""


def on_user_gone() -> None:
    world.update("closest_user", None)


def tick() -> tuple[list[dict[str, str]], list[list[str]], str]:
    with _state_lock:
        while True:
            try:
                reply = _reply_queue.get_nowait()
            except queue.Empty:
                break
            _conversation.append({"role": "assistant", "content": reply})
        conversation = list(_conversation)
        prompt = _latest_prompt
    return conversation, _world_rows(), prompt


# --- Layout --------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WICA — social robot demo") as demo:
        gr.Markdown("# WICA — talk to a social robot")
        if not HAS_KEY:
            gr.Markdown(
                f"> ⚠️ **No API key found.** Set `{WICA_KEY_ENV}` and restart "
                "to enable the robot's reasoning. You can still explore the World panel: sensor "
                "inputs below update the World, but nothing will reason over it yet."
            )
        gr.Markdown(
            "The robot handles **one thought at a time** (v1): while it's thinking or in the "
            "middle of a long action (like a 10s dance), new inputs are *dropped*, not queued — "
            "that's expected, not a bug."
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
                prompt_view = gr.Textbox(
                    label="Prompt sent to the model (last step)",
                    lines=16,
                    interactive=False,
                    value=_latest_prompt,
                )

        send.click(on_send, inputs=msg, outputs=[chatbot, msg])
        msg.submit(on_send, inputs=msg, outputs=[chatbot, msg])
        detect.click(on_detect, inputs=user_id, outputs=user_id)
        gone.click(on_user_gone)

        timer = gr.Timer(0.4)
        timer.tick(tick, outputs=[chatbot, world_view, prompt_view])

    return demo


def main() -> None:
    try:
        build_ui().launch()
    finally:
        if agent is not None:
            agent.stop()


if __name__ == "__main__":
    main()
