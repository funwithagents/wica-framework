"""WICA conversation demo — talk to a simulated social robot in the browser.

Run with:

    uv run --group demo python examples/conversation_demo.py

Set your provider key first (WICA-namespaced, default provider is Anthropic):

    export WICA_ANTHROPIC_API_KEY=sk-...

Override the model/provider with WICA_PROVIDER / WICA_MODEL (the key var follows:
WICA_<PROVIDER>_API_KEY, e.g. WICA_OPENAI_API_KEY). Set WICA_LOG=DEBUG to watch the
Agent drop triggers and retire completed command entries.

This is the first runnable example (see specs/conversation-demo.md). It drives WICA
purely through its public API: speech and sensor events enter the World; the Agent reasons
over the World, issues Commands, and replies; the UI shows the live World state and the exact
prompt sent to the model.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from wica import AgentConfig, Content, TextPart, WorldEntry, get_world
from wica.agent import Agent, CommandExecution

# Keep third-party logs quiet but surface WICA's own. INFO shows agent start/stop, dropped
# triggers, and command failures; DEBUG shows the full World+Agent lifecycle trace (registrations,
# every update and whether it triggered a call, LLM output, command start/end). WICA_LOG overrides.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("wica").setLevel(os.environ.get("WICA_LOG", "INFO").upper())

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
- switch_user_tracking(user_id): follow one specific person; pass no user (null) to stop tracking.

You follow at most one person at a time — the one currently closest to you. Whenever the closest
person changes, switch your tracking to them; and when no one is close to you anymore, stop
tracking by switching to nobody.

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

_state_lock = threading.Lock()
_conversation: list[dict[str, str]] = []
_latest_prompt = "(no prompt sent to the model yet)"


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
    return f"⚡ {key} = {value!r}"


def on_trigger(entry: WorldEntry) -> None:
    """A World entry triggered a step — show it on the input (right) side. Skip the robot's own
    command-completion re-triggers; those are already shown as command calls on the left."""
    if entry.key.startswith("agent:command:"):
        return
    _events.put({"role": "user", "content": _describe_trigger(entry)})


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
    return "\n".join(parts)


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
    """Debug hook — capture the exact messages sent to the model. Runs on the agent loop."""
    global _latest_prompt
    rendered = "\n\n".join(_render_message(m) for m in messages)
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
        on_trigger=on_trigger,
        on_command=on_command,
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


def tick() -> tuple[list[dict[str, str]], list[list[str]], str]:
    with _state_lock:
        while True:
            try:
                event = _events.get_nowait()
            except queue.Empty:
                break
            _conversation.append(event)
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
                prompt_view = gr.Textbox(
                    label="Prompt sent to the model (last step)",
                    lines=16,
                    interactive=False,
                    value=_latest_prompt,
                )

        send.click(on_send, inputs=msg, outputs=msg)
        msg.submit(on_send, inputs=msg, outputs=msg)
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
