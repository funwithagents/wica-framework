# Gradio panels: watch the World, the prompts and the transcript

Three browser panels ship in the package behind the `gradio` extra, lifted from the conversation demo: the **live World-state table**, the **reaction history** (each step's exact prompt and its timing) and the **conversation transcript** (inputs, Command executions and utterances with their live state, the model's private text). Each is a presenter built over your `Wica` plus a panel you drop into your own `gr.Blocks`. Core `wica` never imports Gradio; the design is [specs/gradio-contrib.md](../../specs/gradio-contrib.md).

```
uv add "wica[anthropic,gradio] @ git+https://github.com/funwithagents/wica-framework"
```

## The whole flow

Build, then UI, then wire, then start, then serve, then close:

```python
import gradio as gr

from wica import AgentConfig, CommandExecution, TextPart, Wica, WicaConfig, WorldEntry
from wica.contrib.gradio import (
    EntryDisplay,
    ReactionLog,
    TranscriptLog,
    conversation_panel,
    reaction_panel,
    world_state_panel,
)

# 1. Build the Wica and register your World entries.
config = WicaConfig(
    agent=AgentConfig(
        provider="anthropic",
        model="claude-haiku-4-5",
        system_prompt="You are a friendly social robot.",
    )
)
wica = Wica.init(config)
wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, previous: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,
)


# 2. Build the presenters over the Wica, then the page. One optional hook says how your
#    entries read to a person; return None for the generic default.
def display_entry(entry: WorldEntry) -> EntryDisplay | None:
    value = entry.current.value
    if entry.key == "speech_input":
        return EntryDisplay(f'🗣️ "{value}"')
    if isinstance(value, CommandExecution) and value.name == "say":
        return EntryDisplay("🗣️ say", str(value.args.get("text", "")))
    return None


transcript = TranscriptLog(wica, display_entry)  # the same hook for both presenters,
reactions = ReactionLog(wica, display_entry)  # so the two surfaces label alike

with gr.Blocks() as page:
    with gr.Row():
        with gr.Column():
            conversation_panel(transcript)  # panels go inside the Blocks context
            msg = gr.Textbox(placeholder="Say something…", show_label=False)
        with gr.Column():
            world_state_panel(wica.world)
            reaction_panel(reactions)
    # Input widgets only touch the World; the panels refresh on their own timers.
    msg.submit(
        lambda text: wica.world.update("speech_input", text) or "", inputs=msg, outputs=msg
    )


# 3. Wire: the voice as the output Command, the other Commands. The presenters need no
#    wiring: they subscribed to the Wica's Events when they were built.
async def say(text: str) -> str:
    """Speak out loud to the person in front of you."""
    ...  # your voice
    return "Said it."


async def dance() -> str:
    """Perform a fun little dance."""
    ...
    return "Finished the dance."


wica.set_output_command(say)  # what the model writes in <say>…</say> → the 🗣️ item
wica.register_command(dance)

# 4. Start the system, then serve the page; tear down when the server returns.
wica.start()
try:
    page.launch()
finally:
    wica.close()
```

## The pieces

| | Presenter | Panel | Shows |
|---|---|---|---|
| World state | none (reads the World) | `world_state_panel(world, *, refresh_s=0.2) -> gr.HTML` | Every prompt entry: key, value (a running Command as `name(args) [state]`), last update |
| Reaction history | `ReactionLog(wica, display_entry=None)` | `reaction_panel(reactions, *, refresh_s=0.2, lines=16) -> tuple[gr.Dropdown, gr.Textbox, gr.Textbox]` | One dropdown per reaction, a Prompt tab (the exact messages sent) and an Instrumentation tab (the reaction's measures) |
| Transcript | `TranscriptLog(wica, display_entry=None)` | `conversation_panel(transcript, *, refresh_s=0.2, height=420) -> gr.Chatbot` | The inputs the Agent reacted to, each Command execution and each output (🗣️) as an item that updates with its state, the model's private text as the 💭 item, a declined reaction as the 🚫 item |

A presenter with `wica=None` renders empty and subscribes to nothing, for an explore-only page when the model cannot be built (the demo does this when no key is set). Each panel owns the `gr.Timer` that keeps it live and returns its component(s), so the layout is yours: they are building blocks, not a page.

**`display_entry`** is the one hook: `(WorldEntry) -> EntryDisplay | None`. `EntryDisplay(label, detail=None)` is the one-line label with its icon (the input-side message for a triggering entry, the title of a Command item) and, for a Command item, its body. `None` falls back to the generic rendering (`⚡ key = value`, `🦾 name(args)` for a Command execution). Pass the same hook to both presenters.

## The ordering rules

- **Presenters after `Wica.init`.** They subscribe to the Wica's Events in their constructor. They are plain objects: build them before or outside the `gr.Blocks` context. Panels create Gradio components and **must** be called inside one.
- **Wire before `start()`.** `set_output_command` recomposes the system prompt; the order between the output Command and the Commands does not matter, nor does the order between registering entries and building presenters. The transcript learns which items are outputs from the Events, and the generic rendering of an entry reads the voice's name from the Agent when it renders, so it may be built before `set_output_command` runs.
- **🗣️ is heard, 💭 is not.** The 🗣️ item is one output: one tag of the output Command, shown from its first words (`on_agent_command_started`), its text filled in once the tag closes, and updated as it turns `complete` or `cancelled`. The transcript tells it from a Command by the Event's `is_output`, not by its name. The 💭 item is the model's private text, what it wrote outside its tags; with no output Command set, all its text ([commands.md](commands.md#make-your-agent-speak)).
- **Input widgets update the World, which works only while running**: start the Wica before `launch()`.
- **Tear down from the Gradio thread.** `launch()` blocks until the server exits; call `wica.close()` after it returns, never from a Command, the output Command or an Event subscriber.

The shipped demo does exactly this with a Speaking panel and sensor buttons, once per session: [`app.py`](../../examples/conversation_demo/app.py) builds the presenters, wires and starts a session, and [`app_ui.py`](../../examples/conversation_demo/app_ui.py) draws the panels over it, anew for each session ([specs/conversation-demo.md](../../specs/conversation-demo.md)).
