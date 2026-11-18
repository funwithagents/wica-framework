# Commands

A Command is the agent's unit of action on the world: speech, movement, a display change, an API call. You register functions; the Agent binds them as the model's native tool calls, runs each as a cancellable task, and tracks it as a World entry so a later step can see it running and stop it. One Command is different: the **output Command**, what the model says to its user, which the model issues by writing a tag in its text ([Make your agent speak](#make-your-agent-speak)). The API in one place is [../reference/api.md](../reference/api.md#command); the design is [specs/commands.md](../../specs/commands.md).

## A function is a Command

```python
import asyncio


async def dance() -> str:
    """Perform a fun little dance. Takes about ten seconds."""
    await asyncio.sleep(10)
    return "Finished the dance."


wica.register_command(dance)
```

The function's name is the tool's name, its docstring the description the model reads, its signature the arguments schema: write them for the model. Return a string (or anything printable): it is the result the model observes. Sync and async functions both work ([Cancellation](#cancellation-reaches-the-task-not-always-the-work) says why async is preferred). Register before `start()`.

- **Names are unique**, and `noop`, `next` and `cancel_command` are reserved by WICA: a duplicate or reserved name raises `ValueError`.
- **`Command(...)`** is for the cases a bare function cannot express: `Command(fn, name="sum", description="…")` overrides the inferred metadata; `Command(existing_tool)` registers an off-the-shelf LangChain `BaseTool` unmodified; `Command(fn, triggers_on_completion=False)` is below.

## What happens when the model calls one

Every issued Command runs as an asyncio task on the Wica's loop and is tracked as a World entry **`agent:command:<call_id>`**, where `call_id` is a short id WICA numbers itself (`c1`, `c2`, … over the Agent's lifetime, never the provider's tool-call id), and whose value is a `CommandExecution` (`name`, `args`, `state`, `result`, `error`). It reads `running` while in flight and turns `complete`, `failed` (the function raised; the error text is the model's observation) or `cancelled`.

The model receives an immediate acknowledgement that the Command was **dispatched**, not its result: the result arrives later, when the entry turns terminal, as an observation in a new reasoning step. The runtime primer tells the model so. Two consequences:

- **A completion wakes the Agent** (the entry triggers), so the model sees what happened and can continue, chain a second action, or do nothing ([Staying silent](#staying-silent)). `Command(fn, triggers_on_completion=False)` turns this off for a Command whose outcome never deserves a step (setting an LED): the terminal entry is then observed by the next input-driven step.
- **All the Commands of one response start at once** and run concurrently, in no guaranteed order; WICA does not serialize them. A resource two Commands share (a speaker, a motor) is yours to guard with a lock inside the Commands, and sequencing is the model's job, which the primer tells it: issue the first, issue the next in a later step once the first is observed complete.

A long Command does not block the Agent: once a step has dispatched its Commands, the Agent is free to react to the next input while they run.

## Cancelling

The Agent auto-registers **`cancel_command(call_id)`**: the model can abort its own in-flight Command when it sees it still `running` in a later step (the person said "Stop!" while the dance runs). `stop()` and `close()` cancel every in-flight Command too. A cancelled Command's entry turns `cancelled`, and the completion wakes the Agent like any other. A running output is cancelled the same way, and wakes nothing ([Interruption](#interruption-and-what-was-heard)).

The transcript panel and the `on_agent_command` / `on_agent_command_ended` Events show every issuance and its end ([latency-and-traces.md](latency-and-traces.md)).

## Make your agent speak

Set one **output Command**: the function that receives what the model says to its user, whether it drives a voice, a chat bubble or a robot's mouth. By default the model speaks by writing a tag named after it in its response text (it can call it as a tool instead: [Tag or tool call](#tag-or-tool-call)):

```
<say emotion="happy">Hello there!</say>
```

Only the tag's body reaches your function. Text the model writes outside a tag is private: nobody hears it, and `wica.on_agent_private_text` reports it. The output Command is not a tool and is not in `wica.agent.commands`; speaking with the tag does not depend on the provider's tool calling. With no output Command set, no tag is taught and all the model's text is private, which suits an agent that only acts. The contract in full is [../reference/api.md](../reference/api.md#the-output-command).

Set it **before `start()`**: `set_output_command` recomposes the system prompt (the primer teaches the tag, with your docstring), and setting it after the first step rebuilds the system message and drops the cached prefix.

### The whole text, or the words as they are written

The first parameter receives the text, and its annotation chooses how. Annotated `str`, the function is called once the tag closes, with the whole text:

```python
async def say(text: str) -> str:
    """Speak out loud to the person in front of you."""
    await tts.speak(text)  # your voice
    return "Said it."


wica.set_output_command(say)
```

Annotated `OutputStream`, the function is called at the output's first words and receives them as the model writes them, so a voice can start before the response is complete. It must be `async def`:

```python
from wica import OutputStream


async def say(text: OutputStream) -> str:
    """Speak out loud to the person in front of you."""
    async for piece in text:  # ends when the tag closes
        await tts.feed(piece)
    return "Said it."
```

`text.text` is everything received so far and `await text.read()` waits for the end and returns the whole body. The return value is the execution's result, which the model can observe.

### Attributes

Every other parameter is a tag attribute, for what an utterance carries besides its words. Annotate it `str` or a `Literal` of strings and give it a string default:

```python
from typing import Literal

from wica import OutputStream


async def say(
    text: OutputStream, emotion: Literal["happy", "sad", "neutral"] = "neutral"
) -> None:
    """Speak out loud to the person in front of you."""
    await tts.speak(text, style=emotion)
```

The primer lists each attribute with its allowed values. An attribute the model omits, misspells or gives a value outside the `Literal` falls back to the default, and the text is still delivered. Keep attributes few: each is one more thing for the model to get right.

### Each utterance is a tracked execution

Each tag is its own output, with an `agent:command:<call_id>` entry like any Command: `running` from its first words, then `complete`, `failed` or `cancelled`, its `args` holding the attributes and the `text`. A response with several tags delivers them one after the other, in the order written. `on_agent_command` reports an output once its body is complete, and `on_agent_command_ended` its end.

The model is not obliged to tag: a response with text and no tag at all is delivered whole, as one output, when the response ends. It is not streamed, since until then a tag may still come.

### Interruption and what was heard

There is one voice. When a response speaks, every output still running or waiting from an earlier response is cancelled: the person says "Stop, please", the model answers "Okay", and the story it was telling is cut off. Other Commands are left alone, since speaking is not a request to stop acting. The model can also stop its own speech without saying anything, with `cancel_command` on the output's `call_id`.

A cancelled output raises `asyncio.CancelledError` in your function at its next `await`. Set `text.heard` there to what was actually delivered, and the model's history keeps that instead of the whole body, so the model does not believe it said what nobody heard:

```python
async def say(text: OutputStream) -> str:
    """Speak out loud to the person in front of you."""
    spoken: list[str] = []
    try:
        async for piece in text:
            await tts.speak(piece)
            spoken.append(piece)
    except asyncio.CancelledError:
        text.heard = "".join(spoken)  # what the person heard before the cut
        raise
    return "Said it."
```

A function that leaves `heard` unset keeps its whole text in history. An output of the same response that never started is removed from it.

### Continuing after speaking: the `<next>` note

The end of an output does not call the model again. When something remains to be done once it has finished speaking, the model writes a private note in the same response:

```
<say>Give me a moment.</say>
<next>look it up, then tell them what I found</next>
```

When the response's last output ends, a step runs and shows the model its note. `max_next_steps` (default `1`) caps how many steps in a row a note may wake; `0` turns the mechanism off and `null` removes the cap ([../reference/configuration.md](../reference/configuration.md#the-agent-block)). A cancelled output wakes nothing. A Command issued as a tool call in the same response still wakes a step on completion by its own `triggers_on_completion`, so "speak and act, then speak about the result" needs no note.

### Staying silent

An observation does not oblige a reaction. The model declines by writing `<noop/>` as its whole response; an empty response counts the same. Nothing is dispatched, no Command Event fires, and the reaction's record has `noop=True` ([latency-and-traces.md](latency-and-traces.md)). The primer allows it conservatively: a request or a question must still be answered. `noop` is not a Command; the name stays reserved.

### Tag or tool call

The same output Command can be bound to the model as a tool instead: the model then speaks by calling it, `say(text="Hello there!", emotion="happy")`, and declines by calling a `noop` tool bound beside it.

```python
wica.set_output_command(say, as_tool=True)
```

Which one is better depends on the model, measurably, in both directions: [text-output.md](text-output.md) gives what differs, what was measured on seven models, how to choose and how to compare them on yours.

## Cancellation reaches the task, not always the work

Cancelling cancels the asyncio task running the Command:

- **An `async` Command** unwinds cleanly at its next `await`; `finally` blocks run. Prefer `async` for anything that can run long enough to be worth cancelling.
- **A sync function** is offloaded to a thread, and Python cannot kill a thread: on cancel the entry turns `cancelled` immediately while the function runs to completion in the background, its result discarded. Make a long sync Command **cooperative** (poll a `threading.Event` and return early) if it must actually stop. Blocking work that cannot cooperate belongs in a subprocess you can terminate.

## Observing the World from a Command

A Command is ordinary code on the Wica's loop thread: it may read the World (`wica.world.get(key)`), update an entry (`wica.world.update(...)`, the system is running), start a span ([latency-and-traces.md](latency-and-traces.md)) and await anything. It must not call `wica.stop()` or `wica.close()`: that raises `RuntimeError`, since the loop thread cannot join itself.
