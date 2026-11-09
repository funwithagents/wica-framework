# Commands

A Command is the agent's unit of action on the world: speech, movement, a display change, an API call. You register functions; the Agent binds them as the model's native tool calls, runs each as a cancellable task, and tracks it as a World entry so a later step can see it running and stop it. The API in one place is [../reference/api.md](../reference/api.md#command); the design is [specs/commands.md](../../specs/commands.md).

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

- **Names are unique**, and `noop` and `cancel_command` are reserved by WICA: a duplicate or reserved name raises `ValueError`.
- **`Command(...)`** is for the cases a bare function cannot express: `Command(fn, name="sum", description="…")` overrides the inferred metadata; `Command(existing_tool)` registers an off-the-shelf LangChain `BaseTool` unmodified; `Command(fn, triggers_on_completion=False)` is below.

## What happens when the model calls one

Every issued Command runs as an asyncio task on the Wica's loop and is tracked as a World entry **`agent:command:<call_id>`**, whose value is a `CommandExecution` (`name`, `args`, `state`, `result`, `error`). It reads `running` while in flight and turns `complete`, `failed` (the function raised; the error text is the model's observation) or `cancelled`.

The model receives an immediate acknowledgement that the Command was **dispatched**, not its result: the result arrives later, when the entry turns terminal, as an observation in a new reasoning step. The runtime primer tells the model so. Two consequences:

- **A completion wakes the Agent** (the entry triggers), so the model sees what happened and can continue, chain a second action, or call `noop`. `Command(fn, triggers_on_completion=False)` turns this off for a Command whose outcome never deserves a step (setting an LED): the terminal entry is then observed by the next input-driven step.
- **All the Commands of one response start at once** and run concurrently, in no guaranteed order; WICA does not serialize them. A resource two Commands share (a speaker, a motor) is yours to guard with a lock inside the Commands, and sequencing is the model's job, which the primer tells it: issue the first, issue the next in a later step once the first is observed complete.

A long Command does not block the Agent: once a step has dispatched its Commands, the Agent is free to react to the next input while they run.

## Cancelling

The Agent auto-registers **`cancel_command(call_id)`**: the model can abort its own in-flight Command when it sees it still `running` in a later step (the person said "Stop!" while the dance runs). `stop()` and `close()` cancel every in-flight Command too. A cancelled Command's entry turns `cancelled`, and the completion wakes the Agent like any other.

The transcript panel and the `on_agent_command` / `on_agent_command_ended` Events show every issuance and its end ([latency-and-traces.md](latency-and-traces.md)).

## The output Command

By default the model replies as free text, which the output sink receives. Set an **output Command** and the model must address the user by calling it; its free text becomes private reasoning that still reaches the sink:

```python
async def say(text: str) -> str:
    """Speak out loud to the person in front of you."""
    await tts.speak(text)  # your voice
    return "Said it."


wica.set_output_command(say)
```

What this buys, compared with a sink that calls the voice itself: speech becomes a tracked, **cancellable** action. A new input arriving while `say` runs starts a step that sees `say … (running)` and can `cancel_command` it, which is barge-in; and the model decides per step whether to say anything at all.

- Set it **before `start()`**: it recomposes the system prompt (the primer's reply clause names the Command), and setting it after the first step rebuilds the system message and drops the cached prefix.
- Its name follows the uniqueness rule; its signature is yours (`say(text)` here), the model fills it from the description.
- By default its completion wakes the Agent (one more model call per utterance, so the model can chain or `noop`); pass `Command(say, triggers_on_completion=False)` to make speaking go straight to idle.

## `noop`

The Agent also auto-registers **`noop`**, a zero-argument Command the model calls to react to nothing: it creates no entry, spawns no task and never triggers. It is the terminator of the completion chain (the dance finished, there is nothing to add) and the way the model declines to react to an observation that does not concern it. The primer allows it conservatively: a request or a question must still be answered. `noop` is reported on `on_agent_command` like any issuance, so a UI can show that the agent chose not to react.

## Cancellation reaches the task, not always the work

Cancelling cancels the asyncio task running the Command:

- **An `async` Command** unwinds cleanly at its next `await`; `finally` blocks run. Prefer `async` for anything that can run long enough to be worth cancelling.
- **A sync function** is offloaded to a thread, and Python cannot kill a thread: on cancel the entry turns `cancelled` immediately while the function runs to completion in the background, its result discarded. Make a long sync Command **cooperative** (poll a `threading.Event` and return early) if it must actually stop. Blocking work that cannot cooperate belongs in a subprocess you can terminate.

## Observing the World from a Command

A Command is ordinary code on the Wica's loop thread: it may read the World (`wica.world.get(key)`), update an entry (`wica.world.update(...)`, the system is running), start a span ([latency-and-traces.md](latency-and-traces.md)) and await anything. It must not call `wica.stop()` or `wica.close()`: that raises `RuntimeError`, since the loop thread cannot join itself.
