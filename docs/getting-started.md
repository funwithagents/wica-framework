# Getting started

Your first agent on WICA, then the config edit that takes the same program to a real model. The API's contracts in full are in [reference/api.md](reference/api.md); every config field in [reference/configuration.md](reference/configuration.md).

## Install

Python 3.12+, in a Python project of your own (`uv init my-agent && cd my-agent` creates one). The package is not on PyPI; add it from its git URL with the extra of the model provider you will use. Core `wica` bundles no provider:

```
uv add "wica[anthropic] @ git+https://github.com/funwithagents/wica-framework"
```

The extras are `anthropic`, `openai`, `gemini`, `mistral`, `huggingface-hub` and `openai-compatible` (a model you host) ([reference/configuration.md](reference/configuration.md#providers)); several can be combined (`wica[anthropic,openai]`), and `wica[gradio]` adds the browser panels of [guides/gradio-panels.md](guides/gradio-panels.md). The API changes between commits without a deprecation period while WICA is at 0.1: pin a commit (`@<sha>`) and lock your project's own resolved graph.

## Your first agent

Everything goes through one object, `Wica`. This program runs on the `fake` provider, a scripted model that ships with WICA and needs no key, no extra and no network: its `script` says what the model answers at each reasoning step, so you see the whole loop turn before you spend a token. It registers one **Input** (what the person said), one **Command** (a dance), and the **output Command** (`say`), the function that receives what the model says to the person:

```python
import threading

from wica import AgentConfig, TextPart, Wica, WicaConfig

config = WicaConfig(
    agent=AgentConfig(
        provider="fake",  # a scripted model: no key, no network
        model="scripted",
        system_prompt="You are a friendly social robot.",
        model_kwargs={
            "delay_s": 0,
            "script": [
                {"tool_calls": [{"name": "dance", "args": {}}]},  # step 1: act
                {"text": "<say>Hello! I just danced for you.</say>"},  # step 2: speak
            ],
        },
    )
)

replied = threading.Event()


async def say(text: str) -> None:
    """Speak out loud to the person in front of you."""  # taught to the model
    print("robot says:", text)  # your voice goes here
    replied.set()


async def dance() -> str:
    """Perform a fun little dance."""  # the docstring is the tool description
    print("robot dances")
    return "Finished the dance."


wica = Wica.init(config)  # owns the event loop, the World and the Agent
wica.set_output_command(say)  # the model speaks by writing <say>…</say> in its text
wica.register_command(dance)
wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, previous: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,  # a new value wakes the Agent: this entry is an Input
)
wica.start()

wica.world.update("speech_input", "Hello robot!")  # a perception arrives
replied.wait(timeout=10)  # the system runs on its own thread; wait for the speech
wica.close()
```

Save it as `first_agent.py` and run it:

```
uv run python first_agent.py
```

It prints:

```
robot dances
robot says: Hello! I just danced for you.
```

What happened, in order. `Wica.init` built a **World** (a registry of typed entries) and an **Agent** (the reasoning loop) on one event loop that `start()` runs in a daemon thread. `update()` wrote a new value into the `speech_input` entry; because that entry is registered with `triggers_llm_call=True`, the Agent woke, rendered the World into a prompt (the `serialize_fn` turned the value into the text the model reads) and called the model. Step 1 of the script answered with a `dance` tool call: the Agent dispatched it as a cancellable task and tracked it as a World entry of its own, `agent:command:<call_id>`, which reads `running` and then `complete`. That completion is itself an observation, so the Agent woke again; step 2 wrote a `<say>` tag in its text, and the tag's body went to `say`, as a tracked execution of its own. Nothing else happened: the end of the speech wakes no further step, and the Agent waits for the next input.

Three things to notice. **The prompt is a view of state, not a transcript you append to.** You maintain World entries; every reasoning step renders the current World, and the Agent keeps earlier steps as history. **A Command's outcome arrives later, as an observation.** The model gets an immediate acknowledgement that the Command was dispatched, then sees the result when the entry turns terminal, which is how it can also see an action still running and cancel it ([guides/commands.md](guides/commands.md)). **The model speaks in its text, not through a tool.** With an output Command set, the model is taught a tag of the same name: only the text inside `<say>…</say>` reaches `say`, and anything it writes outside is private. `say` takes the whole text here (`text: str`); annotated `OutputStream`, it receives the words as the model writes them ([guides/commands.md](guides/commands.md#make-your-agent-speak)).

## Running it on a real model

Swap the `AgentConfig` for a real provider and the same program drives a live model; the script is gone, so the robot answers as it likes. The config is plain data, so it can also live in a JSON file next to your prompt ([reference/configuration.md](reference/configuration.md)):

```python
config = WicaConfig(
    agent=AgentConfig(
        provider="anthropic",
        model="claude-haiku-4-5",
        system_prompt="You are a friendly social robot.",
    )
)
```

The key is read from `ANTHROPIC_API_KEY`, the provider's usual variable, when `Wica.init` builds the Agent (`api_key_env` names another); a config never holds the key, so it is safe to commit, and `Wica.init` raises `MissingSecretError` when the variable holds none. The extra for the provider must be installed, or `Wica.init` raises `ImportError` naming it. WICA appends a short **runtime primer** to your system prompt that tells the model how to read observations, that Commands run asynchronously and concurrently, how to speak (the output Command's tag), and that it may write `<noop/>` to not react; your persona stays verbatim before it.

## The lifecycle, in a host of your own

`Wica.init` builds, the `set_*` and `register_*` calls wire, `start()` runs. The order within the wiring does not matter, but wire before you start: `set_output_command` recomposes the system prompt, and setting it after the first step rebuilds the message and drops the model's cached prefix.

```python
wica = Wica.init(config)
wica.set_output_command(say)
wica.register_command(dance)
wica.world.register("speech_input", str, serialize_fn=render, triggers_llm_call=True)
wica.start()  # the loop runs in a daemon thread from here
...
wica.stop()  # a pause: registrations, values and history are kept
wica.start()  # resumes the same system
wica.close()  # terminal: a closed Wica cannot restart
```

- **`update()` is callable from any thread**, a UI callback, a sensor poll loop or a hardware interrupt, but only while the system is running: between `start()` and `stop()` it hops to the shared loop internally; outside, it raises `RuntimeError`. `register`, `get` and the Event subscriptions work at any time.
- **Your callbacks run on the Wica's loop thread**: the output Command, the Commands, the Event subscribers. Never call `stop()` or `close()` from one of them (a thread cannot join itself; the call raises `RuntimeError`); hand it to another thread, or do it after the server or the main loop returns, as the demo's `main()` does.
- **A host with its own asyncio loop** passes it to `Wica.init(config, loop=my_loop)`; WICA then creates no thread and leaves running the loop to you.
- **To reset rather than resume**, close this Wica and `init` a new one: `stop()`/`start()` keeps everything.

## Where next

- [guides/world-and-inputs.md](guides/world-and-inputs.md): what a World entry can carry, how it renders, what wakes the Agent, how bursts are coalesced.
- [guides/commands.md](guides/commands.md): Commands in depth, the output Command (streamed, cancellable speech), staying silent, what runs concurrently.
- [guides/gradio-panels.md](guides/gradio-panels.md): watch the World, the prompts and the transcript in a browser.
- [guides/testing.md](guides/testing.md): the `fake` provider in your own tests, and the harness that measures a running agent.
- [guides/limits.md](guides/limits.md): what v1 does not do yet.
