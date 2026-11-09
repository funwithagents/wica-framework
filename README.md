# WICA

**WICA is an agentic framework for agents that take multimodal inputs and produce multimodal outputs**, such as a social robot that hears, sees and speaks, or an assistant that watches sensors and acts on them.

How it works:

- **The World** holds the agent's state: a registry of typed entries, rendered into the model's prompt at every reasoning step.
- **Inputs** bring perceptions in: speech, a camera frame or a sensor event updates a World entry.
- **The Agent** wakes when the World changes, reasons over it, and decides what to do.
- **Commands** carry its actions out: speech, movement, an API call, each tracked back in the World.

What sets it apart:

- **The prompt is a view of state, not a transcript.** You keep a World of typed entries up to date, and every reasoning step renders it into the model's prompt.
- **Actions run as tasks the model can see and stop.** Commands are asynchronous and cancellable, and their state is part of the World the model reads.

**Start here:**

- [docs/getting-started.md](docs/getting-started.md): your first agent on the scripted `fake` provider, then on a real model.
- [docs/index.md](docs/index.md): the documentation, by task.
- On this page: [what WICA gives you](#what-wica-gives-you), the [quick start](#quick-start), the [conversation demo](#the-conversation-demo).

The name is the model:

| Letter | Pillar | What it is |
|---|---|---|
| **W** | [World](specs/world.md) | The store of current state and context: a registry of typed entries that renders to a prompt |
| **I** | [Inputs](specs/inputs.md) | External multimodal data entering the World (speech, a camera frame, a sensor event) |
| **C** | [Commands](specs/commands.md) | The agent's unit of action on the world: asynchronous, cancellable |
| **A** | [Agent](specs/agent.md) | The reasoning loop that observes the World and issues Commands |

## Project status

> [!WARNING]
> **Under development: the API may change.** WICA is at version 0.1 and is being shaped as it is used. Classes, config fields and module layout can change between commits, without a deprecation period. If you depend on it, pin a commit (`wica[anthropic] @ git+https://github.com/funwithagents/wica-framework@<sha>`). The v1 Agent runs one reasoning step at a time and replies in complete text; the concurrency and streaming model is designed and deferred ([docs/guides/limits.md](docs/guides/limits.md)).

> [!TIP]
> **Feedback is welcome: open an issue.** Bug reports, questions, reports from models the author does not test on, and what you would want the API or the config to do differently all go to [the issue tracker](https://github.com/funwithagents/wica-framework/issues). The project is developed by one person and is not taking pull requests yet; [CONTRIBUTING.md](CONTRIBUTING.md) says why and how to help anyway.

## What WICA gives you

| Feature | What you get |
|---|---|
| **State-first, not chat-first** | You maintain a World of named, typed entries (what the agent heard, who is nearby, what it is doing) and every reasoning step renders the relevant state into the prompt. Each entry is versioned and timestamped, deep-copied on every write and read, and rendered inside an `<entry key="…" id="…">` block the model can reference precisely; an entry can stay out of the prompt, render differently once it is history, or expire ([docs/guides/world-and-inputs.md](docs/guides/world-and-inputs.md)) |
| **Reactive by construction** | Marking an entry `triggers_llm_call=True` is all it takes to wake the Agent when a perception arrives; `update()` is callable from any thread. Bursts are coalesced into one step within a 200 ms window; an entry can bypass the window for a stop button or a barge-in |
| **Multimodal in** | Inputs serialize through a neutral `Content` model (`TextPart`, `ImagePart`): a camera frame can be the inline image while fresh and a line of text afterwards, keeping the deep, cached part of the prompt small and stable ([specs/content.md](specs/content.md)) |
| **Actions that run, and can be stopped** | A Command is a plain sync or async function (or an off-the-shelf LangChain tool) the Agent binds as a native tool call. Each runs as a cancellable task tracked as a World entry that reads `running` then `complete`, `failed` or `cancelled`, so a later step can see an action in flight and cancel it, by the model's own `cancel_command`. All the Commands of one response run concurrently ([docs/guides/commands.md](docs/guides/commands.md)) |
| **Cancellable speech** | Set an output Command (your voice) and the model must address the user through it, its free text turning into private reasoning: speaking becomes a tracked action a new input can interrupt, and the model decides per step whether to say anything, or to `noop` |
| **Provider-agnostic** | Anthropic, OpenAI, Gemini, Mistral or the Hugging Face Hub is a config edit; LangChain is used only at the model-facing edge, and nothing else touches a provider SDK. A built-in scripted `fake` provider drives the whole loop with no key and no network ([docs/reference/configuration.md](docs/reference/configuration.md#providers)) |
| **Bounded history** | History is re-renderable World snapshots, unbounded by default; `history_reactions` keeps the last N reactions and `history_budget` keeps the prompt under a fraction of the model's context window, using the usage each call reports, with hysteretic cuts that keep the cached prefix stable |
| **Observable** | Eight Events surface every trigger, dropped trigger, rendered prompt, issued Command, Command end, text and reaction record; OpenTelemetry spans join your own components' traces; three Gradio panels show the World, the prompts and the transcript live ([docs/guides/latency-and-traces.md](docs/guides/latency-and-traces.md), [docs/guides/gradio-panels.md](docs/guides/gradio-panels.md)) |
| **Testable, measurable, comparable** | The `fake` provider pins exact Commands and utterances in your tests; `wica.testing` probes a running agent, waits for what you expect and reports its latencies and costs; `wica.benchmark` runs scripted sessions across a matrix of models ([docs/guides/testing.md](docs/guides/testing.md), [docs/guides/benchmarking.md](docs/guides/benchmarking.md)) |

## Install

Python 3.12+. The package is not on PyPI; add it from its git URL with the extra of the provider you use. Core `wica` bundles no model provider:

```
uv add "wica[anthropic] @ git+https://github.com/funwithagents/wica-framework"
```

| Extra | Adds | You need it for |
|---|---|---|
| *(none)* | `langchain`, `langchain-core`, `opentelemetry-api` | The framework and the scripted `fake` provider |
| `anthropic` | `langchain-anthropic` | `provider: "anthropic"` |
| `openai` | `langchain-openai` | `provider: "openai"` |
| `google-genai` | `langchain-google-genai` | `provider: "google_genai"`, Gemini through the Gemini Developer API |
| `mistralai` | `langchain-mistralai` | `provider: "mistralai"`, Mistral through La Plateforme |
| `huggingface-hub` | `langchain-huggingface` | `provider: "huggingface-hub"`, the Hub's serverless Inference Providers |
| `gradio` | `gradio` | The three observability panels of `wica.contrib.gradio` |

Selecting a provider whose extra is not installed fails at `Wica.init` with an `ImportError` naming it. **Platforms:** macOS on Apple silicon, Linux (x86_64, arm64) and Windows; WICA is pure Python and CI runs the fast tier on all four.

## Quick start

Everything goes through one object, `Wica`. You build it from a config, wire an output, register the World entries and Commands, start it, and feed it perceptions; it runs on its own thread and the reply reaches your sink:

```python
import threading

from wica import AgentConfig, TextPart, Wica, WicaConfig

config = WicaConfig(
    agent=AgentConfig(
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key_env="WICA_ANTHROPIC_API_KEY",  # read when Wica.init builds the Agent
        system_prompt="You are a friendly social robot.",
    )
)
replied = threading.Event()


async def speak(text: str) -> None:
    """The output sink: receives the model's reply, once per reasoning step."""
    print("robot says:", text)
    replied.set()


wica = Wica.init(config)  # owns the event loop, the World and the Agent
wica.set_output_sink(speak)  # output wiring comes after init, before start
wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, previous: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,  # a new value wakes the Agent: this entry is an Input
)
wica.start()

wica.world.update("speech_input", "Hello robot!")  # a perception arrives, from any thread
replied.wait(timeout=30)  # the Agent wakes, reasons and replies on its own loop
wica.close()
```

Swap the `AgentConfig` for `provider="fake"` and a `script` in `model_kwargs` and the same program runs with no key and no network, the model answering as scripted. The complete first agent, with a Command the robot performs, the lifecycle in a host of your own and the config in a JSON file, continues in [docs/getting-started.md](docs/getting-started.md).

## The conversation demo

The repo ships a browser UI to talk to a simulated social robot and watch the framework work: the conversation with the robot's actions as they run, the live World state, and the exact prompt and timing of every reasoning step, with sensor-input buttons for the perceptions a robot would have. It runs from a checkout with the `demo` dependency group:

```
export WICA_ANTHROPIC_API_KEY=sk-...
uv run --group demo python -m examples.conversation_demo.app
```

then open the address Gradio prints. Without a key it still opens and is explorable; it just cannot run the robot's reasoning. The provider is a config edit: `examples/conversation_demo/agent.config.json`, with one committed variant per provider next to it. Its three panels are reusable in your own app ([docs/guides/gradio-panels.md](docs/guides/gradio-panels.md)); its design is [specs/conversation-demo.md](specs/conversation-demo.md).

## Documentation

Start at [docs/index.md](docs/index.md), the documentation by task. The pages a consumer uses most:

| | |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Your first agent on the scripted provider, then on a real model; the lifecycle in a host of your own |
| [docs/reference/api.md](docs/reference/api.md) | What you import, the `Wica` surface, the World API, `Command`, the eight Events, the errors, the contracts |
| [docs/reference/configuration.md](docs/reference/configuration.md) | Every config field, the loaders, the API key and prompt rules, the history bounds, the providers |
| [docs/guides/](docs/index.md#by-task) | The World and its Inputs, Commands, the Gradio panels, testing, latency and traces, benchmarking, limits, troubleshooting |

## How it is developed

WICA is a spec-driven project: every feature is designed before it is coded, pinned by tests at two levels, and checked by CI on every change.

- **Specs first.** Each concept has one spec under [specs/](specs/), the normative design with its rationale and its open questions, indexed by [specs/_index.md](specs/_index.md) with a status (`Draft`, `Stable`, `Implemented`, …). An implementation plan under [plans/](plans/_index.md) turns a settled spec into code. Each spec names the code and tests it governs, and a test keeps that map, the statuses and the documentation's links and examples honest.
- **Two test tiers.** `tests/` is fast and deterministic: no network, no key, every flow on the scripted `fake` provider. `tests-e2e/` holds the always-run scripted full-loop flows (the framework, the demo, the tester, the benchmark) and the live set, one config per provider, each skipping when its key is unset ([specs/testing.md](specs/testing.md)).
- **CI on every change.** The static gate, the fast tier on Linux (x86_64 and arm64), macOS and Windows, the scripted flows on every pull request, and the live set on pushes to `main` ([specs/ci.md](specs/ci.md)).
- **Measured, not assumed.** The reaction latencies and token costs come from the framework's own instrumentation; the [benchmark](benchmarks/README.md) runs the demo against a matrix of models over scripted sessions, and keeps its results with what was concluded from them.
- **[AGENTS.md](AGENTS.md) is the operating manual** for anyone, or any coding agent, working in the repository: the project map (which module does what, with its spec), the verification gate, how to run the live tests per provider, and how the specs and statuses are kept in step with the code.

The same gate, locally (Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)):

```
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest                  # the fast tier
uv run pytest tests-e2e -k fake   # the scripted full-loop flows, no key
uv run pytest tests-e2e           # the live set: each provider whose key is set runs, the rest skip
```

## License

MIT, see [LICENSE](LICENSE).
