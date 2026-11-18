# Speaking: a tag or a tool call

An agent that talks to a person has two kinds of output: what it **says** and what it **does**. Every tool-calling model already has a channel for doing, the tool call. For saying there are two candidates, and WICA supports both:

- **A tag in the model's text** (the default): `<say emotion="happy">Hello there!</say>`. Speech is the model's reply, as it was trained to write one.
- **A tool call** (`as_tool=True`): `say(text="Hello there!", emotion="happy")`. Speech is one more action, and every response must be a call.

It looks like a detail of syntax. It is not: across the models we measured, the choice moves the results of the same agent, on the same tests, by up to 31 tests out of 48, in **opposite directions for different models**. No one mode is better everywhere, so WICA keeps both, makes the tag the default, and lets the application choose per model, and measure it.

This page explains the two modes, what was measured, and how to choose. The contract of each is in [../reference/api.md](../reference/api.md#the-output-command); wiring the output Command is in [commands.md](commands.md#make-your-agent-speak).

## What is the same in both

Whichever the mode, speech in WICA is a **Command**, not a string returned at the end of a turn: each utterance is a tracked, cancellable execution with a World entry, it runs in the background while the agent keeps reasoning, a newer utterance cuts off the one still playing (one voice), and if it is cut off, the history keeps only what was heard. Your output Command is the same function in both modes:

```python
from typing import Literal


async def say(text: str, emotion: Literal["neutral", "happy", "sad"] = "neutral") -> None:
    """Speak out loud to the person in front of you."""


wica.set_output_command(say)  # the tag
wica.set_output_command(say, as_tool=True)  # the tool call
```

## What differs

| | Tag (default) | Tool call (`as_tool=True`) |
|---|---|---|
| The model speaks by | writing `<say …>…</say>` in its text | calling `say(text=…, …)` |
| It declines by | writing `<noop/>` | calling a `noop` tool |
| Text outside the speech | private | private |
| A reply with no speech in it | plain text is delivered as speech | rejected and retried (`output_command_required`) |
| First words reach your function | as the model writes them (with an `OutputStream`) | once the whole response is complete |
| Speaking, then continuing | only if the model asks, with a `<next>` note | every output's end wakes a step, where it speaks again or calls `noop`, up to `max_next_steps` |
| Depends on the provider's `tool_choice` | no | yes, to force the call (`force=False` to not force) |
| Servers that ignore `tool_choice` (`mlx_lm.server`, Ollama) | work | not helped: the model may answer in text nobody hears |

The trade is visible in the table. The tag lets the model write as it was trained to, gets the first word out as soon as it is written and needs nothing from the server. The tool call puts saying and doing in the same channel and forces a decision at every step: speak, act, or call `noop`.

## What was measured

Seven models on WICA's live regression gate: a test robot playing seven behaviours (greeting, act on request, stay quiet on a background sound, write a long note, be interrupted, remember a name, goodbye), as seven focused tests and as one conversation, three runs each, so 48 tests per model. Measured in November 2026, on the tag output's prototype against the tool call as it was then (where every output's end woke a step, without the cap). Samples are small: large differences are reliable, differences of one or two tests are not.

| Model | Tool call | Tag |
|---|---|---|
| claude-haiku-4-5 | 48 | 48 |
| gpt-4o | 47 | 45 |
| gpt-oss-120b (Cerebras) | 44 | 46 |
| ministral-14b | **47** | 37 |
| gpt-4o-mini | 35 | **44** |
| ministral-3b | 33 | **39** |
| Qwen2.5 7B on Ollama | 5 | **36** |

What the failures were:

- **With the tag, some models describe actions instead of taking them.** Asked to dance, ministral-14b answered "Watch this!" and never called `dance`. Free to write text, it prefers words to any tool; forcing a call hides that.
- **With the tag, some models stay quiet less.** On a background sound nobody addressed, the share of reactions that stayed silent (out of 6) fell from 6 to 1 on ministral-14b and from 6 to about 2 on gpt-4o-mini. When a call is forced, `noop` is the easy answer; with free text, speaking is.
- **With the tool call, some models run away.** gpt-4o-mini lost a whole run to chained `say` calls after a greeting: when every utterance's end wakes the model, a model that always has something more to say never stops. The cap (`max_next_steps`) now bounds this.
- **With the tool call, a server that ignores `tool_choice` cannot be forced.** Qwen2.5 7B on Ollama answered in plain text no one heard, 5 tests of 48; with the tag, the same text is speech.
- **With the tag, the first words arrive earlier.** Median time from the start of the model call to the first words, on short replies: 0.61 s against 0.91 s on claude-haiku-4-5, 0.59 against 0.91 on ministral-14b, 0.63 against 0.96 on gpt-4o-mini, and the gain grows with the length of the reply. The tool call can only be delivered once its arguments are complete.

WICA's own CI now runs ministral-14b with the tool call and gpt-oss-120b in both modes ([below](#how-wica-tests-both)). On the current code, three runs each: ministral-14b with the tool call passed 52 of 54, gpt-oss-120b with the tool call 52 of 54.

## How to choose

- **Start with the tag.** It is the default, it streams, and it works on every server.
- **Try the tool call when the model talks about acting instead of acting**, or speaks when it should stay quiet. Those are the failures the forced call removes.
- **Keep the tag on a server that ignores `tool_choice`** (`mlx_lm.server`, Ollama: [local-models.md](local-models.md)), and when the first words must come fast.
- **`force=False`** binds the tool without forcing it, for Anthropic extended thinking (whose API rejects a forced choice) or a server that rejects `tool_choice`. Unforced, a model may answer in plain text, which the `output_command_required` guard rejects and asks again.
- **Write the persona for neither mode.** The runtime primer teaches the mode: a tag and its attributes, or the tool and `noop`. A persona that says "call `say`" pushes a tag model to a tool call it does not have, and one that shows `<say>` examples pushes a tool model to write tags nobody hears. Say "speak", "say it aloud", "reply briefly" and let the primer say how. If the persona must describe speaking, word that part for the mode in use.
- **Measure on your agent.** Your Commands, your persona and your scenarios decide more than these numbers.

## Compare them on your models

The mode is part of the application's wiring, not of the Agent's config, so it is not a field of `AgentConfig`. In a benchmark ([benchmarking.md](benchmarking.md)), give the matrix a row per mode and let your `Target` read it: any key a matrix entry does not define lands in `entry.extra`.

```json
{"id": "ministral-14b", "tags": ["output-tag"], "output": "tag", "agent": {"provider": "mistral", "model": "ministral-14b-latest", "api_key_env": "WICA_MISTRAL_API_KEY", "system_prompt_file": "persona.md"}},
{"id": "ministral-14b-tool", "tags": ["output-tool"], "output": "tool", "agent": {"provider": "mistral", "model": "ministral-14b-latest", "api_key_env": "WICA_MISTRAL_API_KEY", "system_prompt_file": "persona.md"}}
```

```python
def wire_output(wica: Wica, entry: Entry) -> None:  # called from your Target's open()
    as_tool = entry.extra.get("output") == "tool"
    wica.set_output_command(say, as_tool=as_tool)
```

The summary then shows the two rows of each model side by side: pass rate, pass^k, latencies and token costs.

## How WICA tests both

The live regression gate runs each committed model in one mode, the one an application should use it in, and one model in both. Each e2e config states its mode beside its Agent block (`"output": "tag"` or `"tool"`): Anthropic, OpenAI and Gemini run the tag, Mistral the tool call, and the Hugging Face Hub's gpt-oss-120b both, so a regression in either mode shows on the same model ([specs/testing.md](../../specs/testing.md), "E2e configs").

## How others answer it

Frameworks have answered the same question both ways, and some have changed their answer:

- **Speech as a tool, then not.** MemGPT gave the model a function to send a message and a flag on any function call to ask for another inference step ("heartbeats", [the MemGPT paper](https://arxiv.org/abs/2310.08560), section 2.4). Letta, its successor, deprecated both in its v1 agent: the reply is the model's own message again, not a tool call, to stay close to the multi-step tool calling today's models are trained on ([Letta v1](https://www.letta.com/blog/letta-v1-agent)). WICA's tool mode is close to that earlier design; its `<next>` note is a heartbeat the model asks for in its text, capped.
- **Speech as the model's text.** Voice pipelines send the model's text to the voice: Pipecat places its text processors after the LLM and before the TTS ([Pipecat](https://docs.pipecat.ai/server/utilities/text/pattern-pair-aggregator)), LiveKit Agents' TTS node converts the LLM output into speech ([LiveKit](https://docs.livekit.io/agents/build/nodes/)), and Hugging Face's speech-to-speech pipeline sends the LLM's streamed text to its TTS ([speech-to-speech](https://github.com/huggingface/speech-to-speech)).
- **Tags in the text for how to say it.** Pipecat parses tag pairs such as `<voice>…</voice>` out of the streaming text to switch voices, and LiveKit injects the TTS provider's markup guide into the prompt so the model writes delivery tags. WICA's tag attributes (`emotion="happy"`) are the same idea, with the allowed values read from your function's signature.
- **Channels in one stream.** OpenAI's harmony format for gpt-oss separates the model's chain of thought (`analysis`), function calls (`commentary`) and the message meant for the user (`final`) ([harmony](https://developers.openai.com/cookbook/articles/openai-harmony)): the same split WICA makes between private text, Commands and speech.
- **Tool calls written as text.** Hugging Face's speech-to-speech parses tool calls out of a local model's text (`<code>name(...)</code>`, `<tool_call>{…}</tool_call>`), partly to keep them from being spoken aloud. The failure is real in WICA too: with the tag, Qwen2.5 7B sometimes writes `dance()` in its speech.

None of the projects above publishes the two measured side by side on the same agent. WICA keeps both and makes the comparison one more benchmark row.

## Going further

- The contract of each mode: [../reference/api.md](../reference/api.md#the-output-command) (the tag) and [../reference/api.md](../reference/api.md#the-output-command-as-a-tool-call) (the tool call).
- The design and its rationale: [specs/agent.md](../../specs/agent.md), "Output: tags in the model's text" and "The output Command as a tool call".
