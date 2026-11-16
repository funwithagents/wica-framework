# First-content stamp and the malformed-command guard

**Status:** In progress

Two follow-ups to [streaming the model call](202611161647_streaming-model-call.md). Implements the `first_content_at` stamp and its two measures in [specs/instrumentation.md](../specs/instrumentation.md) and [specs/agent.md](../specs/agent.md) ("Streaming the model call"), and `ModelResponse.malformed` with the `malformed_command` default guard in [specs/agent.md](../specs/agent.md) ("Response guards").

**Why.**

- The stream makes "when did the answer begin to arrive" observable, and nothing records it. Splitting `model_latency` into the wait and the arrival tells a slow start from a long answer, and `generation_time` is the upper bound of what streaming output could save, the number to decide that feature with.
- A tool call cut by the output cap is dropped with a WARNING and the model is never told. A guard turns it into a correction: call again, shorter.

**Measured before writing this plan**, a 200-word reply per provider, forced `say` against free text: free text arrives progressively on Anthropic, OpenAI, Mistral and the Hub; a tool call's arguments arrive progressively on OpenAI but in one burst at the end on Anthropic (3.9 s) and Mistral (6.0 s). Gemini was not measured (the key was over its spending cap). This corrects a sentence of the previous change's "Future improvements", and is why the idle deadline is not built: it would trip on a healthy forced call.

## Scope

- **`wica.instrumentation`**: `ReactionTrace.first_content_at` (defaulted, so a record built by hand stays valid); the `time_to_first_content` and `generation_time` properties.
- **`wica.agent`**:
  - `_stream_response` reports the first piece carrying text or a tool call; `_call_model` stamps `first_content_at` (reset at each attempt, so the record holds the last attempt's) and adds a `wica.first_content` event to the model span;
  - `ModelResponse.malformed` built from the response's `invalid_tool_calls` (streamed or not);
  - a response holding a malformed call is never shown in a retry, whichever guard rejected it.
- **`wica.guards`**: `MalformedCommand`; `ModelResponse.malformed` (defaulted); the `malformed_command` guard, first of the defaults; `output_command_required` leaves a response whose only call is malformed to it.
- **`wica.testing.report`**: the `time_to_first_content` and `generation_time` metrics.
- **`wica.contrib.gradio.reactions`**: both measures on the Instrumentation tab.
- **Docs**: the API reference's default-guard table; the latency guide's stamp and measures; the testing guide's metric list. **AGENTS.md**: the guards row.

Not in scope: the idle deadline (see above); streaming output; a benchmark summary column for the new measures (the per-run reports carry them).

## Tests

- **Guards alone**: `malformed_command` rejects a response with a malformed call as a reminder, naming the Command, and accepts one without; `output_command_required` accepts a response whose only call is malformed and still rejects free text only.
- **Agent, through the streaming double**:
  - a cut `say` is retried with the `malformed_command` feedback, as a reminder (the retry prompt holds no rejected response); `on_response_rejected` carries the malformed call; with the guard removed the call is dropped and the rest of the response used;
  - text and a malformed call rejected by a guard asking to show the response is still retried as a reminder;
  - `first_content_at` falls between `model_started_at` and `model_ended_at`, after a leading chunk without content; it is the last attempt's after a retry; set when the stream then fails; `None` when nothing arrived; set at the end for a model with no streaming path.
- **Record**: the two properties over hand-built records, `None` without the stamp.
- **Report and panel**: the metrics appear for a reaction with the stamp; the panel shows both lines.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- `uv run pytest tests-e2e -k "fake or example or tester"` — the key-less e2e set.
- The live set on every provider with a working key, dispatched on the branch on CI.
- On `Done`: [specs/agent.md](../specs/agent.md) and [specs/instrumentation.md](../specs/instrumentation.md) back to `Implemented`.
