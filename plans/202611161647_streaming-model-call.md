# Streaming the model call

**Status:** In progress

Implements [specs/agent.md](../specs/agent.md) ("Streaming the model call") and the "Streaming" section of [specs/fake-provider.md](../specs/fake-provider.md). Comes out of the analysis in [analysis/_streaming-model-call.md](../analysis/_streaming-model-call.md), option C (always `astream`, no knob).

**Why.** A cancelled non-streamed call reaches the server only if the server checks the connection on its own: `mlx_lm.server` does not, and runs the abandoned generation to its output cap; `llama-server` takes up to a second. A streamed call is noticed within a token on every server. The stream is also the transport that streaming output, a first-token measure, an idle deadline and model-call interruption need.

**Checked before writing this plan**, on the five providers' e2e configs with the Agent's own bindings, `ainvoke` against summed `astream`:

- usage is reported on a stream with each provider's defaults, the input count identical to `ainvoke`;
- the summed stream carries the same tool calls, arguments and ids, the forced `huggingface-hub` binding included;
- a guard retry showing a summed response is accepted by every provider and corrected, Anthropic with extended thinking included;
- a tool call cut by the output cap: invalid with `ainvoke` on OpenAI and Mistral, but **valid with truncated arguments** when streamed — LangChain's lenient partial-JSON parser closes it. Hence the strict parse below;
- the fast tier with the call swapped: two failures, both tests whose stubbed transport answers without streaming; the key-less e2e set passed.

## Scope

- **`wica.agent`**:
  - `_call_model` iterates `bound_model.astream(messages)` under the same deadline and span, summing the chunks; the sum goes through a new `_assemble(chunk) -> AIMessage`;
  - `_assemble` parses each tool-call chunk's raw arguments strictly (`json.loads`; empty → `{}`; not a JSON object → malformed), keeps the well-formed calls as `tool_calls` with the strictly parsed arguments, moves the malformed ones to `invalid_tool_calls` with the parse error, and returns a plain `AIMessage` with the content, usage, response metadata and id of the sum;
  - a WARNING log per malformed call (name and error);
  - nothing else in the step changes.
- **`wica.fake_model`**: `_astream` yielding the scripted step in fixed chunks (text split at word boundaries; each tool call as id + name + first half of its arguments, then the rest; usage on the last chunk), recording the call and validating names like `_agenerate`, the delay awaited before the first chunk.
- **Tests whose stub transport does not stream** move to streamed responses: the `huggingface-hub` forced-binding test (`tests/test_agent.py`), the `openai-chat-completions-compatible` stub server (`tests/test_wica.py`, an SSE body).
- **Docs**: [docs/guides/limits.md](../docs/guides/limits.md) drops "An abandoned model call may keep running on a self-hosted server"; [docs/guides/local-models.md](../docs/guides/local-models.md) — `mlx_lm.server` has one limit left (it ignores `tool_choice`), and "Cancellation" says both servers stop at once because the call is streamed; the "reply is complete text" limit says the call streams but the sink gets the whole text.

Not in scope (follow-ups, in [specs/agent.md](../specs/agent.md) "Future improvements"): streaming output to the sink, a first-token stamp, an idle deadline, an early cut of a runaway call, showing a malformed call to the guards.

## Tests

- **`_assemble`, over hand-built chunks**: text and tool-call chunks split across chunks sum to the same calls and arguments as a whole message; truncated arguments (`{"text": "Once upon`) → no tool call, one `invalid_tool_calls` entry, WARNING logged; empty arguments → `{}`; a JSON array → malformed; usage summed; the result is an `AIMessage`, its list content (a thinking block) kept.
- **Agent, through a streaming double** (`model=` override, `_astream` yielding scripted chunks):
  - a split tool call is dispatched once with the whole arguments; split text reaches the sink once, whole;
  - a truncated `say` under an output Command is not dispatched, and `output_command_required` retries;
  - usage from the last chunk lands in the reaction record;
  - an error raised after the first chunk ends `model_error`, nothing used; a stream that stalls past `model_timeout` after its first chunk ends `model_timeout`; a stream yielding nothing ends `model_error`.
- **Fake provider**: `astream` yields more than one chunk for a step with text and a tool call; the sum equals `ainvoke`'s message (text, calls, ids, usage); the call is recorded once and the cursor advances once; the delay precedes the first chunk and a cancel during it unwinds.
- The two moved stub tests keep their assertions (forced `tool_choice: "required"`; base URL, placeholder key, usage with cached tokens).

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- `uv run pytest tests-e2e -k "fake or example or tester"` — the key-less e2e set.
- The live set on all five providers, dispatched on the branch on CI (`gh workflow run CI --ref streaming-model-call`), since tool-call assembly is where providers differ.
- On `Done`: [specs/agent.md](../specs/agent.md) and [specs/fake-provider.md](../specs/fake-provider.md) back to `Implemented`.
