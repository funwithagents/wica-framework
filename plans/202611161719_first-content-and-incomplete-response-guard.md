# First-content stamp and the incomplete-response guard

**Status:** Done

Two follow-ups to [streaming the model call](202611161647_streaming-model-call.md). Implements the `first_content_at` stamp and its two measures in [specs/instrumentation.md](../specs/instrumentation.md) and [specs/agent.md](../specs/agent.md) ("Streaming the model call"), and `ModelResponse.malformed` / `truncated` with the `incomplete_response` default guard in [specs/agent.md](../specs/agent.md) ("Response guards").

**Why.**

- The stream makes "when did the answer begin to arrive" observable, and nothing records it. Splitting `model_latency` into the wait and the arrival tells a slow start from a long answer, and `generation_time` is the upper bound of what streaming output could save, the number to decide that feature with.
- A tool call cut by the output cap is dropped with a WARNING, or on some providers dispatched with the arguments that survived, and the model is never told. A guard turns it into a correction: call again, within a stated length.

**Measured before writing this plan**, a 200-word reply per provider, forced `say` against free text: free text arrives progressively on Anthropic, OpenAI, Mistral and the Hub; a tool call's arguments arrive progressively on OpenAI but in one burst at the end on Anthropic (3.9 s) and Mistral (6.0 s). Gemini was not measured (the key was over its spending cap). This corrects a sentence of the previous change's "Future improvements", and is why the idle deadline is not built: it would trip on a healthy forced call.

**The guard's feedback was tuned live**, a `say` cut by a 60-token cap through a real `Wica`: "keep them short" left every retry cut again; stating where a malformed call's arguments stopped and asking for half that recovered OpenAI and Mistral. Anthropic returns the cut call well-formed with empty arguments, so only its stop reason shows the cut, and there is no cut text to measure: the limit comes from the cap. Half the cap still overflowed (a tool call's framing costs about thirty tokens); a quarter recovered it. The Hub's reasoning model spends such a cap before any call and cannot recover.

## Scope

- **`wica.instrumentation`**: `ReactionTrace.first_content_at` (defaulted, so a record built by hand stays valid); the `time_to_first_content` and `generation_time` properties.
- **`wica.agent`**:
  - `_stream_response` reports the first piece carrying text or a tool call; `_call_model` stamps `first_content_at` (reset at each attempt, so the record holds the last attempt's) and adds a `wica.first_content` event to the model span;
  - `ModelResponse.malformed` built from the response's `invalid_tool_calls` (streamed or not), `truncated` from the provider's stop reason (`stop_reason: max_tokens`, `finish_reason: length` / `MAX_TOKENS`), `output_tokens` from its usage;
  - a malformed or truncated response is never shown in a retry, whichever guard rejected it.
- **`wica.guards`**: `MalformedCommand`; `ModelResponse.malformed`, `truncated`, `output_tokens` (defaulted); the `incomplete_response` guard, first of the defaults (a malformed call; a truncated response holding a call, or with an output Command bound); `output_command_required` leaves a malformed or truncated response to it, and `invalid_args` skips a truncated response's last call.
- **`wica.testing.report`**: the `time_to_first_content` and `generation_time` metrics.
- **`wica.contrib.gradio.reactions`**: both measures on the Instrumentation tab.
- **Docs**: the API reference's default-guard table; the latency guide's stamp and measures; the testing guide's metric list. **AGENTS.md**: the guards row.

Not in scope: the idle deadline (see above); streaming output; a benchmark summary column for the new measures (the per-run reports carry them).

## Tests

- **Guards alone**: `incomplete_response` rejects a malformed call, a truncated response holding a call and one cut before the output Command, each as a reminder with its length; accepts a whole response and cut free text when text is the reply; the other two guards stay silent about what it covers.
- **The stop reasons**: each provider's cap value marks a response truncated; the ordinary ones do not.
- **Agent, through the streaming double**:
  - a cut `say` is retried with the `incomplete_response` feedback, as a reminder (the retry prompt holds no rejected response); `on_response_rejected` carries the malformed call; with the guard removed the call is dropped and the rest of the response used;
  - text and a malformed call rejected by a guard asking to show the response is still retried as a reminder;
  - a well-formed call in a response cut at the cap is retried with the cap as the limit, `invalid_args` silent; cut free text without an output Command is delivered;
  - `first_content_at` falls between `model_started_at` and `model_ended_at`, after a leading chunk without content; it is the last attempt's after a retry; set when the stream then fails; `None` when nothing arrived; set at the end for a model with no streaming path.
- **Record**: the two properties over hand-built records, `None` without the stamp.
- **Report and panel**: the metrics appear for a reaction with the stamp; the panel shows both lines.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- `uv run pytest tests-e2e -k "fake or example or tester"` — the key-less e2e set.
- The live set on every provider with a working key, dispatched on the branch on CI.
- On `Done`: [specs/agent.md](../specs/agent.md) and [specs/instrumentation.md](../specs/instrumentation.md) back to `Implemented`.

## Result

All of the above clean: ruff, pyright (0 errors), the fast tier (598 passed), the key-less e2e set (14 passed), and CI on the pull request on every platform. The live set, dispatched on the branch (CI run 43909159062), passed on Anthropic, OpenAI, Mistral and the Hugging Face Hub. Gemini was not verified: its key was over its monthly spending cap, so its job could not complete; the stop-reason value it reports (`MAX_TOKENS`) is covered by a unit test only. On an earlier commit of the branch Mistral failed its `long_answer` step once (a long `say` outlasting the 30 s wait, no output cap involved) and passed on the rerun.

Through a real `Wica`, a `say` cut by a 60-token cap was retried and spoken whole on OpenAI, Mistral and Anthropic (3 of 3 each with the final feedback); the Hub's reasoning model spends such a cap before any call and ends `rejected`.
