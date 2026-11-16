# Response guards

**Status:** In progress

Implements [specs/agent.md](../specs/agent.md) ("Response guards"), the `max_retries` field of [specs/config.md](../specs/config.md), the retry fields, `rejected` outcome, `on_response_rejected` Event and `wica.agent.guards` span of [specs/instrumentation.md](../specs/instrumentation.md), and the facade members of [specs/wica.md](../specs/wica.md).

**Why.** A model's response is used as soon as it returns: an unknown Command or bad arguments surface only after dispatch, as a `failed` Command entry, and a free-text reply while an output Command is bound reaches no one when the call is not forced. Checking the response before it is used, and asking again within the same reaction, fixes the mistake before it has any effect — and keeps the failed attempt out of history, so later prompts never show it.

## Scope

- **`wica.guards`** (new module, not re-exported from `wica`): `ProposedCommand`, `ModelResponse`, `Retry`, `ResponseRejection`, the `ResponseGuard` type alias, the three default guards (`unknown_command`, `invalid_args`, `output_command_required`) as plain functions over `ModelResponse`, and the feedback frames (the "rejected, not executed" `tool_result`, the show and reminder frames) as module constants. `invalid_args` validates through `args_schema.model_validate` when the backing tool's `args_schema` is a pydantic model, and skips a raw JSON-schema dict.
- **`wica.config`**: `AgentConfig.max_retries: int = 1` — a non-negative integer; the loaders default an absent key to `1`, reject a bool, a negative number or a non-integer with the key named.
- **`wica.instrumentation`**: `ReactionOutcome` gains `"rejected"`; `ReactionTrace` gains `attempts`, `rejections`, `guard_duration`; `usage` sums over attempts; the `guard_time` measure.
- **`wica.agent`**:
  - guards installed at construction in run order; `add_response_guard` / `remove_response_guard` / `response_guards`; a step reads them once at its start;
  - in `_run_step`, the model call becomes an attempt loop: call (under its own `model_timeout`) → build the `ModelResponse` from the `AIMessage` and the step's bound Commands → run every guard (awaiting awaitables; a raising guard logged at ERROR and counted as accepting) inside a `wica.agent.guards` span → on rejection, combine the feedback, emit `on_response_rejected`, log at WARNING, and either build the retry messages from the step's base messages (show: the raw `AIMessage` + one fixed `tool_result` per tool call + a feedback user message; reminder, or an empty response: a text block appended to the newest observation's user message) or, out of retries, end the reaction `rejected` with an ERROR log;
  - the accepted response goes through the existing path unchanged (history, `on_text`, sink, dispatch); `on_prompt` stays once per reaction, before the first attempt; the Anthropic cache breakpoint is applied to the base messages only;
  - `_apply_history_budget` reads the first attempt's `input_tokens`;
  - `on_response_rejected` Event; `wica.attempts` on the reaction span.
- **`wica.wica`**: `add_response_guard` / `remove_response_guard` / `response_guards` delegated, `on_agent_response_rejected` surfaced.
- **`wica.contrib.gradio.reactions`**: the Instrumentation tab shows attempts and each rejection's feedback when there was a retry.
- **Project map and frontmatter**: `src/wica/guards.py` in the [AGENTS.md](../AGENTS.md) module table; `src/wica/guards.py` and `tests/test_guards.py` in [specs/agent.md](../specs/agent.md)'s frontmatter.
- **Docs**: the configuration reference's `max_retries` row; the API reference — `on_agent_response_rejected`, the facade's guard members, and `wica.guards` under an **Experimental** heading that describes `add_response_guard` without recommending it yet; the latency guide's `rejected` outcome and retry fields; a troubleshooting entry ("reactions end `rejected`": which guard, how to raise `max_retries` or remove a default guard); the cost of a guard (busy time, a sync guard on the loop thread) stated where the API reference describes it.

Not in scope: a `response_rejected` event in the `wica.testing` Probe and a benchmark measure for the retry rate — the reports already carry `attempts` and `rejections` through each `ReactionTrace`; a follow-up once real runs show whether retries need their own numbers.

## Tests

- **Guards alone** (`tests/test_guards.py`), over hand-built `ModelResponse`s: `unknown_command` rejects an unbound name and lists the bound ones, accepts `noop`/`cancel_command`/the output Command; `invalid_args` rejects a missing or mistyped argument with the validation error, accepts valid ones, skips a JSON-schema tool; `output_command_required` rejects free text only and an empty response (as a reminder) while an output Command is bound, accepts any tool call (`noop` included), and accepts free text without an output Command.
- **Config**: default `1`; `0` accepted; `-1`, `true`, `1.5`, `"1"` raise `ConfigError` naming the key — direct construction and `from_dict`.
- **Agent**, over scripted models:
  - a rejected then accepted response: two model calls; history holds only the accepted response; the sink, `on_text` and `on_command` see only it; the next step's prompt shows neither the rejected response nor the feedback; `on_prompt` fired once; `on_response_rejected` once with `will_retry=True`; the trace has `attempts == 2`, one rejection, summed usage;
  - the show retry prompt holds the rejected `AIMessage`, a `tool_result` for each of its tool calls, then the feedback; the reminder retry prompt holds the feedback in the newest observation's user message, no rejected response, and no two consecutive user messages;
  - several guards rejecting one attempt: one combined correction, shown when any asked for it;
  - out of retries: outcome `rejected`, nothing dispatched or delivered, the Observation merged into the next prompt; with `max_retries=0` the first rejection ends the reaction;
  - a retry past its deadline ends `model_timeout`; a raising guard counts as accepting; an async guard is awaited;
  - `remove_response_guard("unknown_command")` lets an unknown Command through to a `failed` entry, as before; adding a duplicate name and removing an unknown one raise `ValueError`.
- **Facade**: the delegated members and the surfaced Event.
- **Fake provider, end to end**: a scripted free-text response followed by an output-Command call, with `force_tool_call: false`, retries through `Wica` and speaks once.
- **Gradio**: the reaction presenter renders attempts and feedback for a retried reaction.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier (including `test_project_map.py` and `test_docs.py`).
- `uv run pytest tests-e2e -k "fake or example or tester"` — the key-less e2e set.
- The live set on every provider with a key, to see that the default guards do not reject what the gate accepts — and how often they retry.
- On `Done`: [specs/agent.md](../specs/agent.md), [specs/config.md](../specs/config.md), [specs/instrumentation.md](../specs/instrumentation.md) and [specs/wica.md](../specs/wica.md) back to `Implemented`.
