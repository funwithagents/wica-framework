# Model-call deadline

**Status:** In progress

Implements [specs/agent.md](../specs/agent.md) ("Model-call deadline"), the `model_timeout` field of [specs/config.md](../specs/config.md) and the `model_timeout` reaction outcome of [specs/instrumentation.md](../specs/instrumentation.md).

**Why.** Nothing bounded the model call: the step awaited `ainvoke` with no deadline, so the only limit was each provider client's default — 600 s per attempt plus retries for Anthropic and OpenAI (about half an hour), 120 s × 5 retries for Mistral, no timeout at all for Gemini. Under the single-in-flight loop a hung call drops every trigger meanwhile, so the Agent goes deaf, not just slow. The `model_kwargs` workaround (each provider's own `timeout`/`max_retries`) is spelled differently per provider and still multiplies by the retries.

## Scope

- `wica.config`: `AgentConfig.model_timeout: float | None = 10.0` — a positive number of seconds or `None` (no deadline); the loaders default an absent key to `10`, accept `null`, reject a bool, zero, a negative number or a non-number with the key named.
- `wica.instrumentation`: `ReactionOutcome` gains `"model_timeout"`.
- `wica.agent`: the model call runs under `asyncio.timeout(model_timeout)`; a `TimeoutError` from it ends the step with outcome `model_timeout` and `error` = `"model call exceeded its <n> s deadline"`, logged at ERROR, nothing dispatched — the same path as a model error. A `stop()` cancellation stays `cancelled`.
- `wica.contrib.gradio.reactions`: the docstring names the new outcome (the error line already shows any `error`).
- Docs: the configuration reference's field table, the latency guide's outcome row, a troubleshooting entry.

## Tests

- Config: default `10.0`; `null` → `None`; an int is accepted; `0`, `-1`, `true`, `"10"` raise `ConfigError` naming the key — direct construction and `from_dict`.
- Agent, over a model that blocks past a short deadline: the reaction ends `model_timeout` with the deadline in `error`, no Command dispatched, no sink call, and the Agent reacts to the next trigger with the earlier observation merged into the next prompt. With `model_timeout=None` the same slow call completes.
- The `fake` provider's `delay_s` past `model_timeout` times out through `Wica` too (the config path end to end).

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- `uv run pytest tests-e2e -k "fake or example or tester"` — the key-less e2e set.
- CI on the pull request.
