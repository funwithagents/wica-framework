# No output-token cap on `huggingface-hub` unless configured

**Status:** Done

Implements the "No output cap unless configured" rule of [specs/config.md](../specs/config.md) ("Providers") and its line in [specs/agent.md](../specs/agent.md) ("Provider-agnostic model, from config").

**Why.** `HuggingFaceEndpoint` defaults `max_new_tokens` to 512 and `ChatHuggingFace` copies it into every request as `max_tokens`. `openai/gpt-oss-120b` reasons before it answers: asked for a long story, or to dance, its reasoning and the tool call ran past 512 tokens, the response stopped on `length` with no tool call, and the step ended empty. In the demo benchmark (k = 5, both scenarios) that stopped 3 of its 10 runs, forced or not. With the cap removed the same model finished a 1,416-token answer on `stop`, and a 4,096 cap took its runs to 10/10 completed with no empty step.

## Scope

- `wica.agent.build_chat_model`, `huggingface-hub` branch: after building `ChatHuggingFace`, set its `max_tokens` to `None` unless `model_kwargs` sets `max_new_tokens` — the request then carries no cap. A configured `max_new_tokens` is kept.
- Docs: the provider row of the configuration reference.

## Tests

- A real `ChatHuggingFace` built offline sends no `max_tokens` by default, and sends `max_new_tokens` when the config sets it.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest`, `uv run pytest tests-e2e -k "fake or example or tester"`, and the live e2e set on `huggingface-hub`.
