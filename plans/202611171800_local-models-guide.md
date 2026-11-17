# Local models: the guide and the benchmark rows

**Status:** Done

Adds the `local` rows of [specs/benchmark.md](../specs/benchmark.md) ("The matrix") and rewrites the user guide for local and self-hosted models. No library code changes.

**Why.** WICA's integration was checked on `llama-server`, `mlx_lm.server` and Ollama with the tag output (Qwen2.5-7B-Instruct, 4 bits, Apple M5 Pro, 2026-11-17): it works on the three. What differs is Ollama's and goes in the guide: its 4096-token default context cuts a longer prompt silently, it unloads a model after five idle minutes, and it ignores the output cap as the OpenAI client sends it (`max_completion_tokens`; 649 tokens out under a cap of 12), which `model_kwargs.extra_body` works around. Sending the cap as `max_tokens` from the provider was considered and left out for now.

## Steps

1. **Benchmark rows** (`benchmarks/matrix.json`): `openai-chat-completions-compatible/qwen2.5-7b-instruct@llama-server`, `@mlx`, `@ollama`, tagged `local`, `open`, `small`, `enabled: false` (no key gates them); the server command in each row's `notes`; the Ollama row's cap in `extra_body`. The `local` tag joins the vocabulary in the matrix's `_comment`, [specs/benchmark.md](../specs/benchmark.md) and [benchmarks/README.md](../benchmarks/README.md).
2. **Docs.** [docs/guides/local-models.md](../docs/guides/local-models.md) rewritten by solution: what every server has in common (no key, `context_window`, `model_timeout` bounding the whole streamed reply, the cap, the warm-up), one section per server run (install, start, config, what was measured, its settings, Ollama's cap in `extra_body`), the servers not run, how to measure a model with the benchmark rows (integration facts only, no model scores), cancellation, the cloud shapes. The configuration reference, troubleshooting and the docs index follow; [specs/config.md](../specs/config.md) names the three servers measured (editorial).

## Verification

- `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`.
- Live, by hand: `extra_body: {"max_tokens": 12}` through WICA on Ollama 0.40.15 gives 12 tokens; the `@llama-server` row ran both benchmark sessions once against a real `llama-server` (both completed).
