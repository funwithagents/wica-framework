# Provider names and the `openai-chat-completions-compatible` provider

**Status:** Done

Implements "Provider names" and the `openai-chat-completions-compatible` paragraph of [specs/config.md](../specs/config.md) ("Providers"), the construction branch and LangChain-id mapping of [specs/agent.md](../specs/agent.md) ("Provider-agnostic model, from config"), and the renames they carry into [specs/project.md](../specs/project.md), [specs/testing.md](../specs/testing.md) and [specs/conversation-demo.md](../specs/conversation-demo.md).

**Why.** Two things, agreed on 2026-11-16 together with tts-engine and asr-engine, so the three repos share one vocabulary:

- **Names.** WICA's `google_genai` and `mistralai` are LangChain package ids, with an underscore, while the sibling repos name hosted providers after what users call them, with hyphens. They become `gemini` and `mistral`; the old names keep working with a deprecation warning.
- **Local and self-hosted models.** A model served by `llama-server`, `mlx_lm.server`, Ollama, LM Studio or vLLM already works through `provider: "openai"` with `model_kwargs.base_url`, but only for a user who knows LangChain: the address is hidden in `model_kwargs`, the OpenAI client refuses to start without a key (so `api_key: "none"`), it reads `OPENAI_API_KEY` otherwise (a real OpenAI key sent to an arbitrary server), streamed usage is off with a `base_url`, and traces read `openai`. A dedicated provider, named after the API it implements, removes all five.

## Scope

**Config** ([src/wica/config.py](../src/wica/config.py)):

- `AgentConfig.base_url: str | None = None`, accepted by the loaders. `__post_init__`: required (a non-empty string) with `openai-chat-completions-compatible`, a `ConfigError` with any other provider; `model_kwargs.base_url` with that provider is a `ConfigError` too.
- The rename table `{"google_genai": "gemini", "mistralai": "mistral"}`, applied in `__post_init__` (so the loaders and direct construction both normalize), logging a WARNING on `wica.config` that names the new name. The stored `provider` is the new name.

**Agent** ([src/wica/agent.py](../src/wica/agent.py), `build_chat_model`):

- `gemini` → `google_genai`, `mistral` → `mistralai` as `init_chat_model`'s `model_provider`; any other value passes through as today.
- A new branch for `openai-chat-completions-compatible`: `ChatOpenAI(model=…, base_url=…, api_key=<resolved key or a placeholder>, stream_usage=True, **model_kwargs)`, imported lazily like the Hub's.
- The history-budget `ConfigError` for an unknown window names the server's context size on this provider.

**Packaging** ([pyproject.toml](../pyproject.toml), `uv.lock`): extras `gemini`, `mistral`, `openai-compatible`; `google-genai` and `mistralai` removed, no aliases.

**Renamed files and values:**

- `tests-e2e/e2e.google-genai.config.json` → `e2e.gemini.config.json`, `e2e.mistralai.config.json` → `e2e.mistral.config.json` (provider values inside), `PROVIDER_CONFIGS`, `KNOWN_FLAKY`, the CI live matrix (`select: gemini`), the `-k` examples and the key table in [AGENTS.md](../AGENTS.md).
- `examples/conversation_demo/agent.google-genai.config.json` → `agent.gemini.config.json`, `agent.mistralai.config.json` → `agent.mistral.config.json`.
- `benchmarks/matrix.json`: provider values, and the ids (`gemini/…`, `mistral/…`, after the `<provider>/<model>` convention). Kept results keep the old names. `tests-e2e/test_demo_bench_fake.py`'s default list.

**Docs** (the local-model route, from the study of 2026-11-15; one home per fact):

- [docs/reference/configuration.md](../docs/reference/configuration.md): the `agent.base_url` field, the provider table (`gemini`, `mistral`, `openai-chat-completions-compatible`), old names, and the key exception.
- New [docs/guides/local-models.md](../docs/guides/local-models.md): start a server (`llama-server`, tried and recommended; `mlx_lm.server`, works with two limits), point WICA at it, what to tune (`context_window`, `model_timeout`, `force_tool_call`, an output cap through `model_kwargs.max_tokens`), the model size to expect to work (7B did, 1.5B did not), cancellation, cloud shapes.
- [docs/index.md](../docs/index.md), [docs/getting-started.md](../docs/getting-started.md), [README.md](../README.md) install table, [docs/guides/troubleshooting.md](../docs/guides/troubleshooting.md), [docs/guides/limits.md](../docs/guides/limits.md).

**Out of scope.** A local entry in the benchmark matrix (decision 3 of the study, not taken); WICA sending an output cap by default while a tool call is forced (documented instead); streaming the model call; `openai-responses-compatible`; the mirrors in wica-interaction.

## Tests

- Config: an old provider name loads as the new one and logs the warning naming it, through `from_dict` and direct construction; `base_url` required with the new provider, rejected with another, `model_kwargs.base_url` rejected with the new provider, wrong type rejected.
- `build_chat_model`: `gemini` and `mistral` build `ChatGoogleGenerativeAI` / `ChatMistralAI`.
- The new provider, driven end to end through `Wica` over an `httpx.MockTransport` (no socket): the request goes to `<base_url>/chat/completions` with the config's model; with no key, the placeholder is sent and `OPENAI_API_KEY` is not, even when set; a configured key is sent; usage from the response lands in the reaction record; the history-budget error names `context_window` and the server.
- `tests/test_docs.py`: the provider tables still match the extras (unchanged checks; the new extra must appear).

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest`, `uv run pytest tests-e2e -k "fake or example or tester"`.
- The live set on `gemini` and `mistral` (renamed configs, `-k gemini`, `-k mistral`) on CI, on the push to `main` or a manual run.
