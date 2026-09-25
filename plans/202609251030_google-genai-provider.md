# Provider: Gemini via `google_genai`

**Status:** In progress

Add **Gemini** (Gemini 3.8 Flash — 2.5 Flash, the original target, is no longer served to new Developer API accounts) as a fully supported provider, through the **Gemini Developer API** (an AI Studio API key) and LangChain's `langchain-google-genai` integration. Implements the `Updated` parts of [config.md](../specs/config.md) ("Providers") and [project.md](../specs/project.md) (provider extras). Flip both `Updated` specs back to `Implemented` when this plan is `Done`.

Design decisions, settled during the analysis that preceded this plan:

- **No construction branch.** The `provider` value is LangChain's own id, `google_genai`, so `build_chat_model` passes it straight to `init_chat_model` like `anthropic`/`openai` — no code in `agent.py` changes. An offline probe of `langchain-google-genai` 4.4.0 confirmed the pass-through meets every WICA requirement: the generic `api_key` kwarg lands on the model's Google key field; the no-argument `noop`/`cancel_command` tools convert to valid function declarations; WICA's standard image blocks convert to inline data; the assistant-with-tool-calls + fixed-ack tool-result history converts; `_agenerate` awaits the async client (cancellation-friendly); `usage_metadata` carries input/output/`cache_read` so `TokenUsage` is filled.
- **Gemini Developer API, not Vertex AI.** Vertex authenticates with Google Cloud credentials that no config field expresses; the Developer API is one key, which is exactly the `api_key`/`api_key_env` surface. `google_vertexai` remains an unsupported pass-through.
- **Thinking minimized in the committed configs.** Gemini thinks by default (dynamic budget), which adds latency to every reaction; `model_kwargs: {"thinking_level": "low"}` (the Gemini 3+ knob — `thinking_budget` is the deprecated 2.5-era one; 3+ cannot fully disable thinking, and 3.8 Flash rejects `minimal` with a 400, so `low` is its floor) caps it through the existing forwarding — no new config field.
- **Extra name `google-genai`** (hyphenated, like `huggingface-hub`); PEP 685 normalization makes `wica[google_genai]` equivalent. Config files use the hyphenated stem (`e2e.google-genai.config.json`), so `-k google` selects the provider.
- **Key env var `WICA_GEMINI_API_KEY`**, following the `WICA_<PROVIDER>_API_KEY` convention. With no key field set, the package reads `GOOGLE_API_KEY` then `GEMINI_API_KEY`.

## Steps

### 1. Packaging (pyproject.toml) — done

- `google-genai = ["langchain-google-genai>=4.4"]` under `[project.optional-dependencies]`; the same package in the `dev` and `demo` groups.
- `uv sync --dev` — the package requires `langchain-core>=1.6.1`, so the lock moves `langchain-core` 1.5.0 → 1.6.x (within every installed provider package's `<2.0` bound).

### 2. Committed configs — done

- `tests-e2e/e2e.google-genai.config.json`, added to `PROVIDER_CONFIGS` in `tests-e2e/support.py` so every live e2e test runs against it (skipping when `WICA_GEMINI_API_KEY` is unset).
- `examples/conversation_demo/agent.google-genai.config.json`, the demo's per-provider variant.
- Both name `gemini-3.8-flash` and set `"model_kwargs": {"thinking_level": "low"}`.

### 3. Specs and docs — done

- [config.md](../specs/config.md): `google_genai` row in the Providers table, a paragraph on the pass-through/Developer-API/thinking decisions; status `Updated`.
- [project.md](../specs/project.md): the extra in the packaging bullet; status `Updated`.
- [testing.md](../specs/testing.md) and [conversation-demo.md](../specs/conversation-demo.md): the new config files in their frontmatter and prose (editorial; status unchanged).
- AGENTS.md key table + `-k google` run line; README and INTEGRATING.md provider tables and install comments.

### 4. Fast test — done

- `tests/test_config.py::test_build_chat_model_google_genai_is_a_pass_through`: builds the real `ChatGoogleGenerativeAI` through `build_chat_model` (no request is made) and asserts the resolved key and `thinking_level` landed on it — the two kwarg mappings that could silently break.

### 5. Live verification — in progress

Run `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k google'` with `WICA_GEMINI_API_KEY` exported.

**Results so far (2026-09-25, `gemini-3.8-flash`, `thinking_level: low`):** each of the four live cases has passed at least once, but never all four in one run —

| Run | Outcome |
|---|---|
| 1 | smoke + tool-calling round trip pass; plain-text + `noop` fail (transient — passed on the immediate rerun) |
| 2 | plain-text + `noop` pass |
| 3 | all four fail: 503 `UNAVAILABLE` "high demand" (transient) |
| 4 | all four fail: 429 `RESOURCE_EXHAUSTED` (key quota) |

Findings that shaped the configs: the Developer API no longer serves `gemini-2.5-flash` to new accounts (404, points at 3.8 Flash); 3.8 Flash rejects `thinking_level: minimal` (400), so `low` is its floor. The `google-genai` SDK logs a harmless "Direct use of automatic function calling (AFC) … is not recommended" warning on every call through `langchain-google-genai`; a consumer can silence the `google_genai.models` logger.

Of the three behaviours only a live run settles: (3) primer steering is confirmed — the direct request was answered as text and the heartbeat observation drew a `noop`. (1) is *not yet* proven: `test_real_tool_calling_round_trip` stops at the Command's terminal entry and does not assert on the follow-up step whose prompt carries the tool-result turn immediately followed by the completion observation. (2) same-Command concurrent calls are not exercised by the live set at all. Both stay listed here as deferrals; neither has produced an error in the runs above.

**To close:** one clean `-k google` run with all four cases passing , then flip [config.md](../specs/config.md) and [project.md](../specs/project.md) back to `Implemented` and mark this plan `Done`. If (1) ever fails, the fix belongs in the renderer in `agent.py` and this plan grows a step for it.
