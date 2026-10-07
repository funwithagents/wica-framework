# Provider: Mistral via `mistralai`

**Status:** Done

Add **Mistral** (`ministral-3b-latest` as the committed model — see the quota decision below) as a fully supported provider, through Mistral's **La Plateforme** API (a Mistral API key) and LangChain's `langchain-mistralai` integration. Implements the `Updated` parts of [config.md](../specs/config.md) ("Providers") and [project.md](../specs/project.md) (provider extras). Flip both `Updated` specs back to `Implemented` when this plan is `Done`.

Design decisions, settled during the study that preceded this plan (2026-10-07, an offline probe of `langchain-mistralai` 1.1.6 against `langchain` 1.3.14 / `langchain-core` 1.6.5, run with a throwaway install so the lock was untouched):

- **No construction branch.** `mistralai` is LangChain's own provider id — `init_chat_model` registers it as a builtin mapped to `ChatMistralAI` — so `build_chat_model` passes it straight through like `anthropic`/`openai`/`google_genai`; no code in `agent.py` changes. The probe built the real model through `build_chat_model` and confirmed every WICA requirement: the generic `api_key` kwarg lands on `mistral_api_key`; `model_kwargs` (probed with `temperature`) land on the model; the no-argument `noop`/`cancel_command` tools bind to valid function declarations; WICA's standard base64 `image` blocks are translated to Mistral's `image_url` shape; `_agenerate` is truly async over an httpx `AsyncClient` (cancellation-friendly); responses carry `usage_metadata` with input/output tokens so `TokenUsage` is filled; the model profile exposes `max_input_tokens` (256k for Small), so `history_budget` works without `context_window`.
- **Tool-call ids are safe.** Mistral only accepts nine-character alphanumeric ids. The Agent reuses the provider's own id for each call (uuid hex only as a fallback), and the integration keeps conforming ids and hashes any other into a conforming one — so the replayed assistant-with-tool-calls + fixed-ack tool-result history needs no change.
- **La Plateforme, one key.** The `api_key`/`api_key_env` surface expresses it fully; with neither set the package reads `MISTRAL_API_KEY`. Self-hosted or other endpoints are out of scope (the `endpoint` kwarg would go through `model_kwargs` if ever needed).
- **Committed model `ministral-3b-latest`, not `mistral-small-latest`** (changed during live verification). On a workspace without a paid tier, Mistral serves the Small/Medium/Magistral/Vibe families with a request quota of **zero** — every completion is a 429 `Rate limit exceeded` (code 1300) with `x-ratelimit-limit-req-minute: 0`, even though the key is valid and `GET /v1/models` lists them — while the Ministral 3b/8b/14b, Codestral and Voxtral families have full quota. Ministral 3B — the smallest of them — has tool calling, vision, a 128k context, 750 req/min, and a LangChain profile (128k `max_input_tokens`, no reasoning output), so it is the committed choice (8B was verified first and passed identically, 256k context, 188 req/min); a paid-tier workspace can switch to Small by editing the config.
- **No thinking knob needed.** `ChatMistralAI` exposes no dedicated reasoning parameter, and Ministral 3B does not reason by default (plain reactions answered in about a second). A reasoning model would be capped through `model_kwargs` — the same forwarding Gemini's `thinking_level` uses — never a new config field.
- **Extra name `mistralai`** — the provider id verbatim, no hyphen to normalize. Config files use the same stem (`e2e.mistralai.config.json`), so `-k mistral` selects the provider.
- **Key env var `WICA_MISTRAL_API_KEY`**, following the `WICA_<PROVIDER>_API_KEY` convention.
- **Lock impact is nil.** `langchain-mistralai` 1.1.6 requires `langchain-core>=1.4.7,<2`; the lock already sits at 1.6.5. It adds `httpx-sse` and `tokenizers` as new transitive dependencies.

## Steps

### 1. Packaging (pyproject.toml) — done

- `mistralai = ["langchain-mistralai>=1.1"]` under `[project.optional-dependencies]`; the same package in the `dev` and `demo` groups.
- `uv sync --dev`; the lock diff added only `langchain-mistralai` 1.1.6 and `httpx-sse` 0.4.3 (no `langchain-core` move).

### 2. Committed configs — done

- `tests-e2e/e2e.mistralai.config.json` — `provider: "mistralai"`, `model: "ministral-3b-latest"`, `api_key_env: "WICA_MISTRAL_API_KEY"`, the terse test system prompt; added to `PROVIDER_CONFIGS` in `tests-e2e/support.py` so every live e2e test runs against it (skipping when the key is unset).
- `examples/conversation_demo/agent.mistralai.config.json`, the demo's per-provider variant.

### 3. Specs and docs — done

- [config.md](../specs/config.md) and [project.md](../specs/project.md): the `mistralai` row, paragraph and extra are already written (this plan's companion change); they return to `Implemented` at the end.
- [testing.md](../specs/testing.md) and [conversation-demo.md](../specs/conversation-demo.md): add the two new config files to their frontmatter and prose (editorial; status unchanged). `tests/test_project_map.py` checks every listed path exists, so this lands with step 2, not before.
- AGENTS.md: key table row + `-k mistral` run line; README and INTEGRATING.md provider tables and install comments.

### 4. Fast test — done

- `tests/test_config.py::test_build_chat_model_mistralai_is_a_pass_through`, next to the Gemini one: builds the real `ChatMistralAI` through `build_chat_model` (no request is made) and asserts the resolved key landed on `mistral_api_key` and a `model_kwargs` value (`temperature`) landed on the model — the two kwarg mappings that could silently break.

### 5. Live verification — done

Run `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k mistral'` with `WICA_MISTRAL_API_KEY` exported. Record per-case results here as the Gemini plan did; watch specifically for (a) reasoning latency on plain reactions (see the thinking decision above) and (b) any 400 on the replayed tool-result history, which would be the first provider to reject the Agent's renderer and would add a step here.

**Results so far (2026-10-07):** steps 1–4 landed with ruff, pyright and the fast tier (334 tests) green, the always-run fake flows passing, and `-k mistral` selecting the new config — all four live cases skip cleanly with `WICA_MISTRAL_API_KEY not set`, which is the expected behaviour without a key. A key was then exported and the set run again: all four cases fail with 429 `Rate limit exceeded` (code 1300) before any model output. A direct probe showed the key is valid — `GET /v1/models` returns 200 and lists `mistral-small-latest` — but every `POST /v1/chat/completions` answers 429 with `x-ratelimit-limit-req-minute: 0`: the La Plateforme workspace has no active plan (neither the free Experiment tier nor billing), so its request quota is zero. Not a WICA behaviour. A per-model probe of all 27 chat-capable models then showed the zero quota is **per model family**, not per workspace: Small/Medium/Magistral/Vibe at 0 req/min, Ministral/Codestral/Voxtral served normally. The committed configs were switched to `ministral-8b-latest` (live set passed clean, 6 s total), then to `ministral-3b-latest` at the user's request, which passed identically:

| Case | Outcome (3B) |
|---|---|
| smoke (`build_chat_model` + one call) | pass, 1.0 s |
| plain-text round trip | pass, 0.9 s |
| tool-calling round trip | pass, 1.3 s |
| `noop` on a heartbeat | pass, 1.5 s |

Neither watch item fired: no reasoning latency, and the replayed tool-result history was accepted. [config.md](../specs/config.md) and [project.md](../specs/project.md) flipped back to `Implemented`.
