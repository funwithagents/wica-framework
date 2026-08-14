# Multi-provider: OpenAI + Hugging Face Hub

**Status:** Done

Add two model providers alongside the existing `anthropic`: **`openai`** (via `langchain-openai`, the uniform `init_chat_model` path) and **`huggingface-hub`** (via `langchain-huggingface`, a dedicated construction branch targeting the Hub's serverless Inference Providers). Implements the `Updated` parts of [config.md](../specs/config.md) ("Providers") and [project.md](../specs/project.md) (provider extras), and the `huggingface-hub` construction/cancellation notes added to [agent.md](../specs/agent.md). Flip both `Updated` specs back to `Stable` when this plan is `Done`.

Scope is deliberately reduced (decided in design): Gemini is **out**; there is **no** capability probe (runtime errors only); the local `transformers` pipeline path is **never** used (breaks async cancellation); there is **no** `base_url` field.

## Steps

### 1. Packaging (pyproject.toml)

- Add `[project.optional-dependencies]` with one extra per provider, named to match the `provider` value:
  - `anthropic = ["langchain-anthropic>=1.0"]`
  - `openai = ["langchain-openai>=1.0"]`
  - `huggingface-hub = ["langchain-huggingface>=1.2"]`
- Add `langchain-openai` and `langchain-huggingface` to **both** the `dev` and `demo` dependency groups (alongside the existing `langchain-anthropic`), so e2e tests and the demo can switch providers by editing config alone.
- Bump the `langchain`/`langchain-core` floors if needed to match what's installed (repo is on 1.x; the `>=0.3.0` floor is stale — align it).
- `uv sync --dev` and confirm all three integration packages import.

### 2. Config: `hf_provider` field ([config.py](../src/wica/config.py))

- Add `hf_provider: str = "auto"` to `AgentConfig`.
- Accept it in the strict loader: add `"hf_provider"` to the agent block's allowed optional keys (`_AGENT_OPTIONAL_COMMON` or a sibling set), parsed with `_require_str` when present, defaulting to `"auto"`.
- Keep it **lenient/harmless-when-unused**: it's a plain optional string accepted regardless of `provider` (no strict "only with huggingface-hub" coupling — only the HF branch in `from_config` reads it). Note this choice in a code comment.
- Thread it through `from_dict`/`from_json` like the other fields.

### 3. Agent construction branch ([agent.py](../src/wica/agent.py))

Extract model construction into a **single shared function** so there's one provider-branch path, not two. Add a module-level `build_chat_model(config: AgentConfig) -> BaseChatModel` in `agent.py` (agent.py owns the LangChain boundary; `config.py` must stay import-free of LangChain). `Agent.from_config` calls it instead of inlining `init_chat_model`, and — crucially — the e2e `real_chat_model` helper is repointed at it too (Step 5), retiring its own duplicate `init_chat_model` call, which would build the *wrong* (local-pipeline) model for `huggingface-hub`.

`build_chat_model` branches on provider:

- `provider == "huggingface-hub"` → build the hosted, cancellable path:
  ```python
  from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
  endpoint_kwargs = dict(config.model_kwargs)
  if config.api_key is not None:
      endpoint_kwargs["huggingfacehub_api_token"] = config.api_key   # not api_key
  endpoint = HuggingFaceEndpoint(
      repo_id=config.model,
      provider=config.hf_provider,           # "auto" | "fireworks-ai" | ...
      task="text-generation",
      **endpoint_kwargs,
  )
  model = ChatHuggingFace(llm=endpoint)
  ```
  Import `langchain_huggingface` **inside** the branch so core `wica` (and the `openai`/`anthropic` users) never require it — an unselected extra fails only when its provider is actually chosen.
- else → existing `init_chat_model(config.model, model_provider=config.provider, **model_kwargs)` with `api_key` forwarded as today.
- Verify at implementation time (a real design risk noted in agent.md): (a) `ChatHuggingFace`/`HuggingFaceEndpoint` overrides an async `_agenerate` so `task.cancel()` truly aborts, and (b) `ChatHuggingFace(...)` construction doesn't do a blocking network fetch of the chat template in a way that breaks the sync `from_config` call path — if it does, decide whether that's acceptable or needs handling.
  - **Verified (a):** `langchain_huggingface.ChatHuggingFace` overrides `_agenerate`, so the hosted path is genuinely async and cancellable — no executor fallback. `HuggingFaceEndpoint` exposes `repo_id`/`provider`/`huggingfacehub_api_token`/`task` as expected.

### 4. Example + e2e config files

- Add committed example configs demonstrating each provider (using `api_key_env`, so they carry no secret), e.g. `examples/agent.openai.config.json` and `examples/agent.huggingface-hub.config.json` (the latter with `provider: "huggingface-hub"`, a Hub `repo_id`, and an `hf_provider`). Keep the existing anthropic config as the demo default.
- Leave `tests-e2e/e2e.config.json` on anthropic as the default; the e2e tests parametrize/override provider (Step 5).

### 5. Tests

- **Deterministic ([tests/](../tests/)):**
  - `hf_provider` parsing/validation in `test_config.py`: present → carried onto `AgentConfig`; absent → defaults to `"auto"`; wrong type → `ConfigError`; unknown-key rejection still fires.
  - `build_chat_model` **without network**: mock `HuggingFaceEndpoint`/`ChatHuggingFace` and assert the HF branch's kwarg mapping — `repo_id=model`, `provider=hf_provider`, `huggingfacehub_api_token=<api_key>`, `model_kwargs` forwarded — and that the `openai`/`anthropic` path still routes through `init_chat_model`. Tests the one branch, not the live provider.
- **e2e helper ([tests-e2e/support.py](../tests-e2e/support.py)):** repoint `real_chat_model` at `build_chat_model(cfg)`, retiring its own `init_chat_model` call. This gives the whole e2e tier one construction path and makes `test_smoke.py` exercise the real (HF-aware) builder rather than a divergent one.
- **Live ([tests-e2e/](../tests-e2e/)):** a provider-parametrized (or per-config) smoke test that a **Command round-trips** — register a trivial tool, drive one step, assert the model issues the tool call and its `agent:command:<id>` entry goes terminal — for `openai` and `huggingface-hub` (using a known tool-capable Hub model+backend). Each skips without its key, per the existing `support.py` mechanics.

### 6. Verification & status flip

- `uv run ruff check .`, `uv run pyright`, `uv run pytest` all green.
- Optionally run the live tier against each provider: `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e'` (needs `WICA_OPENAI_API_KEY` / `WICA_HF_TOKEN` set).
- Flip [config.md](../specs/config.md) and [project.md](../specs/project.md) `Updated → Stable` (both the `**Status:**` line and their [specs/_index.md](../specs/_index.md) rows), and mark this plan `Done` here and in [plans/_index.md](_index.md).

## Notes / out of scope

- **No capability probe** for tool calling — runtime errors only (agent.md "Commands"). A hard-rejecting provider surfaces as an `ainvoke` exception (the pre-existing agent-level error gap, agent.md Open question #7); silent degradation is uncaught. Neither is handled here.
- **No `base_url`** — dropped with the OpenAI-compat shim; `huggingface-hub` routes via `HuggingFaceEndpoint(provider=…)`, not a base URL.
- **No new `src/wica/` module** — changes live in existing `config.py`/`agent.py`, so the AGENTS.md project map is unaffected.
