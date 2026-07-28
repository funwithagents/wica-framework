# Framework config from JSON

**Status:** Done

## Goal

Implement [specs/config.md](../specs/config.md): let WICA (or at least an Agent) start from a single JSON file carrying **provider / model / api key / system prompt / model params**. Closes the deferred agent.md item ("`AgentConfig` from env/file"). No behavioral change to the Agent loop — this is a new loading layer plus api-key resolution (`api_key` literal / `api_key_env` reference) and one convenience entry point.

## Scope

New `src/wica/config.py` owning both config dataclasses and the JSON loader; small edits to `agent.py`, `__init__.py`, `AGENTS.md`; tests. Plus migrating **two first consumers** to committed config files: the **conversation demo** (touches the Stable [conversation-demo.md](../specs/conversation-demo.md) spec) and the **e2e test helper**. Both committed configs use `api_key_env` so they carry no secret. Env-based startup (no key in the file) must keep working exactly as today.

## Steps

### 1. New module `src/wica/config.py`

- **Move `AgentConfig` here** from [agent.py](../src/wica/agent.py) and add `api_key: str | None = None`. Fields: `provider`, `model`, `system_prompt`, `api_key`, `model_kwargs`. Pure dataclass — imports nothing from `agent.py` (keeps the dependency one-way; model construction stays on the Agent side).
- **`WicaConfig`** dataclass: `agent: AgentConfig`, `logging: str = "INFO"`.
- **`ConfigError(ValueError)`** — WICA's own error so a bad file gives an actionable message instead of a raw `KeyError`/`json.JSONDecodeError`.
- **`MissingEnvError(ConfigError)`** — raised specifically when a referenced env var (e.g. `api_key_env`) is unset, naming the variable. A distinct subclass so callers can catch *"env not set"* (degrade) separately from a structural `ConfigError` (a real bug in the file).
- **Strict parsing helpers** (shared, private) enforcing:
  - required keys present (`provider`, `model` in `agent`; exactly one of `system_prompt`/`system_prompt_file`),
  - **at most one** of `api_key`/`api_key_env` (both → `ConfigError`),
  - **unknown keys rejected** at both the top level (`agent`, `logging`) and inside `agent`,
  - coarse type checks (`model_kwargs` is an object, `logging`/`api_key_env` are strings, etc.),
  each raising `ConfigError` naming the offending key and its block.
- **Api-key resolution** (in the shared agent-block parser, so **both** `from_dict` and `from_json` do it — env needs no base dir): `api_key` passes through literally; `api_key_env` reads `os.environ[name]`, raising `MissingEnvError(name)` if unset; neither → `api_key=None`. Result is a plain resolved `api_key: str | None` on `AgentConfig`.
- **`AgentConfig.from_dict(data)`** — inline `system_prompt` only; a `system_prompt_file` key here is a `ConfigError` (no base dir to resolve against). Honours `api_key`/`api_key_env`.
- **`WicaConfig.from_dict(data)`** — parses `logging` + delegates the `agent` block to `AgentConfig.from_dict` (inline-only).
- **`WicaConfig.from_json(path)`** — reads/parses the file, then builds the `agent` block resolving `system_prompt_file` **relative to `path.parent`** (absolute paths used as-is), reading the referenced file and storing the result as the resolved `system_prompt: str`. A missing/unreadable prompt file raises `ConfigError` naming the resolved path. This is the only path that accepts `system_prompt_file`.
- **`apply_logging(level: str)`** — sets the level on the `wica` logger (mirrors the demo's `logging.getLogger("wica").setLevel(...)`). Kept a pure, explicit helper, not a side effect of parsing.

### 2. `src/wica/agent.py`

- Remove the local `AgentConfig`; import it from `config.py`.
- `Agent.from_config` — forward `api_key` to `init_chat_model(..., api_key=config.api_key)` **only when set**; when `None`, pass nothing so `init_chat_model` reads the provider env var as today.
- Add **`Agent.from_config_file(path, **kwargs)`**: `cfg = WicaConfig.from_json(path)`, `config.apply_logging(cfg.logging)`, then `Agent.from_config(cfg.agent, **kwargs)`. `**kwargs` carries the code-only wiring a JSON file can't express (World, loop, output sink, hooks).

### 3. `src/wica/__init__.py`

- Re-export `AgentConfig` and `WicaConfig` from `config.py` (drop the `AgentConfig` import from `agent.py`); add both to `__all__`.

### 4. Docs + committed config files

- **`AGENTS.md` Project map** — add a `config.py` row (drift-guard test `tests/test_project_map.py` enforces it).
- **`examples/prompts/wica.md`** — the demo's persona (moved out of the inline `SYSTEM_PROMPT` in [conversation_demo.py](../examples/conversation_demo.py)), demonstrating `system_prompt_file`.
- **`examples/agent.config.json`** (committed) — `anthropic` / `claude-sonnet-5`, `"api_key_env": "WICA_ANTHROPIC_API_KEY"`, `"system_prompt_file": "prompts/wica.md"` (resolves relative to the config file → `examples/prompts/wica.md`). Env-ref ⇒ no secret ⇒ checked in; no `.example.json`/gitignore dance needed for it.
- **`.gitignore`** — add a documented local-override name (e.g. `**/agent.config.local.json`) as the git-ignored slot for a *literal-key* config a user creates locally, keeping config.md's "literal key ⇒ git-ignored" guidance real without ignoring the committed env-ref configs.

### 5. Migrate the conversation demo to a config file

- **Load from the committed file.** Replace the env block ([conversation_demo.py:42-62](../examples/conversation_demo.py#L42-L62): `PROVIDER`/`MODEL`, the `_STANDARD_KEY_ENV_BY_PROVIDER` routing, WICA-namespaced key) with `Agent.from_config_file("examples/agent.config.json", output_sink=..., on_prompt=..., on_trigger=..., on_command=...)`; the hooks/sink ride in as `**kwargs`. Commands are still registered after construction, then `agent.start()`.
- **Preserve "no key, still usable."** The gate is no longer file-presence (the config is committed) but key resolution: wrap the build in `try/except MissingEnvError` — on it, fall back to today's explore-only path (`register_world()` only, no agent) and print a clear hint naming `WICA_ANTHROPIC_API_KEY` (or "add a literal key to a local config"). The old `HAS_KEY` gate becomes this catch.
- **Move the persona** out to `examples/prompts/wica.md` (step 4); delete the inline `SYSTEM_PROMPT`.
- **Logging.** `Agent.from_config_file` applies the config's `logging` level; keep a `logging.basicConfig(WARNING)` for third-party quieting and a sensible default on the explore-only fallback.
- Update the module docstring: run with `WICA_ANTHROPIC_API_KEY` set (config's `api_key_env`), or edit the committed config to a literal key / different provider.

### 6. Spec: conversation-demo.md (Stable → Updated → Stable)

- Rewrite the **Configuration** section (lines 97-106): provider/model/persona now come from a committed JSON config (`examples/agent.config.json`), persona via `system_prompt_file`, and the api key via `api_key_env` (`WICA_ANTHROPIC_API_KEY`) so the file carries no secret. Keep the **"No key, still usable"** guarantee, reworded to "when that env var is unset". The WICA-namespacing survives — now as *what `api_key_env` points at*, not a bespoke internal route.
- Per the status discipline this is a code-affecting edit to a Stable spec, so it flips to **Updated** with the edit and back to **Stable** when this plan lands (code + spec in sync). Update the [specs/_index.md](../specs/_index.md) row to match.

### 7. Migrate the e2e tier through the full config pipeline (Option B)

Goal: e2e drives **file → `WicaConfig` → `Agent`** via `Agent.from_config_file`, not just a model builder — so the whole config path is exercised against a live provider, not only in the deterministic unit tests.

- **`tests-e2e/e2e.config.json`** (committed) — `anthropic` / `claude-haiku-4-5`, `"api_key_env": "WICA_ANTHROPIC_API_KEY"`, `"logging": "WARNING"` (keep the tier quiet), and a **shared, load-bearing** inline `system_prompt`: `"You are a terse test assistant. Use tools when appropriate."` It serves both Agent tests (the plain-text one never calls a tool; the tool one is driven by the world prompt), so no per-test prompt override is needed and `from_config_file` needn't grow a `system_prompt=` kwarg.
- **`tests-e2e/support.py`** — drop `require_env` and the ad-hoc `WICA_ANTHROPIC_API_KEY`/`WICA_E2E_MODEL` reads; add two helpers over the committed config, each catching `MissingEnvError → pytest.skip(...)` to preserve "skip without credentials":
  - `real_agent(**kwargs) -> Agent` = `Agent.from_config_file(E2E_CONFIG, **kwargs)` — the full pipeline; the two Agent tests build through this, passing `world=`/`output_sink=` as `**kwargs`.
  - `real_chat_model(**kwargs) -> BaseChatModel` = build a raw model from the same config's provider/model/`api_key` — still needed by `test_smoke.py`, which exercises a bare model (`.invoke(...)`), not an Agent.
- **e2e test bodies** ([test_agent.py](../tests-e2e/test_agent.py)) — the two Agent tests swap `Agent(real_chat_model(), system_prompt="…", world=…, output_sink=…)` for `real_agent(world=…, output_sink=…)`; the tool test still calls `agent.register_command(add)` post-construction before `start()`. `test_smoke.py` keeps `real_chat_model()`. Model/provider now change by editing the config, not `WICA_E2E_MODEL`.

### 8. Tests — `tests/test_config.py`

Functional, driving the public API the way a caller would (no network):

- **Happy path** `from_dict`: valid dict → `WicaConfig` with the expected `AgentConfig` fields and `logging`.
- **`from_json` inline**: temp file with inline `system_prompt` parses correctly.
- **`from_json` file reference**: temp config + sibling prompt file; assert `system_prompt` is the file's contents, and that resolution is **relative to the config file** (run with a different CWD to prove it's not CWD-relative).
- **Exactly-one enforcement**: both `system_prompt` and `system_prompt_file` → `ConfigError`; neither → `ConfigError`.
- **Api-key resolution**: literal `api_key` → passthrough; `api_key_env` with the var set (via `monkeypatch.setenv`) → resolves to that value; `api_key_env` **unset** → `MissingEnvError` naming the var; both `api_key`+`api_key_env` → `ConfigError`; neither → `api_key is None`. Assert `MissingEnvError` is a `ConfigError` subclass (so the demo/e2e catch is sound).
- **Strict validation**: missing `provider`/`model` → `ConfigError`; unknown key (`"modl"`, stray top-level key) → `ConfigError`; wrong type for `model_kwargs`/`logging` → `ConfigError`.
- **`system_prompt_file` missing on disk** → `ConfigError` naming the resolved path.
- **`AgentConfig.from_dict` rejects `system_prompt_file`** (no base dir) but honours `api_key_env`.
- **api_key forwarding** (`tests/test_agent.py` or here): monkeypatch `init_chat_model` and assert `Agent.from_config` passes `api_key` when set and omits it when `None`.

## Verification

`uv run ruff check .`, `uv run pyright`, `uv run pytest` all pass. Smoke-test the demo both ways: with `WICA_ANTHROPIC_API_KEY` set (agent runs) and unset (app opens explore-only via the `MissingEnvError` catch). Run the e2e tier once with a key (`uv run pytest tests-e2e`) to confirm the config-driven `real_chat_model` works, and confirm it *skips* (not errors) with the key unset. On completion: flip [specs/config.md](../specs/config.md) **Draft → Stable** and [specs/conversation-demo.md](../specs/conversation-demo.md) back to **Stable** (both index rows too), and this plan **Todo → Done** (and its index row).

## Post-landing refinement: no `Agent.from_config_file`

Before this plan's changes were committed, `Agent.from_config_file(path, **kwargs)` (steps 2/5/7 above) was removed as unwarranted API surface for a two-line composition. Callers now write `wica_config = WicaConfig.from_json(path)` then `apply_logging(wica_config.logging)` then `Agent.from_config(wica_config.agent, **kwargs)` themselves — see [specs/config.md](../specs/config.md) "Flow into the Agent". All three call sites (the demo, `tests-e2e/support.py`'s `real_agent`, and `tests/test_config.py`) were updated to the explicit form; behavior is unchanged.

**Follow-up:** `real_agent()` in `tests-e2e/support.py` does **not** call `apply_logging` (unlike the demo) — `apply_logging`'s own correctness is already covered deterministically by `tests/test_config.py`, and calling it in a test helper would reset the `wica` logger on every test, fighting any level a developer sets by hand while debugging a live e2e run. `tests-e2e/e2e.config.json`'s `"logging"` field was removed accordingly (it was unread).
