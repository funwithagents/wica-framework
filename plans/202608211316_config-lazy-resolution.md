# Config: pure-data objects, lazy resolution

**Status:** Done

Turns [specs/config.md](../specs/config.md) and [specs/agent.md](../specs/agent.md) (both `Updated`) back into `Implemented`. Makes the config dataclasses a 1:1 mirror of the JSON — plain data that reads no environment and no files — and moves the two resolutions (env-var read, prompt-file read) from **load time** to **Agent build time**.

## Goal

Today `from_dict`/`from_json` eagerly resolve two indirections away, so `AgentConfig` is a *resolved* object:

- `api_key` / `api_key_env` → collapsed to `api_key: str | None` (env read at load; `MissingEnvError` at load).
- `system_prompt` / `system_prompt_file` → collapsed to `system_prompt: str` (file read at load, relative to the config dir).

After this change `AgentConfig` mirrors the JSON — it holds `api_key`, `api_key_env`, `system_prompt`, `system_prompt_file` all verbatim — and the reads happen when the config is *used* (`Agent.from_config` / `build_chat_model`).

## Settled design (from the spec update)

- **Load = validation only, eager.** Structural checks stay at load: unknown-key rejection, type checks, required `provider`/`model`, and both mutual-exclusion rules (`system_prompt` xor `system_prompt_file`; not-both `api_key`/`api_key_env`). No env access, no file read.
- **`api_key_env` is context-free** → stored verbatim, resolved at build (`os.environ` read). Same in `from_dict` and `from_json`.
- **`system_prompt_file` is a path** → `from_json` **locates** it (joins the config dir → absolute path) but does **not read** it; `from_dict` stores it verbatim. The **read** happens at build. Absolutizing is a locate, not a read — it folds the ephemeral base dir into the path itself, so no `source_dir` field is needed and the relocatable-config property is preserved. A relative path via `from_dict` resolves against CWD at read time (caller's responsibility). The old "`from_dict` rejects `system_prompt_file`" carve-out is dropped.
- **Resolution helpers live in `config.py`, called by the agent-side consumers.** Config owns *how* a field resolves; the consumer owns *when*.
- **Errors move to build.** `MissingEnvError` (unset key env) and unreadable-prompt-file `ConfigError` fire at `Agent.from_config`, not at load. Both callers that degrade move their `try` from around `from_json` to around the build call. `both-given` / typo'd-key / missing-required stay plain `ConfigError` at load.

## Steps

### 1. `src/wica/config.py` — data + validation, resolvers

- **`AgentConfig` fields** become the JSON mirror:
  ```python
  provider: str
  model: str
  system_prompt: str | None = None
  system_prompt_file: str | None = None
  api_key: str | None = None
  api_key_env: str | None = None
  model_kwargs: dict[str, Any] = field(default_factory=dict)
  hf_provider: str = "auto"
  ```
- **`_parse_agent_block`** stops resolving. It keeps: unknown-key rejection, `provider`/`model` required-string checks, `model_kwargs`/`hf_provider` type checks, and moves the two "at most one of" checks here as *structural* validation (they currently live inside `_resolve_api_key`/`_resolve_system_prompt`). It also keeps the "exactly one of `system_prompt`/`system_prompt_file` present" check (required-ish). It passes `system_prompt`, `system_prompt_file`, `api_key`, `api_key_env` through onto the dataclass unchanged — except:
  - When `base_dir is not None` (i.e. `from_json`) **and** `system_prompt_file` is a relative path, replace it with `str((base_dir / value).resolve())` (or `.absolute()` — no `.resolve()` symlink surprises; pick one and note it). Absolute paths pass through. This is the only place `base_dir` is still used, and only to *locate*, never to read.
- **Delete** `_resolve_api_key` and `_resolve_system_prompt` (the reading versions). Replace with two pure resolver functions over an `AgentConfig`:
  ```python
  def resolve_api_key(config: AgentConfig) -> str | None: ...   # literal → env read (MissingEnvError) → None
  def resolve_system_prompt(config: AgentConfig) -> str: ...     # inline → read system_prompt_file (ConfigError)
  ```
  `resolve_system_prompt` can assume exactly one of the pair is set (load-time validation guarantees it); read via `Path(config.system_prompt_file).read_text(...)`, wrapping `OSError` in `ConfigError` naming the path.
- `MissingEnvError` / `ConfigError` classes unchanged.
- `from_dict` / `from_json` bodies barely change — they still call `_parse_agent_block`/`_parse_wica_block`; only the base-dir threading now means "locate the prompt path" instead of "read it", and env is untouched.

### 2. `src/wica/agent.py` — resolve at build

- `build_chat_model(config)`: replace every `config.api_key` read with a single `api_key = resolve_api_key(config)` at the top, then use `api_key` in all three branches (fake ignores it; huggingface-hub → `huggingfacehub_api_token`; init_chat_model → `api_key`).
- `Agent.from_config(config, **kwargs)`: `system_prompt = resolve_system_prompt(config)` then `cls(build_chat_model(config), system_prompt=system_prompt, **kwargs)`.
- Import `resolve_api_key`, `resolve_system_prompt` from `wica.config`.

### 3. Callers move their `try` to build time

- **[examples/conversation_demo.py](../examples/conversation_demo.py)** (~line 238): `WicaConfig.from_json` no longer raises `MissingEnvError`; wrap `Agent.from_config(...)` in the `try`/`except MissingEnvError` instead. `from_json` + `apply_logging` run unconditionally; only the build degrades to explore-only.
  - **Note the CWD consequence:** the demo's four `examples/agent*.config.json` use `"system_prompt_file": "prompts/wica.md"`. `from_json` absolutizes it against `examples/`, so the prompt still resolves from any launch dir — **no demo regression** (this is why we absolutize rather than defer the locate). Verify by launching the demo from the repo root.
- **[tests-e2e/support.py](../tests-e2e/support.py)** (~line 25): `load_agent_config` no longer raises `MissingEnvError` (nothing resolves at load), so the `pytest.skip` moves. Options: keep `load_agent_config` as the plain loader and move the skip into `real_chat_model`/`real_agent` (wrap `build_chat_model` / `Agent.from_config` in `try/except MissingEnvError → pytest.skip`). This keeps the "unset key env → skip" behavior, now triggered at build.

### 4. Tests — `tests/test_config.py`

Rewrite the resolution-timing tests to assert the new split:

- **Stay at load (validation):** unknown key, wrong type, `both api_key+api_key_env`, `both system_prompt+system_prompt_file`, neither prompt, missing `provider`/`model`. These still raise `ConfigError` from `from_dict`/`from_json`.
- **Now deferred (no longer raise at load):**
  - `api_key_env` pointing at an **unset** var: `from_dict`/`from_json` succeed and store the name; `MissingEnvError` raises from `resolve_api_key(cfg)` / `Agent.from_config(cfg)`.
  - `system_prompt_file`: `from_json` stores an **absolute** path (assert it's absolute and ends with the given relative path); the file is **not** read at load (a config whose prompt file is missing still *loads*); `resolve_system_prompt` / `Agent.from_config` reads it, and a missing file raises `ConfigError` there.
  - `from_dict` with `system_prompt_file`: no longer rejected — it loads and stores the string (drop/replace `test_agent_config_from_dict_rejects_system_prompt_file`).
- **Resolvers directly:** `resolve_api_key` (literal → env → None) and `resolve_system_prompt` (inline vs file read) as unit tests.
- **`build_chat_model` / `Agent.from_config` forwarding** tests (openai api_key, hf token, omit-when-unset) still hold — they exercise the resolved value; adjust any that pre-set a resolved `api_key` on `AgentConfig` to instead go through `api_key`/`api_key_env`.
- Confirm `tests-e2e/test_fake_flows.py` (uses `from_dict`, inline prompt, no key) still passes unchanged.

### 5. Statuses + map

- Flip [specs/config.md](../specs/config.md) and [specs/agent.md](../specs/agent.md) `Updated → Implemented`, and their rows in [specs/_index.md](../specs/_index.md).
- Mark this plan `Done` and its [plans/_index.md](_index.md) row.
- No new/renamed `src/wica/` module and no spec frontmatter path changes, so the Project map and spec `code:`/`tests:` lists are unaffected — but re-run `tests/test_project_map.py` to confirm.

## Verification

```
uv run ruff check .
uv run pyright
uv run pytest
uv run pytest tests-e2e -k fake
```

Plus a manual demo launch from the repo root to confirm the absolutized `system_prompt_file` resolves (the one behavior most at risk from this change).

## Out of scope

- General `${VAR}` interpolation for arbitrary fields (config.md OQ #1) — still deferred; only `api_key_env` resolves an env var.
- Any change to the JSON shape, provider set, or `hf_provider` semantics.
