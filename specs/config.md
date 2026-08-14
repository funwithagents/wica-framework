---
code:
  - src/wica/config.py
tests:
  - tests/test_config.py
---

# Config

**Status:** Stable

## Purpose

A **framework config** lets you start WICA — or at least an Agent — from a single JSON file rather than constructing config objects in code. It holds the data needed to stand up an Agent against a provider: **provider**, **model**, **api key**, **system prompt**, and any extra **model params**. The goal is that switching provider/model, changing the persona, or moving between deployments is a file edit, not a code change.

This closes the deferred item in [agent.md](agent.md) ("Future improvements": *"`AgentConfig` from env/file. v1 constructs `AgentConfig` directly in code; loading it from a config file or environment variables is unbuilt."*) and settles the "source TBD (env / file)" note under agent.md's "Provider-agnostic model, from config".

## Settled

### Framework-level, not just agent-level

The config is a **top-level framework config** with an `agent` block nested inside it, not a bare `AgentConfig` at the root. Today the only block is `agent` plus a `logging` level, but nesting from the start means later concerns (World seeding, Command wiring, multiple agents) get added as sibling blocks without reshaping every existing file. The top-level object is `WicaConfig`.

```json
{
  "logging": "INFO",
  "agent": {
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "api_key": "sk-...",
    "system_prompt": "You are Wica, a friendly social robot...",
    "model_kwargs": { "temperature": 0.7 }
  }
}
```

| Field | Location | Required | Notes |
|---|---|---|---|
| `logging` | top level | no (default `"INFO"`) | Level applied to the `wica.*` loggers, mirroring the demo's `WICA_LOG` handling ([conversation_demo.py](../examples/conversation_demo.py)) |
| `provider` | `agent` | **yes** | Selects the model backend. Most values pass straight through as LangChain's `model_provider` to `init_chat_model` (`anthropic`, `openai`, …); the special value `huggingface-hub` takes WICA's own construction path (see "Providers" below) |
| `model` | `agent` | **yes** | Model name passed to `init_chat_model` (for `huggingface-hub`, the Hub `repo_id`, e.g. `meta-llama/Llama-3.3-70B-Instruct`) |
| `api_key` | `agent` | no | Literal key (see "API key"); at most one of `api_key`/`api_key_env` |
| `api_key_env` | `agent` | no | Name of an env var the key is read from at load time (see "API key"); at most one of `api_key`/`api_key_env` |
| `system_prompt` | `agent` | **one of** | Inline persona string (see "System prompt") |
| `system_prompt_file` | `agent` | **one of** | Path to a file holding the persona, resolved relative to the config file (see "System prompt") |
| `model_kwargs` | `agent` | no (default `{}`) | Extra params forwarded to `init_chat_model` (e.g. `temperature`) |
| `hf_provider` | `agent` | no (default `"auto"`) | **Only** for `provider: "huggingface-hub"`: the Hub Inference **backend** provider (`auto`/`fireworks-ai`/`together`/…), forwarded to `HuggingFaceEndpoint(provider=…)`. Ignored by other providers (see "Providers") |

### Providers

WICA recognizes a small set of `provider` values, each backed by an optional integration package installed as an **extra** (packaging lives in [project.md](project.md)):

| `provider` | Extra → package |
|---|---|
| `anthropic` | `wica[anthropic]` → `langchain-anthropic` |
| `openai` | `wica[openai]` → `langchain-openai` |
| `huggingface-hub` | `wica[huggingface-hub]` → `langchain-huggingface` |

Core `wica` bundles **no** provider; selecting one whose extra isn't installed fails at runtime with a clear `ImportError`, never silently.

Two of these are ordinary LangChain providers — `anthropic`, `openai`, and any other LangChain-supported value are passed through as `model_provider` to `init_chat_model`, so switching between them is a pure config edit. **`huggingface-hub` is WICA-specific:** it targets the Hugging Face Hub's serverless Inference Providers and the Agent constructs it on its own path rather than via `init_chat_model`. Config-wise that adds exactly one field — **`hf_provider`**, naming the Hub **backend** (`auto`/`fireworks-ai`/…). *How* that model is built and *why* it bypasses `init_chat_model` (including how the resolved API key is forwarded) is an Agent concern — see [agent.md](agent.md), "Provider-agnostic model, from config".

### JSON, loaded via plain dataclasses

The config objects stay **plain dataclasses** — `AgentConfig` plus `WicaConfig`, both now in [config.py](../src/wica/config.py) (see "Module placement" — `AgentConfig` moved here out of `agent.py`) — with hand-written `from_dict(data)` / `from_json(path)` classmethods. No pydantic dependency is added; agent.md's "pydantic-style" phrasing is treated as intent (a small validated settings object), not a mandate to adopt pydantic.

Loading is **strict**, so a broken file fails loudly at load time rather than silently doing nothing:

- **Missing required keys** (`provider`, `model`, and exactly one of `system_prompt`/`system_prompt_file` — see "System prompt") raise a clear error naming the missing key and its block.
- **Unknown keys are rejected** — a typo like `"modl"` or `"systemprompt"` raises rather than being ignored, which would otherwise leave the real field on its silent default.
- **Wrong types** (e.g. `model_kwargs` not an object) raise with the offending key named.

The errors are WICA's own (not raw `KeyError`/`json.JSONDecodeError` leaking through), so a misconfigured file gives an actionable message.

### API key: literal or env reference

The key can be given two ways, **at most one** of them:

- **Literal** — `"api_key": "sk-..."`, the raw key in the file.
- **Env reference** — `"api_key_env": "WICA_ANTHROPIC_API_KEY"`, the *name* of an environment variable the key is read from at load time.

Both resolve, during loading, to a plain `api_key: str | None` on `AgentConfig`, which `Agent.from_config` passes straight to `init_chat_model(..., api_key=...)` — the same path `tests-e2e/support.py` already uses — when set. (`huggingface-hub` bypasses `init_chat_model` and forwards this same resolved value under its own token kwarg — an Agent construction detail, see [agent.md](agent.md).) When **neither** is given, nothing is passed and `init_chat_model` reads the provider's standard env var (`ANTHROPIC_API_KEY`, …) exactly as the code does today, so existing env-based setups keep working and env stays the zero-config default. Providing **both** is a `ConfigError`. Unlike `system_prompt_file`, env resolution needs no base directory, so **both `from_dict` and `from_json`** honour `api_key_env` (only the file indirection is `from_json`-only).

**A referenced env var that is unset raises `MissingEnvError`** — a `ConfigError` subclass naming the variable — so a misconfiguration is loud rather than silently keyless. Callers that prefer to degrade catch it: the demo falls back to explore-only, the e2e helper skips. A *structural* problem (typo'd field, missing required key) is a plain `ConfigError`, so the two are distinguishable.

**This is what makes a config committable.** An env-reference config carries no secret, so the conversation-demo and e2e config files are **checked in** using `api_key_env`. A config that uses a **literal** `api_key` is a plaintext secret on disk and should be **git-ignored** (with a committed `*.example.json` placeholder alongside). Literal keys are the convenience for a quick local start; env reference is the path for anything shared or committed — and it reframes the project's WICA-namespaced key (`WICA_<PROVIDER>_API_KEY`) as simply *what `api_key_env` points at*, not a bespoke routing step.

### System prompt: inline or file

The persona can be given **inline** (`system_prompt`) or **by reference** (`system_prompt_file`), so prompts can live as their own files in a `prompts/` folder instead of being escaped into JSON — personas are long and awkward to embed. **Exactly one** must be present: providing both, or neither, is a load-time error (see strict loading above), so there's never ambiguity about which wins.

```json
{
  "agent": {
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "system_prompt_file": "prompts/wica.md"
  }
}
```

- **Path resolution is relative to the config file's own directory**, not the process CWD. A config at `deploy/agent.config.json` with `"system_prompt_file": "prompts/wica.md"` reads `deploy/prompts/wica.md` regardless of where the process is launched from — so a config-plus-prompts folder is relocatable and CWD-independent. Absolute paths are used as-is.
- Resolution and file reads happen **at load time**, inside `from_json`: `AgentConfig` still ends up holding a resolved `system_prompt: str`, so nothing downstream (`Agent.from_config`, the Agent loop) knows or cares whether it came from inline text or a file. `from_dict` alone — which has no file location to resolve against — accepts only inline `system_prompt`; the file indirection is a `from_json` concern.
- A missing or unreadable `system_prompt_file` raises a clear WICA error naming the resolved path.

### Flow into the Agent

- `WicaConfig.from_json(path)` → a `WicaConfig` holding an `AgentConfig`.
- `AgentConfig` gains an `api_key: str | None = None` field; `Agent.from_config` forwards it to `init_chat_model` only when set (see "API key").
- **No `Agent.from_config_file` convenience method.** "Start the agent from a file" is the caller composing two calls itself: `wica_config = WicaConfig.from_json(path)` then `Agent.from_config(wica_config.agent, **kwargs)` — the same `**kwargs` carrying the code-only wiring that doesn't belong in a JSON file (World instance, event loop, output sink, instrumentation hooks). Keeping this as two explicit calls, rather than folding it into one, keeps `Agent`'s API surface from growing a second construction path for what is otherwise a two-line composition; it also makes the `apply_logging` call (below) visible at the call site instead of implicit inside a helper.
- Applying `logging` is the caller's job at startup (`apply_logging(wica_config.logging)`, typically right after `from_json` and before `Agent.from_config`), not something either config-loading or `Agent.from_config` does implicitly.

### Module placement

A new `src/wica/config.py` module owns **both config dataclasses** — `AgentConfig` (moved out of [agent.py](../src/wica/agent.py)) and the new `WicaConfig` — plus the file-loading layer. Config is one concept and lives in one place.

Direction of dependency: `config.py` holds only **pure data + loading** and imports nothing from `agent.py`; `agent.py` imports `AgentConfig` from `config.py`. This keeps the dependency one-way (no import cycle): model construction (`init_chat_model`) stays in `Agent.from_config` in `agent.py`, so `config.py` never needs to know about `Agent`. `WicaConfig` therefore carries no `build_agent()` method, and `Agent` carries no `from_config_file` — "file → running Agent" is the two-call composition described above ("Flow into the Agent"), not a single convenience method on either side.

The public API (`__init__.py`) re-exports the config surface a caller composes an Agent from a file with — `AgentConfig`, `WicaConfig`, the `apply_logging` startup helper, and the `ConfigError`/`MissingEnvError` types a caller catches (the demo falls back to explore-only on `MissingEnvError`, the e2e helper skips) — all from `config.py`. Adding a top-level `src/wica/` module means the **Project map in AGENTS.md** and its drift-guard test (`tests/test_project_map.py`) are updated in the same change.

## Open questions

1. **Env references beyond the api key.** Env resolution exists for the key (`api_key_env`) but is a dedicated field, not general `"${VAR}"` string interpolation any field (`model`, `system_prompt`, …) could use. Generalizing to a single interpolation syntax is deferred until something needs it; `api_key_env` covers the one case that matters now.
2. **Format beyond JSON.** JSON only for now. TOML/YAML (comments, less escaping) may be nicer for a human-edited config; deferred until there's a reason to add a parser dependency.
3. **How far "framework config" reaches.** The `agent` block is the only real content today. Whether World seeding, Command registration, or multiple-agent wiring ever move into config — versus staying code, since they're callables/objects a JSON file can't express — is unresolved. The nesting exists so that's an additive decision, not a reshape.
4. **Schema/validation depth.** Hand-written strict validation (required keys, unknown-key rejection, coarse type checks) is the v1 bar. Whether that grows into a published schema (JSON Schema) or richer validation is deferred; it interacts with Open question #2 (adopting pydantic would come "for free" with a format/parser change).
