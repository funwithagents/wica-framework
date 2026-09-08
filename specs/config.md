---
code:
  - src/wica/config.py
tests:
  - tests/test_config.py
---

# Config

**Status:** Implemented

## Purpose

A **framework config** is the plain data used to start WICA — or at least an Agent — regardless of whether a caller constructs it directly in Python, parses it from a dictionary embedded in larger application settings, or loads it from a dedicated JSON file. It holds the data needed to stand up an Agent against a provider: **provider**, **model**, **api key**, **system prompt**, and any extra **model params**. The object is the stable boundary; dictionary and JSON loaders are input adapters that add strict runtime validation and, when a source directory exists, locate a relative prompt path. This is the config-driven Agent construction described in [agent.md](agent.md), "Provider-agnostic model, from config".

## Settled

### Framework-level, not just agent-level

The config is a **top-level framework object** with an `agent` field, not a bare `AgentConfig`. Its dictionary/JSON representation therefore has an `agent` block at the root. Today that is the only block, but nesting from the start means later concerns (World seeding, Command wiring, multiple agents) can be added as sibling fields without reshaping every existing representation. The top-level object is `WicaConfig`.

```json
{
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
| `provider` | `agent` | **yes** | Selects the model backend. Most values pass straight through as LangChain's `model_provider` to `init_chat_model` (`anthropic`, `openai`, …); the special value `huggingface-hub` takes WICA's own construction path (see "Providers" below) |
| `model` | `agent` | **yes** | Model name passed to `init_chat_model` (for `huggingface-hub`, the Hub `repo_id`, e.g. `meta-llama/Llama-3.3-70B-Instruct`) |
| `api_key` | `agent` | no | Literal key (see "API key"); at most one of `api_key`/`api_key_env` |
| `api_key_env` | `agent` | no | Name of an env var the key is read from at **Agent build** (see "API key"); at most one of `api_key`/`api_key_env` |
| `system_prompt` | `agent` | **one of** | Inline persona string (see "System prompt") |
| `system_prompt_file` | `agent` | **one of** | Path to the persona file; **located** relative to the config file at load, **read** at Agent build (see "System prompt") |
| `model_kwargs` | `agent` | no (default `{}`) | Extra params forwarded to `init_chat_model` (e.g. `temperature`) |
| `hf_provider` | `agent` | no (default `"auto"`) | **Only** for `provider: "huggingface-hub"`: the Hub Inference **backend** provider (`auto`/`fireworks-ai`/`together`/…), forwarded to `HuggingFaceEndpoint(provider=…)`. Ignored by other providers (see "Providers") |

### Providers

WICA recognizes a small set of `provider` values, each backed by an optional integration package installed as an **extra** (packaging lives in [project.md](project.md)):

| `provider` | Extra → package |
|---|---|
| `anthropic` | `wica[anthropic]` → `langchain-anthropic` |
| `openai` | `wica[openai]` → `langchain-openai` |
| `huggingface-hub` | `wica[huggingface-hub]` → `langchain-huggingface` |
| `fake` | none — built in (test-only) |

Core `wica` bundles **no** provider; selecting one whose extra isn't installed fails at runtime with a clear `ImportError`, never silently.

**`fake` is a testing provider**, not a real backend: it builds a deterministic, network-free, key-less scripted model (`FakeChatModel`) whose responses come from `model_kwargs` (`script`/`default`/`loop`/`delay_s`) rather than any API. `api_key`/`api_key_env` are unnecessary and ignored, and `model` is a free-text label. It exists to drive the Agent's whole loop deterministically in tests — see [fake-provider.md](fake-provider.md).

Two of these are ordinary LangChain providers — `anthropic`, `openai`, and any other LangChain-supported value are passed through as `model_provider` to `init_chat_model`, so switching between them is a pure config edit. **`huggingface-hub` is WICA-specific:** it targets the Hugging Face Hub's serverless Inference Providers and the Agent constructs it on its own path rather than via `init_chat_model`. Config-wise that adds exactly one field — **`hf_provider`**, naming the Hub **backend** (`auto`/`fireworks-ai`/…). *How* that model is built and *why* it bypasses `init_chat_model` (including how the resolved API key is forwarded) is an Agent concern — see [agent.md](agent.md), "Provider-agnostic model, from config".

### Logging is not framework config

WICA is a **library**, and by the Python stdlib convention a library only *emits* log records — every module holds `logging.getLogger(__name__)` under the `wica.*` tree — and never *configures* logging: no `setLevel`, no handlers, no `basicConfig`. Deciding what is shown, at what level, and where, is the **embedding application's** policy. So there is deliberately **no `logging` field on `WicaConfig`** and no `apply_logging` helper: the framework config carries only what stands up the Agent, not what the host app chooses to surface. The one library-appropriate action lives in [__init__.py](../src/wica/__init__.py) — a `logging.NullHandler()` on the top-level `wica` logger, so records don't hit the stdlib last-resort handler when the app hasn't configured logging. The emitted-level convention (DEBUG/INFO/WARNING) is documented in [agent.md](agent.md) as *what the library emits*, distinct from what the application shows. A demo or app configures its own logging in application code (the conversation demo calls `basicConfig` and sets the `wica` level itself — see [conversation_demo.py](../examples/conversation_demo.py)).

### Config objects mirror the JSON; resolution is deferred to use

The config dataclasses are a **1:1 mirror of the dictionary/JSON representation** — plain data, holding the same fields without resolving their references. Each supported indirection keeps *both* of its fields on the object rather than collapsing to one resolved value:

- `api_key` **and** `api_key_env` (the env var *name*) both survive onto `AgentConfig`.
- `system_prompt` **and** `system_prompt_file` both survive onto `AgentConfig`.

The **resolution** — reading the env var, reading the prompt file — happens **when the config is used**, at Agent build (`Agent.__init__` / `build_chat_model`), not at load. A loaded `AgentConfig` is therefore inert, side-effect-free, and reusable: constructing one never touches the environment or the filesystem, and the same object can be built inline in code (set `system_prompt`/`api_key` directly — the `*_file`/`*_env` fields are simply the file-driven alternatives, absent inline).

The two references are resolved by the same principle, with one asymmetry driven by *what the reference means*:

- **`api_key_env` is context-free.** An env var name means the same thing anywhere, so resolving it needs no base directory. The object holds the name verbatim; the consumer reads `os.environ` at build.
- **`system_prompt_file` is a path**, meaningful only relative to the file that declared it — and the config file's own directory exists only during `from_json`. So `from_json` **locates** it there — joining the config directory to produce an **absolute** path stored back on the object — but does **not read** it; the read defers to build like the env var. Locating is not resolution: no I/O, deterministic, and it folds the ephemeral base-dir context into the path itself rather than a separate `source_dir` field. `from_dict(data, *, base_dir=None)` has no config directory of its own, so it takes one optionally: given a `base_dir`, it locates a relative `system_prompt_file` against that directory — the same absolutization `from_json` performs against the config file's own directory; given none (the default), it stores the path as given and a *relative* one resolves against the process CWD when read — a code-path caller's own responsibility, where `from_json`'s (and `base_dir`'s) absolutization is the CWD-independent path.

### Plain dataclasses with dictionary and JSON loaders

The config objects stay **plain dataclasses** — `AgentConfig` plus `WicaConfig`, both in [config.py](../src/wica/config.py) — with hand-written `from_dict(data)` / `from_json(path)` classmethods. Callers may construct the dataclasses directly; that path relies on the caller and static type checking rather than running the loaders' validation. No pydantic dependency is added — the config objects are a small settings layer, not a pydantic model.

Parsing through `from_dict` / `from_json` performs strict **structural validation** but no environment lookup or prompt-file read (those are deferred to build, above), so broken external configuration fails loudly rather than silently doing nothing:

- **Missing required keys** (`provider`, `model`, and exactly one of `system_prompt`/`system_prompt_file` — see "System prompt") raise a clear error naming the missing key and its block.
- **Unknown keys are rejected** — a typo like `"modl"` or `"systemprompt"` raises rather than being ignored, which would otherwise leave the real field on its silent default.
- **Wrong types** (e.g. `model_kwargs` not an object) raise with the offending key named.
- **Mutually exclusive pairs** — `api_key` with `api_key_env`, or `system_prompt` with `system_prompt_file` — raise; these are structural (about the file's shape, not the values), so they stay eager even though the *values* resolve later.

The errors are WICA's own (not raw `KeyError`/`json.JSONDecodeError` leaking through), so misconfigured external data gives an actionable message. Errors that depend on the *outside world* — an unset env var, an unreadable prompt file — are not structural and surface later, at build (see "API key", "System prompt").

### API key: literal or env reference

The key can be given two ways, **at most one** of them — both stored **verbatim** on `AgentConfig` (`api_key: str | None`, `api_key_env: str | None`), neither read at load:

- **Literal** — `"api_key": "sk-..."`, the raw key in the file.
- **Env reference** — `"api_key_env": "WICA_ANTHROPIC_API_KEY"`, the *name* of an environment variable.

**The pair resolves at Agent build, inside `build_chat_model`**, to an effective key: literal `api_key` if set; else the value of the named env var; else `None`. A resolved key is passed straight to `init_chat_model(..., api_key=...)` — the same path `tests-e2e/support.py` already uses — when set. (`huggingface-hub` bypasses `init_chat_model` and forwards this same resolved value under its own token kwarg — an Agent construction detail, see [agent.md](agent.md).) When **neither** is given, nothing is passed and `init_chat_model` reads the provider's standard env var (`ANTHROPIC_API_KEY`, …) exactly as the code does today, so existing env-based setups keep working and env stays the zero-config default. Because resolution needs no base directory, `from_dict` and `from_json` store `api_key_env` identically — the field is fully code-path-agnostic.

Providing **both** is a `ConfigError` at **load** — a structural mistake in the file, caught eagerly. A referenced env var that is **unset** raises `MissingEnvError` (a `ConfigError` subclass naming the variable) at **build**, when the key is actually read — so callers that prefer to degrade catch it around Agent build (`Wica.init`, or a direct `Agent(...)`): the demo falls back to explore-only, the e2e helper skips. A *structural* problem (typo'd field, both-given, missing required key) is a plain `ConfigError` at load, so the two remain distinguishable.

**This is what makes a config committable.** An env-reference config carries no secret, so the conversation-demo and e2e config files are **checked in** using `api_key_env`. A config that uses a **literal** `api_key` is a plaintext secret on disk and should be **git-ignored** (with a committed `*.example.json` placeholder alongside). Literal keys are the convenience for a quick local start; env reference is the path for anything shared or committed — and it reframes the project's WICA-namespaced key (`WICA_<PROVIDER>_API_KEY`) as simply *what `api_key_env` points at*, not a bespoke routing step.

### System prompt: inline or file

The persona can be given **inline** (`system_prompt`) or **by reference** (`system_prompt_file`), so prompts can live as their own files in a `prompts/` folder instead of being escaped into JSON — personas are long and awkward to embed. Both fields are kept on `AgentConfig`; **exactly one** must be present: providing both, or neither, is a load-time `ConfigError` (structural — see strict loading above), so there's never ambiguity about which wins.

```json
{
  "agent": {
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "system_prompt_file": "prompts/wica.md"
  }
}
```

- **The path is located relative to the config file's own directory at load, then read at Agent build.** `from_json` joins its own directory to the relative path and stores the resulting **absolute** path back on `AgentConfig` (a *locate*, no file I/O); `Agent.__init__` reads that path when it builds the Agent. A config at `deploy/agent.config.json` with `"system_prompt_file": "prompts/wica.md"` therefore reads `deploy/prompts/wica.md` regardless of where the process is launched from — the path is bound to the config's location, not the process CWD, so a config-plus-prompts folder stays relocatable and CWD-independent — while the file read itself stays deferred like the api key. Absolute paths in the file are stored as-is.
- **`from_dict(data, *, base_dir=None)` locates against a caller-supplied directory, or stores verbatim.** It has no config directory of its own, so it accepts one: with a `base_dir` (a `str` or `Path`), a relative `system_prompt_file` is absolutized against it — the same locate `from_json` does against the config file's directory (absolute paths are stored as-is either way). Without one, the path is stored verbatim and a *relative* one resolves against the process CWD when read. Inline `system_prompt` is the ordinary code-path choice; a code caller that wants a file passes `base_dir` (e.g. `os.path.dirname(path)` when the dict came from a file), gives an absolute path, or uses `from_json`.
- `Agent.__init__` resolves the pair to the effective **persona**: inline `system_prompt` if set, else the contents of `system_prompt_file`. `AgentConfig` never holds a resolved `system_prompt` merged from the file — the field stays what the JSON carried — so resolution is the consumer's, but nothing downstream (`Agent.__init__`, the Agent loop) knows or cares whether the text came from inline or a file. Config resolution yields the persona *only*; the Agent then **composes** the final system prompt as `persona + WICA runtime primer` (see [agent.md](agent.md), "System prompt composition"), so the string the model receives is not strictly the config's verbatim text.
- A missing or unreadable `system_prompt_file` raises a clear WICA `ConfigError` naming the path — at **build**, when it's read.

### Flow into the Agent

The config is consumed through the `Wica` facade (see [wica.md](wica.md)), which owns the World+Agent it builds:

- `WicaConfig(agent=AgentConfig(...))` constructs the plain objects directly when Python owns the settings. It does not run loader validation.
- `WicaConfig.from_dict(data, base_dir=...)` parses and validates a WICA-shaped mapping, optionally locating a relative prompt path against the caller-supplied directory. A larger application passes its WICA subsection rather than unrelated sibling settings.
- `WicaConfig.from_json(path)` reads a dedicated JSON document, parses and validates that same shape, and derives the prompt-path base directory from the JSON file's location.
- Every path yields a `WicaConfig` holding an **unresolved** `AgentConfig`: no API-key environment lookup or prompt-file read has happened yet.
- `Wica.init(wica_config, **kwargs)` is where the config is put to work. It builds a `World`, constructs `Agent(wica_config.agent, world=…, **code_only_kwargs)`. **`Agent.__init__` is where resolution happens**: it resolves the system prompt (see "System prompt") and calls `build_chat_model`, which resolves the api key (see "API key"), before constructing the model. This is the point where `MissingEnvError` or an unreadable prompt file surfaces — so a caller that degrades wraps `Wica.init`, not `from_json`.
- **Config-driven construction *is* the constructor: exactly one build path.** `Agent.__init__` takes the `AgentConfig` directly. `Wica.init` likewise takes the already-created `WicaConfig`; the `**kwargs` carry code-only wiring a config object should not express (event loop, output sink/output Command, `coalesce_window`). There is deliberately no path-taking convenience such as `Wica.from_json`: it would privilege one input adapter and hide the config object the caller often wants.

### Module placement

A new `src/wica/config.py` module owns **both config dataclasses** — `AgentConfig` (moved out of [agent.py](../src/wica/agent.py)) and the new `WicaConfig` — plus the file-loading layer. Config is one concept and lives in one place.

The **resolution helpers** (`resolve_api_key(config)`, `resolve_system_prompt(config)`) live in `config.py` too — the env-read and file-read semantics are config's concern — as **pure functions over an `AgentConfig`**, *called* by the agent-side consumers (`build_chat_model`, `Agent.__init__`) at build. So config owns *how* a field resolves, while the consumer owns *when* — keeping the reads out of load without scattering the resolution logic into `agent.py`.

Direction of dependency: `config.py` holds only **pure data + parsing + resolution helpers** and imports nothing from `agent.py`; `agent.py` imports `AgentConfig` and the resolvers from `config.py`. This keeps the dependency one-way (no import cycle): model construction (`init_chat_model`/`build_chat_model`) stays in `agent.py`, so `config.py` never needs to know about `Agent`. `WicaConfig` therefore carries no `build_agent()` method — every source first produces a config object and then passes it to `Wica.init`, as described above ("Flow into the Agent").

The public API (`__init__.py`) re-exports the complete config surface a caller uses to compose a `Wica` — `AgentConfig`, `WicaConfig`, and the `ConfigError`/`MissingEnvError` types a caller catches around parsing or `Wica.init` (the demo falls back to explore-only on `MissingEnvError`, the e2e helper skips) — all from `config.py`. Adding a top-level `src/wica/` module means the **Project map in AGENTS.md** and its drift-guard test (`tests/test_project_map.py`) are updated in the same change.

## Open questions

1. **Env references beyond the api key.** Env resolution exists for the key (`api_key_env`) but is a dedicated field, not general `"${VAR}"` string interpolation any field (`model`, `system_prompt`, …) could use. Generalizing to a single interpolation syntax is deferred until something needs it; `api_key_env` covers the one case that matters now.
2. **Format beyond JSON.** JSON only for now. TOML/YAML (comments, less escaping) may be nicer for a human-edited config; deferred until there's a reason to add a parser dependency.
3. **How far "framework config" reaches.** The `agent` block is the only real content today. Whether World seeding, Command registration, or multiple-agent wiring ever move into config — versus staying code, since they're callables/objects a JSON file can't express — is unresolved. The nesting exists so that's an additive decision, not a reshape.
4. **Schema/validation depth.** Hand-written strict validation (required keys, unknown-key rejection, coarse type checks) is the v1 bar. Whether that grows into a published schema (JSON Schema) or richer validation is deferred; it interacts with Open question #2 (adopting pydantic would come "for free" with a format/parser change).
