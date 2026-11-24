---
code:
  - src/wica/setup.py
  - src/wica/fields.py
  - src/wica/config.py
tests:
  - tests/test_setup.py
  - tests/test_fields.py
---

# Setup

**Status:** Implemented

## Purpose

Everything an application needs to set the agent up for a user, without knowing any provider: which providers exist, what the config looks like, which extra a config needs, what it needs on the machine (the provider's package, its key, the prompt file), and one cheap call to find out whether the key and the model are good. An application draws a form from it, installs what is missing, and only then calls `Wica.init`.

Today this knowledge is spread over the loader, `build_chat_model` and the error raised when something is missing. A config can only be tried by building the Agent, which imports the provider's package and reads its key. This spec makes the same knowledge available as data, before anything is installed.

These functions implement the **module contract** of the `module-setup` project (its `specs/module-contract.md`), version 1. That spec is the reference for the shapes exchanged; this one says what WICA puts in them. WICA does not depend on `module-setup` at runtime: the contract is plain data, and `module-setup` is a dev dependency only, for the conformance check its tests run.

## The `setup` module

`wica.setup` exposes seven functions over the **WICA block** — the document `WicaConfig.from_dict` takes, `{"agent": {...}}` ([config.md](config.md)):

```python
def config_schema() -> dict: ...
def required_extras(config: dict) -> list[str]: ...
def check_config(config: dict) -> list[dict]: ...
def check_setup(config: dict) -> list[dict]: ...
def prepare(config: dict, on_progress: Callable[[dict], None] | None = None) -> None: ...
def options(name: str, config: dict) -> list[dict]: ...
def probe(config: dict, *, secrets: Mapping[str, str] | None = None) -> dict: ...
```

They take and return JSON-compatible data only, are synchronous, and import nothing outside WICA at module level. `config_schema`, `required_extras` and `check_config` read nothing but the config; `check_setup` also tries the provider's imports, reads the environment and looks for the prompt file. Only `probe` uses the network. `wica.setup` is reached as a submodule and not re-exported from `wica`.

| Function | Returns |
|---|---|
| `config_schema()` | `{"contract": 1, "distribution": "wica", "title": "Agent", "schema": …}` — the WICA block as a JSON Schema (below). The same value on every call |
| `required_extras(config)` | The extra the selected provider needs, as a requirement string: `["wica[anthropic]"]`; `[]` for `fake`, for a provider WICA does not know, and for anything that selects no provider. Reads `agent.provider` only, never raises |
| `check_config(config)` | Every problem that makes the loader refuse the config, each `{"path": "/agent/model", "message": …}` with a JSON Pointer into the block; `[]` when it is accepted |
| `check_setup(config)` | What this machine is missing to run it: the provider's extra, the key variable, the prompt file; each `{"id", "title", "status", "detail", "preparable"}` |
| `prepare(config, on_progress)` | Nothing: WICA downloads nothing (no item is preparable). Returns at once |
| `options(name, config)` | `[]` for every name: no list is offered (a model list per provider would cost requests) |
| `probe(config, *, secrets=None)` | One authenticated, free call to the provider, `{"ok", "message", "duration_s"}`. Never raises |

## Fields are declared once

The `agent` block's fields are declared as data, and the loader, `check_config` and `config_schema` read the same declarations, so the form and the validation cannot drift apart.

`src/wica/fields.py` provides `Field`, `read_fields` (each field's value, raising on the first invalid one), `field_problems` (every invalid field, same rules, same wording) and `fields_schema` (the JSON Schema properties). It uses the standard library only and knows nothing about agents; it is meant to move to a shared core library, which several libraries will import instead of carrying a copy. Its rules: a present string is non-empty unless the field allows it, a number is finite and within its bounds, a bool is never a number, an integer is never a float; without a `default` an explicit `null` counts as absent, with one it is rejected; `implied` is the value a form shows for a field whose absence the code fills in (or leaves to the provider), never used to validate.

`src/wica/config.py` declares:

- **The common fields**, valid with every provider and drawn on the `agent` object: `model` (required), `system_prompt` (may be empty), `system_prompt_file`, `history_reactions`, `history_budget`, `context_window`, `model_timeout`, `max_next_steps`, `max_retries`, `stream_output`; every tuning field `advanced`. `model_kwargs` is a free-form object, which `Field` has no kind for: it is checked by hand (an object) and drawn as raw JSON (`additionalProperties: true`, advanced).
- **The providers**, one `Provider` each: its `name`, `title`, `kind` (`hosted`, `local` or `fixture`), `extra`, the packages its extra provides, and its own fields — `api_key_env` (from `key_env_field`, defaulting to the provider's usual variable), `base_url` (the compatible server, required), `hf_provider` (the Hub, default `"auto"`).

| `provider` | Title | Kind | Extra | `api_key_env` default | Own fields |
|---|---|---|---|---|---|
| `anthropic` | Anthropic | hosted | `anthropic` | `ANTHROPIC_API_KEY` | `api_key_env` |
| `openai` | OpenAI | hosted | `openai` | `OPENAI_API_KEY` | `api_key_env` |
| `gemini` | Gemini | hosted | `gemini` | `GEMINI_API_KEY` | `api_key_env` |
| `mistral` | Mistral | hosted | `mistral` | `MISTRAL_API_KEY` | `api_key_env` |
| `huggingface-hub` | Hugging Face Hub | hosted | `huggingface-hub` | `HF_TOKEN` | `api_key_env`, `hf_provider` |
| `openai-chat-completions-compatible` | OpenAI-compatible server | local | `openai-compatible` | none | `base_url`, `api_key_env` |
| `fake` | Scripted fake | fixture | none | none (no key) | none |

**A provider WICA does not know** (any other LangChain id, whose package a library user installs) is still accepted, as today: it has no extra, no default key variable, and takes `api_key_env` without a default. It is not in the schema, so a form cannot offer it.

**Validation** of a block, used by the loader (first problem raised), by `AgentConfig.__post_init__` (direct construction, first problem raised) and by `check_config` (every problem):

1. not an object;
2. `api_key` present: refused, whatever its value ([config.md](config.md), "API key");
3. each unknown key;
4. `provider` missing or not a non-empty string;
5. `model` missing; then every field of the provider's fields (the common ones, its own, and `api_key_env`, `base_url` and `hf_provider` for a provider that does not declare them: still accepted and type-checked, as today, but not drawn);
6. `model_kwargs` not an object;
7. once the fields are valid, the rules across fields: exactly one of `system_prompt`/`system_prompt_file`; `base_url` present exactly with `openai-chat-completions-compatible`, and not in `model_kwargs` then.

Messages keep the `agent: ` prefix: `"agent: 'history_reactions' must be an integer >= 1"`, `"agent: missing required key 'model'"`, `"agent: unknown key 'modl'"`. A required field that is absent says so, rather than the generic "must be". An old provider name (`google_genai`) is renamed quietly by validation, with the deprecation warning logged once by `AgentConfig` as today.

## The schema

`config_schema()["schema"]`:

- **The root** — an object with one required property, `agent`.
- **`agent`** — titled **"Provider"**, with a description saying what it chooses; the common fields as its `properties` (`required: ["model"]`), and one `oneOf` branch per known provider, in the table's order. Each branch has the title `"<Title> (<kind>)"` ("Anthropic (hosted)", "Scripted fake (fixture)"), `provider` as a `const`, and its own fields; its `required` is `provider` plus its own required fields (`base_url`). Switching provider in a form drops the old branch's fields, so `api_key_env` goes back to the new provider's default.
- **`api_key_env`** carries `x-secret-env` and, where the provider has one, the default variable, so the Secrets tab lists it for a fresh config.
- A renamed provider has no branch: the schema offers current names only, `check_config` still accepts the old one.

## `check_setup`

Asked only about a config `check_config` accepts. Its items, in order:

1. **The extra** — `extra:wica[<extra>]`, titled `"<Title> extra"`, never preparable: `ready` ("Installed.") when the provider's packages import (the imports are tried, not looked up: a broken install shows), else `missing` with the install hint `build_chat_model` raises (one home: `import_provider`). `fake` and an unknown provider have none.
2. **The key** — `key:<VARIABLE>`, for the variable `api_key_env` names or, absent, the provider's default; `ready` when set and non-empty in the environment, else `missing` with a sentence naming the variable. None for `fake`, nor for a provider without a default when no variable is named.
3. **The prompt file** — `file:system_prompt_file`, when the config names one: `ready` when the file exists, else `missing`, never preparable; the detail gives the path looked at. A relative path is looked up against the process's working directory, as `from_dict` without `base_dir` does: the contract passes no directory (see Open questions). It is reported whether or not the extra imports: finding it needs no provider package.

Local and fast: no network, nothing built.

## `probe`

The **light call** rule: one authenticated, free request per provider, checking the key and that the configured model exists, never a chat request. It builds no Agent.

| Provider | Call | `ok` when |
|---|---|---|
| `fake` | none | at once: "The scripted fake needs nothing to run." |
| `anthropic` | the SDK's `models.retrieve(model)` | the model is found |
| `openai` | `models.retrieve(model)` | the model is found |
| `gemini` | `google.genai`'s `models.get(model=…)` | the model is found |
| `mistral` | `GET /v1/models` | the model is listed, under its id or an alias |
| `huggingface-hub` | the Hub's `model_info(model, expand=["inferenceProviderMapping"])` with the token | the model is served by an inference provider (by `hf_provider` when it is not `auto`) |
| `openai-chat-completions-compatible` | `GET <base_url>/models` with the key or the placeholder | the server answers; a model it does not list is said in the message, not a failure (several servers serve whatever model they loaded under any name) |
| unknown | none: the chat model is built (its package imported, its constructor run) | it builds; the message says nothing was sent |

- The key comes from `resolve_api_key(config, secrets)`, like the Agent's; no key is `ok: false` with the `MissingSecretError` message. A config `check_config` refuses is `ok: false` with its first problem; a missing extra, with the install hint.
- Each request has the config's `model_timeout` (10 s without one) and no retry.
- A refusal is reported in words: the key was refused (401/403), the model is not found (404), or the error's own message.
- `duration_s` covers the whole probe. It runs on the calling thread: no event loop is needed.

## Testing

`tests/test_fields.py` — the shared helper's own tests: `read_fields` and `field_problems` agree on every rule (an empty string, `NaN`, a bool as a number, a float as an integer, out of bounds, `null` with and without a default, a value outside `choices`); `fields_schema` emits each attribute and annotation.

`tests/test_setup.py`:

- The conformance check (`module_setup.contract.check_contract`) passes over every example config under `examples/configs/`.
- `config_schema()` is stable, has one branch per known provider in order, `api_key_env` with its default and `x-secret-env`, no `api_key` anywhere.
- Every example config is accepted, and names only keys the schema names.
- `check_config` reports each problem at its path (non-object, `api_key`, unknown key, missing provider or model, a bad field, both or neither prompt, `base_url` with the wrong provider), several at once; and for each, the loader raises the first of them, word for word.
- `config_schema`, `check_config` and `required_extras` import no provider package: run in a subprocess with every provider import blocked, where `check_setup` reports each provider's extra `missing` with the install hint.
- `required_extras`: the extra per provider, `[]` for `fake`, an unknown provider, `{}`, a non-object; every extra named is declared in `pyproject.toml`.
- `check_setup`: the installed extras `ready`; the key `missing` until set (empty counts as unset), under the default or the named variable; no key item for `fake` or a compatible server naming none; the prompt file `ready`/`missing`.
- `probe`: `fake` is ok; a missing key, a refused config and a missing extra are `ok: false` with their messages and raise nothing; the hosted probes over a stubbed SDK client: ok when the model is found, `ok: false` naming the model or the refused key otherwise, and nothing but the listing call is made.

## Open questions

1. **Relative paths in `check_setup`.** The contract passes no directory, so a relative `system_prompt_file` is looked up against the working directory, while an app builds the Agent with its folder as `base_dir`. An app avoids the mismatch with an absolute path (the conversation demo's default does); the contract may later carry the app's folder.
2. **A model list (`options("models")`)** is deferred: it would cost one request per refresh and the lists are long.
3. **`fields.py` is a copy** of the helper the family's libraries share, until a core library holds it.
