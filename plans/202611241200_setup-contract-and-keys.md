# The setup contract and keys never in a config

**Status:** Done

Builds [specs/setup.md](../specs/setup.md) and the key handling of [specs/config.md](../specs/config.md) ("API key: never in the config").

**Why.** An application that sets the agent up for its users (a form, a setup page) needs to know, before anything is installed, what the config looks like, which extra it needs, what is missing on the machine, and whether the key works. And a config that can hold a key cannot be shared: the key moves out of it, next to it.

## Steps

1. **`src/wica/fields.py`**, a verbatim copy of the family's shared field helper (`Field`, `read_fields`, `field_problems`, `fields_schema`), with its tests (`tests/test_fields.py`).
2. **The fields declared once** in `config.py`: the common fields, one `Provider` per known provider (title, kind, extra, packages, own fields with `key_env_field`), `agent_problems` / `wica_problems` as the one validation the loaders, `AgentConfig.__post_init__` and `check_config` share. Messages keep the `agent: ` prefix.
3. **Keys.** `api_key` removed from `AgentConfig` and refused at load; `api_key_env` defaults to the provider's variable; `resolve_api_key(config, secrets)`; `MissingSecretError` with `MissingEnvError` as a deprecated alias (module `__getattr__`, `.env_var`); `secrets` through `Wica.init` → `Agent` → `build_chat_model`; the key always passed explicitly; `import_provider` gives the install hint.
4. **`src/wica/setup.py`**: the seven functions; `probe` as one free listing/retrieval call per provider.
5. **Example configs** moved to `examples/configs/<provider>.json` (no key, the default variable), plus `fake.json` and `openai-chat-completions-compatible.json`; the demo reads `examples/configs/anthropic.json` until its rebuild.
6. **`module-setup`** as a dev dependency (git, `main`) for the conformance check.
7. **Callers and docs**: the benchmark matrix (the default variable gates an entry), the e2e support (`MissingSecretError`), the configuration and API references, the guides, the README.

8. **After review**: `model` declared in each provider's branch (a switch empties it); `check_setup`, `prepare`, `options` and `probe` take `base_dir` (module-setup's contract, its PR #7), a relative `system_prompt_file` located against it as `WicaConfig.from_dict` does; module-setup at `rev = "base-dir"` until that PR is merged.

## Tests

`tests/test_setup.py` (the conformance check over the examples, the schema, every problem at its path in the loader's words, the base install with every provider import blocked, `check_setup`'s items, `probe` over stubbed SDK clients); `tests/test_config.py` (the refused `api_key`, the default variables, `secrets` first, `MissingSecretError` and its alias, direct construction validated like the loader).

## Breaking changes

- `api_key` is refused in a config and gone from `AgentConfig`.
- With no `api_key_env`, the provider's usual variable is read by WICA and passed explicitly (`GEMINI_API_KEY` only for Gemini, `HF_TOKEN` only for the Hub); no package reads its own variable any more.
- `MissingEnvError` is deprecated (same class as `MissingSecretError`); its message changed.
- Direct construction of `AgentConfig` validates field types and bounds; an empty `model` or `hf_provider` is refused. Some validation messages are reworded.
- The demo's configs moved from `examples/conversation_demo/agent*.config.json` to `examples/configs/`.
