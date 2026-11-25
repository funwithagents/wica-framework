# Setup items one by one: required, prepared each on its own

**Status:** Done

Brings [specs/setup.md](../specs/setup.md) onto version 1 of the setup contract as it now stands: every `check_setup` item says whether it is `required`, and `prepare` prepares one item, named by its id.

**Why.** The contract's project was renamed (`app-setup`, imported as `app_setup`), and its contract changed: an app shows each item's preparation on its own, and starts a session while an item it does not need yet is still missing. WICA has nothing to prepare, so the change is small, but its functions must take the new shapes or the conformance check refuses them.

## Steps

1. **The dependency renamed**: `app-setup` in the `dev` group and `[tool.uv.sources]` (git, `main`), `app_setup` in the imports, the mentions in the specs, docs and the project map; `uv.lock` pins its current `main`.
2. **`check_setup`**: `"required": True` on every item (the extra, the key, the prompt file: a session cannot start without any of them, and only the user can provide them).
3. **`prepare(config, item, *, on_progress=None, base_dir=None)`**: no item is preparable, so every id raises `ValueError` naming it; `on_progress` is never called.
4. **The spec** says so; back to `Implemented` once the gate passes.

## Tests

`tests/test_setup.py`: the conformance check over the examples (it refuses a `prepare` not taking the item); every item carries the six keys, required and not preparable, for a known provider, a compatible server and an unknown provider; `prepare` raises `ValueError` naming the id, for an item of the config and for one that is not, without reporting progress. No concurrency test: there is no item to prepare.
