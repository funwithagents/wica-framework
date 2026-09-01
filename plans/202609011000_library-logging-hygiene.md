# Library logging hygiene: stop configuring logging, drop the config level

**Status:** Done

## Motivation

WICA is a library, and the Python stdlib guidance for libraries is narrow: a library
should only *emit* records (via `logging.getLogger(__name__)`) and must **not** configure
logging — no `setLevel`, no handlers, no `basicConfig`. Deciding what is shown, and where,
is the embedding application's policy.

The current design inverts this:

- `config.py` exposes `apply_logging(level)` which does
  `logging.getLogger("wica").setLevel(...)` — the library setting its own level policy.
- `WicaConfig` carries a top-level `logging` field, baking that policy into the framework's
  config surface.
- `Wica.init` calls `apply_logging(config.logging)`.
- Meanwhile the one thing a library *should* do — attach a `logging.NullHandler()` to its
  top-level `wica` logger so records don't hit the stdlib last-resort handler when the app
  hasn't configured logging — is **missing**.

The good part stays: every module already uses `logging.getLogger(__name__)`, and the
DEBUG/INFO/WARNING discipline (see [agent.md](../specs/agent.md)) is a fine convention — it
describes what the library *emits*, not what it *shows*.

## Changes

### Library (`src/wica/`)

1. `config.py`: remove the `logging` field from `WicaConfig`; remove `logging` from
   `_WICA_ALLOWED` (so an unknown `logging` key is now *rejected* by the strict loader);
   drop the level parsing/validation in `_parse_wica_block`; delete `apply_logging` and the
   now-unused `import logging`.
2. `wica.py`: drop the `apply_logging` import and the `apply_logging(config.logging)` call in
   `Wica.init`; update the docstrings that mention applying logging.
3. `__init__.py`: stop exporting `apply_logging`; add
   `logging.getLogger("wica").addHandler(logging.NullHandler())` — the one
   library-appropriate configuration.

### Application (`examples/`)

4. `conversation_demo.py`: the demo is an *application*, so configuring logging is correct
   here. Keep `basicConfig(WARNING)` + `getLogger("wica").setLevel(INFO)` **hardcoded** (no
   env var for now, per decision); stop reading `wica_config.logging`; drop the
   `apply_logging` import and its use in the explore-mode fallback; fix the header comment.
5. Strip `"logging": ...` from the four `examples/*.config.json` files (the e2e configs
   never carried it).

### Tests

6. `tests/test_config.py`: drop the logging-field assertions/cases
   (`_wica_dict` no longer injects `logging`, `test_wica_config_from_dict_defaults_logging`,
   `test_logging_wrong_type_raises`, the `logging`-level assertion in
   `test_from_json_then_wica_init_composes`). No test is added for the absent field — the
   generic unknown-key rejection is already covered by `test_unknown_top_level_key_raises`.
7. `tests/test_wica.py`: remove `test_init_applies_logging` (no replacement — logging is
   not the library's concern to test), drop its now-unused `logging` import, and drop the
   `logging_level` parameter from the `fake_config` helper.

### Specs

8. [config.md](../specs/config.md): remove the `logging` field from the JSON example and
   field table, the "Applying `logging`" flow bullet, and the `apply_logging` mention in the
   public-API paragraph. Note logging is app policy, not framework config.
9. [wica.md](../specs/wica.md): `init` no longer applies logging; drop the "Calls
   `apply_logging`" construction step and the `logging` references.
10. [agent.md](../specs/agent.md): clarify the DEBUG/INFO/WARNING levels are what the library
    *emits*; the embedding application owns handlers/levels (what is shown), and the library
    installs only a `NullHandler`.

All three specs are `Implemented`; they flip to `Updated` while code lags, then back to
`Implemented` once this plan is `Done`.

## Verification

`uv run ruff check .` · `uv run ruff format .` · `uv run pyright` · `uv run pytest`
