# `from_json` string parser + `from_json_file` loader

**Status:** Done

## Motivation

`WicaConfig.from_json(path)` today takes a **file path**, but by Python convention `from_json`
names a **string** parser (cf. `json.loads`, pydantic `model_validate_json(str)`) — the name is
misleading. Adopt the conventional triad of input adapters, each a thin layer over the next:

- `from_dict(data, *, base_dir=None)` — already-parsed mapping (unchanged).
- `from_json(text, *, base_dir=None)` — a JSON **string**.
- `from_json_file(path)` — a dedicated JSON **file**.

This also makes "locate relative to the config directory" uniformly just `base_dir` instead of a
behavior welded to file loading: `from_json_file` is the one adapter that fills `base_dir` in from
the file's own location; the other two accept it optionally (a string/dict carries no location).

Clean rename, no deprecated alias (pre-1.0, self-contained). Breaking change to the public
`WicaConfig.from_json` signature.

## Changes

### Code — [src/wica/config.py](../src/wica/config.py)

- **`WicaConfig.from_json(cls, text: str, *, base_dir=None)`** — `json.loads` the string (wrapping
  `JSONDecodeError` as `ConfigError("invalid JSON config: …")`), reject a non-object with
  `ConfigError`, then `_parse_wica_block(data, base_dir=_as_base_dir(base_dir))`. Same optional
  `base_dir` semantics as `from_dict`.
- **`WicaConfig.from_json_file(cls, path)`** — the old `from_json` body: read the file (wrap `OSError`
  as `ConfigError`), then delegate to `from_json(text, base_dir=config_path.parent)`, wrapping any
  `ConfigError` with an `in config file <path>:` prefix so file-load errors still name the file.
- Fix internal docstrings that named `from_json` as the file locator (`_as_base_dir`,
  `_validate_system_prompt`, `resolve_system_prompt`) → `from_json_file`.

### Callers

- [examples/conversation_demo.py](../examples/conversation_demo.py): `from_json` → `from_json_file`.
- [tests-e2e/support.py](../tests-e2e/support.py): two call sites → `from_json_file`.

### Tests — [tests/test_config.py](../tests/test_config.py)

- Rename the five `WicaConfig.from_json(config_path)` file-path calls → `from_json_file`, plus the
  prose/var-name touch-ups (`test_from_dict_base_dir_matches_from_json_file_location`).
- Add string-parser coverage for the new `from_json`: happy-path parse, `base_dir` locating a
  relative `system_prompt_file` (parity with `from_dict`), invalid-JSON `ConfigError`, non-object
  `ConfigError`.

### Specs

- [specs/config.md](../specs/config.md) — "Plain dataclasses with dictionary and JSON loaders"
  rewritten around the triad; `from_json` → `from_json_file` throughout the locate/flow prose.
- [specs/wica.md](../specs/wica.md), [specs/conversation-demo.md](../specs/conversation-demo.md),
  [specs/testing.md](../specs/testing.md) — reference `from_json_file` for file loading; wica.md
  lists all three adapters and renames the "no `Wica.from_json_file`" note. These are editorial
  (no code in those modules changes).

## Verification

`uv run ruff check .` · `uv run ruff format .` · `uv run pyright` · `uv run pytest` all pass.
`config.md` stays `Implemented` (design + code in sync at completion).
