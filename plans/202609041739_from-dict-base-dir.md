# Optional `base_dir` on `from_dict`

**Status:** Done

Governs [specs/config.md](../specs/config.md). Additive extension: let `WicaConfig.from_dict` /
`AgentConfig.from_dict` locate a relative `system_prompt_file` against a caller-supplied base
directory — the one thing `from_json` does that `from_dict` cannot.

## Why

A sibling app (`vocal-interaction-wica`) composes three engine configs from a single JSON file and
hands each section to its engine as an in-memory dict, with no temp files. For WICA that requires
`from_dict` to resolve a relative `agent.system_prompt_file` against a base directory the caller
supplies (it will pass `os.path.dirname(path)`). The base-dir plumbing already exists end-to-end
(`_parse_wica_block(data, base_dir=…)` is exactly what `from_json` passes) — this only exposes it.

## The change

`src/wica/config.py`:

- `WicaConfig.from_dict(cls, data, *, base_dir: str | Path | None = None)` — pass
  `Path(base_dir) if base_dir is not None else None` into `_parse_wica_block`.
- `AgentConfig.from_dict(cls, data, *, base_dir: str | Path | None = None)` — same, into
  `_parse_agent_block`. (Kept symmetric with `WicaConfig`; both are public and both currently pass
  `base_dir=None`.)

Behavior:

- **No `base_dir` (default):** unchanged — a relative `system_prompt_file` is stored verbatim and
  resolves against the process CWD when read.
- **With `base_dir`:** a relative `system_prompt_file` is absolutized against it — the same *locate*
  (no I/O, deferred read) `from_json` performs against the config file's directory.
- **Absolute `system_prompt_file`:** stored as-is, `base_dir` ignored (matches `from_json`, via the
  existing `_validate_system_prompt` `is_absolute()` guard).
- Accept `str | Path` for ergonomics (the app passes a `str`).

Purely additive: no existing caller passes `base_dir`, so nothing changes for them.

## Spec edits (`specs/config.md`)

Set **Status: Updated** while the code lags, mirror in `specs/_index.md`. Revise the two prose spots
that assert `from_dict` "has no config directory":

1. "Config objects mirror the JSON…" — the `system_prompt_file` bullet: `from_dict(data, *,
   base_dir=None)` locates against a caller-supplied directory when passed (same absolutization as
   `from_json`), stores verbatim when not.
2. "System prompt: inline or file" — the `from_dict` bullet: same revision.

Flip spec + index back to **Implemented** once this plan is `Done`.

## Tests (`tests/test_config.py`)

- Relative `system_prompt_file` + `base_dir` → `AgentConfig.system_prompt_file` is the absolute path
  under `base_dir`, matching what `from_json` produces for a config located there (locate, no read).
- Relative `system_prompt_file`, no `base_dir` → stored verbatim (regression guard on the default).
- Absolute `system_prompt_file` + `base_dir` → stored as-is (`base_dir` ignored for absolute paths).
- Cover both `WicaConfig.from_dict` and `AgentConfig.from_dict` (at least one each).

## Verification

Standard gate (AGENTS.md): `ruff check`, `ruff format`, `pyright`, `pytest`. Mark plan `Done` /
spec `Implemented` only once all pass.
