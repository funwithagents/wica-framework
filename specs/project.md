# Project

**Status:** Stable

## Purpose

Structure and tooling for the WICA project itself: Python version, dependency/packaging management with `uv`, repo layout conventions, and development tooling.

## Decided

- **Python version:** 3.12+ minimum
- **Package layout:** `src/` layout — `src/wica/...` — not flat, to avoid accidentally importing an uninstalled package from the repo root
- **Dependency/venv management:** `uv`
- **Linting/formatting:** `ruff`
- **Testing:** `pytest`. Two tiers, kept in separate directories: the default run (`tests/`) is fast, deterministic, and touches no real network; a small live/e2e tier (`tests-e2e/`) calls a real LLM provider and is opt-in only (see below)
- **Live/e2e tests:** tests that call a real model provider (network + API key required, non-deterministic output, costs money) live in `tests-e2e/`, a directory separate from `tests/`. Since `testpaths = ["tests"]` in `pyproject.toml`, the default `uv run pytest` never collects them — no marker or opt-out flag needed. Run them explicitly with `uv run pytest tests-e2e`. Each such test also skips itself (never fails) via `pytest.skip` when the relevant provider API key isn't set, so contributors without credentials — and CI, unless deliberately configured — aren't broken by its absence
- **Type checking:** `pyright`, dev dependency, run via `uv run pyright`. Config lives in `[tool.pyright]` in `pyproject.toml` (`standard` mode, targets `src`, `tests`, and `tests-e2e`, pinned to the `.venv`). VS Code: install the Pylance extension, which bundles pyright and picks up the same settings via `.vscode/settings.json`
- **Distribution intent:** internal framework for now, not published to PyPI. Structure should not preclude publishing later, but no release tooling is set up now
- **Repo shape:**
  - `tests/` at repo root, mirroring the `src/wica/` module structure
  - `tests-e2e/` at repo root, for the live/e2e tier above — not collected by the default `pytest` run
  - `examples/` at repo root, for runnable demo scripts
  - `docs/` reserved for future user-facing documentation (usage guides, API reference) — distinct from `specs/`, which holds pre-implementation design/planning docs and continues to exist alongside `docs/` once that appears

## Open questions

None currently.
