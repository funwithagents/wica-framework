# Project

**Status:** Stable

## Purpose

Structure and tooling for the WICA project itself: Python version, dependency/packaging management with `uv`, repo layout conventions, and development tooling.

## Decided

- **Python version:** 3.12+ minimum
- **Package layout:** `src/` layout — `src/wica/...` — not flat, to avoid accidentally importing an uninstalled package from the repo root
- **Dependency/venv management:** `uv`
- **Linting/formatting:** `ruff`
- **Testing:** `pytest`
- **Type checking:** `pyright`, dev dependency, run via `uv run pyright`. Config lives in `[tool.pyright]` in `pyproject.toml` (`standard` mode, targets `src` and `tests`, pinned to the `.venv`). VS Code: install the Pylance extension, which bundles pyright and picks up the same settings via `.vscode/settings.json`
- **Distribution intent:** internal framework for now, not published to PyPI. Structure should not preclude publishing later, but no release tooling is set up now
- **Repo shape:**
  - `tests/` at repo root, mirroring the `src/wica/` module structure
  - `examples/` at repo root, for runnable demo scripts
  - `docs/` reserved for future user-facing documentation (usage guides, API reference) — distinct from `specs/`, which holds pre-implementation design/planning docs and continues to exist alongside `docs/` once that appears

## Open questions

None currently.
