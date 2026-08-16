---
code:
  - pyproject.toml
tests:
  - tests/test_project_map.py
---

# Project

**Status:** Implemented

## Purpose

Structure and tooling for the WICA project itself: Python version, dependency/packaging management with `uv`, repo layout conventions, and development tooling.

## Decided

- **Python version:** 3.12+ minimum
- **Package layout:** `src/` layout — `src/wica/...` — not flat, to avoid accidentally importing an uninstalled package from the repo root
- **Dependency/venv management:** `uv`
- **Linting/formatting:** `ruff`
- **Testing:** `pytest`, in two physically-separated tiers — a fast, deterministic, no-network default run (`tests/`, the only tier `testpaths` collects) and an opt-in live tier (`tests-e2e/`) that calls real providers, parametrized over one committed config per provider. Full strategy — tier rationale, functional-test philosophy, `reset_world` isolation, the provider-parametrized live tier — is specced in [testing.md](testing.md)
- **Type checking:** `pyright`, dev dependency, run via `uv run pyright`. Config lives in `[tool.pyright]` in `pyproject.toml` (`standard` mode, targets `src`, `tests`, and `tests-e2e`, pinned to the `.venv`). VS Code: install the Pylance extension, which bundles pyright and picks up the same settings via `.vscode/settings.json`
- **Distribution intent:** distributed via **GitHub, not PyPI** — consumed by separate agent-app repositories that depend on WICA through a git URL (e.g. `"wica[openai] @ git+https://…"`). Structure should not preclude PyPI later, but no release tooling is set up now
- **Provider integrations are optional extras.** Core `wica` depends only on `langchain`/`langchain-core` and ships with **no** model provider. Each provider's LangChain integration is an install-time **extra** under `[project.optional-dependencies]` — `wica[anthropic]` → `langchain-anthropic`, `wica[openai]` → `langchain-openai`, `wica[huggingface-hub]` → `langchain-huggingface` — so a consumer pulls only what it wires up, and an unselected provider fails at runtime with a clear `ImportError` rather than silently. Extras (not dependency groups) are the mechanism because downstream repos consume WICA *as a dependency*, and groups aren't visible to a dependant. The same provider packages are **also** listed in the `dev`/`demo` dependency groups, so this repo's own e2e tests and demo can switch providers by editing config alone — extras serve downstream consumers, groups serve in-repo work (provider surface in [config.md](config.md), "Providers")
- **Repo shape:**
  - `tests/` at repo root, mirroring the `src/wica/` module structure
  - `tests-e2e/` at repo root, for the live/e2e tier above — not collected by the default `pytest` run
  - `examples/` at repo root, for runnable demo scripts
  - `docs/` reserved for future user-facing documentation (usage guides, API reference) — distinct from `specs/`, which holds pre-implementation design/planning docs and continues to exist alongside `docs/` once that appears

## Open questions

None currently.
