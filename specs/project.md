---
code:
  - pyproject.toml
tests:
  - tests/test_project_map.py
  - tests/test_docs.py
---

# Project

**Status:** Implemented

## Purpose

Structure and tooling for the WICA project itself: Python version, dependency/packaging management with `uv`, repo layout conventions, and development tooling.

## Decided

- **Python version:** 3.12+ minimum
- **Platforms:** **macOS (arm64), Linux (x86_64 and arm64 — the on-robot target) and Windows (x86_64).** WICA is pure Python with no platform branches, and every dependency ships wheels for each; Intel macOS is untested but expected to work; the support is stated by CI running the fast tier on each ([ci.md](ci.md) "Platforms"), and a platform-only failure is a bug to fix
- **Package layout:** `src/` layout — `src/wica/...` — not flat, to avoid accidentally importing an uninstalled package from the repo root
- **Dependency/venv management:** `uv`
- **Linting/formatting:** `ruff`
- **Testing:** `pytest`, in two physically-separated tiers — a fast, deterministic, no-network default run (`tests/`, the only tier `testpaths` collects, run in parallel with `pytest-xdist` from the `dev` group) and an opt-in full-loop tier (`tests-e2e/`) containing both deterministic scripted-fake flows and live cases parametrized over one committed config per provider. Full strategy — tier rationale, functional-test philosophy, fresh-World isolation, and the provider-parametrized live set — is specced in [testing.md](testing.md)
- **Type checking:** `pyright`, dev dependency, run via `uv run pyright`. Config lives in `[tool.pyright]` in `pyproject.toml` (`standard` mode, targets `src`, `tests`, and `tests-e2e`, pinned to the `.venv`). VS Code: install the Pylance extension, which bundles pyright and picks up the same settings via `.vscode/settings.json`
- **Distribution intent:** distributed via **GitHub, not PyPI** — consumed by separate agent-app repositories that depend on WICA through a git URL (e.g. `"wica[openai] @ git+https://…"`). Structure should not preclude PyPI later, but no release tooling is set up now
- **Provider integrations are optional extras.** Core `wica` depends only on `langchain`/`langchain-core` and ships with **no** model provider. Each provider's LangChain integration is an install-time **extra** under `[project.optional-dependencies]` — `wica[anthropic]` → `langchain-anthropic`, `wica[openai]` → `langchain-openai`, `wica[huggingface-hub]` → `langchain-huggingface`, `wica[google-genai]` → `langchain-google-genai`, `wica[mistralai]` → `langchain-mistralai` — so a consumer pulls only what it wires up, and an unselected provider fails at runtime with a clear `ImportError` rather than silently. Extras (not dependency groups) are the mechanism because downstream repos consume WICA *as a dependency*, and groups aren't visible to a dependant. The same provider packages are **also** listed in the `dev`/`demo` dependency groups, so this repo's own e2e tests and demo can switch providers by editing config alone — extras serve downstream consumers, groups serve in-repo work (provider surface in [config.md](config.md), "Providers")
- **Repo shape:**
  - `tests/` at repo root, mirroring the `src/wica/` module structure
  - `tests-e2e/` at repo root, for the full-loop fake/live tier above — not collected by the default `pytest` run
  - `examples/` at repo root, for runnable demo scripts
  - `.github/workflows/` for continuous integration — one `ci.yml`, specced in [ci.md](ci.md)
  - `README.md` (the landing page: what WICA is, its status, install, the quick start, the demo, the links into `docs/`, how it is developed), `CONTRIBUTING.md` (how the project is run: one maintainer, no external pull requests for now, feedback through issues) and `docs/` (the consumer documentation, entered by task through `docs/index.md`: `getting-started.md`, `reference/` for the API and the configuration, `guides/` for one page per task). One home per fact: a reference page owns it, every other page links there; `specs/` stays the normative design the docs are written from, and `AGENTS.md` the operating manual for working in the repo — two entry points, consumer and contributor, the shape the sibling repos (`tts-engine`, `reachy-mini-bridge`) use
- **Documentation checks:** `tests/test_docs.py`, in the fast tier, keeps the Markdown honest the way `tests/test_project_map.py` keeps the AGENTS.md map honest. Every relative link in the root Markdown files and under `docs/`, `specs/`, `benchmarks/`, `examples/`, `tests/` and `tests-e2e/` must resolve, and an anchor into a Markdown file must name one of its headings (GitHub's slugs); links inside code are examples and are skipped. Not scanned: `plans/`, which record what was built and are not rewritten when a file they link to later moves; `analysis/` (untracked); the benchmark's git-ignored `runs/`. The tables that mirror code are checked against it: the provider rows of the README and of `docs/reference/configuration.md` against the extras of `pyproject.toml` (plus the built-in `fake`), the configuration page's `agent` field table against `AgentConfig`'s fields, the API reference's import table against `wica.__all__` and its Events table against the `on_*` attributes of `Wica`. The Python examples a reader copies are executed: the README's quick start, with a scripted `FakeChatModel` standing in for the provider it names and its key variable set, so the imports, names and calls it shows cannot drift from the library; the getting-started application and the testing guide's examples run as written on `provider: "fake"`. Every `owner.method(` in the python fences of `README.md` and `docs/` must name a method that exists on the public API

## Open questions

None currently.
