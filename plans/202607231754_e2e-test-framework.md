# E2E test framework

Introduces the reusable scaffolding for **live** tests — tests that call a real LLM provider over the network — per [specs/project.md](../specs/project.md) "Live/e2e tests". This plan ships the framework only: the `tests-e2e/` directory, skip-without-credentials mechanics, and a helper to build a real chat model from env. It adds no concrete e2e test cases for any particular module — those belong to whichever feature plan needs live coverage, built on top of this framework.

## Why a separate plan

The mechanics of "a test tier that never runs in the normal dev loop and skips cleanly without credentials" are generic — useful to any module that touches a real provider (a LangChain adapter, a tool-calling loop, anything else that can only be fully verified against the real thing), not tied to one specific feature. Standing this up on its own, independent of whichever feature happens to need it first, keeps that plumbing out of any one feature plan's diff and makes it reusable from day one.

## Scope

- `tests-e2e/` — **new** top-level directory, sibling to `tests/`. `pyproject.toml`'s `testpaths = ["tests"]` already excludes it from the default `uv run pytest` run — no marker or `addopts` needed, physical separation does the job.
- `tests-e2e/conftest.py` — **new**: the same autouse World-reset fixture `tests/conftest.py` has (duplicated, not imported — `tests-e2e` isn't a package that imports from `tests`, and the fixture is a few lines).
- `tests-e2e/support.py` — **new**, shared helpers (no `test_` prefix, so pytest doesn't collect it as a test module):
  - `require_env(name: str) -> str` — returns the env var's value, or calls `pytest.skip(...)` if unset.
  - `real_chat_model(**kwargs) -> BaseChatModel` — calls `require_env("ANTHROPIC_API_KEY")`, then `init_chat_model(model, model_provider="anthropic", **kwargs)` where `model` defaults to `os.environ.get("WICA_E2E_MODEL", "claude-haiku-4-5")` (cheapest/fastest available, overridable without a code change since model names/aliases drift).
- `tests-e2e/test_smoke.py` — **new**, one throwaway test proving the framework itself works (see "Implementation steps").
- `pyproject.toml` — add `langchain-anthropic` to the `dev` dependency group (needed to actually construct a real chat model), and add `tests-e2e` to `[tool.pyright]`'s `include`.
- `AGENTS.md` — new subsection under "Testing" documenting the tier and how to run it (see below).

## `tests-e2e/support.py`

```python
from __future__ import annotations

import os

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set — skipping e2e test")
    return value


def real_chat_model(**kwargs) -> BaseChatModel:
    require_env("ANTHROPIC_API_KEY")
    model = os.environ.get("WICA_E2E_MODEL", "claude-haiku-4-5")
    return init_chat_model(model, model_provider="anthropic", **kwargs)
```

- `pytest.skip` (not a hard failure) inside `require_env` is the point: running `uv run pytest tests-e2e` without `ANTHROPIC_API_KEY` set produces a skip, not a red build.
- `WICA_E2E_MODEL` exists so the pinned default model name can be bumped via env instead of a code change once it's deprecated/renamed by the provider — model aliases have historically drifted.

## `pyproject.toml` changes

```toml
[tool.pyright]
include = ["src", "tests", "tests-e2e"]
```

```toml
[dependency-groups]
dev = [
    "langchain-anthropic>=...",
    "pyright>=1.1.411",
    "pytest>=8.0.0",
    "ruff>=0.8.0",
]
```

`testpaths` is unchanged (`["tests"]`) — that's the whole exclusion mechanism.

## Usage convention (for consumers of this framework)

A test file under `tests-e2e/` builds whatever it needs via `real_chat_model()` and asserts on **behavior**, not exact text (real model output varies run to run) — e.g. "the output sink received non-empty text" or "the registered tool was actually invoked with plausible args," not a specific string.

## `AGENTS.md` addition

Under "Testing", add:

```markdown
### Live/e2e tests

Some tests call a real LLM provider over the network. They live in `tests-e2e/`, a directory
separate from `tests/`, so the default `uv run pytest` never runs them — no network access or
API key is needed for the normal dev loop. Run them explicitly, and only when you actually want
to verify against a live provider:

\```
uv run pytest tests-e2e
\```

Each e2e test skips itself (does not fail) if its required API key isn't set in the environment
— see `tests-e2e/support.py`.
```

## Implementation steps

1. `pyproject.toml`: add `langchain-anthropic` to `dev`; add `tests-e2e` to `[tool.pyright]` `include`; `uv sync --dev`.
2. `tests-e2e/conftest.py`: World-reset fixture. `tests-e2e/support.py`: `require_env`, `real_chat_model`.
3. `tests-e2e/test_smoke.py`: one throwaway test (e.g. `real_chat_model().invoke("say hi").content` is non-empty) confirming the mechanics actually behave as described before anything else builds on top: `uv run pytest` (no args) does **not** collect anything under `tests-e2e/`, and `uv run pytest tests-e2e` with `ANTHROPIC_API_KEY` unset produces a **skip**, not a failure or error.
4. `AGENTS.md`: add the "Live/e2e tests" subsection.

## Out of scope / deferred

- Any concrete e2e test for a specific module — added by whichever feature plan needs one, on top of this.
- Multi-provider e2e coverage (e.g. also testing against OpenAI/Gemini) — one reference provider (Anthropic) is enough to catch adapter bugs for now.
- CI wiring to actually run the `tests-e2e` tier on a schedule/secrets-configured runner — nothing here assumes or sets that up; today it's a manual, local, opt-in command only.
