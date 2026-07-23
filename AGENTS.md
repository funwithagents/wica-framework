# Agent instructions

Start at [specs/_index.md](specs/_index.md) for an overview of the specs and their status before making design decisions or writing code — it lists each spec and whether it's settled ("Stable") or still open ("Draft"/"Not started"). For what's been (or is being) built, see [plans/_index.md](plans/_index.md), which lists each implementation plan and its status ("Todo"/"In progress"/"Done").

## Project map

Where things live. This is a coarse, module-level map — for the full file inventory use `git ls-files`; for design detail follow the spec links.

### Top-level layout

| Path | What's there |
|---|---|
| `src/wica/` | The library itself — one module per core concept (see below) |
| `specs/` | Pre-implementation design docs, one per concept, each with a `**Status:**` — indexed by [specs/_index.md](specs/_index.md) |
| `plans/` | Implementation plans turning settled specs into buildable steps — indexed by [plans/_index.md](plans/_index.md) |
| `tests/` | Fast, deterministic, no-network tests; mirrors the `src/wica/` module structure |
| `tests-e2e/` | Opt-in live tests that call a real LLM provider (not collected by default `pytest`) |

### `src/wica/` modules

| Module | Role | Spec |
|---|---|---|
| [content.py](src/wica/content.py) | Provider-agnostic multimodal content model (`TextPart`/`ImagePart`/`Content`), shared everywhere | [content.md](specs/content.md) |
| [world.py](src/wica/world.py) | The World state registry: typed entries, register/update/get API, rendering to `Content`, `get_world()` singleton | [world.md](specs/world.md) |
| [agent.py](src/wica/agent.py) | The Agent reasoning loop and Commands: LangChain-backed inference over the World, snapshot history, async cancellable Commands, output sink | [agent.md](specs/agent.md), [commands.md](specs/commands.md) |
| [__init__.py](src/wica/__init__.py) | Public API surface — re-exports the names above | — |

**Keep this map current:** when you add, rename, or remove a top-level `src/wica/` module or a root directory, update the map in the same change — same discipline as keeping spec/plan statuses honest (below). A test (`tests/test_project_map.py`) enforces that every `src/wica/*.py` module appears here and vice-versa.

## Keeping statuses current

Specs and plans both carry a status, and you are responsible for keeping it honest as work progresses — update it in the same change that does the work, not as an afterthought:

- **Spec status** (`**Status:**` line near the top of each spec, and the Status column in [specs/_index.md](specs/_index.md)) tracks both *design maturity* and *whether the code reflects the current spec*: `Not started` → `Draft` (open questions remain) → `Stable` (settled **and** fully implemented — design and code in sync). Promote a spec to `Stable` only once its core design is settled, its remaining open questions are genuine deferrals (not load-bearing unknowns), **and** a plan implementing it is `Done`. Keep the `**Status:**` line and the index row in sync.
  - **When you edit a `Stable` spec in a way that requires new code, set its status to `Updated` in the same change.** `Updated` means the design is settled but the implementation now lags it. Then write a new implementation plan for the gap (see below) and, once that plan is `Done`, flip the spec back to `Stable`. This `Stable → Updated → Stable` loop is what keeps a spec's status an honest signal of whether the code actually matches it — never leave a re-designed spec sitting at `Stable`.
  - A purely editorial edit to a `Stable` spec (typos, clarifications, reordering — nothing that changes what the code should do) stays `Stable`; it does **not** need `Updated`.
- **Plan status** (`**Status:**` line near the top of each plan, and the Status column in [plans/_index.md](plans/_index.md)) tracks *implementation progress*: `Todo` → `In progress` → `Done`. Mark a plan `Done` only once it's implemented and verified (lint, type check, tests all pass — see Verification). Keep the `**Status:**` line and the index row in sync.
- Whenever you add a spec or plan, add its row to the relevant `_index.md`; whenever you change a status, change it in both the file and the index.

## Testing

- Write functional tests: exercise what a feature/function actually does (inputs → outputs, state changes, side effects), not just that it runs or matches its signature.
- Avoid trivial/tautological tests — e.g. asserting a constant, asserting an object is not `None`, asserting a mock was called. If a test would pass for a broken implementation, it's not worth writing.
- Prefer driving the public API the way a real caller would over asserting on internals.

### Live/e2e tests

Some tests call a real LLM provider over the network. They live in `tests-e2e/`, a directory separate from `tests/`, so the default `uv run pytest` never runs them — no network access or API key is needed for the normal dev loop. Run them explicitly, and only when you actually want to verify against a live provider:

```
uv run pytest tests-e2e
```

Each e2e test skips itself (does not fail) if its required API key isn't set in the environment — see `tests-e2e/support.py`.

## Implementation plans

- Write implementation plans as files in the [plans](plans/) folder.
- Name each file `YYYYMMDDHHmm_plan-title.md`: a compact date-time prefix, then an underscore, then a kebab-case title (words separated by `-`).
  - Example: `202607201830_world-registry-refactor.md`
- Give each plan a `**Status:**` line just under its title (`Todo`/`In progress`/`Done`) and add a row for it to [plans/_index.md](plans/_index.md). Keep both current as work progresses (see "Keeping statuses current" above).

## Verification

After any code change, run linting, type checking, and tests, and fix any failures before considering the work done.

## Commands

```
uv sync --dev
uv run ruff check .
uv run pyright
uv run pytest
```
