# Agent instructions

Start at [specs/_index.md](specs/_index.md) for an overview of the specs and their status before making design decisions or writing code — it lists each spec and whether it's still open ("Draft"/"Not started"), design-validated ("Stable"), or built ("Implemented"). For what's been (or is being) built, see [plans/_index.md](plans/_index.md), which lists each implementation plan and its status ("Todo"/"In progress"/"Done").

## Project map

Where things live. This is a coarse, module-level map — for the full file inventory use `git ls-files`; for design detail follow the spec links.

### Top-level layout

| Path | What's there |
|---|---|
| `src/wica/` | The library itself — one module per core concept (see below) |
| `specs/` | Pre-implementation design docs, one per concept, each with a `**Status:**` — indexed by [specs/_index.md](specs/_index.md) |
| `plans/` | Implementation plans turning settled specs into buildable steps — indexed by [plans/_index.md](plans/_index.md) |
| `tests/` | Fast, deterministic, no-network tests; mirrors the `src/wica/` module structure |
| `tests-e2e/` | Opt-in full-loop tests: deterministic scripted-fake flows plus live provider cases (not collected by default `pytest`) |
| `examples/` | Runnable example apps demonstrating the framework — e.g. the Gradio conversation demo ([specs/conversation-demo.md](specs/conversation-demo.md)); deps live in the `demo` uv group, not core |

### `src/wica/` modules

| Module | Role | Spec |
|---|---|---|
| [content.py](src/wica/content.py) | Provider-agnostic multimodal content model (`TextPart`/`ImagePart`/`Content`), shared everywhere | [content.md](specs/content.md) |
| [events.py](src/wica/events.py) | Generic, project-agnostic `Event[T]` pub/sub primitive (subscribe/unsubscribe/emit); a standalone leaf with zero project imports, re-exported as public API | [events.md](specs/events.md) |
| [world.py](src/wica/world.py) | The World state registry: typed entries, register/update/get API, `start`/`stop`/`is_running` lifecycle, sync/async listener + `on_trigger` Event dispatch on the shared loop, rendering to `Content` (constructed per `Wica`, no singleton) | [world.md](specs/world.md), [inputs.md](specs/inputs.md) |
| [config.py](src/wica/config.py) | Framework config: `AgentConfig`/`WicaConfig` dataclasses, strict JSON loading (`from_dict`/`from_json`), `system_prompt_file`, `api_key`/`api_key_env` resolution | [config.md](specs/config.md) |
| [agent.py](src/wica/agent.py) | The Agent reasoning loop and Commands: built from `AgentConfig` on the injected loop, LangChain-backed inference over the World, snapshot history, async cancellable Commands, output sink, `on_trigger`/`on_prompt`/`on_command` Events | [agent.md](specs/agent.md), [commands.md](specs/commands.md) |
| [wica.py](src/wica/wica.py) | The `Wica` facade — single entry point owning the shared loop + a `World`+`Agent` pair, restartable `init`/`start`/`stop` lifecycle plus terminal `close`, `register_command`, and four surfaced instrumentation `Event`s | [wica.md](specs/wica.md) |
| [fake_model.py](src/wica/fake_model.py) | Deterministic, network-free `FakeChatModel` for tests: a scripted `provider: "fake"` model driving the loop over canned responses; test tooling, not re-exported into the runtime `wica` namespace | [fake-provider.md](specs/fake-provider.md) |
| [__init__.py](src/wica/__init__.py) | Public API surface — re-exports the names above | — |

**Keep this map current:** when you add, rename, or remove a top-level `src/wica/` module or a root directory, update the map in the same change — same discipline as keeping spec/plan statuses honest (below). A test (`tests/test_project_map.py`) enforces that every `src/wica/*.py` module appears here and vice-versa — and that the spec frontmatter (see below) stays honest too.

## Keeping statuses current

Specs and plans both carry a status, and you are responsible for keeping it honest as work progresses — update it in the same change that does the work, not as an afterthought:

- **Spec status** (`**Status:**` line near the top of each spec, and the Status column in [specs/_index.md](specs/_index.md)) tracks *design maturity* and *whether the code reflects the spec*, as a lifecycle: `Not started` → `Draft` (open questions remain) → `Stable` (design settled, reviewed and validated — open questions are deferrals only — but **not necessarily implemented yet**) → `Implemented` (a `Done` plan has built it and the code matches the spec). Keep the `**Status:**` line and the index row in sync.
  - **`Stable` is the design-review gate, not an implementation claim.** Promote `Draft` → `Stable` once the core design is settled and its remaining open questions are genuine deferrals (not load-bearing unknowns) — this is where the design is validated *before* code is written. No implementation is required to be `Stable`.
  - **`Implemented` means code matches.** Promote `Stable` → `Implemented` only once a plan implementing it is `Done` (lint, type check, tests all pass — see Verification). This is the one transition that asserts design and code are in sync.
  - **When you edit an `Implemented` spec in a way that requires new code, set its status to `Updated` in the same change.** `Updated` means the design is settled but the existing implementation now lags it — a stronger warning than `Stable`, because there is stale code to fix, not just code to write. Then write a new implementation plan for the gap (see below) and, once that plan is `Done`, flip the spec back to `Implemented`. This `Implemented → Updated → Implemented` loop keeps a spec's status an honest signal of whether the code actually matches it — never leave a re-designed spec sitting at `Implemented`.
  - A purely editorial edit to a `Stable` or `Implemented` spec (typos, clarifications, reordering — nothing that changes what the code should do) keeps its status; it does **not** need `Updated`.
- **Plan status** (`**Status:**` line near the top of each plan, and the Status column in [plans/_index.md](plans/_index.md)) tracks *implementation progress*: `Todo` → `In progress` → `Done`. Mark a plan `Done` only once it's implemented and verified (lint, type check, tests all pass — see Verification). Keep the `**Status:**` line and the index row in sync.
- Whenever you add a spec or plan, add its row to the relevant `_index.md`; whenever you change a status, change it in both the file and the index.

## Spec frontmatter

Every spec opens with a YAML frontmatter block naming the code and tests it governs:

```
---
code:
  - src/wica/world.py
tests:
  - tests/test_world.py
---
```

This is the **spec → code/tests** mapping — the inverse of the module → spec column in the Project map above. Its job is to give the **spec-drift checks** an explicit, version-controlled scope: the exact files to diff a spec against, so a checker never has to guess which code implements a given spec. `code:` names the implementation the spec specifies; `tests:` names the tests that pin its behavior (may be empty/absent, e.g. the demo spec).

The mapping is **many-to-many**: a file can be governed by several specs — `agent.py` by both [agent.md](specs/agent.md) and [commands.md](specs/commands.md), `world.py` by both [world.md](specs/world.md) and [inputs.md](specs/inputs.md) — so the same path legitimately appears in more than one spec's frontmatter.

**Keep it current** (same discipline as statuses): when you move, rename, or delete a file a spec governs — or add a new `src/wica/` module — update the affected spec's `code:`/`tests:` in the same change. `tests/test_project_map.py` enforces three invariants: every listed path exists, every spec declares a non-empty `code:` list, and every concept module in `src/wica/` is named by at least one spec (`__init__.py` is exempt as package glue).

## Testing

- Write functional tests: exercise what a feature/function actually does (inputs → outputs, state changes, side effects), not just that it runs or matches its signature.
- Avoid trivial/tautological tests — e.g. asserting a constant, asserting an object is not `None`, asserting a mock was called. If a test would pass for a broken implementation, it's not worth writing.
- Prefer driving the public API the way a real caller would over asserting on internals.

### Live/e2e tests

Some tests call a real LLM provider over the network. They live in `tests-e2e/`, a directory separate from `tests/`, so the default `uv run pytest` never runs them — no network access or API key is needed for the normal dev loop. Run them explicitly, and only when you actually want to verify against a live provider.

**The live e2e set is parametrized over one config per provider.** Each committed config names its `provider`/`model` (and `hf_provider` for the Hub) and points at its own `api_key_env`; they're wired together as `PROVIDER_CONFIGS` in `tests-e2e/support.py`, so **every live e2e test runs once per config**. A config whose key env var is unset **skips** (it does not fail — see `tests-e2e/support.py`), so you only exercise the providers you have keys for. The provider/config surface is specced in [specs/config.md](specs/config.md) ("Providers").

| Config | Key env var |
|---|---|
| `tests-e2e/e2e.anthropic.config.json` | `WICA_ANTHROPIC_API_KEY` |
| `tests-e2e/e2e.openai.config.json` | `WICA_OPENAI_API_KEY` |
| `tests-e2e/e2e.huggingface-hub.config.json` | `WICA_HF_TOKEN` |

**All three keys live in `~/.zshrc`**, but the shell tool runs non-interactive `bash`/`zsh`, which doesn't source it — a plain `uv run pytest tests-e2e` in that shell sees no keys and every case skips. Source it explicitly in an interactive `zsh` invocation. Run **all providers** (each whose key is set runs; the rest skip):

```
zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e'
```

Run **one provider** by filtering on its config-filename stem with `-k` (the configs are named symmetrically, so the provider name works directly):

```
zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k openai'
zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k huggingface'
zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k anthropic'
```

Never `echo`/print a key itself; when checking whether one is set, redact the value (e.g. `env | grep WICA | sed -E 's/=.*/=<set>/'`).

**The `tests-e2e/` tier also holds an always-run, deterministic set: the scripted-fake flows** (`test_fake_flows.py`, over `provider: "fake"` — see [specs/fake-provider.md](specs/fake-provider.md)). These are network-free and key-less, so they sit **outside** `PROVIDER_CONFIGS` and never skip. Run just them — no keys, no `~/.zshrc` sourcing needed — with `-k fake`:

```
uv run pytest tests-e2e -k fake
```

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
