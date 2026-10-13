---
code:
  - .github/workflows/ci.yml
tests:
---

# Continuous integration

**Status:** Stable

## Purpose

Run the verification gate of [AGENTS.md](../AGENTS.md) — lint, type check, tests — on GitHub's hosted runners for every pull request and every push to `main`, so it is a machine's verdict on each change and not only a local command. CI runs what the tiers of [testing.md](testing.md) are designed to run: the fast tier on every supported platform, the always-run scripted-fake flows of the e2e tier, and — on `main` only — the live provider set with the repository's keys. Like [testing.md](testing.md) this is a cross-cutting practice, not a runtime concept: nothing here ships in the library, and the one file that implements it is the workflow.

## Decided

### The runners

- **GitHub-hosted runners, the free tier.** The repo is public, so every runner is free, macOS and Windows included. No self-hosted runner.
- **Images pinned by name** — `ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-15`, `windows-2025` — never `*-latest`, so the platform a job was validated on moves only when a pin is bumped on purpose.
- **One Python, 3.12** — the floor of `requires-python`, the version `.python-version` names; `uv` installs it on the runner. A newer Python is not in the matrix: the floor is the version most likely to break, and the library has no version branches.
- **`uv sync --locked`** in every job, so a `pyproject.toml` edit without its lock update fails CI rather than resolving silently. The default sync installs the `dev` group, which carries every provider package and Gradio (see [project.md](project.md)) — what pyright, the contrib's fast tests and the live set all need. `astral-sh/setup-uv` caches uv's downloads keyed on `uv.lock`.

### Platforms: the fast tier on each

WICA supports **macOS (arm64), Linux (x86_64 and arm64) and Windows (x86_64)** ([project.md](project.md)), and the fast tier is what states it: one job per platform.

- **Why Windows.** WICA is pure Python with no native code, subprocess or signal handling, and every dependency ships Windows wheels — so supporting it is nearly free, while apps built on WICA (a robot bridge among them) do run there. What differs on Windows is exactly what the fast tier exercises: asyncio's default Proactor event loop under the World's shared loop and its thread hand-offs, and Python 3.12's coarse Windows clocks (about 15.6 ms per tick for `datetime.now`/`time.monotonic`, the clocks of [instrumentation.md](instrumentation.md) and the tests' deadlines). A Windows-only failure is a WICA bug to fix — or a test whose timing margin assumed a fine clock — never a reason to drop the platform.
- **Why macOS.** It is where development happens, so it is rarely the first to break; it stays in the matrix because it is cheap and because a laptop's run is not a clean checkout's.
- **Why Linux arm64.** It is where a WICA app runs *on* a robot — a Raspberry Pi (the wireless Reachy Mini's own computer) or a Jetson — rather than on a laptop beside it. WICA's own code is architecture-neutral; what the job proves is that every locked dependency (Gradio's stack, `tokenizers`, `pydantic-core`, …) has an arm64 wheel, or builds, so `uv sync --locked` succeeds there. A dependency bump that drops arm64 fails this entry, not a user's robot.
- **Not Intel macOS.** It runs the same macOS code paths as the arm64 entry, so it could only catch a missing x86_64 macOS wheel — a platform Apple no longer ships, that packages are starting to drop and whose GitHub runner images are being retired. **Untested, expected to work**: no native code of WICA's own, same OS as the tested entry. Revisited only if a user reports a break there.
- **`fail-fast: false`**, so one platform's failure leaves the others running and the log says which platforms broke.

### The static gate on Linux only

`ruff check .`, `ruff format --check .` and `pyright` give the same verdict on every platform, so they run once, on Linux. `ruff format --check .` is the CI form of the `ruff format .` the AGENTS.md command list runs (Ruff formats no Markdown here, so `.` is safe). **Gradio's type stubs come first:** Gradio's wheel ships no `.pyi` for its components — `component_meta` writes them into site-packages the first time `gradio` is imported — so on a fresh venv pyright cannot see generated event methods (`Timer.tick`, `Dropdown.change`) and reports errors on the contrib. The job imports `gradio` once before `pyright`; the same applies to a fresh local clone (see [AGENTS.md](../AGENTS.md) "Commands").

### The e2e tier, split by its two kinds

[testing.md](testing.md) gives `tests-e2e/` two kinds of test with opposite costs, and CI treats them as two jobs, selected **by file** (not `-k`, which matches on test names and would drift as tests are added):

- **Scripted-fake flows — every event, Linux.** `tests-e2e/test_fake_flows.py` and `tests-e2e/test_example_flow.py`: no network, no key, deterministic, never skip. They run on every pull request like the fast tier. One platform: their platform-sensitive paths are the fast tier's, already covered on all three.
- **Live provider set — push to `main` and manual runs, Linux.** `tests-e2e/test_smoke.py` and `tests-e2e/test_wica.py`, parametrized over `PROVIDER_CONFIGS`. Never on a pull request: every case is a paid call, the outcome is non-deterministic, and a fork's pull request receives no secret anyway. On `main` it gives each merged change a live verdict against every provider whose key the repository holds. **One platform:** the live set is HTTP to a provider — the operating system changes nothing it measures.

### The workflow

One file, `.github/workflows/ci.yml`. It triggers on `pull_request`, on `push` to `main`, and on `workflow_dispatch`. Concurrency is one run per ref; a newer push cancels an older run **on a pull request only** — every push to `main` keeps its full verdict, live set included.

| Job | When | What |
|---|---|---|
| `check` | every event | `uv sync --locked`, then `ruff check .`, `ruff format --check .`, `pyright` |
| `fast-tier` | every event | `uv sync --locked`, then `pytest` (the default `tests/` tier, in parallel — [testing.md](testing.md) "Tooling") — **once per platform**: `ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-15`, `windows-2025` |
| `e2e-fake` | every event | `uv sync --locked`, then `pytest tests-e2e/test_fake_flows.py tests-e2e/test_example_flow.py -rs` |
| `live` | push to `main`, `workflow_dispatch` | `uv sync --locked`, then `pytest tests-e2e/test_smoke.py tests-e2e/test_wica.py -rs` with the provider keys mapped from secrets |

None waits on another, so a run takes as long as its slowest job — about a minute and a half end to end, most of it the sync. The fast tier's parallel run helps a runner less than a laptop: three or four cores, not fifteen. Every job carries a `timeout-minutes` well under GitHub's default, so a hung loop fails in minutes, not hours.

### Secrets

- **One repository secret per committed e2e config, under the config's own `api_key_env` name** — `WICA_ANTHROPIC_API_KEY`, `WICA_OPENAI_API_KEY`, `WICA_HF_TOKEN`, `WICA_GEMINI_API_KEY`, `WICA_MISTRAL_API_KEY` (the table in [AGENTS.md](../AGENTS.md) "Live/e2e tests") — mapped onto the same environment variable in the `live` job's test step only.
- **No secret is required.** A secret the repository does not hold comes out as an empty variable, which `resolve_api_key` treats as unset (`MissingEnvError`), so that provider's cases **skip** — the job stays green with any subset of keys, none included. `-rs` lists the skips in the job log; a provider that should have run and skipped is a missing or misnamed secret.
- **Adding a provider** adds a config to `PROVIDER_CONFIGS`, a row to the AGENTS.md key table, and a line to the `live` job's `env` — the same hand-kept pairing [testing.md](testing.md) open question 2 already notes.
- **Required status checks on `main`** are `check`, the four `fast-tier` entries and `e2e-fake` — the jobs that run on a pull request. `live` cannot be required (it never runs on one). Requiring them is a repository setting on GitHub, outside the repo.

## Relationship to the other specs

- **[testing.md](testing.md):** the two tiers, the fake/live split inside the e2e tier and the skip-without-a-key rule CI runs on. CI is the hosted run of that strategy, with no rule of its own about what a test does.
- **[project.md](project.md):** the supported platforms CI's matrix verifies, the `dev` group the sync installs, `.github/workflows/` in the repo shape.
- **[config.md](config.md):** `api_key_env` and the empty-is-unset resolution that make a missing secret a skip.
- **[AGENTS.md](../AGENTS.md):** the verification gate CI automates and the key-env table the secrets mirror.

## Open questions

1. **Live flakiness on `main`.** A provider outage or a model's off-script answer turns `main` red with no code at fault. Accepted for now — the live set's assertions are coarse ([testing.md](testing.md) open question 1) — and revisited if it happens often enough to train people to ignore the badge (options: a retry, or moving `live` to a schedule).
2. **A Python version matrix.** Only the 3.12 floor runs. Adding the newest Python is deferred until a dependency or a language change gives a reason.
