# CI on GitHub Actions

**Status:** Done

Build [ci.md](../specs/ci.md): one workflow running the static gate, the fast tier on Linux, macOS and Windows, the scripted-fake flows on every pull request, and the live provider set on pushes to `main`. Also states the supported platforms in [project.md](../specs/project.md) (`Updated` until the Windows and macOS fast-tier jobs are green, then back to `Implemented`) and closes [testing.md](../specs/testing.md)'s "CI wiring" open question. Now `Done`: `ci.md` is `Implemented` and `project.md` is back to `Implemented`.

Shape taken from `reachy-mini-bridge`'s CI (its `specs/testing/ci.md`): pinned runner images, `astral-sh/setup-uv` with a cache on `uv.lock`, `uv sync --locked`, `fail-fast: false` on the platform matrix, `timeout-minutes` on every job, no secret required for a green run. Two differences come from WICA's own tiers. The bridge's live tier runs a free local simulator on Linux, while WICA's live set makes paid provider calls, so it runs on `main` only. And WICA's e2e tier also holds key-less scripted-fake flows, which get a job of their own on every pull request.

## Steps

### 1. Workflow: done

`.github/workflows/ci.yml` with the four jobs of the spec's table (`check`, `fast-tier` × `ubuntu-24.04`/`ubuntu-24.04-arm`/`macos-15`/`windows-2025`, `e2e-fake`, `live`). The `e2e-fake` and `live` jobs select their files by path. `live` is gated by `if: github.event_name != 'pull_request'` and maps the five `WICA_*` secrets. Concurrency cancels superseded runs on pull requests only.

Checked locally (macOS, 2026-10-13): `actionlint` clean. `uv sync --locked` and `ruff format --check .` pass. The `e2e-fake` command passes (7 tests). The `live` command, with every key set to the empty string (what an absent secret gives), skips all 20 cases with `-rs` naming each variable, which confirms that an empty secret is a skip and not a failure.

### 2. Specs and docs: done

- New [ci.md](../specs/ci.md) (`Stable`), with frontmatter `code: .github/workflows/ci.yml`, and its row in [specs/_index.md](../specs/_index.md).
- [project.md](../specs/project.md): the **Platforms** bullet (macOS, Linux, Windows) and `.github/workflows/` in the repo shape; status `Updated`.
- [testing.md](../specs/testing.md): open question "CI wiring" closed (later questions renumbered), and a pointer to how the tiers map onto CI jobs. This is editorial, so the status stays `Implemented`.
- [AGENTS.md](../AGENTS.md): a `.github/workflows/` row in the top-level layout, the CI note under "Live/e2e tests" (secrets under the key-env names) and under "Verification".

### 3. First runs on GitHub: done

- Push the branch and open a pull request. Expect `check`, the `fast-tier` entries and `e2e-fake` to run, and `live` not to run.
- **Windows** is the first run ever on that platform. The watch-list from the spec: the coarse 15.6 ms clock under Python 3.12 (stamps that tie, durations of zero) and the tight wall-time assertions in `tests/test_agent.py` (`< 0.15` s for the zero-window case, `< 0.5` s for the early flush). A failure there is fixed in WICA or in the test's margin, and recorded here.
- **First run (PR #1, 2026-10-13):** Linux and macOS fast tier, `e2e-fake` green; `live` skipped as designed. Two failures, both fixed on the branch:
  - `check`: pyright reported 5 errors in `wica.contrib.gradio` (`Timer.tick`, `Dropdown.change`, `State.change`). Gradio's wheel ships no component `.pyi`; `component_meta` writes them on first import, so a fresh venv has none until something imports `gradio`. Reproduced on a clean `git archive` checkout (5 errors → 0 after `import gradio`). Fix: a step importing `gradio` before `pyright`; the same note in AGENTS.md "Commands" for fresh clones.
  - `fast-tier (windows-2025)`: `test_get_prompt_entries_omits_excluded_and_orders_by_timestamp` — the updated entry did not move last. The coarse Windows clock gave two versions the same stamp, and the stable sort fell back to registration order. A WICA gap, not a test margin: the prompt order is meant to put the latest update last. Fix in `world.py`: a World-wide version counter (bumped under the lock by every `register`/`update`) breaks timestamp ties in `get_prompt_entries`/`get_prompt_snapshot`; [world.md](../specs/world.md) states the tie rule (code and spec change together, so its status stays `Implemented`). New test `test_prompt_order_breaks_timestamp_ties_by_update_order` freezes the World's clock so every version ties; it fails against the old `world.py` (`['a', 'b']`, Windows' failure) and passes with the fix.
- **Second run:** every job green, Windows included (fast tier: Linux 58 s, macOS 1 m 17 s, Windows 1 m 22 s; `check` 17 s; `e2e-fake` 16 s).
- **Linux arm64 added** (`ubuntu-24.04-arm`) after the second run, as the on-robot target; Intel macOS recorded in [ci.md](../specs/ci.md) as untested, expected to work. [project.md](../specs/project.md) names the architectures.
- **Fourth run (the e2e rename):** `fast-tier (macos-15)` failed `test_noop_renders_as_native_call_with_plain_ack` (`sink.event.wait` timed out), green on the three runs before. A test race, not a WICA bug: it waited for the first model call, not the end of the reaction, so on a slow runner the second trigger arrived while the step was still in flight and was dropped by design (single-in-flight, [agent.md](../specs/agent.md)). Fixed with the suite's own idiom (`len(model.calls) == 1 and not agent._busy`); the `_step` helper (9 call sites) had the same race and now takes the agent and waits for it to be idle. Not reproduced locally — the drop rule and the identical idiom in other tests are the evidence.
- **Fast tier in parallel** (`pytest-xdist` in the `dev` group, `addopts = ["-n", "logical", "--maxprocesses", "8"]`, as reachy-mini-bridge and naoqi-sim): locally 43 s → about 17 s, three runs green, and three more green with 24 workers on 15 cores. `tests-e2e/conftest.py` forces that tier serial (`pytest_cmdline_main` → `numprocesses = 0`); checked that the fake flows run without workers and the fast tier with 8. [testing.md](../specs/testing.md) "Tooling" (plus `tests-e2e/conftest.py` in its frontmatter), [project.md](../specs/project.md), AGENTS.md "Commands", [ci.md](../specs/ci.md).
- **Final run (parallel tier), three attempts, all green:** fast tier Linux 39 s, Linux arm64 31 s, macOS 52 s, Windows 46 s (the test run itself 20–30 s each, the rest the sync); `check` 21 s; `e2e-fake` 19 s.
- **Secrets:** all five set as repository secrets (2026-10-13): `WICA_ANTHROPIC_API_KEY`, `WICA_OPENAI_API_KEY`, `WICA_HF_TOKEN`, `WICA_GEMINI_API_KEY`, `WICA_MISTRAL_API_KEY`.
- **`live` before merge: not possible.** GitHub offers `workflow_dispatch` only for a workflow on the default branch, and `ci.yml` reaches `main` with this PR, so the manual run planned here cannot happen on the branch. The live set itself passed against all five providers locally on this branch (27 passed, 2026-10-13); its first CI run is the merge's own push to `main`, whose `-rs` should list no provider skip.
- **Required status checks** (`check`, the four `fast-tier` entries, `e2e-fake`): a repository setting, set on GitHub after merge — outside the repo and outside this plan.
