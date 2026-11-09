# One failed outcome, and the completed rate before compliance

**Status:** Done

Revises [specs/wica-tester.md](../specs/wica-tester.md) ("Scenarios") and [specs/benchmark.md](../specs/benchmark.md) ("The summary") after the first batch in the step format (`checks-baseline`, 2026-11-09, not kept).

**Why.** Ten of 38 runs stopped. Three of them stopped because the model never did what the session needed — ministral-3b never answered "Hi!", gpt-oss-120b returned empty responses, gpt-oss-20b sent malformed tool-call arguments — and a stopped run is not scored, so those models kept a flattering compliance (gpt-oss-120b: 100%) on their other runs. Separately, "stalled" and "failed" behaved identically and only split the counts.

## Scope

- `wica.testing`: `RunOutcome` is `completed` or `failed`; `TestRun.stop(at)` always fails the run; a timed-out wait stops a scenario (and the scripted tester's chain) as failed. The finding's kind (`wait` or `expect`) and the stopping step still say which kind of miss it was.
- `wica.benchmark`: `RunResult.completed`; `Cell.completed` replaces the scored / failed / stalled counts; every table shows **completed** (`n/N`) right before pass^k and compliance; "stopped at" labels drop the outcome.
- Specs, AGENTS.md, `benchmarks/README.md` and the tests follow.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- The e2e-fake set — key-less, passes.
