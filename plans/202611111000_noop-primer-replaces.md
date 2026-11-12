# The `noop` clause names what `noop` replaces

**Status:** Done

Extend the runtime primer's `noop` clause so a model woken by a change that needs nothing calls `noop` instead of checking in or repeating itself ([specs/agent.md](../specs/agent.md), "System prompt composition", part 4). The live gate's `background` step found two committed models speaking when a background sound woke them (`gpt-oss-120b` checks in, `ministral-14b` repeats the last request); both were `KNOWN_FLAKY` ([plans/202611101812_e2e-regression-gate.md](202611101812_e2e-regression-gate.md)).

## Steps

1. `_NOOP_PROMPT` in `src/wica/agent.py`: the current clause, plus "rather than checking in, filling the silence, or repeating something you already said or did"; the guard against skipping a request unchanged.
2. `tests-e2e/test_robot.py`: drop the pairs that pass reliably from `KNOWN_FLAKY`; [specs/testing.md](../specs/testing.md) says so.
3. Verify: ruff, ruff format, pyright, the fast tier; the whole live set on every provider; a plain request answered in text (no output Command) on every provider, before and after; the demo benchmark at k = 5 on the five gate models, before and after; CI on the branch.
4. [specs/agent.md](../specs/agent.md) back to `Implemented`.

## Results

Three wordings were tried, 2026-11-11 (the current code alongside each, same runs):

| Wording | `background` passed, HF / Mistral (8 focused + 5 in conversation) | Plain request answered with `noop` (no output Command) |
|---|---|---|
| Current | 8/13 / 11/13 | `gpt-4o` 4/35, others 0 |
| "You are called whenever your World changes … being called is not a cue to speak or act …" | 13/13 / 13/13 | `gpt-4o` 11/15, Gemini 6/30 → rejected |
| The same, answering first | 13/13 / 12/13 | `gpt-4o` 13/15 → rejected |
| **Current + "rather than checking in, filling the silence, or repeating …"** | **12/13 / 13/13** | `gpt-4o` 3/20, others 0 |

The kept wording: the whole live set green on every provider; fast tier, lint and types green.

The demo benchmark (k = 5, five gate models, `commands` + `conversation`) does not confirm the gate's gain:

| | Current | Kept wording |
|---|---|---|
| `quiet_on_background_change` | 23/25 (`gpt-4o`, `gpt-oss-120b` ×1 each) | 22/25 (`gpt-4o` ×2, `gpt-oss-120b` ×1) |
| `no_noop_on_input` | 50/50 | 49/50 — Haiku `noop`s the user walking away instead of `switch_user_tracking` |
| `cancels_when_asked` | 50/50 | 49/50 (Mistral) |

On the demo, the background failures are a model speaking when its own `emotion` changes; the phrase does not reach them. One `noop` on input in 50 is within noise at k = 5, but it is the failure a clause pushing towards silence risks.
