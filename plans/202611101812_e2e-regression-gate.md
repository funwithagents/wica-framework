# Live e2e set as a regression gate

**Status:** In progress

Turn the live e2e set into a regression alarm for the behaviours WICA promises ([specs/testing.md](../specs/testing.md) "The live set as a regression gate", [specs/ci.md](../specs/ci.md)), after two framework fixes it depends on: WICA-minted call ids ([specs/commands.md](../specs/commands.md), [specs/agent.md](../specs/agent.md)) and a narrower `no_colliding_cancel` ([specs/wica-tester.md](../specs/wica-tester.md)). Background and the maintainer's decisions: `analysis/_e2e-improvements.md` (2026-11-10).

## Steps

### 1. Short, WICA-minted call ids

- `Agent` mints `c1`, `c2`, … for every tool call it receives (noop included), counting over its lifetime and skipping a registered `agent:command:c<n>` key; the provider's id stays on `CommandRecord.tool_call_id` / `NoReactionRecord.tool_call_id`. Replaces `_world_call_id`.
- Fast tier: tests that assumed `call_id == tool_call_id` move to the minted ids; new tests for the numbering (order, noop takes a number, a taken key is skipped, a restart does not reset it) and for `cancel_command` with a minted id.
- Scripted fake flows that cancel by id (`tests-e2e/test_example_flow.py`) use the minted id.

### 2. Narrow `no_colliding_cancel`

- Fails only when a reaction cancels a Command and issues a new call of that same Command; the target is resolved from the window's events by the `call_id` the cancel names (bare or prefixed); an unresolved target is not judged.
- Fast tier: the cancel + `set_emotion` case passes, cancel `dance` + `dance` fails, an unresolved target passes.

### 3. The gate

- `tests-e2e/robot_app.py`: the test robot (Inputs, Commands, persona; `say` timed per word with a cap; `write_note` marking `(long)` from 150 words), a `build_robot(config_path, **agent_overrides)` and the "someone walks up" setup.
- `tests-e2e/scenarios/*.json`: `greeting`, `act_on_request`, `background`, `long_answer`, `interrupt`, `memory`, `goodbye`.
- `tests-e2e/test_robot.py`: focused tests per (provider, step) and `TestConversation` per provider (`history_reactions = 4`, a `TestRun` per step, settle-and-reset between steps, the history-cut assertion at the end); reports per step.
- `tests-e2e/support.py`: `WICA_E2E_FAIL_ON_MISSING` turns a missing-key skip into a failure.
- `tests-e2e/test_wica.py`: drop `test_output_command_live` and `test_real_tool_calling_round_trip` (now `greeting`, `act_on_request`).
- `tests-e2e/e2e.mistralai.config.json`: `ministral-8b-latest`.
- `.github/workflows/ci.yml`: `live` runs `test_robot.py` too, sets `WICA_E2E_FAIL_ON_MISSING=1`, writes and uploads the reports, timeout sized for it.

### 4. Docs

- `docs/guides/testing.md` where it describes the live set; `AGENTS.md` "Live/e2e tests" (the new file, `WICA_E2E_FAIL_ON_MISSING`, the reports).

### 5. Verify and vet

- ruff, ruff format, pyright, the fast tier, the key-less e2e set.
- Each step ≥ 3 times on every provider, by hand; flaky (provider, step) pairs investigated, then `xfail(strict=False)` with the reason if they stay flaky.
- The CI jobs run manually on the branch (`gh workflow run CI --ref <branch>`), `live` included.
- Specs back to `Implemented` (commands, agent, testing, ci) once done.

## Results

_To fill in: vetting runs per provider and step, what was flaky, the CI runs._
