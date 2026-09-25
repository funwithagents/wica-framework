# History budget: cut history from the provider's reported token usage

**Status:** Done

Add a usage-driven, context-window-relative trigger to the history cut built by [202609251800_history-window.md](202609251800_history-window.md): two new `AgentConfig` fields, `history_budget` (fraction of the context window) and `context_window` (token override, else the model's LangChain profile), a cut computed from each call's reported `input_tokens`, and token usage on the fake model so the deterministic tier can exercise it. Implements the `Updated` parts of [agent.md](../specs/agent.md) ("History budget", primer clause 2), [config.md](../specs/config.md) (two field rows) and [fake-provider.md](../specs/fake-provider.md) ("Token usage"). Flip agent.md and fake-provider.md back to `Implemented` when this plan is `Done` (config.md stays `Updated` for its other pending gap).

Design decisions (settled in the spec):

- **Signal = `usage_metadata.input_tokens` of the response just received**, read in `_run_step` after the call. No pre-call token counting.
- **Context window resolved once in `Agent.__init__`**: `config.context_window`, else `self.model.profile["max_input_tokens"]`, else `ConfigError` (only when `history_budget` is set).
- **high = window × budget; low = high // 2.** Over `high` → keep `⌊(low − fixed) / per_reaction⌋` reactions (at least one), where `fixed` is the smallest input count reported so far (the first call's: fixed part + one reaction) and `per_reaction = (input_tokens − fixed) / (observations − 1)`. Cut at the end of the step through the shared `_cut_history(keep)` helper the fixed window also uses. (Revised from a plain average during implementation: with a large system prompt the average charged every reaction a share of the fixed part and cut far too deep.)
- **A single over-budget reaction cannot be cut** → WARNING log, no cut; the fixed part alone over the low mark → WARNING, one reaction kept.
- **Primer clause** present when either field is set.

## Steps

### 1. Config (`src/wica/config.py`, `tests/test_config.py`)

- `history_budget: float | None = None` — accept `int`/`float` (not `bool`) with `0 < value <= 1`; `context_window: int | None = None` — positive int (not `bool`). Both parsed from the dict with the same null/absent handling as `history_reactions`, both validated in `__post_init__`.
- Tests: defaults; parsed values; `0`, `1.5`, `"0.5"`, `True` rejected for the budget; `0`, `-1`, `"4096"`, `True` rejected for the window; `context_window` alone accepted.

### 2. Agent (`src/wica/agent.py`, `tests/test_agent.py`)

- `__init__`: after the model is built, resolve `self._history_high`/`self._history_low` (tokens, or `None`) from `config.history_budget` and the window; raise `ConfigError` when the budget is set and no window is known. Primer condition becomes "either field set".
- Refactor `_apply_history_window` into `_cut_history(keep: int, reason: str)` (deletes before the `keep`-th newest Observation, logs) plus the fixed-window check that calls it at step start.
- In `_run_step`, after the successful call and the response processing: if `builder.usage` and `self._history_high` and `usage.input_tokens > high` → compute `keep` and call `_cut_history`, or log WARNING when `keep` would not shrink history.
- Tests (ProgrammableChatModel with `usage_metadata` on its `AIMessage`s, `context_window` passed via `make_agent`):
  - Under the mark nothing is cut; over it the oldest reactions go and the next prompt starts at the expected Observation, with the computed keep count (e.g. window 1000, budget 0.5 → high 500/low 250; 6 reactions reporting 600 → avg 100 → keep 2).
  - Below `high` again after the cut, no further cut on the next call (hysteresis).
  - A response without usage never trips the budget.
  - A single reaction over the mark: WARNING logged, history intact.
  - Both policies set: the fixed cap still applies at step start.
  - Build without a known window raises `ConfigError` mentioning `context_window`; with `context_window` set it builds; a model whose `profile` supplies `max_input_tokens` builds without the override (a ProgrammableChatModel with `profile={"max_input_tokens": …}`).
  - Primer clause present with only `history_budget` set.

### 3. Fake model (`src/wica/fake_model.py`, `tests/test_fake_model.py`, `tests-e2e/test_fake_flows.py`)

- `_to_result` attaches `usage_metadata`: from the step's `"usage"` when given, else `input_tokens = sum(len(m.text) for m in messages) // 4` and `output_tokens = len(text) // 4`. `_generate`/`_agenerate` pass the messages through.
- Unit tests: the estimate grows with the prompt and is deterministic; the override wins.
- One always-run fake flow: `history_budget` + `context_window` in the JSON config, a scripted `usage` over the mark at a chosen step, then assert via `on_agent_prompt` that the following prompt holds fewer observations.

### 4. Docs and statuses

- [agent.md](../specs/agent.md) and [fake-provider.md](../specs/fake-provider.md): `Updated` → `Implemented`; [specs/_index.md](../specs/_index.md) rows in sync.
- AGENTS.md project map: agent.py and fake_model.py role lines.
- This plan: `Done`; [plans/_index.md](../plans/_index.md) row in sync.

### 5. Verification

`uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`, `uv run pytest tests-e2e -k "fake or example"`.
