# History window: bound history to the last `history_reactions` reactions

**Status:** Done

Bound the Agent's append-only history with a lossy, hysteretic window driven by a new `AgentConfig.history_reactions` field, and tell the model in the runtime primer that its context holds only its latest reactions. Implements the `Updated` parts of [agent.md](../specs/agent.md) ("History window", "System prompt composition" clause 2, "History record shape" wording, open question 3) and the `history_reactions` row of [config.md](../specs/config.md). Flip agent.md back to `Implemented` when this plan is `Done` (config.md stays `Updated` until its other pending gap is closed by its own plan).

Design decisions (settled in the spec, restated here for the implementer):

- **One field, `history_reactions: int | None = None`.** `None` = unbounded (today's behavior). A positive integer `X` is the number of reactions the model is guaranteed to always see. The high-water mark is fixed at `2X`, not a second knob.
- **Unit = reaction = `ObservationRecord`.** Failed, empty, and `noop` steps count, since each appends exactly one Observation.
- **Cut at step start, from storage.** In `_append_observation`, after appending: if the Observation count is `>= 2X`, delete every record before the `X`-th-newest Observation. Records are removed from `self._history`, not hidden at render time. The renderer is untouched.
- **Primer clause only when configured**, composed in `_compose_system_prompt` between the perception/acting part and the output-mode clause; decided by config, so `set_output_command` recomposition behaves as before.

## Steps

### 1. Config (`src/wica/config.py`, `tests/test_config.py`)

- Add `history_reactions: int | None = None` to `AgentConfig`; in `__post_init__` reject a non-`None` value that is not a positive `int` (a `bool` is not an `int` here) with a `ConfigError` naming the field.
- `_parse_agent_block`: add `"history_reactions"` to `_AGENT_OPTIONAL_COMMON`; accept absent or `null` as `None`, otherwise require a positive integer (same wrong-type/wrong-value message shape as `hf_provider`).
- Tests: default is `None`; parsed from dict/JSON; `null` accepted; `0`, `-1`, `"3"`, `true` rejected with `ConfigError` matching `history_reactions`; direct construction enforces the same.

### 2. Agent history cut (`src/wica/agent.py`, `tests/test_agent.py`)

- Read `config.history_reactions` in `Agent.__init__`.
- `_append_observation`: after appending the new Observation, call a small `_apply_history_window()` that counts `ObservationRecord`s and, at `>= 2X`, finds the index of the `X`-th-newest Observation and does `del self._history[:index]`, logging at DEBUG how many reactions and records were dropped. Keep the terminal-Command retirement loop as is (it runs on the just-captured snapshot, independent of the cut).
- Tests (drive the public API with the programmable fake, then inspect `on_prompt` messages and, where it is the only way, `agent._history`):
  - With `history_reactions=2`, run 4 steps: after the 4th step's prompt the history holds exactly 2 Observations and the first message after the system prompt is the 3rd step's observation as a `HumanMessage`; after the 3rd step (count 3 < 4) nothing was cut yet — the hysteresis, asserted by comparing the second message of the 2nd and 3rd prompts (byte-identical) versus the 4th (changed).
  - Failed model call and `noop` steps count as reactions (a step whose model raises still advances the count and gets cut).
  - The cut never leaves a dangling `tool_call`: after a cut, no `ToolMessage` precedes the first `AIMessage` and every `ToolMessage` follows an `AIMessage` carrying its id.
  - A Command dispatched in a cut reaction and completing afterwards still renders its terminal `agent:command:<call_id>` entry in the next observation (the orphan case is valid, not dropped).
  - With `history_reactions=None`, a 5-step run keeps 5 Observations (regression guard).

### 3. Primer clause (`src/wica/agent.py`, `tests/test_agent.py`)

- Add a `_HISTORY_WINDOW_PROMPT` constant (text per agent.md clause 2: latest reactions only, older exchanges gone, current observation is the authority, persistent facts come from entries) and append it in `_compose_system_prompt` right after `_RUNTIME_PRIMER` when `history_reactions` is set.
- Tests: the system message contains the clause iff `history_reactions` is set; it stays in place across `set_output_command` recomposition; ordering is perception → window → output-mode → noop.

### 4. Docs and statuses

- [agent.md](../specs/agent.md): `Updated` → `Implemented`; [specs/_index.md](../specs/_index.md) row in sync.
- AGENTS.md project map: the agent.py role line mentions the history window (no new module, so `test_project_map` is unaffected).
- This plan: `Done`; [plans/_index.md](../plans/_index.md) row in sync.

### 5. Verification

`uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`, and `uv run pytest tests-e2e -k fake` (the scripted-fake flows exercise the whole loop with unbounded history and must be unaffected).
