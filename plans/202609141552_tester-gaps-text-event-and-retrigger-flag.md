# WicaTester gaps: `on_text` instrumentation Event and per-Command `triggers_on_completion`

**Status:** Done

Closes the two framework gaps the WicaTester study still lists after the prerequisites shipped
([specs/_wica-tester-study.md](../specs/_wica-tester-study.md), §2 G7–G8; decision §8.7 —
prerequisites first). Both are additive; the only behavior change is opt-in (a Command may now
choose not to re-trigger).

This plan is written to be executed step by step, in order, by an agent with no other context.
Read [AGENTS.md](../AGENTS.md) first (verification and status rules), then the spec sections named
in each step. The specs are already edited and marked `Updated`
([agent.md](../specs/agent.md), [commands.md](../specs/commands.md), [wica.md](../specs/wica.md)).

## Goal

1. **G7 — `Agent.on_text: Event[str]`**, surfaced as `Wica.on_agent_text`. Emitted with the
   step's complete free text, when non-empty, immediately before the output sink is awaited; a
   raising sink does not affect the emission; subscribers are isolated by `Event.emit` as usual.
2. **G8 — `Command.triggers_on_completion: bool`** (default `True`, keyword-only constructor
   argument, read-only property; allowed when wrapping a `BaseTool`). `_dispatch_command`
   registers the `agent:command:<call_id>` entry with `triggers_llm_call` equal to the flag. For
   the output Command, `set_output_command(bare_callable)` wraps it as
   `Command(fn, triggers_on_completion=_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION)`; an explicit
   `Command` keeps its own flag. A name that matches no registered Command registers with `True`.
3. Specs back to `Implemented`; INTEGRATING.md and the AGENTS.md project map already say "five
   Events".

Out of scope: the `wica.testing` subpackage (next plan), exposing `Agent._history` (G4, not
needed), Event timestamps (G5, the harness stamps arrivals itself), a broadcast
`agent:command_activity` entry ([commands.md](../specs/commands.md) OQ#4).

## What the code looks like today

In [src/wica/agent.py](../src/wica/agent.py):

- `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION = True` (≈line 213) is a module constant.
- `__init__` (≈line 420) creates `self.on_trigger`, `self.on_prompt`, `self.on_command`.
- `set_output_command` (≈line 462) wraps a bare callable as `Command(fn)`.
- `_run_step` (≈line 786): `if text:` appends `AssistantTextRecord(text)` then awaits the sink
  inside a `try/except Exception`.
- `_dispatch_command` (≈line 834) computes `triggers_llm_call` from the constant when
  `name == self._output_command_name`, else `True`, then registers the entry and emits
  `on_command`.

In [src/wica/command.py](../src/wica/command.py): `Command.__init__(fn, *, name=None,
description=None)` and the `tool`/`name` properties.

In [src/wica/wica.py](../src/wica/wica.py): `__init__` surfaces the Agent's three Events.

## Ground rules

- Work the steps in order; after every code step run `uv run ruff check .`,
  `uv run ruff format .`, `uv run pyright`, `uv run pytest`. Do not move on with a failure.
- Do not rename or remove `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION`; it becomes the output
  Command's default, nothing else.
- Test style (AGENTS.md): drive the public API, assert on observable results. Fixtures and
  helpers in `tests/test_agent.py`: `loop`, `world`, `sink`, `make_agent`,
  `ProgrammableChatModel`, `sequence`, `text_response`, `tool_call_response`, `wait_until`.

## Step 1 — `Command.triggers_on_completion`

File: [src/wica/command.py](../src/wica/command.py).

Add the keyword-only parameter `triggers_on_completion: bool = True` to `__init__` (after
`description`), store it as `self._triggers_on_completion`, and expose:

```python
    @property
    def triggers_on_completion(self) -> bool:
        """Whether this Command's terminal completion wakes a reasoning step (default True).
        The per-Command form of the output Command's re-trigger knob; False removes only the
        trigger — the execution entry still exists and is observed by the next step. See
        specs/commands.md ("The `Command` object")."""
        return self._triggers_on_completion
```

The `BaseTool` branch keeps rejecting `name`/`description` but accepts the flag (it is a WICA
option, not tool metadata). Update the module and class docstrings' constructor examples.

Tests (`tests/test_command.py`): the default is `True`; `Command(fn, triggers_on_completion=False)`
reports `False`; `Command(existing_tool, triggers_on_completion=False)` is accepted and reports
`False` while `Command(existing_tool, name="x")` still raises.

## Step 2 — Honor the flag in `_dispatch_command`; make the constant the output default

File: [src/wica/agent.py](../src/wica/agent.py).

1. In `set_output_command`, replace `command = fn if isinstance(fn, Command) else Command(fn)`
   with
   ```python
   command = (
       fn
       if isinstance(fn, Command)
       else Command(fn, triggers_on_completion=_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION)
   )
   ```
   and update the constant's comment (≈line 209) to say it is the *default* for a bare-callable
   output Command, overridable per `Command`.
2. In `_dispatch_command`, replace the `triggers_llm_call = (...)` expression with
   ```python
   registered = self._commands.get(name)
   triggers_llm_call = registered.triggers_on_completion if registered is not None else True
   ```
   and reword the comment above it: every Command re-triggers by default; a Command registered
   with `triggers_on_completion=False` (the output Command's default comes from the constant)
   completes without waking a step, and its terminal entry is observed and retired by the next
   step that runs for any other reason.
3. Update the bullet in the `Commands` docstring/comment block near `register_command` if it
   mentions the constant.

Tests (`tests/test_agent.py`):

- `test_command_with_triggers_on_completion_false_does_not_retrigger`: register
  `Command(ping, triggers_on_completion=False)` where `ping` is a quick async function; script
  one tool call to it; after the entry goes terminal (per-key listener, as in
  `test_on_command_fires_once_the_execution_entry_exists_and_carries_its_call_id`), sleep 0.2 s
  and assert `len(model.calls) == 1`; then update the input again and assert the second step's
  prompt contains the completed `ping` entry (`"complete"` in the rendered human message), proving
  the outcome was observed and retired, not lost.
- `test_output_command_passed_as_command_keeps_its_own_retrigger_flag`: `set_output_command(
  Command(speak, triggers_on_completion=False))`; script `speak` then a text; assert only one
  model call after `speak` completes (no re-trigger), while a bare-callable output Command still
  re-triggers (extend the existing output-Command test or add a parametrized pair).

## Step 3 — `Agent.on_text`, surfaced as `Wica.on_agent_text`

Files: [src/wica/agent.py](../src/wica/agent.py), [src/wica/wica.py](../src/wica/wica.py).

1. In `Agent.__init__`, next to the other Events: `self.on_text: Event[str] = Event()`, with a
   comment line in the Events block: `on_text(text): the step's complete free text, right
   before the sink receives it (observation; the sink is delivery).`
2. In `_run_step`, inside `if text:` and before the `try` that awaits the sink:
   `self.on_text.emit(text)`.
3. In `Wica.__init__`: `self.on_agent_text: Event[str] = agent.on_text`.

Tests:

- `tests/test_agent.py::test_on_text_fires_with_the_step_text_before_the_sink`: subscribe a
  recorder and use a sink that appends to the same list, script `text_response("hello")`; assert
  the list is `["event:hello", "sink:hello"]` (order pins "before the sink"). Also assert an empty
  text response emits nothing.
- `tests/test_agent.py::test_on_text_fires_even_when_the_sink_raises`: sink raises; the Event
  still delivered the text (extend `test_output_sink_failure_is_logged_and_commands_still_dispatch`
  or add a sibling).
- `tests/test_wica.py`: extend
  `test_input_drives_world_and_agent_triggers_and_a_command_event` (or add a sibling) to subscribe
  `wica.on_agent_text` and assert it carried the scripted `"hi"`.

## Step 4 — Flip statuses

1. `specs/agent.md`, `specs/commands.md`, `specs/wica.md`: `**Status:** Updated` →
   `**Status:** Implemented`; their rows in `specs/_index.md` likewise.
2. This plan: `Todo` → `Done`; its row in [plans/_index.md](_index.md).

Final verification: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
`uv run pytest`, and `uv run pytest tests-e2e -k "fake or example"` all pass.
