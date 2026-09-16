# Runtime primer: a response's Commands run concurrently

**Status:** Done

Implements the concurrency line added to the WICA runtime primer in [specs/agent.md](../specs/agent.md) ("System prompt composition", part 1; "Concurrency and long-running Commands in v1") and echoed in [specs/commands.md](../specs/commands.md) ("Command execution as a World entry"). One sentence of framework behavior the model could not infer: several tool calls in one response all start at once and run concurrently, in no guaranteed order, so an action that must wait for another belongs in a later step. Deliberately leaves out any dispatch change — no sequential mode, no resource-exclusive Commands, no provider-level `parallel_tool_calls=False` — the behavior stays as it was; only the model is now told about it.

## Why

WICA dispatches every tool call of a response at once, as concurrent tasks. Most agent loops run a response's calls one after another, so a model assumes that calls listed in order run in order, and puts dependent or conflicting actions (two Commands using the same output device, say) in a single response — under WICA they then run at the same time. The primer said a tool call "runs asynchronously" and nothing about several calls running together, so that assumption stood. The primer's charter is exactly "what changes how the model chooses actions", so this is where the fact belongs; guarding a shared resource remains the application's job (a lock inside its Commands) or the model's (sequence across steps).

## Scope

- `src/wica/agent.py` — one bullet appended to `_RUNTIME_PRIMER` (perception + acting part), plus the constant's comment.
- `specs/agent.md` — part 1 of the primer composition names the line and its rationale; the concurrency section states that a response's Commands dispatch at once with no ordering/exclusivity and who guards a shared resource.
- `specs/commands.md` — one paragraph in "Command execution as a World entry" pointing at the same fact.
- `tests/test_agent.py` — the primer-composition test also pins the concurrency line.

## Steps

1. Append the bullet to `_RUNTIME_PRIMER` (after the deferred-outcome bullet, before the output-mode clause).
2. Edit the two spec sections and the commands.md paragraph.
3. Extend `test_system_prompt_composes_persona_and_runtime_primer_without_output_clause` to assert the line is present.

## Verification

`uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest` — all pass.
