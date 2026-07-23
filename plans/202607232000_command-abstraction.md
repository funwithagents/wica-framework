# Command abstraction (tools become the under-the-hood primitive)

**Status:** Done

## Goal

Elevate **Command** to a first-class WICA concept — the **C** in World / Inputs / Commands / Agents — and demote LangChain *tools* to the implementation primitive a Command runs on, matching the vocabulary already used informally in [agent.md](../specs/agent.md) (`cancel_activity` as "a model-issued command") and paralleling how LangChain and provider blocks are already quarantined to the Agent's I/O boundary ([content.md](../specs/content.md)).

Before this change the acting mechanism was called "tools" everywhere (spec section title, `register_tool`, `ToolCallStatus`, `agent:tool_call:<id>` keys), and the named `Commands` pillar had no spec at all.

## Scope

Specs, this plan, and code — a rename plus one new spec. No behavioral change: the event-driven execution-as-World-entry model is unchanged; only naming and framing move.

### Specs

- **New [specs/commands.md](../specs/commands.md) (Draft)** — owns the Command concept: Command = WICA's unit of agent action, backed under the hood by a LangChain tool; the `agent:command:<call_id>` execution-as-World-entry lifecycle (`running → complete|failed|cancelled`, terminal-triggers-a-step); `CommandExecution` value model; generic call description; WICA-native (non-tool) Commands (`cancel_activity`, future `speak`); the deferred sync/async optimization; open questions (failure rendering, mixed-speed parallel Commands, stale-`id` handling, generic command-activity listening, `render_result`). Much of this moved out of agent.md's old "Tools" section.
- **[specs/agent.md](../specs/agent.md)** — "Tools" section reframed to "Commands", delegating the concept/lifecycle to commands.md and keeping only Agent-loop concerns; all "tool"-as-WICA-concept references updated (history record kind, `cancel_activity`, `speak`, open questions #1/#7/#12, future improvements). Stays **Draft**.
- **[specs/_index.md](../specs/_index.md)** — added the `commands.md` row (Draft).
- **[specs/world.md](../specs/world.md)** — editorial link from open question #3 to commands.md's stale-`id` open question. Stays **Stable** (no code-affecting change).

### Code ([src/wica/agent.py](../src/wica/agent.py), tests)

| Before | After |
|---|---|
| `register_tool` | `register_command` |
| `ToolCallStatus` | `CommandExecution` |
| `ToolCallRecord` | `CommandRecord` |
| `_serialize_tool_call_status` | `_serialize_command_execution` |
| `_dispatch_tool_call` / `_run_tool` | `_dispatch_command` / `_run_command` |
| `cancel_tool_call` / `_cancel_tool_call_on_loop` | `cancel_command` / `_cancel_command_on_loop` |
| `self._tools` / `self._tool_call_keys` | `self._commands` / `self._command_keys` |
| World key `agent:tool_call:<id>` | `agent:command:<id>` |

`bind_tools` and `response.tool_calls` stay as-is — they are the genuine LangChain primitive at the I/O boundary. Tests (`tests/test_agent.py`, `tests-e2e/test_agent.py`) updated to the new API and key prefix.

## Verification

`uv run ruff check .`, `uv run pyright`, and `uv run pytest` all pass (50 tests). e2e tier not run here (opt-in, needs a live provider).
