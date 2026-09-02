# Output Command, `noop`, and the `Command` definition object

**Status:** Done

## Motivation

Two coupled additions, driven by wanting a WICA agent whose user-facing output is a
**cancellable, observable action** (e.g. TTS you can barge-in on) rather than a plain string
handed to a sink:

1. **The output Command** — an optional, application-supplied Command that becomes the agent's
   user-facing output channel. When set, the model's free text becomes *private reasoning* and
   the Command is how it speaks. This realizes the once-deferred `speak()`
   ([agent.md](../specs/agent.md), "Future improvements"), generalized: any Command can be the
   channel. See [commands.md](../specs/commands.md), "The output Command".
2. **`noop`** — an always-registered, zero-arg WICA-native Command by which the model reliably
   declares *no reaction*, replacing the fragile "return empty text" (models do that poorly).
   It is the terminator for the output Command's re-trigger chain and, independently, the way
   any agent declines to react. See [commands.md](../specs/commands.md), "`noop`".

Both want a clean way to pass a command definition around (to `register_command` *and* to
`Wica.init(output_command=…)`). That surfaced a gap: WICA is named for Commands but has no
`Command` *type* — only `CommandExecution`/`CommandRecord`/`CommandIssued`. So this plan also
introduces:

3. **The `Command` definition object** — the concrete "registration/wrapper layer" the specs
   already name, wrapping a callable *or* an off-the-shelf `BaseTool` and holding the backing
   `BaseTool` internally. It **completes the LangChain quarantine** [agent.md](../specs/agent.md)
   claims: the command-definition surface (`register_command`, `output_command`) speaks
   `Command | Callable` only — `BaseTool` no longer appears there. See
   [commands.md](../specs/commands.md), "The `Command` object".

## Locked design decisions

Settled in discussion; recorded here so the build doesn't re-litigate them:

- **Output is a Command, not sink-routing** — for cancellation (barge-in) and observability,
  which an opaque inline `await output_sink(...)` cannot give.
- **`output_sink` is retained and its call site is unchanged** — it still receives the model's
  free text every step. Only the *meaning* shifts (utterance → private reasoning) when an
  output Command is set. No branching in the free-text path.
- **Re-trigger on output-command completion is on by default but one-line-reversible** — a
  module constant `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION = True`. `True` enables utterance
  chaining/self-continuation (≥2 LLM calls per utterance, terminated by `noop`); `False` makes
  speak → idle. `include_in_prompt` stays `True` either way (barge-in works regardless).
- **`noop` never triggers and creates no World entry** — it is the one Command that is not a
  World action. Its non-triggering is not tunable. It is recorded in history and rendered as a
  native tool call + *plain acknowledgement* tool_result (keeps the provider message stream
  valid; sidesteps consecutive-observation rendering).
- **Prompt injection is gated on the output Command** — the only place WICA augments the
  configured prompt. `noop` needs no injection (discovered from its tool description); a live
  test verifies models reach for it unaided.
- **`register_command(fn)` takes one argument** (`Callable | Command`); no `name`/`description`
  kwargs. Overrides go through `Command(fn, name=…, description=…)`; off-the-shelf tools via
  `Command(tool)`. Every existing external/test call site is already the bare form.

## Changes

### New module `src/wica/command.py`

1. Add `Command`:
   - `__init__(self, fn_or_tool: Callable[..., Any] | BaseTool, *, name=None, description=None)`.
     Move the `tool(...)`/`isinstance(BaseTool)` wrapping logic currently inside
     `Agent.register_command` ([agent.py:316-322](../src/wica/agent.py#L316)) here. A `BaseTool`
     is held as-is (name/description overrides rejected or ignored when a built tool is passed —
     pick "ignored" for simplicity, or raise; decide in build); a callable is wrapped with
     `tool(...)` (name from `__name__`, description from docstring unless overridden). An
     undocumented callable with no `description` keeps LangChain's loud `ValueError` (verified
     behavior) — optionally rewrap it as a clearer WICA error.
   - Expose `.tool` (the backing `BaseTool`, for the Agent to `bind_tools`) and `.name`.
   - This is the only file besides `agent.py` that imports LangChain tool machinery.
2. Re-export `Command` from `__init__.py` (it is runtime API, unlike `fake_model`), and add it
   to `__all__`.
3. **Project map**: add a `command.py` row to the module table in [AGENTS.md](../AGENTS.md)
   (and mention it under the `commands.md` spec column). Add `src/wica/command.py` to
   [commands.md](../specs/commands.md) frontmatter `code:`. `tests/test_project_map.py` enforces
   both; run it.

### `src/wica/agent.py`

4. `register_command(fn: Callable[..., Any] | Command) -> None` — drop the `name`/`description`
   parameters. If `fn` is a `Command`, use it; else wrap as `Command(fn)`. Store commands as
   `dict[str, Command]` (or keep the tool dict but source it from `Command.tool`); bind with
   `self.model.bind_tools([c.tool for c in self._commands.values()])`.
5. Constructor: add `output_command: Callable[..., Any] | Command | None = None`. Build it into
   a `Command` in `__init__` (so its `.name` is known), store `self._output_command` and
   `self._output_command_name`.
5a. **System-prompt composition** (see [agent.md](../specs/agent.md), "System prompt
    composition"): after resolving the persona, `self.system_prompt = persona + primer`, where the
    primer is composed in **three parts**: (1) always-on perception + acting (observations, and the
    async-command / deferred-outcome explanation); (2) an **output-mode clause** — default
    "reply in plain text" when there is no output Command, or the "free text is private, speak via
    `<name>`" clause when there is; (3) the always-on **`noop`** clause, written conservatively so a
    request/question is still answered. Constants: `_RUNTIME_PRIMER`, `_DEFAULT_OUTPUT_PROMPT`,
    `_OUTPUT_COMMAND_PROMPT` (template naming `<name>`), `_NOOP_PROMPT`. Compose once in `__init__`
    so the combined string is cache-stable. **Note (empirical, gpt-4o):** the output-mode clause is
    *not optional* — without a "how to reply" line, some models treat a plain request as an
    ignorable observation and `noop` it; part (2) is what keeps direct requests answered. The primer
    is fixed and non-optional in v1 (opt-out/override is a deferred open question).
6. `start()`: register the control Commands via `Command(...)`:
   - `cancel_command` → `register_command(Command(self._cancel_command_action, name=_CANCEL_COMMAND_NAME, description=_CANCEL_COMMAND_DESCRIPTION))`.
   - `noop` → `register_command(Command(self._noop_action, name=_NOOP_COMMAND_NAME, description=_NOOP_COMMAND_DESCRIPTION))` (a trivial zero-arg backing that is never actually invoked — the loop intercepts noop before dispatch — but is needed for the bound tool schema).
   - the output Command (if `self._output_command`) → `register_command(self._output_command)`.
7. Add constants: `_NOOP_COMMAND_NAME = "noop"`, `_NOOP_COMMAND_DESCRIPTION` (must state *when*
   to call it — it stands alongside the primer's `noop` clause), `_NOOP_ACK`, `_RUNTIME_PRIMER`,
   `_DEFAULT_OUTPUT_PROMPT`, `_OUTPUT_COMMAND_PROMPT`, `_NOOP_PROMPT`, and
   `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION = True`.
8. Add a **dedicated `NoReactionRecord(call_id: str)`** to the `HistoryRecord` union (decision
   pinned in [agent.md](../specs/agent.md), "History record shape" — *not* an overloaded
   `CommandRecord`). `_run_step` tool-call loop ([agent.py:534-540](../src/wica/agent.py#L534)):
   special-case `noop` — append a `NoReactionRecord(call_id)` and emit `on_command`
   (`CommandIssued("noop", {})`, for observability), but **do not** `_dispatch_command` it (no
   entry, no task, no trigger; it is not a World action). All other calls unchanged.
9. `_dispatch_command`: when `name == self._output_command_name`, register the
   `agent:command:<call_id>` entry with `triggers_llm_call=_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION`
   instead of the hardcoded `True` ([agent.py:558](../src/wica/agent.py#L558)). Everything else
   (`include_in_prompt=True`, the cancellable task, `_running_tasks` tracking) is identical, so
   `cancel_command` reaches it for barge-in.
10. `_render_messages` / `flush_assistant` ([agent.py:662-667](../src/wica/agent.py#L662)):
    render a `NoReactionRecord` as a native `noop` tool call whose `tool_result` is a **plain
    acknowledgement** (e.g. "Acknowledged — no action taken."), selected by record type
    (`isinstance` — the reason for the dedicated record) rather than `_command_ack(call_id)`
    (there is no World entry to point at).

### `src/wica/wica.py`

11. `Wica.init(..., output_command: Callable[..., Any] | Command | None = None, …)` — forward to
    `Agent(config.agent, …, output_command=output_command, …)`. Update the construction
    docstring/comment.
12. `Wica.register_command(fn)` — drop `name`/`description`; forward `fn` verbatim.

### Tests (fast tier, `tests/`)

13. New `tests/test_command.py`: `Command(fn)` derives name from `__name__` and description
    from docstring; `Command(fn, name=…, description=…)` overrides; `Command(existing_tool)`
    wraps unmodified (`.tool is existing_tool`, `.name` preserved); undocumented callable with
    no description raises. Add `tests/test_command.py` to [commands.md](../specs/commands.md)
    frontmatter `tests:`.
14. `tests/test_agent.py`:
    - `register_command(add)` still works (regression); `register_command(Command(add))` works;
      `register_command(Command(some_tool))` binds the tool.
    - System-prompt composition: the persona is present verbatim *and* the runtime primer is
      appended for every agent (assert the primer's perception/acting/`noop` content is in the
      rendered `SystemMessage`); the output clause naming the command is present **only** when
      `output_command` is set, absent otherwise.
    - Output-command dispatch registers its entry with `triggers_llm_call` matching
      `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION` (drive a scripted/fake model; assert the follow-up
      step happens when `True`). Assert `output_sink` still receives the step's free text when an
      output command is set.
    - `noop`: issuing it records history, creates **no** `agent:command:*` entry, spawns no
      task, fires **no** re-trigger, and ends the step (but *does* emit `on_command` for
      observability); its rendered tool_result is the plain ack.

### E2e (`tests-e2e/`)

15. `test_fake_flows.py` (always-run, scripted): 
    - **Output-command flow** — register a capturing `output_command`; script the model to emit
      free text + call the output Command; assert the capture fn received the user-facing text,
      `output_sink` received the free text (thinking), and (with the default flag `True`) a
      follow-up step runs and is terminated by a scripted `noop`.
    - **`noop` flow** — script the model to call `noop`; assert no `agent:command:*` entry, no
      re-trigger, history records it, step ends cleanly.
16. Live tests (parametrized over `PROVIDER_CONFIGS` — see [testing.md](../specs/testing.md)):
    - **`noop` live** — the one the user specifically wants: set up a "nothing to do" scenario
      (e.g. a passive observation, or a completed action needing no follow-up) and assert the
      model reliably declines by invoking `noop` (coarse house-style assertion: `noop` was
      called / no spurious command or output was produced). Verifies models discover it from its
      description without prompt injection.
    - Optionally an **output-command live** test — capturing output Command, assert user-facing
      text arrives via it and free text via the sink.
17. Add the new e2e test file(s) touched to the relevant spec frontmatter `tests:`
    ([commands.md](../specs/commands.md) / [agent.md](../specs/agent.md)) once they exist.

### Demo (`examples/`) — optional, nice-to-have

18. Not required for the feature. If useful, demonstrate an `output_command` in the conversation
    demo (a "robot speech" channel) with the sink relabeled as the thinking stream — but this can
    be a follow-up; the tests above are the acceptance surface.

### Specs

19. Already edited to `Updated` in this change: [commands.md](../specs/commands.md) (the
    `Command` object, output Command, `noop`), [agent.md](../specs/agent.md) (Output section,
    system-prompt composition + runtime primer, the control Commands, `NoReactionRecord`, resolved
    OQ#1, built `speak()`, new OQ#7 on primer opt-out), [wica.md](../specs/wica.md)
    (`output_command`, one-arg `register_command`). [config.md](../specs/config.md) is also lightly
    updated (resolution yields the *persona*; the Agent composes persona + primer) but stays
    `Implemented` — its own code (`config.py`) doesn't change, only the cross-reference. Flip
    commands/agent/wica back to `Implemented` once this plan is `Done`, and update their
    frontmatter `code:`/`tests:` as the new files land (steps 3, 13, 17).

## Verification

`uv run ruff check .` · `uv run ruff format .` · `uv run pyright` · `uv run pytest`
(then the always-run fake flows: `uv run pytest tests-e2e -k fake`; and, for the live `noop`
check, `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k noop'` against a
provider whose key is set).
