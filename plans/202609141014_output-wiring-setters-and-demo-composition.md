# Output wiring setters (`set_output_sink` / `set_output_command`) and demo composition

**Status:** Done

Implements the 2026-09-14 updates of [specs/wica.md](../specs/wica.md) ("Output wiring is
delegated"), [specs/agent.md](../specs/agent.md) ("Output wiring", "System prompt composition"),
[specs/commands.md](../specs/commands.md) (the output Command's name check moves to the setter) and
[specs/conversation-demo.md](../specs/conversation-demo.md) ("Composition: build, then wire, then
start"). Once `Done`, flip those four specs from `Updated` back to `Implemented` (file + index).

## Motivation

`output_sink` and `output_command` are constructor arguments of `Agent` and keyword arguments of
`Wica.init`. That forces an application to build its output objects *before* the `Wica` exists,
while those objects usually need the `Wica` — the World for per-Command listeners, the `Event`s to
subscribe, the Agent's `output_command_name`. The conversation demo absorbed that cycle with two
workarounds: a two-phase `TranscriptLog` (`__init__` then `attach(wica)`, with a nullable `_world`
and a duplicated `output_command_name=` kwarg) and late-bound module globals (`world`, `state`)
that the Commands close over "at call time, always after that binding".

Dedicated setters make output wiring symmetric with `register_command` and allow the natural
order: **build the Wica → build the objects that need it → wire → start**.

**Breaking change, accepted**: the two kwargs are removed, not deprecated. All in-repo callers
(tests, e2e helpers, demo) are migrated in this plan; the user updates external code.

## Goal

### Framework

1. **`Agent.__init__(config, *, world, loop, coalesce_window=0.2, model=None)`** — drop
   `output_sink` and `output_command`. Keep the resolved persona (`self._persona`) and factor the
   primer assembly into `_compose_system_prompt()` so `self.system_prompt` (still a public
   attribute; tests read it) can be recomposed. Initial state: no output Command, no-op sink.
2. **`Agent.set_output_sink(sink: Callable[[str], Awaitable[None]] | None) -> None`** — assign;
   `None` restores `_noop_output_sink`. No other effect (read once per step at the delivery site).
3. **`Agent.set_output_command(fn: Callable[..., Any] | Command | None) -> None`**, per
   [agent.md](../specs/agent.md) "Output wiring":
   - wrap a callable as `Command(fn)`; validate *before* touching state — reserved name
     (`_RESERVED_COMMAND_NAMES`) or a name in `_commands` that is not the current output Command →
     `ValueError`;
   - detach the previous output Command if attached (`_commands.pop(name)` + rebind);
   - set `_output_command` / `_output_command_name`; recompose `system_prompt`;
   - if `_started`, attach now via `_register_builtin` (rebinds tools); otherwise `start()`
     attaches it as today.
   - `None`: detach if attached, clear both fields, recompose (default clause).
   - `register_command` keeps rejecting `_output_command_name` (now symmetric).
4. **`Wica.init(config, *, coalesce_window=0.2, loop=None)`** — drop the two kwargs;
   **`Wica.set_output_sink`** / **`Wica.set_output_command`** delegate verbatim to the Agent.
5. Docstrings/comments in `agent.py` and `wica.py` updated to the new flow (the constructor no
   longer "builds the output Command before prompt composition so its name is known").
6. `AGENTS.md` project map: the `wica.py` row lists `set_output_sink`/`set_output_command` next to
   `register_command`.

### Demo (`examples/conversation_demo/`)

7. **`transcript.py`** — `TranscriptLog(wica: Wica | None, display_entry: DisplayEntry | None = None)`:
   subscribes the three `Event`s and binds the World in the constructor; drop `attach`, the
   nullable-until-attached caveat and the `output_command_name=` kwarg. The 🗣️ icon reads
   `wica.agent.output_command_name` **live** at render time (the UI is built before
   `set_output_command` runs). `wica=None` is the explore-only case: nothing subscribes, Command
   items are created but never followed (no Agent runs anyway).
8. **`app_state.py` → `speaking.py`** — keeps `Speaking`/`SpeakingSlot` unchanged (Gradio-free, the
   Speaking panel's model); `DemoState` is deleted.
9. **`app_ui.py`** — `build_ui(wica: Wica | None, world: World, display_entry, config_error) -> DemoUi`,
   where `DemoUi` is a small dataclass `{blocks: gr.Blocks, transcript: TranscriptLog, speaking: SpeakingSlot}`.
   `build_ui` constructs both presenters (the UI owns its state) and lays out the panels as today;
   the rendering code is otherwise unchanged.
10. **`app.py`** — no module globals, no `AppHandle`, no `build_app`, no `COMMANDS`:
    - `register_world(world: World)` takes the World;
    - `class Robot(world: World, speaking: SpeakingSlot)` with methods `say` (output Command,
      drives `self.speaking`), `dance`, `set_emotion`, `switch_user_tracking` (the last two write
      `self.world`), and `commands` → the three non-output Commands. Bound methods are valid
      `Command`s (name from the method, description from its docstring, `self` excluded from the
      schema — verified). `_SAY_WORD_DELAY_S` stays a module constant read at call time (tests
      monkeypatch it).
    - `build_system(config) -> tuple[Wica | None, World, str | None]`: `Wica.init` with the
      `MissingEnvError` → World-only fallback as today; registers the World entries in both cases.
    - `wire(wica: Wica, transcript: TranscriptLog, speaking: SpeakingSlot) -> Robot`: builds the
      Robot, `wica.set_output_sink(transcript.output_sink)`, `wica.set_output_command(robot.say)`,
      registers `robot.commands`. Gradio-free; does **not** start.
    - `main()`: logging → `build_system` → `build_ui` → `wire` (if a Wica) → `wica.start()` →
      `ui.blocks.launch()` → `wica.close()` in `finally`. The module docstring describes this order.

### Tests

11. `tests/test_agent.py`: `make_agent` / `_dummy_agent` call `agent.set_output_sink(sink)` after
    construction; the output-Command tests use `set_output_command`. New functional tests:
    - set before `start()` → attached at start, `output_command_name`, prompt clause names it;
    - set after `start()` → attached immediately, an issued call to it dispatches and re-triggers
      per `_TRIGGER_ON_OUTPUT_COMMAND_COMPLETION`;
    - replace → old name gone from `agent.commands`, new one bound, prompt clause updated;
    - `set_output_command(None)` → detached, `output_command_name is None`, default clause back;
    - reserved name / already-registered application name → `ValueError`, `agent.commands` and
      `system_prompt` unchanged; `register_command` of the output name still rejected;
    - `set_output_sink` after start → next step's text reaches the new sink; `None` → no delivery.
12. `tests/test_wica.py`: migrate to the setters; one facade test that text and an output Command
    flow through `wica.set_output_sink` / `wica.set_output_command`.
13. `tests-e2e/support.py`: `real_wica` no longer forwards output kwargs (callers set after);
    `tests-e2e/test_wica.py`, `tests-e2e/test_fake_flows.py` migrated (rename
    `test_output_command_flow_through_wica_init` → `…_through_setters`).
14. `tests/test_conversation_demo.py`: `TranscriptLog` tests build an **unstarted** `Wica.init`
    over a `provider: "fake"` config (key-less, no thread until `start()`) instead of the
    `output_command_name=` kwarg; the `say` tests build `Robot(world, SpeakingSlot())` directly and
    drop the `monkeypatch.setattr(app, "state", …, raising=False)` fixture; imports follow the
    rename to `speaking.py`.
15. `tests-e2e/test_example_flow.py`: plays the UI's role — `build_system(config)`,
    `TranscriptLog(wica, display_entry)`, `SpeakingSlot()`, `wire(...)`, `wica.start()`; assertions
    unchanged. Still imports no Gradio.

### Specs and indexes

16. [specs/conversation-demo.md](../specs/conversation-demo.md) frontmatter: replace
    `app_state.py` with `speaking.py` once the file exists (the path was removed from the list in
    the spec update so `tests/test_project_map.py` stays green throughout).
17. On `Done`: `wica.md`, `agent.md`, `commands.md`, `conversation-demo.md` → `Implemented` in the
    file and in [specs/_index.md](../specs/_index.md); this plan → `Done` here and in
    [plans/_index.md](_index.md).

## Non-goals

- Deprecation shims for the removed kwargs (explicitly not wanted).
- Locking the model rebinding for setters called while running — same contract as
  `register_command` today, documented in the spec.
- The explore-only fallback's shape (`Wica | None` + a standalone World) — unrelated to output
  wiring; a "Wica without a live model" is a separate question.
- Lifting the transcript into `wica.contrib` ([gradio-contrib.md](../specs/gradio-contrib.md)
  keeps it out of scope).

## Steps

1. Framework: `agent.py` (items 1–3), `wica.py` (4), docstrings (5), `AGENTS.md` (6).
2. Framework tests: items 11–13; run the default tier and `uv run pytest tests-e2e -k fake`.
3. Demo: items 7–10 (add `speaking.py`, delete `app_state.py`, rewrite `app.py` standup,
   `build_ui` returns `DemoUi`).
4. Demo tests: items 14–15; run `uv run pytest tests-e2e -k "fake or example"`.
5. Manual check: `uv run --group demo python -m examples.conversation_demo.app` with and without
   the key (explore-only mode still opens).
6. Statuses (items 16–17).

## Verification

```
uv run ruff check .
uv run ruff format .
uv run pyright
uv run pytest
uv run pytest tests-e2e -k "fake or example"
```

Optionally one live provider: `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k anthropic'`.
