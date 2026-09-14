# Gradio contrib: lift the World-state, prompt-history and transcript panels into `wica.contrib.gradio`

**Status:** Done

Implements [specs/gradio-contrib.md](../specs/gradio-contrib.md) (`Stable`, validated 2026-09-14)
and the matching 2026-09-14 update of [specs/conversation-demo.md](../specs/conversation-demo.md)
("Reused from the Gradio contrib", "Composition" step 2). Once `Done`, flip `gradio-contrib.md`
`Stable` → `Implemented` and `conversation-demo.md` `Updated` → `Implemented` (file + index).

## Motivation

The conversation demo grew three surfaces that show *the framework*, not the robot: the live
World-state table, the browsable prompt history, and the generic conversation transcript
(`TranscriptLog` + one `display_entry` hook, since commits 9d5a488 and c87845d). They live in
`examples/conversation_demo/` (`transcript.py`, and the table/prompt/chatbot wiring in `app_ui.py`),
so any other Gradio application on WICA would copy them. This plan moves them into the package as
an opt-in contrib, `wica.contrib.gradio`, behind a `wica[gradio]` extra, and turns the demo into
their reference consumer. It carries over, rather than re-derives, the three Gradio lessons the
demo paid for: `gr.HTML` instead of `gr.Dataframe` so fleeting Command rows render; a version
`gr.State` so the chatbot is pushed only on a real change and autoscroll stops fighting the
reader; and snap-to-newest for prompts, then hands off.

Decisions settled in the spec and accepted by the user: presenters take a `Wica` and subscribe in
their constructor; `PromptLog` takes the same `display_entry` hook as the transcript; every contrib
module imports Gradio at the top and `gradio` joins the `dev` group; `CommandExecution` is
re-exported from `wica`; each panel owns its own `gr.Timer`.

## Goal

### Packaging and public API

1. **`pyproject.toml`** — new extra `gradio = ["gradio>=5.0.0"]` under
   `[project.optional-dependencies]` (comment: contrib, not a provider; see
   `specs/gradio-contrib.md` "Packaging"); add `gradio>=5.0.0` to the `dev` group (pyright
   `include`s `src`, and the fast tier imports the contrib); the `demo` group keeps it. Re-lock
   (`uv lock`, `uv sync --dev`).
2. **`src/wica/contrib/__init__.py`** — docstring only: "optional, batteries attached, not core;
   never re-exported from `wica`".
3. **`src/wica/contrib/gradio/__init__.py`** — re-exports the public names of the four modules
   below (`EntryDisplay`, `DisplayEntry`, `TranscriptLog`, `TranscriptSnapshot`,
   `conversation_panel`, `PromptLog`, `PromptSnapshot`, `render_messages`, `prompt_panel`,
   `world_rows`, `world_html`, `world_state_panel`) with an `__all__`.
4. **`src/wica/__init__.py`** — export `CommandExecution` (from `wica.agent`) next to
   `CommandIssued`; add to `__all__`. Shape unchanged.

### `src/wica/contrib/gradio/display.py` — the shared hook

5. `EntryDisplay` (frozen dataclass `label`, `detail=None`) and `DisplayEntry`
   (`Callable[[WorldEntry], EntryDisplay | None]`), moved from the demo's `transcript.py`;
   `format_args(args) -> str` (the `k=v!r` join used by every surface);
   `display_entry_or_default(entry, hook, output_command_name) -> EntryDisplay` — the hook if it
   returns a value, else the generic default: `⚡ key = value!r`; for a `CommandExecution` value
   `🗣️ name` when `name == output_command_name` else `🦾 name`, detail `name(args)`.

### `src/wica/contrib/gradio/world_state.py`

6. Move `_format_value` / `_world_rows` / `_world_html` from `app_ui.py` as public `world_rows` and
   `world_html` (with the fixed-width, theme-variable CSS and the `_WORLD_TABLE_HEADERS` /
   `_WORLD_COL_WIDTHS` constants). **Command-row detection switches from the `agent:command:` key
   prefix to `isinstance(version.value, CommandExecution)`** (the row still gets `class="cmd"`).
   `None` → `—`, lists → `[a, b]`, else `str`; every cell `html.escape`d.
7. `world_state_panel(world, *, refresh_s=0.2) -> gr.HTML`: `gr.HTML(value=world_html(world))` plus
   its own `gr.Timer(refresh_s)` whose tick returns `world_html(world)`. Must be called inside a
   `gr.Blocks` context (document it; Gradio raises otherwise).

### `src/wica/contrib/gradio/prompts.py`

8. `render_messages(messages: list[BaseMessage]) -> str` — the demo's `_render_message` (+
   `_flatten_content`) over each message, joined by blank lines. Unchanged output: `### ROLE`
   header, concatenated text blocks, `[image <mime>]`, `🔧 tool_call name(args)  [id=…]`,
   `🔧 result for [id=…] => …`.
9. `PromptLog(wica: Wica | None, display_entry: DisplayEntry | None = None)`:
   - constructor subscribes `wica.on_agent_trigger → self.on_trigger` and
     `wica.on_agent_prompt → self.on_prompt` (nothing when `wica is None`);
   - `on_trigger(entry)`: label = `display_entry_or_default(...)`.label, or `f"{label} finished"`
     when the value is a `CommandExecution`; kept as `_last_trigger_label` (initial `"start"`; no
     lock — both Events fire sequentially on the agent loop, trigger before prompt);
   - `on_prompt(messages)`: append `{"label": f"{HH:MM:SS} — {trigger}", "text": render_messages(...)}`
     under a lock (append-only, dicts never mutated);
   - `snapshot() -> PromptSnapshot(count: int, choices: list[tuple[str, int]], newest_text: str | None)`
     under the lock; `text(index: int | None) -> str | None` with bounds check.
10. `prompt_panel(log, *, refresh_s=0.2, lines=16) -> tuple[gr.Dropdown, gr.Textbox]`: a
    `gr.Dropdown(label="Prompt sent to the model (newest shown automatically)", choices=[],
    interactive=True)` and a read-only `gr.Textbox(lines=lines, value="(no prompt sent to the
    model yet)")`; its own timer whose tick holds `last_shown_count` in a closure and returns
    `gr.update(choices=..., value=newest)` + `gr.update(value=newest_text)` only when the count
    grew, bare `gr.update()`s otherwise; `dropdown.change(lambda i: log.text(i) or gr.update(), …)`.

### `src/wica/contrib/gradio/transcript.py`

11. Move `TranscriptLog` and `TranscriptSnapshot` from the demo, **minus the prompt history**:
    drop `_prompts`, `prompt_text`, and the `prompt_count` / `prompt_choices` /
    `newest_prompt_text` snapshot fields (`TranscriptSnapshot(conversation, conv_sig)`); `on_prompt`
    keeps opening the reaction group (`💬 reaction N · <trigger>`) and therefore keeps
    `_last_trigger_label`. Rendering goes through `display.display_entry_or_default` (the
    `display()` method remains as the public per-entry renderer). Replace the duplicated
    `NOOP_COMMAND_NAME` / `COMMAND_KEY_PREFIX` constants with imports of `wica.agent`'s
    `_NOOP_COMMAND_NAME` / `_COMMAND_KEY_PREFIX` (in-package; not promoted). Everything else —
    the ordered queue, per-Command `add_listener` on `agent:command:<call_id>` with the async
    `on_command_update`, pending → terminal edits with `status` removed, duration, `🚫 noop`,
    `💭 output sink`, the change signature — is carried over verbatim with its comments.
12. `conversation_panel(log, *, refresh_s=0.2, height=420) -> gr.Chatbot`: `gr.Chatbot(label=
    "Conversation", height=height)`, a `gr.State(0)` version, its own timer whose tick compares
    `snapshot().conv_sig` to the last one (closure) and returns the bumped version only on change,
    and `version.change(lambda: log.snapshot().conversation, outputs=chatbot,
    show_progress="hidden")`. The chatbot is **never** an output of the timer — keep the demo's
    autoscroll comment on the function.

### Demo (`examples/conversation_demo/`)

13. **Delete `transcript.py`.**
14. **`app.py`** — `from wica import CommandExecution`; `from wica.contrib.gradio import EntryDisplay`
    for the `display_entry` hook's return type. Update the module docstring: importing `app` now
    pulls the contrib (and so Gradio, present in the `dev` group); the lazy `app_ui` import in
    `main()` can go. `register_world`, `Robot`, `build_system`, `wire` unchanged.
15. **`app_ui.py`** — remove the World table, prompt wiring, chatbot/version code and
    `_NO_PROMPT_YET`; keep `_speaking_html` and the Speaking panel's own 0.2 s timer, the sensor
    inputs, the Markdown copy and the layout. `build_ui` constructs `TranscriptLog(wica,
    display_entry)`, `PromptLog(wica, display_entry)` and `SpeakingSlot()`, and places
    `conversation_panel(transcript)` in the left column above the Speaking HTML and inputs,
    `world_state_panel(world)` and `prompt_panel(prompts)` in the right column. `DemoUi` keeps
    `blocks`, `transcript`, `speaking` (the app wires only those two). Refresh the module
    docstring.

### Tests

16. **`tests/contrib/`** (fast tier; import the contrib, build no `gr.Blocks`):
    - `conftest.py`: the `unstarted_wica` factory fixture moved from `tests/test_conversation_demo.py`
      (unstarted `Wica.init` over `provider: "fake"`, optional output Command, closed on teardown).
    - `test_gradio_transcript.py`: the moved `TranscriptLog` tests (generic default + hook
      fallback, `on_trigger` sides and Command re-trigger labelling, reaction group opened on
      `on_prompt`, `output_sink` blank/recorded, `on_command` pending item, `noop`, running →
      complete via a real World listener, failure/cancellation, foreign entry ignored, signature
      stability) — dropping their prompt-history assertions.
    - `test_gradio_prompts.py`: `render_messages` (moved flatten/render tests); new `PromptLog`
      tests — record two prompts after two triggers and assert labels (`HH:MM:SS — 🗣️ "hi"` via a
      hook, `… finished` for a `CommandExecution` trigger, `start` before any trigger), order,
      `snapshot()` count/choices/newest, `text()` bounds/`None`.
    - `test_gradio_world_state.py`: new — a `World` with two plain entries (`None`, a list, a
      string containing `<b>`) and one `agent:command:x` entry holding a running
      `CommandExecution`; assert `world_rows` values/formatting, that `world_html` puts the
      Command on a `class="cmd"` row formatted `name(args) [running]`, and that `<b>` is escaped.
17. **`tests/test_conversation_demo.py`** keeps only the demo's pieces: `display_entry`
    labelling, `SpeakingSlot`, `say` driving the slot; docstring updated.
18. **`tests-e2e/test_example_flow.py`** — import `TranscriptLog` from `wica.contrib.gradio`; also
    build a `PromptLog(wica, display_entry)` as `build_ui` does, and assert after the first step
    that `snapshot().count == 1` and its label ends with `🗣️ "…"` (the contrib exercised against
    the live loop). Docstring: no longer "imports no Gradio".
19. **`tests/test_project_map.py`** — extend to subpackages: `_actual_modules` globs
    `src/wica/**/*.py` and yields paths relative to `src/wica/` (`contrib/gradio/transcript.py`);
    `_MODULE_LINK` accepts subdirectories; `_NON_CONCEPT_MODULES` adds `contrib/__init__.py` and
    `contrib/gradio/__init__.py`; the frontmatter-governance check compares the same relative
    paths. Keep the messages actionable.

### Docs, specs and indexes

20. **`AGENTS.md`** — project map: one row `src/wica/contrib/gradio/` linking each of its four
    modules (`display.py`, `world_state.py`, `prompts.py`, `transcript.py`) → role → spec
    `gradio-contrib.md`; note that `contrib` is reached explicitly, not via `wica`. "Testing":
    mention `tests/contrib/` and that the fast tier now imports Gradio (in the `dev` group).
21. **`README.md`** — repo-layout row for `src/wica/` mentions the contrib; a short "Gradio
    panels" paragraph under "The conversation demo" with the `wica[gradio]` extra and a three-line
    usage snippet. **`INTEGRATING.md`** — a row "Show the World / prompts / transcript in a Gradio
    app → `specs/gradio-contrib.md`".
22. **`specs/gradio-contrib.md`** — fill the frontmatter: `code:` the four contrib modules, both
    `__init__.py`, `pyproject.toml`, `src/wica/__init__.py`; `tests:` the three `tests/contrib/`
    modules; delete the "frontmatter names only pyproject.toml" note. **`specs/commands.md`** —
    editorial note in "CommandExecution value model" that the type is re-exported from `wica`
    (status unchanged). **`specs/conversation-demo.md`** — frontmatter drops `transcript.py`;
    remove the "Updated (2026-09-14)" banner.
23. On `Done`: `gradio-contrib.md` → `Implemented`, `conversation-demo.md` → `Implemented` (files
    + [specs/_index.md](../specs/_index.md)); this plan → `Done` here and in
    [plans/_index.md](_index.md).

## Deviations (as built)

- The `unstarted_wica` fixture lives in `tests/conftest.py` (not `tests/contrib/conftest.py`)
  because the demo's `say` tests share it; the plain entry helpers (`make_entry`,
  `command_entry`, `item_titled`) moved to `tests/support.py`, mirroring `tests-e2e/support.py`.
- `tests-e2e/test_example_flow.py` had a latent race (asserting the `noop` item right after the
  second prompt appeared, although the prompt fires before the model call); it now waits for the
  item.
- Step 6 was run headlessly: the real page built over the fake provider, launched with
  `prevent_thread_lock`, fetched (HTTP 200), driven through a full turn (`say ✅`, re-trigger,
  `noop`), with the four timers present; plus the explore-only page over a World-only system.

## Non-goals

- A one-call `observability_panels(...)` wrapper, a shared `timer=` argument, an `all_entries`
  option on the World table, a styling API — all listed as deferrals in the spec.
- Gradio-free presenters (a `wica.contrib` module without the Gradio import).
- Moving the Speaking panel / `SpeakingSlot` out of the demo.
- Promoting `_NOOP_COMMAND_NAME` / `_COMMAND_KEY_PREFIX` to public names.
- Any behavioural change to the three surfaces: this is a move plus a split, and the transcript's
  rendered items, the prompt text, and the table rows must come out byte-identical to today's.

## Steps

1. Packaging + export (items 1–4); `uv lock && uv sync --dev`; confirm `import gradio` works in
   the dev env and `pyright` sees it.
2. Contrib modules (items 5–12): move code out of `transcript.py` / `app_ui.py` first, then add
   the three panel functions.
3. Contrib tests (item 16): move, split, add; run `uv run pytest tests/contrib`.
4. Demo migration (items 13–15) and its tests (items 17–18); run `uv run pytest` and
   `uv run pytest tests-e2e -k "fake or example"`.
5. Drift guard + docs (items 19–21); run `uv run pytest tests/test_project_map.py`.
6. Manual check: `uv run --group demo python -m examples.conversation_demo.app`, with and without
   the key — the transcript groups and Command states, the World table showing a `say` row while
   it speaks, the prompt dropdown snapping then staying put, and scrolling up in the chat while
   the robot talks, all as before.
7. Statuses (items 22–23).

## Verification

```
uv run ruff check .
uv run ruff format .
uv run pyright
uv run pytest
uv run pytest tests-e2e -k "fake or example"
```

Optionally one live provider: `zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run pytest tests-e2e -k anthropic'`.
