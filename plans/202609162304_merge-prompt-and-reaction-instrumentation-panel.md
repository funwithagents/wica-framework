# Merge the prompt-history and reaction-instrumentation views into one `ReactionLog`

**Status:** Done

Builds [specs/gradio-contrib.md](../specs/gradio-contrib.md) ("Component 2: Reaction history",
status `Updated`) in full: replaces the contrib's `PromptLog`/`prompt_panel` with `ReactionLog`/
`reaction_panel` — one dropdown of reactions (unchanged labelling/snap-to-newest behavior) with two
tabs underneath, "Prompt" (today's rendered messages, unchanged) and "Instrumentation" (the
reaction's `ReactionTrace`, rendered once `wica.on_agent_reaction_ended` fires for it). Also lands
the small wording updates [specs/conversation-demo.md](../specs/conversation-demo.md) (status
`Updated`) needs at the wiring layer it describes. Both specs' Consumers/Open-questions
reconciliation against [specs/instrumentation.md](../specs/instrumentation.md) is **already done**
(editorial only, that spec's status is unchanged at `Implemented`) — this plan is code + the
remaining prose docs (`AGENTS.md`, `INTEGRATING.md`, `README.md`) only.

This plan is written to be executed step by step, in order, by an agent with no other context.
Read [AGENTS.md](../AGENTS.md) first (verification and status rules), then the spec sections named
in each step.

**No change to `src/wica/agent.py`/`world.py`/`wica.py`/`instrumentation.py`.** Everything this
needs already exists and is `Implemented`: `wica.on_agent_trigger`, `wica.on_agent_prompt`,
`wica.on_agent_reaction_ended` (`ReactionTrace`), and `wica.instrumentation.reaction_latency`. This
is a `wica.contrib.gradio` + demo + docs change only.

Out of scope: the rolling/aggregate metrics view (instrumentation.md open question #1 — a
different shape, a strip across *many* reactions, not this one-reaction-at-a-time browser); the
`observability_panels()` one-call wrapper (gradio-contrib.md open question, still deferred);
sharing one `gr.Timer` across panels (same, still deferred). Do not touch the historical plans
[202607291530_demo-prompt-history.md](202607291530_demo-prompt-history.md) or
[202609141200_gradio-contrib-panels.md](202609141200_gradio-contrib-panels.md) — they are
immutable record of what was already built, not what this plan changes.

## What the code looks like today

Read these before step 1; the plan refers to them by name.

- [src/wica/contrib/gradio/prompts.py](../src/wica/contrib/gradio/prompts.py) (203 lines): module
  constant `_NO_PROMPT_YET`; `_flatten_content`, `_render_message`, public `render_messages`;
  `@dataclass(frozen=True) class PromptSnapshot(count, choices, newest_text)`; `class PromptLog`
  with `__init__(wica, display_entry=None)` (subscribes `wica.on_agent_trigger` → `on_trigger`,
  `wica.on_agent_prompt` → `on_prompt`, when `wica is not None`), `output_command_name` property,
  `on_trigger(entry)` (sets `self._last_trigger_label` via `display_entry_or_default`, labelling a
  `CommandExecution` trigger `"<label> finished"`), `on_prompt(messages)` (appends
  `{"label": f"{stamp} — {self._last_trigger_label}", "text": rendered}` to `self._prompts` under
  `self._lock`), `snapshot()` (returns `PromptSnapshot`), `text(index)` (bounds-checked read); then
  `prompt_panel(log, *, refresh_s=0.2, lines=16) -> tuple[gr.Dropdown, gr.Textbox]` — a `gr.Dropdown`
  + read-only `gr.Textbox`, a closure `last_shown_count` snapping the dropdown+textbox to the
  newest prompt only when `snapshot().count` has grown, and `on_select(index)` reading
  `log.text(index)` on the dropdown's `.change`.
- [src/wica/contrib/gradio/__init__.py](../src/wica/contrib/gradio/__init__.py): re-exports
  `PromptLog, PromptSnapshot, prompt_panel, render_messages` from `prompts.py` alongside the
  transcript/display/world_state names; module docstring names "the live World-state table, the
  prompt history and the conversation transcript".
- [src/wica/contrib/gradio/transcript.py](../src/wica/contrib/gradio/transcript.py) line ~241, one
  comment: `"is kept by \`PromptLog\`, not here.) Opens *pending* ..."`.
- [examples/conversation_demo/app_ui.py](../examples/conversation_demo/app_ui.py): module docstring
  says `build_ui` assembles "the five surfaces (Conversation / Speaking / World state / Prompt /
  Sensor inputs)" and names `prompt_panel`/`PromptLog`; imports `PromptLog`, `prompt_panel` from
  `wica.contrib.gradio`; `build_ui()` does `prompts = PromptLog(wica, display_entry)` next to
  `transcript = TranscriptLog(wica, display_entry)`; the right column places
  `gr.Markdown("### World state (live)")`, `world_state_panel(world)`, then `prompt_panel(prompts)`
  directly underneath, no header, no wrapper. `DemoUi` (the returned handle) carries only
  `blocks`/`transcript`/`speaking` — `prompts` is never returned, it's local to `build_ui`.
- [tests-e2e/test_example_flow.py](../tests-e2e/test_example_flow.py): imports
  `from wica.contrib.gradio import PromptLog, TranscriptLog`; `@dataclass(frozen=True) class Stack`
  has a `prompts: PromptLog` field; `stand_up(config)` builds
  `prompts = PromptLog(wica, display_entry)` and passes it into `Stack(...)`;
  `test_speech_input_drives_say_action_and_world_state` asserts, after `spoke_and_felt()`:
  `prompts = handle.prompts.snapshot()`, `prompts.count >= 1`,
  `prompts.choices[0][0].endswith('— 🗣️ "hello"')`, `"hello" in (handle.prompts.text(0) or "")`,
  then waits for `handle.prompts.snapshot().count >= 2` and checks
  `handle.prompts.snapshot().choices[1][0].endswith("— 🗣️ say finished")`.
- [tests/contrib/test_gradio_prompts.py](../tests/contrib/test_gradio_prompts.py) (129 lines): the
  rendering tests (`_flatten_content`, `render_messages` — unaffected, keep as-is) plus the log
  tests: `test_empty_log_snapshot_and_text` (checks `snap.newest_text`, `log.text(None)`,
  `log.text(0)`), `test_prompts_are_labelled_by_time_and_hook_rendered_trigger` (checks
  `snap.newest_text`), `test_command_completion_retrigger_is_labelled_finished`,
  `test_voice_is_labelled_from_the_agents_output_command` (uses the `unstarted_wica` fixture from
  [tests/conftest.py](../tests/conftest.py)), `test_text_indexes_the_history_and_bounds_check`
  (uses `log.text(...)`). Local helpers: `hook`, `_LABEL` regex, `trigger_part(label)`.
- [tests/contrib/test_gradio_transcript.py](../tests/contrib/test_gradio_transcript.py) has its own
  local `reaction_trace(reaction_id, busy_time) -> ReactionTrace` helper (fixes every other field)
  — narrower than what this plan's new tests need (varying `outcome`/`usage`/`error`/etc.); leave
  that file and its helper untouched, add a more general builder to `tests/support.py` instead (see
  step 7).
- [tests/support.py](../tests/support.py): shared plain helpers (`make_entry`, `command_entry`,
  `titles`, `item_titled`) for the contrib's fast tests.
- `ReactionTrace`/`TokenUsage`/`reaction_latency` are re-exported from `wica` and defined in
  [src/wica/instrumentation.py](../src/wica/instrumentation.py) — see
  [specs/instrumentation.md](../specs/instrumentation.md) ("Layer 1", "Metrics") for the exact
  field list and derived properties (`coalescing_wait`, `render_time`, `model_latency`,
  `busy_time`).
- [AGENTS.md](../AGENTS.md) project map, the `contrib/gradio` row (line ~33), and
  [INTEGRATING.md](../INTEGRATING.md) "4. Add a Gradio UI" (a full worked example importing
  `PromptLog`/`prompt_panel`, a panel-signatures block, and an ordering-rules bullet list), and
  [README.md](../README.md) "Gradio panels for your own app" (a shorter worked example, same
  imports) all reference `PromptLog`/`prompt_panel` and need the rename.

## Ground rules

- Work the steps in order. After every code step run `uv run ruff check .`, `uv run ruff format .`,
  `uv run pyright`, `uv run pytest`; from step 6 on also `uv run pytest tests-e2e -k example`. Do
  not move on with a failure.
- **This is a rename-and-extend, not a redesign.** Every existing `PromptLog`/`prompt_panel`
  behavior (labelling, snap-to-newest-then-leave-the-user-alone, `BaseMessage` rendering) is
  preserved exactly under its new name; only the Instrumentation tab and its own narrower
  refresh rule are new.
- Every new/renamed module, class or function keeps (or gets) a docstring naming the spec section
  it implements.
- `render_messages` keeps its name (it renders `list[BaseMessage]`, not reactions — nothing about
  it changes).

## Step 1 — Rename and extend the presenter: `src/wica/contrib/gradio/reactions.py`

Spec: [specs/gradio-contrib.md](../specs/gradio-contrib.md) "Component 2: Reaction history" (the
`ReactionLog` bullets).

1. `git mv src/wica/contrib/gradio/prompts.py src/wica/contrib/gradio/reactions.py`.
2. Update the module docstring to describe the merge (name `ReactionLog`, the two-tab shape, that
   a record is opened by `on_prompt` and completed later by `on_reaction_ended`; keep the "See
   specs/gradio-contrib.md" pointer, updated to "Component 2").
3. Add to the imports: `from wica import ReactionTrace` (already re-exported from `wica/__init__.py`
   — see [src/wica/__init__.py](../src/wica/__init__.py)) and
   `from wica.instrumentation import reaction_latency`.
4. Rename `_NO_PROMPT_YET` → `_NO_REACTION_YET = "(no reaction yet)"`. Add
   `_INSTRUMENTATION_PENDING = "(reaction still in progress — instrumentation appears once it ends)"`.
5. Add, near `render_messages`:

```python
def render_reaction_trace(trace: ReactionTrace) -> str:
    """The reaction's ReactionTrace as readable text: outcome, the derived measures (only the ones
    whose underlying stamp exists — a cancelled reaction has no model_latency), token usage,
    dispatched commands, noop, the error on a model_error outcome, and the OpenTelemetry trace id
    when an SDK is installed. See specs/instrumentation.md ("Layer 1", "Metrics")."""
    lines = [f"reaction {trace.reaction_id} — {trace.outcome}"]
    latency = reaction_latency(trace)
    if latency is not None:
        lines.append(f"reaction latency: {latency:.3f}s")
    lines.append(f"coalescing wait: {trace.coalescing_wait:.3f}s")
    if trace.render_time is not None:
        lines.append(f"render time: {trace.render_time:.3f}s")
    if trace.model_latency is not None:
        lines.append(f"model latency: {trace.model_latency:.3f}s")
    if trace.sink_duration is not None:
        lines.append(f"sink duration: {trace.sink_duration:.3f}s")
    lines.append(f"busy time: {trace.busy_time:.3f}s")
    if trace.usage is not None:
        cached = (
            f" ({trace.usage.cache_read_tokens} cached)"
            if trace.usage.cache_read_tokens
            else ""
        )
        lines.append(
            f"tokens: {trace.usage.input_tokens} in / {trace.usage.output_tokens} out{cached}"
        )
    if trace.command_call_ids:
        lines.append(f"commands: {', '.join(trace.command_call_ids)}")
    if trace.noop:
        lines.append("noop")
    if trace.error is not None:
        lines.append(f"error: {trace.error}")
    if trace.trace_id is not None:
        lines.append(f"trace: {trace.trace_id}")
    return "\n".join(lines)
```

6. Replace `PromptSnapshot` with:

```python
@dataclass(frozen=True)
class ReactionSnapshot:
    """A consistent read of the history for one UI tick: how many reactions there are, and the
    dropdown choices (label, index). Unlike the old PromptSnapshot there is no "newest text" field
    — the panel reads whichever index it needs (newest or currently selected) directly, since the
    Instrumentation tab needs that for the *selected* index too, not just the newest."""

    count: int
    choices: list[tuple[str, int]]
```

7. Replace `class PromptLog` with `class ReactionLog`:

```python
class ReactionLog:
    """Append-only, thread-safe history of the framework's reasoning steps: one record per
    reaction, holding both the exact prompt sent to the model (from `on_agent_prompt`) and, once
    the reaction ends, its ReactionTrace (from `on_agent_reaction_ended`) — merged because they
    describe the same step from two angles. See specs/gradio-contrib.md ("Component 2").

    Built over a `Wica`, it subscribes in its constructor to `on_agent_trigger` (to label the
    coming reaction by what triggered it), `on_agent_prompt` (to open the record) and
    `on_agent_reaction_ended` (to fill in that same record's trace, matched by `reaction_id`).
    `wica=None` is the explore-only case: nothing subscribes and the history stays empty.

    Threading: the three handlers run on the agent loop; `on_prompt` and `on_reaction_ended` fire
    at different points of the same reaction's life (`on_prompt` opens the record, well before
    `on_reaction_ended` fills it in), so both take the lock. Records are appended once and mutated
    exactly once (to attach the trace); the UI reads them on its own thread."""

    def __init__(
        self, wica: Wica | None, display_entry: DisplayEntry | None = None
    ) -> None:
        self._wica = wica
        self._display_entry = display_entry
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._last_trigger_label = "start"
        if wica is not None:
            wica.on_agent_trigger.subscribe(self.on_trigger)
            wica.on_agent_prompt.subscribe(self.on_prompt)
            wica.on_agent_reaction_ended.subscribe(self.on_reaction_ended)

    @property
    def output_command_name(self) -> str | None:
        return None if self._wica is None else self._wica.agent.output_command_name

    def on_trigger(self, entry: WorldEntry) -> None:
        """A World entry triggered a step — remember its label for the reaction that follows. A
        Command-completion re-trigger is labelled `<item> finished`."""
        display = display_entry_or_default(
            entry, self._display_entry, self.output_command_name
        )
        if isinstance(entry.current.value, CommandExecution):
            self._last_trigger_label = f"{display.label} finished"
        else:
            self._last_trigger_label = display.label

    def on_prompt(self, messages: list[BaseMessage]) -> None:
        """Open this reaction's record: the exact messages sent to the model, labelled by time +
        trigger. Its instrumentation is filled in later by on_reaction_ended."""
        rendered = render_messages(messages)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        with self._lock:
            self._records.append(
                {
                    "label": f"{stamp} — {self._last_trigger_label}",
                    "prompt_text": rendered,
                    "trace": None,
                }
            )

    def on_reaction_ended(self, trace: ReactionTrace) -> None:
        """Attach this reaction's trace to the record on_prompt opened for it, matched by
        reaction_id (on_agent_prompt fires exactly once per reaction that starts, in the same
        order ReactionTrace.reaction_id counts them, so reaction_id - 1 is that record's index).
        An index out of range is a defensive no-op — it should not happen."""
        with self._lock:
            index = trace.reaction_id - 1
            if 0 <= index < len(self._records):
                self._records[index]["trace"] = trace

    def snapshot(self) -> ReactionSnapshot:
        with self._lock:
            return ReactionSnapshot(
                count=len(self._records),
                choices=[(r["label"], i) for i, r in enumerate(self._records)],
            )

    def prompt_text(self, index: int | None) -> str | None:
        """The exact prompt text at `index`, or None if out of range."""
        if index is None:
            return None
        with self._lock:
            if 0 <= index < len(self._records):
                return self._records[index]["prompt_text"]
        return None

    def instrumentation_text(self, index: int | None) -> str | None:
        """The rendered ReactionTrace at `index`, the pending placeholder if the reaction hasn't
        ended yet, or None if `index` is out of range — the same three-way split `prompt_text`
        would have if a reaction could have no prompt, made explicit here because "not ended yet"
        is a real, common state (unlike an unset prompt)."""
        if index is None:
            return None
        with self._lock:
            if not (0 <= index < len(self._records)):
                return None
            trace = self._records[index]["trace"]
        return _INSTRUMENTATION_PENDING if trace is None else render_reaction_trace(trace)
```

8. **Two drift guards are checked against the literal `git mv`, not against this plan's pace —
   update both now, in the same step, or `tests/test_project_map.py` fails red for the rest of the
   plan's execution:**
   - [AGENTS.md](../AGENTS.md) project map, the `contrib/gradio` row: replace the
     `[prompts.py](src/wica/contrib/gradio/prompts.py)` link with
     `[reactions.py](src/wica/contrib/gradio/reactions.py)` (`test_agents_md_maps_exactly_the_wica_modules`
     diffs this map against every actual `src/wica/**/*.py` file — a stale or missing link fails it
     immediately, independent of what the row's *prose* says, which is updated in step 8 below).
   - [specs/gradio-contrib.md](../specs/gradio-contrib.md)'s frontmatter (see
     [AGENTS.md](../AGENTS.md), "Spec frontmatter"): `code:` —
     `src/wica/contrib/gradio/prompts.py` → `src/wica/contrib/gradio/reactions.py`. Leave `tests:`
     alone for now (`tests/contrib/test_gradio_prompts.py` still exists and is still this spec's
     test file until step 7 renames it too) — `test_spec_frontmatter_paths_all_exist` checks the
     `code:` and `tests:` lists together, so touch only the one path that actually moved in this
     step.
9. Verify: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, and
   `uv run pytest tests/test_project_map.py` (should be green again after step 8's two edits;
   the rest of `pytest` still fails on every file still referencing `PromptLog`/`prompt_panel` —
   fine at this point, fixed by the following steps).

## Step 2 — `reaction_panel`

Spec: [specs/gradio-contrib.md](../specs/gradio-contrib.md) "Component 2" (the "One dropdown, two
tabs" and "Snaps to the newest ... plus one exception" bullets).

Replace `prompt_panel` with:

```python
def reaction_panel(
    log: ReactionLog, *, refresh_s: float = 0.2, lines: int = 16
) -> tuple[gr.Dropdown, gr.Textbox, gr.Textbox]:
    """Create the reaction-history dropdown, its two tabs (Prompt / Instrumentation), and the timer
    that keeps them live. Call inside a `gr.Blocks` context.

    The dropdown snaps to the newest reaction exactly as the old prompt_panel did — selecting it
    and showing both its (known) prompt and its (pending) instrumentation — and otherwise sends no
    update, so the user can browse older reactions freely. The one addition: every tick also
    re-reads the Instrumentation tab for whichever reaction is *currently selected* and, if its
    text changed (pending -> the rendered trace, once that reaction ends), pushes just that pane —
    never the dropdown, never the Prompt tab — so a reaction being watched updates in place once
    its trace lands, the same way the transcript's own reaction group loses its spinner in place.
    """
    selector = gr.Dropdown(
        label="Reaction (newest shown automatically)",
        choices=[],
        interactive=True,
    )
    with gr.Tabs():
        with gr.Tab("Prompt"):
            prompt_view = gr.Textbox(
                show_label=False, lines=lines, interactive=False, value=_NO_REACTION_YET
            )
        with gr.Tab("Instrumentation"):
            instrumentation_view = gr.Textbox(
                show_label=False, lines=lines, interactive=False, value=_NO_REACTION_YET
            )

    last_shown_count = 0
    last_instrumentation: dict[int, str] = {}

    def tick(selected: int | None) -> tuple[Any, Any, Any]:
        nonlocal last_shown_count
        snap = log.snapshot()
        if snap.count > last_shown_count:
            last_shown_count = snap.count
            newest = snap.count - 1
            instrumentation = log.instrumentation_text(newest)
            if instrumentation is not None:
                last_instrumentation[newest] = instrumentation
            return (
                gr.update(choices=snap.choices, value=newest),
                gr.update(value=log.prompt_text(newest)),
                gr.update(value=instrumentation),
            )
        if selected is not None:
            current = log.instrumentation_text(selected)
            if current is not None and last_instrumentation.get(selected) != current:
                last_instrumentation[selected] = current
                return gr.update(), gr.update(), gr.update(value=current)
        return gr.update(), gr.update(), gr.update()

    def on_select(index: int | None) -> tuple[Any, Any]:
        prompt = log.prompt_text(index)
        instrumentation = log.instrumentation_text(index)
        if instrumentation is not None and index is not None:
            last_instrumentation[index] = instrumentation
        return (
            gr.update() if prompt is None else prompt,
            gr.update() if instrumentation is None else instrumentation,
        )

    selector.change(on_select, inputs=selector, outputs=[prompt_view, instrumentation_view])
    timer = gr.Timer(refresh_s)
    timer.tick(
        tick, inputs=selector, outputs=[selector, prompt_view, instrumentation_view]
    )
    return selector, prompt_view, instrumentation_view
```

Verify with `uv run ruff check .`, `uv run ruff format .`, `uv run pyright` (still expect
downstream import errors, fixed next).

## Step 3 — `src/wica/contrib/gradio/__init__.py`

1. Replace the `from wica.contrib.gradio.prompts import (...)` block with:

```python
from wica.contrib.gradio.reactions import (
    ReactionLog,
    ReactionSnapshot,
    reaction_panel,
    render_messages,
    render_reaction_trace,
)
```

2. Update `__all__`: remove `"PromptLog"`, `"PromptSnapshot"`, `"prompt_panel"`; add
   `"ReactionLog"`, `"ReactionSnapshot"`, `"reaction_panel"`, `"render_reaction_trace"`. Keep
   `__all__` sorted (matches the existing convention).
3. Update the module docstring's surface list ("the live World-state table, the prompt history and
   the conversation transcript" → "... the reaction history (prompt + instrumentation) and the
   conversation transcript").

## Step 4 — `transcript.py` cross-reference

[src/wica/contrib/gradio/transcript.py](../src/wica/contrib/gradio/transcript.py): update the one
comment referencing `PromptLog` (in `TranscriptLog`'s class docstring, near "is kept by
`PromptLog`, not here") to `ReactionLog`.

Verify: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest` (expect
failures only in `tests/contrib/test_gradio_prompts.py` and `tests-e2e/test_example_flow.py`,
fixed in steps 6-7).

## Step 5 — Demo: `examples/conversation_demo/app_ui.py`

1. Module docstring: "the five surfaces (Conversation / Speaking / World state / Prompt / Sensor
   inputs)" → "... / Reaction history / ..."; "`conversation_panel`, `world_state_panel`,
   `prompt_panel`" → "..., `reaction_panel`"; "`PromptLog`" → "`ReactionLog`".
2. Imports: `PromptLog` → `ReactionLog`, `prompt_panel` → `reaction_panel`.
3. In `build_ui()`: `prompts = PromptLog(wica, display_entry)` → `reactions = ReactionLog(wica, display_entry)`.
4. In the right-hand `gr.Column`, replace:

```python
                gr.Markdown("### World state (live)")
                world_state_panel(world)
                prompt_panel(prompts)
```

   with:

```python
                gr.Markdown("### World state (live)")
                world_state_panel(world)
                gr.Markdown("### Reaction history")
                reaction_panel(reactions)
```

   (`reaction_panel` returns three components the demo doesn't need to reference further, so the
   call site doesn't need to capture its return value — same as today's `prompt_panel(prompts)`.)
5. `DemoUi` is unaffected (it never carried `prompts`).

Verify: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`.

## Step 6 — `tests-e2e/test_example_flow.py`

1. Import: `from wica.contrib.gradio import PromptLog, TranscriptLog` →
   `from wica.contrib.gradio import ReactionLog, TranscriptLog`.
2. `Stack`: `prompts: PromptLog` → `reactions: ReactionLog`.
3. `stand_up()`: `prompts = PromptLog(wica, display_entry)` → `reactions = ReactionLog(wica, display_entry)`;
   update the `Stack(...)` call's keyword from `prompts=prompts` to `reactions=reactions`.
4. In `test_speech_input_drives_say_action_and_world_state`, replace:

```python
        # The prompt history captured this step's prompt, labelled by the speech trigger, and
        # the prompt text carries the rendered World (the speech entry the model saw).
        prompts = handle.prompts.snapshot()
        assert prompts.count >= 1
        assert prompts.choices[0][0].endswith('— 🗣️ "hello"')
        assert "hello" in (handle.prompts.text(0) or "")

        # `say` completing re-triggered a second step (ended by noop), so a second prompt appears,
        # labelled by the Command's completion.
        wait_until(lambda: handle.prompts.snapshot().count >= 2)
        assert handle.prompts.snapshot().choices[1][0].endswith("— 🗣️ say finished")
```

   with:

```python
        # The reaction history captured this step's prompt, labelled by the speech trigger, and
        # the prompt text carries the rendered World (the speech entry the model saw).
        reactions = handle.reactions.snapshot()
        assert reactions.count >= 1
        assert reactions.choices[0][0].endswith('— 🗣️ "hello"')
        assert "hello" in (handle.reactions.prompt_text(0) or "")

        # Its instrumentation starts pending and resolves once the reaction ends — it must have
        # ended well before this point, since the sink and the say Command it dispatched already
        # completed (spoke_and_felt() above waited for that).
        instrumentation = handle.reactions.instrumentation_text(0)
        assert instrumentation is not None and "in progress" not in instrumentation
        assert "reaction 1 — ok" in instrumentation

        # `say` completing re-triggered a second step (ended by noop), so a second reaction
        # appears, labelled by the Command's completion.
        wait_until(lambda: handle.reactions.snapshot().count >= 2)
        assert handle.reactions.snapshot().choices[1][0].endswith("— 🗣️ say finished")
```

Verify: `uv run pytest tests-e2e -k example` (fake, key-less, always-run — see
[specs/fake-provider.md](../specs/fake-provider.md)).

## Step 7 — Tests: rename and extend `tests/contrib/test_gradio_prompts.py`

Spec: [specs/gradio-contrib.md](../specs/gradio-contrib.md) "Testing".

1. `git mv tests/contrib/test_gradio_prompts.py tests/contrib/test_gradio_reactions.py`.
2. Update the module docstring: name `ReactionLog`/`render_reaction_trace`, point at "Component 2".
3. Imports: `from wica.contrib.gradio import EntryDisplay, PromptLog, render_messages` →
   `from wica.contrib.gradio import EntryDisplay, ReactionLog, render_messages, render_reaction_trace`;
   `from wica.contrib.gradio.prompts import _flatten_content` →
   `from wica.contrib.gradio.reactions import _flatten_content`.
4. Keep the rendering tests (`_flatten_content`, `render_messages` — the three
   `test_render_messages_*` and three `test_flatten_content_*`) unchanged; they test unrelated
   functions.
5. Rewrite the log tests:
   - `test_empty_log_snapshot_and_text` → assert `(snap.count, snap.choices) == (0, [])`,
     `log.prompt_text(None) is None`, `log.prompt_text(0) is None`,
     `log.instrumentation_text(None) is None`, `log.instrumentation_text(0) is None` (nothing
     recorded yet, so still "out of range", not "pending").
   - `test_prompts_are_labelled_by_time_and_hook_rendered_trigger`: same trigger/prompt sequence,
     `PromptLog` → `ReactionLog`; drop the `snap.newest_text` assertion (no longer a snapshot
     field), replace with `"after emotion" in (log.prompt_text(2) or "")`.
   - `test_command_completion_retrigger_is_labelled_finished`,
     `test_voice_is_labelled_from_the_agents_output_command`: rename `PromptLog` → `ReactionLog`,
     otherwise unchanged (both only exercise labelling).
   - `test_text_indexes_the_history_and_bounds_check` → rename to
     `test_prompt_text_indexes_the_history_and_bounds_check`, `log.text(...)` →
     `log.prompt_text(...)`.
6. Add new tests for the merge, using `make_reaction_trace` (add to
   [tests/support.py](../tests/support.py) first — see below):
   - `test_instrumentation_is_pending_until_the_reaction_ends`: `log = ReactionLog(None)`;
     `log.on_prompt([HumanMessage(content="hi")])`; assert
     `log.instrumentation_text(0) is not None and "in progress" in log.instrumentation_text(0)`;
     `log.on_reaction_ended(make_reaction_trace(reaction_id=1))`; assert
     `log.instrumentation_text(0) == render_reaction_trace(make_reaction_trace(reaction_id=1))`.
   - `test_on_reaction_ended_for_an_unknown_index_is_ignored`: `log = ReactionLog(None)`;
     `log.on_prompt([HumanMessage(content="hi")])`; `before = log.snapshot()`;
     `log.on_reaction_ended(make_reaction_trace(reaction_id=7))`; assert
     `log.snapshot() == before` and `log.instrumentation_text(0)` still reads pending.
   - `test_render_reaction_trace_shows_only_present_measures`: a trace with `model_started_at=None,
     model_ended_at=None` (a `cancelled` outcome) → `"model latency"` not in the rendered text; a
     normal trace → `"model latency"` **is** present.
   - `test_render_reaction_trace_shows_usage_commands_noop_and_error`: one trace with
     `usage=TokenUsage(3, 2, None)`, `command_call_ids=("c1",)` → both appear; one with `noop=True`
     → `"noop"` present; one with `outcome="model_error", error="boom"` → `"error: boom"` present.
7. In [tests/support.py](../tests/support.py), add (imports: `from datetime import datetime,
   timedelta, timezone`; `from typing import Any`; `from wica import ReactionTrace, TokenUsage`):

```python
def make_reaction_trace(**overrides: Any) -> ReactionTrace:
    """A hand-built ReactionTrace with sensible defaults (an ordinary, fast, textful reaction), any
    field overridable by keyword — shared by the reaction-history and (future) other contrib tests
    that need a fully-formed trace rather than one driven through a live Agent."""
    t0 = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
    defaults: dict[str, Any] = {
        "reaction_id": 1,
        "triggers": (),
        "window_opened_at": t0,
        "window_closed_at": t0,
        "prompt_ready_at": t0 + timedelta(seconds=0.05),
        "model_started_at": t0 + timedelta(seconds=0.05),
        "model_ended_at": t0 + timedelta(seconds=0.5),
        "outcome": "ok",
        "error": None,
        "text_length": 5,
        "sink_duration": 0.01,
        "command_call_ids": (),
        "noop": False,
        "usage": None,
        "ended_at": t0 + timedelta(seconds=0.6),
        "trace_id": None,
        "span_id": None,
    }
    return ReactionTrace(**{**defaults, **overrides})
```

Verify: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`.

## Step 8 — Docs: `AGENTS.md` row prose, `INTEGRATING.md`, `README.md`

1. [AGENTS.md](../AGENTS.md) project map, the `contrib/gradio` row: the link itself
   (`[reactions.py](src/wica/contrib/gradio/reactions.py)`) was already updated in step 1.8 — here,
   update only the row's **prose**: "the prompt history (`PromptLog` + `prompt_panel`)" → "the
   reaction history (`ReactionLog` + `reaction_panel`, prompt + instrumentation tabs)".
2. [specs/gradio-contrib.md](../specs/gradio-contrib.md)'s frontmatter `tests:` list: now that step
   7 has renamed the test file, `tests/contrib/test_gradio_prompts.py` →
   `tests/contrib/test_gradio_reactions.py` (the other half of the frontmatter update deferred from
   step 1.8).
3. [INTEGRATING.md](../INTEGRATING.md) "4. Add a Gradio UI":
   - Heading and intro prose: "the three observability panels" list still reads correctly (World
     state, prompt history, transcript → World state, reaction history, transcript) — reword "the
     prompt history" to "the reaction history".
   - Code sample: import `PromptLog, prompt_panel` → `ReactionLog, reaction_panel`;
     `prompts = PromptLog(wica, display_entry)` → `reactions = ReactionLog(wica, display_entry)`
     with the comment updated ("so transcript and reaction labels agree"); `prompt_panel(prompts)`
     → `reaction_panel(reactions)`.
   - Panel-signatures block: `prompt_panel(prompts, *, refresh_s=0.2, lines=16) -> tuple[gr.Dropdown, gr.Textbox]`
     → `reaction_panel(reactions, *, refresh_s=0.2, lines=16) -> tuple[gr.Dropdown, gr.Textbox, gr.Textbox]`;
     `TranscriptLog(wica, display_entry=None); PromptLog(wica, display_entry=None)` →
     `TranscriptLog(wica, display_entry=None); ReactionLog(wica, display_entry=None)`.
   - Ordering-rules bullets: "`TranscriptLog`/`PromptLog` subscribe..." → "`TranscriptLog`/`ReactionLog`
     subscribe..."; "One hook for both presenters" bullet's "the prompt history's labels" →
     "the reaction history's labels".
4. [README.md](../README.md) "Gradio panels for your own app": same renames in its shorter code
   sample (`PromptLog, prompt_panel` imports; `prompts = PromptLog(...)`; `prompt_panel(prompts)`)
   and the prose line "the live World-state table, the prompt history and the conversation
   transcript" → "..., the reaction history ...".

Verify: `uv run ruff check .`, `uv run ruff format .` (docs aren't linted, but re-run anyway as the
final sanity pass), `uv run pyright`, `uv run pytest`, `uv run pytest tests-e2e -k "fake or example"`.

## Step 9 — Statuses and final verification

1. [specs/gradio-contrib.md](../specs/gradio-contrib.md): `Updated` → `Implemented`.
2. [specs/conversation-demo.md](../specs/conversation-demo.md): `Updated` → `Implemented`.
3. [specs/_index.md](../specs/_index.md): flip both rows' Status column back to `Implemented`
   (their description text was already updated when this plan's specs were written — no further
   edit needed there beyond the status word).
4. This plan: `Todo` → `Done`, and its row in [plans/_index.md](_index.md).
5. Final verification: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
   `uv run pytest`, `uv run pytest tests-e2e -k "fake or example"` all pass, and
   `uv run pytest tests/test_project_map.py` is green (the project map row from step 8 must match
   `reactions.py`'s existence and the spec frontmatter from this same step).
