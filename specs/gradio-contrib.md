---
code:
  - src/wica/contrib/__init__.py
  - src/wica/contrib/gradio/__init__.py
  - src/wica/contrib/gradio/display.py
  - src/wica/contrib/gradio/world_state.py
  - src/wica/contrib/gradio/reactions.py
  - src/wica/contrib/gradio/transcript.py
  - src/wica/__init__.py
  - pyproject.toml
tests:
  - tests/contrib/test_gradio_transcript.py
  - tests/contrib/test_gradio_reactions.py
  - tests/contrib/test_gradio_world_state.py
---

# Gradio contrib components

**Status:** Implemented

## Purpose

Reusable Gradio components for the three **framework-observability surfaces** an application
building a Gradio UI on WICA wants, so it gets them by importing a component, not by rebuilding
the wiring:

1. a **live World-state view** — what the World holds right now;
2. a **reaction history** — one entry per reasoning step, browsable step by step, showing both how
   that World became the step's LLM prompt and how the step performed (latency, tokens, outcome);
3. a **conversation transcript** — the reasoning loop as a chat: the entry that triggered each
   step on one side, and on the other every Command the model issued with its live execution
   state, its free text, and its explicit "no reaction".

All three are about *the framework itself* — the World, the Agent's prompts, the Agent's
instrumentation `Event`s (see [wica.md](wica.md)) — rather than any particular persona, so they
generalize across deployments. They were built and validated inside the conversation demo (see
[conversation-demo.md](conversation-demo.md)); this spec **lifts them into the package** so the
demo becomes one consumer among others, keeping only what is specific to its robot.

These are **contrib** components: opt-in, behind an install extra, and adding no dependency to
core `wica`. Core stays UI-free and provider-agnostic — Gradio is never imported by core and never
appears in `[project.dependencies]`.

## Packaging

- **Install extra `wica[gradio]`** under `[project.optional-dependencies]` in `pyproject.toml`,
  pulling `gradio>=5.0.0`. An application that wants the panels installs `wica[gradio]`; core
  `wica` never pulls Gradio. Importing the contrib package without the extra installed fails at
  import with Gradio's own `ImportError`, the same contract as the provider extras (see
  [project.md](project.md), "Provider integrations are optional extras").
- **Subpackage `src/wica/contrib/gradio/`**, in a `contrib` namespace that marks "optional,
  batteries attached, not core." One module per surface plus the shared hook:

  | Module | Holds |
  |---|---|
  | `display.py` | The `display_entry` hook contract (`EntryDisplay`, `DisplayEntry`) and the generic default rendering of a World entry / Command execution for a person, shared by the transcript and the reaction history |
  | `world_state.py` | `world_state_panel` and its pure rendering functions |
  | `reactions.py` | `ReactionLog` presenter (prompt text + `ReactionTrace`, one record per reaction), the message-to-text and trace-to-text renderers, `reaction_panel` |
  | `transcript.py` | `TranscriptLog` presenter and `conversation_panel` |
  | `__init__.py` | Re-exports the public names above |

  Every module imports `gradio` at the top (there is no Gradio-free half to reach without it),
  so the package is only imported by code that installed the extra.
- **`contrib` is not re-exported from `wica/__init__.py`.** It is reached explicitly
  (`from wica.contrib.gradio import world_state_panel`), keeping core's public API free of any
  Gradio type.
- **`gradio` joins the `dev` dependency group** so this repo's own tooling covers the contrib:
  `pyright` (which `include`s `src`) resolves the import, and the fast test tier can import the
  presenters and pure rendering functions. The `demo` group keeps it as well — the runnable demo
  now depends on the contrib. Extras serve downstream consumers, groups serve in-repo work, as for
  the providers.
- **Project map and drift guard.** The AGENTS.md project map gains a row for
  `src/wica/contrib/gradio/` pointing at this spec. `tests/test_project_map.py` only globs
  top-level `src/wica/*.py` today; the implementation extends it to subpackage modules so the
  contrib is held to the same "every module is mapped and governed" rule.

## Common shape: presenter + panel

Each surface has the same two-layer shape, taken from the demo's validated design
([conversation-demo.md](conversation-demo.md), "Composition: build, then wire, then start"):

- A **presenter** — a plain, thread-safe object that holds the surface's state and knows nothing
  about layout. It is **built over a `Wica` and subscribes to the `Wica`'s `Event`s in its
  constructor** (`TranscriptLog(wica)`, `ReactionLog(wica)`), so the subscription is a consequence
  of construction and the application never wires callbacks by hand. This is the order the
  framework's output-wiring setters exist for: build the `Wica`, build the presenters over it,
  hand their callables back (`wica.set_output_sink(transcript.output_sink)`), then `start()`.
  `wica=None` is allowed and means *explore-only*: nothing subscribes and the presenter renders an
  empty surface — the demo's no-key mode. The World-state view needs no presenter: it reads the
  `World` directly.
- A **panel** — a function called inside a `gr.Blocks` context that creates the surface's Gradio
  component(s), owns the `gr.Timer` that keeps it live, and returns the component(s) so the
  application can place or reference them. An application drops one call into its layout and gets
  a self-refreshing panel. Each panel owns its own timer so the surfaces never share one
  round-trip (the demo learned that driving the World table from the transcript's tick made new
  entries appear only as fast as the growing chat payload could travel).

**Threading contract.** Presenter handlers run on the agent loop (Events and World listeners are
dispatched there — [wica.md](wica.md), "The event loop"); the panel's timer reads on Gradio's
thread through the presenter's `snapshot()`-style methods. Everything shared sits behind the
presenter's lock or a thread-safe queue, and reads return copies, so the UI never observes a
half-updated state.

The panels are **building blocks, not a page**: the application lays out its own `gr.Blocks` and
places `world_state_panel(world)`, `reaction_panel(reactions)` and `conversation_panel(transcript)`
where it wants them. The contrib does not dictate page structure. Signatures below pin the
public surface; parameter names may still be tuned by the implementation plan.

## The `display_entry` hook (shared)

The one place an application injects persona knowledge into these generic surfaces. Every
transcript item and every reaction-history label that comes from a World entry is rendered through
it:

```python
@dataclass(frozen=True)
class EntryDisplay:
    label: str            # one line with its icon: the input-side message, or a Command item's title
    detail: str | None    # a Command item's body (ignored for the input side)

DisplayEntry = Callable[[WorldEntry], EntryDisplay | None]
```

Returning `None` falls back to the **generic default**: `⚡ key = value` for an ordinary entry;
for a Command execution (an entry whose value is a `CommandExecution`) `🦾 name` as label with
`name(args)` as detail, or `🗣️ name` when the Command is the `Wica`'s configured output Command
(read **live** from `wica.agent.output_command_name`, so it is right whether the presenter or
`set_output_command` came first). Because a Command's execution *is* a World entry, the same hook
customizes both sides of the transcript — the demo's hook turns `speech_input` into `🗣️ "hello"`
and a running `say` into `🗣️ say` with the spoken text as body. The hook lives in the application
next to its entries' `serialize_fn`s: the same per-entry knowledge, rendered for a person instead
of for the model.

## Component 1: World-state panel

```python
def world_rows(world: World) -> list[list[str]]: ...        # [key, value, updated] per entry
def world_html(world: World) -> str: ...                     # the whole table, self-styled
def world_state_panel(world: World, *, refresh_s: float = 0.2) -> gr.HTML: ...
```

- Reads live state directly from the public `World` API (`world.get_prompt_entries()`, a
  deep-copied snapshot taken under the World lock) and renders every entry as a table: key, value,
  and when it last changed (`HH:MM:SS`, local time).
- **Renders fleeting entries reliably.** A short-lived Command entry (a `say` that is `running`
  only for the second it speaks) appears and is retired between frames. The panel re-renders the
  whole table as HTML (`gr.HTML`) on each refresh rather than driving a diffing `gr.Dataframe`,
  whose frontend row-diffing dropped such rows, so a row that appears and disappears between two
  ticks still shows.
- **Stable layout.** Fixed column widths (`table-layout: fixed`, wrapping long values) so a wide
  value never reflows the table; styled through Gradio theme variables
  (`--border-color-primary`, `--color-accent-soft`) so it follows the host theme.
- **Recognizes a running Command by type** — `isinstance(value, CommandExecution)` — to format it
  as `name(args) [state]` and highlight its row, rather than inspecting the key. `None` renders as
  `—`, lists as `[a, b]`, everything else via `str`. All cell text is HTML-escaped.
- The pure functions are public so an application (or a test) can render the same rows or HTML
  without a timer, e.g. into its own component.
- Stateless: it owns no framework state, only its component and timer.

## Component 2: Reaction history

**Merged with the reaction's instrumentation, not a separate fourth component.** A prompt exists
only for the instant `on_agent_prompt` fires; a reaction's `ReactionTrace`
([instrumentation.md](instrumentation.md), "Layer 1") exists only once `on_agent_reaction_ended`
fires, later, for the same reaction. Both describe the *same* reasoning step from two angles —
what the model was shown, and how the step performed — so one presenter keeps one append-only,
thread-safe record per reaction, and one panel shows both facets of whichever reaction is
selected, as two tabs under a single dropdown, rather than forcing the person to keep two
independently-scrolling histories in sync by hand:

```python
class ReactionLog:
    def __init__(self, wica: Wica | None, display_entry: DisplayEntry | None = None) -> None: ...
    # subscribes on_agent_trigger (labels), on_agent_prompt (opens a record) and
    # on_agent_reaction_ended (fills in the same record's trace) in the constructor
    def snapshot(self) -> ReactionSnapshot: ...              # count, dropdown choices — one consistent read
    def prompt_text(self, index: int | None) -> str | None: ...
    def instrumentation_text(self, index: int | None) -> str | None: ...

def render_messages(messages: list[BaseMessage]) -> str: ...
def render_reaction_trace(trace: ReactionTrace) -> str: ...
def reaction_panel(
    log: ReactionLog, *, refresh_s: float = 0.2, lines: int = 16
) -> tuple[gr.Dropdown, gr.Textbox, gr.Textbox]: ...     # selector, prompt view, instrumentation view
```

- **One record per reaction, opened early, filled in late.** `on_prompt` appends the record (label
  + rendered prompt text) the moment the step's prompt is known — reusing exactly the labelling
  rule `PromptLog` had (`HH:MM:SS — <trigger>` through the `display_entry` hook, `<label> finished`
  for a Command-completion re-trigger, `start` before any trigger). `on_reaction_ended` then finds
  that *same* record by `reaction_id` (reactions and prompts are both counted 1, 2, 3 … in lockstep
  — `on_agent_prompt` fires exactly once per reaction that starts, `ReactionTrace.reaction_id`
  counts the same reactions the same way) and attaches its `ReactionTrace`. A reaction whose trace
  hasn't arrived yet (the model is still thinking, or it's still running) reads as *pending* —
  `instrumentation_text` returns a placeholder, never `None`, so a caller never has to special-case
  "not yet" versus "out of range" (`None` is reserved for an invalid index).
- **One dropdown, two tabs.** The dropdown lists reactions exactly as the old prompt dropdown did
  (same labels, same snap-to-newest-then-leave-the-user-alone rule below); a `gr.Tabs()` underneath
  holds a "Prompt" tab (unchanged rendering) and an "Instrumentation" tab (the selected reaction's
  formatted trace, or the pending placeholder). Selecting a different reaction updates both tabs
  together, so the person is always looking at one reaction's two facets, never mismatched ones.
- **Snaps to the newest, then leaves the user alone — plus one exception.** When a new reaction
  appears the panel selects it and shows both its (known) prompt text and its (pending)
  instrumentation; between reactions the timer otherwise sends no update, so browsing an older
  reaction is undisturbed. The one addition: every tick also re-checks the **instrumentation text
  of whichever reaction is currently selected** (read back from the dropdown's own value) and, if
  it changed since the last tick — the common case being *pending → the rendered trace*, once that
  reaction ends — pushes just that update to the Instrumentation tab, leaving the dropdown and the
  Prompt tab untouched. This is not "yanking the user around": it only ever changes the content of
  the pane already on screen, for the reaction already selected, the same way the transcript's own
  reaction group loses its spinner in place once `on_agent_reaction_ended` fires. Read-only text
  boxes start with a "no reaction yet" note.
- **Renders `BaseMessage` to readable text** — unchanged from the old `PromptLog`: a `### ROLE`
  header per message; content blocks flattened by concatenation (each World-rendered block already
  carries its own newlines, so the panel shows the exact text the model receives with no phantom
  blank lines); images as `[image <mime>]`; and tool calls / tool results shown explicitly
  (`🔧 tool_call name(args) [id=…]`, `🔧 result for [id=…] => …`) so a call-only assistant message
  isn't blank. `BaseMessage` is the type `wica.on_agent_prompt` already carries, the framework's
  LangChain I/O boundary; rendering it here introduces no new dependency.
- **Renders `ReactionTrace` to readable text** — the reaction's outcome and id; the derived
  measures as seconds (`coalescing_wait`, `render_time`, `model_latency`, `sink_duration`,
  `busy_time`), each shown only when its underlying stamp exists (a cancelled reaction has no
  `model_latency`); `reaction_latency(trace)` ([instrumentation.md](instrumentation.md), "Metrics")
  when it applies (not for a reaction whose only trigger was a Command completion); token usage
  when reported; the dispatched `command_call_ids` and whether the reaction ended in `noop`; the
  `error` on a `model_error` outcome; and the OpenTelemetry `trace_id` when an SDK is installed —
  the exact identifier a person pastes into their tracing backend to see the same reaction as a
  span tree. `ReactionTrace` is the type `wica.on_agent_reaction_ended` already carries; rendering
  it here, like `render_messages`, introduces no new dependency.

## Component 3: Conversation transcript

The demo's generic transcript (see [conversation-demo.md](conversation-demo.md), "What the user
sees" 1.), moved into the package unchanged in behaviour, minus the prompt/instrumentation history
it used to carry (now `ReactionLog`):

```python
class TranscriptLog:
    def __init__(self, wica: Wica | None, display_entry: DisplayEntry | None = None) -> None: ...
    # subscribes on_agent_trigger, on_agent_prompt, on_agent_command, on_agent_reaction_ended
    async def output_sink(self, text: str) -> None: ...      # the application passes this to wica.set_output_sink
    def snapshot(self) -> TranscriptSnapshot: ...            # conversation (Gradio "messages" format) + change signature

def conversation_panel(log: TranscriptLog, *, refresh_s: float = 0.2, height: int = 420) -> gr.Chatbot: ...
```

What it renders, from the framework's uniform signals only:

- **Input side (right):** the World entry each step observed (`on_agent_trigger`), rendered by
  the hook. A Command-completion re-trigger is not shown (the Command item already carries its
  outcome) but still labels the step. A trigger dropped by the busy single-in-flight loop never
  fires the Event, so it never appears and no reply follows — faithful to what happened.
- **Assistant side (left):** each reasoning step is one **collapsible reaction group**
  (`💬 reaction N · <trigger>`, opened *pending* — a spinner — on `on_agent_prompt` and, on
  `on_agent_reaction_ended`, losing the spinner and gaining a `duration` — the reaction's
  `busy_time`, the same `metadata.duration` its Command items already use — matched by
  `reaction_id`, which counts reactions exactly as `on_agent_prompt` does), and its outputs nest
  under it in order:
  - **every Command the model issued** (`on_agent_command`), titled through the hook (`🗣️` for the
    output Command, `🦾` otherwise), **with its execution state live**: the item is created the
    moment the Command is issued, marked *pending* (a spinner), and the log listens to the
    Command's `agent:command:<call_id>` World entry — the seam `CommandIssued.call_id` exists for
    ([agent.md](agent.md), "Instrumentation") — to edit the item in place as it goes running →
    **✅ complete** (with its result) / **❌ failed** (with the error) / **⏹ cancelled**, plus how
    long it ran. A finished item **stays open**: its `status` is removed rather than set to
    `done`, because Gradio auto-collapses a `done` item and the reader could no longer see what
    was said. The listener is async so terminal states are delivered in version order after
    `running`.
  - **💭 output sink** — the model's free text for the step (blank text is skipped);
  - **🚫 noop** — the auto-registered "no reaction" Command, rendered directly since it has no
    World entry.
- **Only pushed on a real change.** The panel does *not* list the chatbot as an output of its
  timer: every event that outputs to a component flips its loading status, and `gr.Chatbot`'s
  autoscroll re-runs on that flip, yanking a reader who scrolled up back to the bottom every tick.
  Instead the timer bumps a `gr.State` version only when the snapshot's **change signature**
  (content, title and status of every item) differs from the last one, and that State's `.change`
  is what pushes the transcript. Autoscroll then follows genuinely new content and leaves reading
  alone. This is a requirement of the panel, not an implementation detail: a reimplementation
  that outputs the chatbot from the timer regresses the demo's scroll fix.

The presenter's `snapshot().conversation` is the Gradio "messages" format
(`{"role", "content", "metadata": {"title", "id"/"parent_id", "status", "duration"}}`), which is
also what the tests assert on.

## Public-API dependencies on core

- **`CommandExecution` becomes public.** The World-state panel recognizes a Command entry by type,
  the transcript reads its `state`/`result`/`error`, and — decisively — an application's
  `display_entry` hook must `isinstance` against it to customize a Command item (the demo's hook
  already does, importing it from `wica.agent`). A supported hook contract cannot depend on a
  non-exported internal, so the implementation **re-exports `CommandExecution` from
  `wica/__init__.py`** alongside `CommandIssued`, and [commands.md](commands.md) gets an editorial
  note that the value model is public. Its shape is unchanged.
- **`noop` name and the `agent:command:` key prefix.** The demo duplicated these as module
  constants "matching `wica.agent`'s". Inside the package the contrib imports `wica.agent`'s own
  constants instead of duplicating them; whether to promote them to public names is left to the
  implementation (no consumer outside the package needs them today).
- **`Wica.agent.output_command_name`** and **`Wica.world`** are the only other facade members the
  presenters touch, both already public ([wica.md](wica.md)).

## What the demo keeps

After the move, [conversation-demo.md](conversation-demo.md) governs only what is specific to the
robot: its World entries and `serialize_fn`s, the `display_entry` hook, the `Robot` and its
Commands, the **Speaking panel** (the simulated TTS, `SpeakingSlot` + its HTML), the sensor-input
widgets, the page layout, and the build → UI → wire → start composition. Its `app_ui.py` calls the
three contrib panels; `examples/conversation_demo/transcript.py` is deleted. The demo is then the
reference consumer of this contrib, and the composition it demonstrates is exactly what another
application does.

## Testing

Fast, deterministic, no-network, in the default tier — the contrib's tests are the demo's
existing presenter tests ([tests/test_conversation_demo.py](../tests/test_conversation_demo.py))
relocated and split per module, `tests/contrib/test_gradio_<module>.py` (the fast tier mirrors
the package layout — [testing.md](testing.md)). They import the contrib (hence `gradio` in the
`dev` group) but construct no `gr.Blocks`, timers or browser:

- **`TranscriptLog`** driven the way the Agent drives it: `on_trigger`/`on_prompt`/`on_command`
  calls and an `on_command_update` sequence, asserting the transcript items, reaction nesting,
  the pending → terminal edits (state suffix, result/error, duration, `status` removed), `noop`,
  the output-sink item, and that the change signature moves on an in-place edit and is stable
  otherwise. Where a log needs a `Wica` (live output Command name, per-Command World listeners),
  the tests build an **unstarted** one over the key-less `provider: "fake"` model, as today.
- **`ReactionLog`** as a plain object: record a few `on_agent_prompt`-shaped message lists after
  triggers, assert the labels (time + hook-rendered trigger, `… finished`, `start`), ordering,
  the newest-selection data the panel reads, and `prompt_text(index)` bounds — the old `PromptLog`
  coverage, unchanged in shape — plus the merge itself: `instrumentation_text(index)` reads as the
  pending placeholder before `on_agent_reaction_ended` fires for that index and as the rendered
  trace after, matched by `reaction_id`; an `on_agent_reaction_ended` for an index out of range
  (defensive; shouldn't happen given the framework's `reaction_id` invariant) is ignored rather
  than raising.
- **`render_reaction_trace`** over a hand-built `ReactionTrace`: every derived measure shown only
  when its stamp is present (a cancelled reaction with no `model_latency`), token usage, dispatched
  command ids, `noop`, `error` on a `model_error` outcome, and the `trace_id` line.
- **`render_messages`** over AI/Human/Tool messages: role headers, block flattening without extra
  newlines, image markers, explicit tool calls and tool results.
- **`world_rows`/`world_html`** over a real `World`: register ordinary entries and one whose
  value is a `CommandExecution`, and assert the running Command shows as `name(args) [state]` on a
  highlighted row, `None`/list formatting, and that special characters are escaped.
- The demo's own fast tests keep only the demo's pieces (its hook, `SpeakingSlot`, `say`), and
  the always-run scripted-fake flow `tests-e2e/test_example_flow.py` stays the end-to-end check —
  it builds the contrib presenters itself, as `build_ui` does, so it also exercises the contrib
  against the live loop.

## Open questions

Deferrals; none blocks the design above.

- **Sharing one timer.** Each panel creates its own `gr.Timer`. If three concurrent timers prove
  costly on a page, an optional `timer=` argument could let an application pass one it owns;
  wait for the need.
- **All entries vs. prompt entries.** The World-state view shows `include_in_prompt` entries only
  ("the robot's mind as the model sees it"). `World.keys()`/`get_config()` now make a full
  listing possible; an `all_entries=True` option could mark non-prompt rows. Deferred.
- **A one-call `observability_panels(world, reactions, transcript)`** laying the three out in a
  column. Building blocks first; add the wrapper only if asked.
- **Styling hooks.** Fixed, theme-variable-driven CSS for now; no public style API.
- **UI-agnostic presenters.** `TranscriptLog`/`ReactionLog` are Gradio-free in logic (their only
  Gradio coupling is the "messages" dict format / the two-textbox tab shape) but live in a package
  that imports Gradio. If a second UI toolkit ever wants them, they move to a Gradio-free
  `wica.contrib` module; not before.
- **An aggregate/rolling metrics view.** [instrumentation.md](instrumentation.md) ("Consumers")
  anticipated a distinct fourth component for this — current state (idle / coalescing / thinking /
  speaking), rolling last/median/p95 of the measures across *many* reactions. That is a different
  shape from this component (one reaction at a time, on demand) and is not addressed here; still
  open, wait for the need.

## Out of scope

- **The Speaking panel.** Word-by-word progress of the simulated TTS depends on how the demo's
  `say` is written — persona/product-specific, it stays in the demo.
- **Input widgets.** Sensor-input buttons and the speech textbox are application content.
- **Non-Gradio UIs.** This module is Gradio-specific; a Streamlit/FastAPI equivalent would be its
  own contrib module and spec.
- **Any core dependency on Gradio.** The coupling is one-directional (contrib → core) by design.
