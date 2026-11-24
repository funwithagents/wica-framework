---
code:
  - examples/conversation_demo/app.py
  - examples/conversation_demo/speaking.py
  - examples/conversation_demo/app_ui.py
  - examples/conversation_demo/session.py
  - examples/conversation_demo/prompts/wica.md
  - examples/conversation_demo/default_config.json
  - examples/configs/anthropic.json
  - pyproject.toml
tests:
  - tests/test_conversation_demo.py
  - tests-e2e/test_example_flow.py
---

# Conversation demo

**Status:** Implemented

## Purpose

WICA's first runnable **example application**: a browser conversation UI that lets a person talk to
a simulated social robot and *watch the framework work*. It exists to make the four pillars
(World / Inputs / Commands / Agent) tangible and pokeable without writing code — and, secondarily,
as a manual smoke-test harness that drives the library through its public API only.

This spec describes the **product** — what the user sees and does, and how the simulated robot
behaves. The wiring (Gradio, the `Wica` facade, the instrumentation `Event`s the panels subscribe
to) is an implementation concern, covered by the plan, not here. The three surfaces that show
*the framework* rather than the robot — the conversation transcript, the World-state view and the
prompt history — are the package's reusable Gradio components
([gradio-contrib.md](gradio-contrib.md)); the demo is their reference consumer and specifies here
only how it uses them.

Scope is deliberately small: it demonstrates the **v1 Agent** (single reasoning call in flight at a
time; speech streamed to the robot's voice as the model writes it). It is not meant to show off
concurrent or interruptible reasoning — that isn't in v1.

## The persona

A friendly **social robot** that can talk, express an emotion, do a little dance, and keep track of
people around it. Its persona and behaviour come from the Agent's configurable system prompt, so a
deployment can change who the robot *is* without touching the demo.

## What the user sees

Five surfaces:

1. **Conversation.** A chat transcript that doubles as a trace of the reasoning loop. The user
   types an utterance (their "speech"); more broadly, whatever World entry triggers a reasoning
   step shows on the **input (right) side** — a typed utterance or a sensor event like a user being
   detected. On the **assistant (left) side**, each **reasoning step (a "reaction") is one
   collapsible group** (`💬 reaction N`, labelled by the trigger that caused it), showing a spinner
   while the robot thinks and, once it has reacted, its duration; inside that group its outputs
   appear in order, each **labelled by the framework channel it came from** — so the demo makes the
   **output Command** feature concrete (see [agent.md](agent.md), "Output: tags in the model's
   text"):
   - **🗣️ say** — the robot's spoken reply: what the model wrote inside a `<say>` tag, delivered to
     the `say` **output Command** (what the person actually hears); one item per tag, showing the
     full text the robot set out to say;
   - **💭 private text** — what the model wrote outside its `<say>` tags in the step: notes nobody
     hears, shown apart from the voice;
   - **🦾** command calls — its other actions like `dance`;
   - **🚫 noop** — the robot explicitly *choosing not to react* (the model answered `<noop/>`),
     typically to a change it was not asked about.

   **Every Command item carries its execution state, live.** A 🦾 item appears the moment the
   robot issues the Command, and a `say` item as soon as its text is complete (its first words are
   already in the Speaking panel by then), marked *in progress* (a spinner on its title); when the Command
   ends, the same item is updated in place: **✅ complete** (with its result), **❌ failed** (with
   the error) or **⏹ cancelled**, plus how long it ran. A finished item **stays open** — the
   spinner goes, the body (e.g. what was said) remains readable without expanding it. So a 10-second dance is visibly "running"
   in the transcript until it finishes, and a barge-in reads as the interrupted `🗣️ say` flipping
   to *cancelled* — after a `🦾 cancel_command` item, or simply as the robot's new `🗣️ say`
   appears, since speaking again replaces what it was saying ([agent.md](agent.md), "One voice"). This is the transcript's view of the
   `agent:command:<call_id>` lifecycle ([commands.md](commands.md), "Command execution as a World
   entry"): the World-state panel (3.) only shows a command entry while it exists, and a finished
   one is retired at the very next step, so the transcript is where the outcome stays visible.

   So a turn reads as "input → (the robot thinks…) → the robot says X → the robot does Y". An input
   that's *dropped* because a reasoning call is already in flight doesn't appear and gets no reply —
   faithful to what actually happened.

   The transcript is **generic**: nothing in it knows the robot. It is the contrib's
   conversation panel ([gradio-contrib.md](gradio-contrib.md), "Component 3"); what it renders
   comes from the framework's uniform signals (a triggering `WorldEntry`, an issued Command and
   its `CommandExecution` states, the step's prompt, private text and reaction record), and the only persona-specific
   knowledge — which icon and wording an entry gets — is injected through the contrib's one
   `display_entry` hook, which the demo implements (see "Reused from the Gradio contrib" below).
2. **Speaking (now).** A small panel directly under the conversation showing the **state of the
   current `say`**: the sentence being spoken with the words already "spoken" set apart from the
   ones still to come, a `spoken/total` word count, and the utterance's state — *speaking*,
   *complete*, or *cancelled*. It updates word by word while the robot speaks and keeps showing the
   last utterance (as complete or cancelled) until the next one starts. The words reach it as
   the model writes them, so the panel starts before the response is complete. This is the demo's
   **simulated TTS** made visible on its own, away from the transcript — the one surface that is
   specific to this robot's voice rather than to the framework.
3. **World state.** A live view of the current World — every entry the demo tracks, shown as raw
   values (key, value, when it last changed). This is the robot's whole mind laid bare:
   what it heard, who's nearby, how it feels, who it's tracking, and any command currently running.
   It updates in real time as inputs arrive and the robot acts — including a short-lived command
   entry (e.g. a `say` that is "running" only while it speaks), so a fleeting action still shows.
   This is the contrib's World-state panel over the demo's World
   ([gradio-contrib.md](gradio-contrib.md), "Component 1").
4. **Prompt.** The exact prompt sent to the model, shown as read-only text — so the user can see
   *how* World state becomes an LLM prompt, the core idea of WICA. Every reasoning step's prompt is
   kept, not just the last one: a dropdown above the text lists them (labelled by time and the
   trigger that caused the step, e.g. `14:03:12 — 🗣️ "hello"`) so the user can scroll back through
   the history. When a new step runs, its prompt is appended and **automatically shown** — the view
   always snaps to the newest, even if the user had an older one selected. Between steps the user is
   free to browse earlier prompts. This is the contrib's prompt-history panel
   ([gradio-contrib.md](gradio-contrib.md), "Component 2"), labelling triggers through the same
   `display_entry` hook as the transcript.
5. **Inputs.** Buttons that inject sensor-style events into the World (below), simulating a robot's
   perception without real hardware.

## Interaction model

- **Speech in.** What the user types is treated as speech heard by the robot: it enters the World
  and prompts the robot to think and (usually) reply.
- **Sensing in.** The input buttons simulate perception events. They change World state and can
  prompt the robot to react on their own — the robot may say something in response to *who walked
  up*, not only to what was said.
- **Speech out.** The robot speaks by writing a `<say>` tag in its response; the tag's text is
  delivered to the `say` **output Command** — so its user-facing voice is a real, observable,
  cancellable Command execution, and whatever the model writes outside the tag is private, shown
  apart from the voice (see [agent.md](agent.md), "Output: tags in the model's text"). Usually one
  utterance per turn; a second one replaces the first if it is still being spoken. The demo
  configures `say` via `wica.set_output_command(say)`; without an output Command nobody would hear
  the robot. **The demo simulates speaking**: `say` takes the text as an `OutputStream`, so it
  starts on the first words the model writes, and reveals them one by one in the **Speaking panel**
  at a speaking pace while the Command stays `running` (and cancellable) in the World; the
  transcript shows the `say` item with its full text and its live state (in progress →
  complete/cancelled), like any other Command. Cut off part-way, `say` reports the words it got to
  (`stream.heard`), so the robot's history keeps what was heard and not the whole text. The
  streaming is the framework's; the word-by-word pace is the *demo's* simulated TTS — a slow
  backing function updating its own panel.
- **Acting.** The robot may perform robot actions (Commands, below) alongside a spoken reply.
  Their effects show up in the World state view, and a longer action remains visible while it runs.

### One reasoning call at a time (a visible v1 trait)

The robot handles **one LLM reasoning call at a time**. An input that arrives while the model is
currently thinking is **dropped rather than queued**. This is a real v1 Agent limitation, and the
demo makes it observable on purpose so it reads as designed behaviour, not a bug.

A long-running Command is different: the reasoning step ends once the Command is dispatched, so
the Command may continue in the background while a later input starts a new reasoning step. For
example, talking to the robot mid-dance does reach the Agent; the next prompt shows the dance as
still running, and the model may continue talking or issue `cancel_command` for it. The demo thus
makes the distinction between a busy reasoning call and an in-flight physical action visible.

## Sensor inputs (the buttons)

| Input | Effect on the World | Prompts the robot? |
|---|---|---|
| **Closest user detected `<id>`** | Records `<id>` as the nearest person. | Yes |
| **Closest user gone** | Clears the nearest-person state (no one nearby). | Yes |

"Closest user" models a robot noticing who is directly in front of it. Setting it, changing it to a
different id, or clearing it are all perception events the robot can react to.

## Robot actions (Commands the robot can choose)

The robot decides when to use these; the user does not press them — they're how the robot *acts*,
and the point is to watch the model choose them in context.

| Action | What it does | Notable |
|---|---|---|
| **Say `<text>`** | Speaks to the person. | The **output Command** (`wica.set_output_command(say)`), which the model issues as a `<say>` tag in its text, not as a tool call: the robot's voice, shown in the transcript labelled **🗣️ say** (full text, live state) and word by word in the Speaking panel. Text the model writes outside the tag is private (the **💭 private text** item). |
| **Dance** | Performs a ~10-second dance. | Long-running: visibly "in progress" in the transcript and the World state for its whole duration. New inputs may start reasoning while it runs, letting the model observe or cancel the action. |
| **Set emotion `<emotion>`** | Sets the robot's current emotional state. | Reflected in World state and in the robot's subsequent prompt/behaviour. |
| **Switch tracking to user `<id>` (or nobody)** | Follows one specific person, or stops tracking when called with no user. | The robot follows **at most one** person at a time — a single `tracked_user` entry, not a set. Passing no user (null) clears it. |

Tracking models the robot deciding to follow a specific person over time — distinct from the
transient "closest user" perception input. Emotion and tracking persist in the World until changed;
"dance" is a momentary action.

The robot is instructed (via its system prompt) to **only follow the person currently closest to
it**: when the closest person changes it switches tracking to them, and when no one is close it
stops tracking (switches to nobody). This is LLM-driven, not a hard rule — it demonstrates the
robot reasoning from the `closest_user` perception (which triggers a step when it changes, including
when it clears to "no one") and issuing the `switch_user_tracking` command in response, rather than
the demo wiring the effect deterministically behind the agent's back.

## Reused from the Gradio contrib

The conversation, World-state and reaction-history surfaces are not the demo's code: they are the
three components of `wica.contrib.gradio` ([gradio-contrib.md](gradio-contrib.md)) —
`TranscriptLog` + `conversation_panel`, `world_state_panel`, `ReactionLog` + `reaction_panel` — and
the demo uses them exactly as any application would. What the demo contributes is the persona side
of the seam:

- **The `display_entry(entry: WorldEntry)` hook.** The contrib renders every transcript item and
  prompt label that comes from a World entry through this function, which returns an
  `EntryDisplay(label, detail)` — `label` is the one-line text with its icon (the input side's
  message, or a Command item's title), `detail` the Command item's body — or `None` to fall back
  to the contrib's generic default (`⚡ key = value` for an entry; `🦾 name` / `name(args)` for a
  Command execution, whose value is a `CommandExecution`). Because a Command's execution *is* a
  World entry, the same hook customises both sides: the demo's hook turns `speech_input` into
  `🗣️ "hello"`, `closest_user` into `👤 Closest user detected: alice`, and a running `say` into
  `🗣️ say` with the spoken text as body. The hook lives in the demo next to the entries'
  `serialize_fn`s — the same per-entry knowledge, rendered for a person instead of for the model.
  The demo passes the one hook to both presenters (`TranscriptLog(wica, display_entry)`,
  `ReactionLog(wica, display_entry)`) so the transcript and the reaction-history labels agree.
- **The Speaking panel is the demo's, not the transcript's.** `say` reports its word-by-word
  progress to the Speaking slot the UI owns (handed to the robot at wiring — see "Composition"
  below); the transcript only ever sees `say` as a Command with a state, like the others. The
  slot (`examples/conversation_demo/speaking.py`) and its HTML rendering stay in the demo.
- **What the contrib renders by itself** — a declined reaction (`🚫 noop`, from the reaction
  record, never through the hook since no entry exists), the step's private text (`💭`) and the
  configured output Command (labelled 🗣️ by default, read live from the Agent, so it is correct
  whichever is wired first) — needs nothing from the demo.

## The app: a folder, a Setup tab, a session

The demo is an app built on the `app-setup` project's Setup and Secrets panels, which own the
app's folder, the config form, the checks and the keys; this spec says what the demo puts around
them. `app-setup[ui]` is in the `demo` dependency group (and the `dev` group, whose fast tier
imports the demo), never a dependency of the published package.

```
uv run --group demo python -m examples.conversation_demo.app [--folder PATH] [--host ADDRESS]
    [--port N] [--start | --no-start]
```

- **One section, `wica`**, whose document is the WICA block, described by `wica.setup`
  ([setup.md](setup.md)), titled "Agent". The host is `wica` itself, so the Setup panel installs
  this project's own extras.
- **The app's folder** holds its `config.json`, its `.env` (the keys the Secrets panel saves) and
  its `launch.json`: `~/.config/wica-conversation-demo` unless `--folder` names another. The page
  listens on `127.0.0.1` unless `--host` names another address; `--port` picks the port.
- **The default config** is a committed file, `examples/conversation_demo/default_config.json`,
  loaded into `App(default_config=...)` (app-setup writes it into an empty folder at the first
  launch): provider `anthropic`, model `claude-haiku-4-5`, `"system_prompt_file": "wica.md"`, and
  no key nor `api_key_env` — the key is read from `ANTHROPIC_API_KEY`, the provider's default,
  from the shell or the folder's `.env`. The scripted fake stays an example config
  (`examples/configs/fake.json`), to paste in the Setup tab for a key-less try.
- **The persona lives in the folder.** At launch, when the app's folder has no `wica.md`, the
  bundled `examples/conversation_demo/prompts/wica.md` is copied there; it is never overwritten,
  so the user edits the robot's persona in the folder. The default config's relative `wica.md`
  resolves against the folder: the session builds with it as `base_dir`, and app-setup passes
  the same folder to the setup checks (`base_dir`, [setup.md](setup.md)), so Check and Start look
  at the same file. The app never changes its working directory.
- **The page**: the demo's **Conversation** tab first, then the **Setup** panel, then the
  **Secrets** panel following the Setup panel's draft. It opens on the Conversation tab when the
  session starts at launch (`starts_at_launch`, whether it actually started or showed its
  blockers), else on Setup.
- **A session with a lock.** A session is one `Wica` built from the saved config and running
  (`examples/conversation_demo/session.py`, `Conversation`). **Start** checks the saved config
  (`AppModel.blockers`: its problems, then what the machine is missing — the extra, the key, the
  prompt file) and lists them on the page instead of starting; otherwise it builds the session
  from `model.used_config(saved)["wica"]` with the folder as `base_dir`. A build that still fails
  is said on the page and leaves nothing running. **Stop** stops and closes the Wica. The config is
  locked from Start until the close has finished (`AppModel(..., locked=conversation.locked)`), so
  the Setup panel disables Save and Reset while a session runs. One session at a time: Start while
  one runs does nothing. Leaving the page's process stops it.
- **Autostart.** The demo starts its session at launch (`App(autostart=True)`); the user's choice
  is app-setup's, kept in the folder's `launch.json` and drawn on the Setup panel as a checkbox.
  `--start` / `--no-start` override it for one launch. A launch whose saved config has blockers
  shows them in the Conversation tab instead of starting. **The first launch with no key does
  exactly that, by design**: the default config needs `ANTHROPIC_API_KEY`, so the Conversation
  tab lists the missing key; the user sets it in the Secrets tab (or the shell) and presses Start. `config.json` holds only the `wica`
  section.

## The Conversation tab

A status line, **Start** and **Stop**, the reading guide, and — while a session runs — the five
surfaces above over that session.

- **The status line** says what the session is doing: running (with its provider and model),
  starting, stopping, or stopped — with the reasons the last Start did not start (its blockers,
  or the build's error) until a Start succeeds.
- **The buttons are greyed by that status**: Start can be pressed only while stopped, Stop only
  while running; both are greyed while starting or stopping, and greyed at once on a click,
  before the start or stop runs.
- Both are refreshed on every change and polled every second, so a start or a stop from another
  browser tab, or at launch, shows.
- Each session is a new `Wica`, so its surfaces are drawn anew (a `gr.render` keyed on the
  session's version): a start or a stop redraws them. With no session, the tab says so; there is
  no explore-only mode.

## Composition of a session: build, then wire, then start

A session is composed in one place — `open_session(config)` in the app — in a fixed order that the
framework's output wiring exists to allow (see [wica.md](wica.md), "Output wiring is delegated"):

1. **Build the Wica** from the config (`build_system`), and register the demo's World entries on
   its World. A missing key, a missing extra or an unreadable prompt raises here.
2. **Build the presenters on top of it**: the contrib's `TranscriptLog` and `ReactionLog` (each
   subscribed to the Wica's `Event`s in its constructor, both given the demo's `display_entry`
   hook) and the Speaking slot. The page places the three contrib panels plus its own Speaking
   panel and sensor inputs over them.
3. **Wire the robot.** The robot's Commands are built over the Wica's World and the Speaking slot
   (`say` drives it). The app sets `say` as the output Command on the Wica and registers the other
   Commands: `wire(wica, speaking)`. The transcript needs no wiring — it subscribed to the Wica's
   `Event`s when it was built. This step is Gradio-free and takes the slot as its argument, so the
   tests and the benchmark run the same wiring against a slot of their own.
4. **Start** the Wica. A failure after step 1 closes it again.

Consequences: no session state exists before its Wica does, so nothing is bound late — no module
globals, no two-phase attach — and everything is wired before `start()`, the setters' intended
zone.

## Configuration

- **The config is the app's**, edited in the Setup tab (or by hand in the folder's `config.json`)
  and read at each Start: provider, model and persona all live there. The example configs in
  `examples/configs/` — one per provider, the persona by a path relative to them — show the same
  demo against each supported provider (see [config.md](config.md), "Providers"); their `agent`
  block can be pasted into the Setup tab's raw view.
- **The API key is referenced, not stored.** A config names no key: it is read from the
  provider's usual variable (`ANTHROPIC_API_KEY`, …) or the one `api_key_env` names, from the
  shell or from the folder's `.env`, which the Secrets tab writes (the shell wins). No secret lives
  in a config (see config.md, "API key: never in the config").
- **No key, no start.** A missing key is one of Start's blockers, said on the page; the Setup
  tab's Test makes one free call to the provider to check the key and the model
  ([setup.md](setup.md), "probe"). Nothing contacts a model without credentials.
- **`WICA_DEMO_TRACES=console`** installs an OpenTelemetry SDK `TracerProvider` with a
  `ConsoleSpanExporter` before the first session, printing every span (`wica.world.update`,
  `wica.agent.reaction`/`model`/`guards`/`command`) to stdout — an example of the exporter wiring
  [instrumentation.md](instrumentation.md) ("Consumers") leaves to the application. Unset or any
  other value: no SDK import, no change.

## Testing

`tests/test_conversation_demo.py` (fast tier, no page built): the committed default file is the
app's default and is written into an empty folder; the persona is copied into the folder once
and never overwritten; with no key, the default config's Start lists the missing key and starts
nothing; Start runs the saved config and locks it until Stop, which closes the Wica; the buttons'
state per status (stopped, starting, running, stopping) and the status line's; a failed build is
said and leaves the config unlocked; a relative prompt file is read from the app's folder; the
command line's `--start` / `--no-start`. `tests-e2e/test_example_flow.py`: a session's wiring
over the scripted fake, end to end.

## Benchmark

The demo is also the framework's **reference benchmark** ([benchmark.md](benchmark.md)): `benchmarks/bench.py` stands it up as a session does minus the presenters (`build_system` → `wire` over a `SpeakingSlot` of its own → `start`), once per run, for each entry of `benchmarks/matrix.json` (the per-provider models of `examples/configs/` as its `default` rows, persona included) against the session workload in `benchmarks/scenarios/` — `conversation`, written to the `speech_input` + `say` shape so it runs unchanged against any application of that shape, and `commands`, the demo's own actions. Nothing in the demo changes for it.

## Out of scope (for this demo)

- Real audio/vision — inputs are simulated via UI, not a mic/camera.
- Concurrent or interruptible reasoning — a post-v1 Agent feature. (Cutting off the robot's speech or its dance is in: that cancels a Command, not a reasoning call.)
- Persistence across sessions — the World starts empty at each Start.
