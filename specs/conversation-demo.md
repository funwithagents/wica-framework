---
code:
  - examples/conversation_demo.py
  - examples/prompts/wica.md
  - examples/agent.config.json
  - examples/agent.anthropic.config.json
  - examples/agent.openai.config.json
  - examples/agent.huggingface-hub.config.json
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
to) is an implementation concern, covered by the plan, not here.

Scope is deliberately small: it demonstrates the **v1 Agent** (single reasoning call in flight at a
time; one complete spoken reply per step). It is not meant to show off concurrency/interruption or
streaming — those aren't in v1.

## The persona

A friendly **social robot** that can talk, express an emotion, do a little dance, and keep track of
people around it. Its persona and behaviour come from the Agent's configurable system prompt, so a
deployment can change who the robot *is* without touching the demo.

## What the user sees

Four surfaces, side by side:

1. **Conversation.** A chat transcript that doubles as a trace of the reasoning loop. The user
   types an utterance (their "speech"); more broadly, whatever World entry triggers a reasoning
   step shows on the **input (right) side** — a typed utterance or a sensor event like a user being
   detected. On the **assistant (left) side**, each **reasoning step (a "reaction") is one
   collapsible group** (`💬 reaction N`, labelled by the trigger that caused it), and inside that
   group its outputs appear in order, each **labelled by the framework channel it came from** — so
   the demo makes the **output Command** feature concrete (see [agent.md](agent.md), "Output"):
   - **🗣️ say** — the robot's spoken reply, delivered through the `say` **output Command** (what the
     person actually hears), not free text;
   - **💭 output sink** — the model's free text for the step, which an output Command turns into
     private reasoning (delivered to the `output_sink`), shown apart from the voice;
   - **🦾** command calls — its other actions like `dance`, italicised;
   - **🚫 noop** — the robot explicitly *choosing not to react* (the auto-registered `noop`
     Command); it commonly ends a turn, since speaking through the output Command re-triggers the
     agent and the model then declines to add more.

   So a turn reads as "input → (the robot thinks…) → the robot says X → the robot does Y". An input
   that's *dropped* because a reasoning call is already in flight doesn't appear and gets no reply —
   faithful to what actually happened.
2. **World state.** A live view of the current World — every entry the demo tracks, shown as raw
   values (key, value, when it last changed). This is the robot's whole mind laid bare:
   what it heard, who's nearby, how it feels, who it's tracking, and any command currently running.
   It updates in real time as inputs arrive and the robot acts — including a short-lived command
   entry (e.g. a `say` that is "running" only while it speaks), so a fleeting action still shows.
3. **Prompt.** The exact prompt sent to the model, shown as read-only text — so the user can see
   *how* World state becomes an LLM prompt, the core idea of WICA. Every reasoning step's prompt is
   kept, not just the last one: a dropdown above the text lists them (labelled by time and the
   trigger that caused the step, e.g. `14:03:12 — 🗣️ "hello"`) so the user can scroll back through
   the history. When a new step runs, its prompt is appended and **automatically shown** — the view
   always snaps to the newest, even if the user had an older one selected. Between steps the user is
   free to browse earlier prompts.
4. **Inputs.** Buttons that inject sensor-style events into the World (below), simulating a robot's
   perception without real hardware.

## Interaction model

- **Speech in.** What the user types is treated as speech heard by the robot: it enters the World
  and prompts the robot to think and (usually) reply.
- **Sensing in.** The input buttons simulate perception events. They change World state and can
  prompt the robot to react on their own — the robot may say something in response to *who walked
  up*, not only to what was said.
- **Speech out.** The robot speaks by calling a `say` **output Command** — so its user-facing voice
  is a real, observable, cancellable Command, and the model's free text becomes private *thinking*
  shown apart from the voice (see [agent.md](agent.md), "Output"). One (or a short chain of)
  utterance(s) per turn. The demo configures `say` via `Wica.init(output_command=…)`; without an
  output Command the free text would itself be the voice. **The demo simulates speaking**: `say`
  takes time and reveals its words one by one, streaming them into the transcript while the Command
  stays `running` (and cancellable) in the World. This is a *demo* effect — a slow backing function
  updating its own UI message — **not** framework token streaming (which is post-v1; see
  [agent.md](agent.md), "Future improvements").
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
| **Say `<text>`** | Speaks to the person. | The **output Command** (`Wica.init(output_command=…)`): the robot's voice, shown in the transcript labelled **🗣️ say**. Because it is a Command, the model's own free text becomes private reasoning (the **💭 output sink**) instead of speech. |
| **Dance** | Performs a ~10-second dance. | Long-running: visibly "in progress" in the World state for its whole duration. New inputs may start reasoning while it runs, letting the model observe or cancel the action. |
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

## Configuration

- **The demo loads from a committed JSON config**, `examples/agent.config.json`, via
  `WicaConfig.from_json` followed by `Wica.init` (see [config.md](config.md)) — provider,
  model, and persona (`system_prompt_file`, pointing at `examples/prompts/wica.md`) all live
  there. Switching providers/models, or editing the persona, is a file edit, not a code change.
  Committed per-provider variants sit alongside it — `agent.anthropic.config.json`,
  `agent.openai.config.json`, `agent.huggingface-hub.config.json` — showing the same demo against
  each supported provider (see [config.md](config.md), "Providers"); copy one over
  `agent.config.json` to switch backend.
- **The API key is referenced, not stored.** The committed config uses
  `"api_key_env": "WICA_ANTHROPIC_API_KEY"` — the WICA-namespaced env var (so it never collides
  with a provider key another tool in the environment already uses) that `api_key_env` reads at
  Agent build. No secret lives in the committed file. A literal key (e.g. to try a different
  provider without exporting an env var) goes in a local `*.local.json` copy instead, which is
  git-ignored (see config.md, "API key: literal or env reference").
- **No key, still usable.** With `WICA_ANTHROPIC_API_KEY` unset, `Wica.init` raises
  `MissingEnvError` when it builds the Agent (the env read is deferred to build, not load — see
  config.md, "API key"); the demo catches it, still opens, and clearly says what to set — it just
  can't run the robot's reasoning until the variable is present. Nothing contacts a model without
  credentials.

## Out of scope (for this demo)

- Real audio/vision — inputs are simulated via UI, not a mic/camera.
- Concurrent or interruptible reasoning, streamed speech, barge-in — all post-v1 Agent features.
- Persistence across restarts — the World starts empty each run.
