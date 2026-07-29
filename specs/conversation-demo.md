# Conversation demo

**Status:** Stable

## Purpose

WICA's first runnable **example application**: a browser conversation UI that lets a person talk to
a simulated social robot and *watch the framework work*. It exists to make the four pillars
(World / Inputs / Commands / Agent) tangible and pokeable without writing code — and, secondarily,
as a manual smoke-test harness that drives the library through its public API only.

This spec describes the **product** — what the user sees and does, and how the simulated robot
behaves. The wiring (Gradio, thread bridging, the `on_prompt` hook) is an implementation concern,
covered by the plan, not here.

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
   detected. The robot's **spoken replies and its command calls** show on the **assistant (left)
   side**, in the order they happen, so a turn reads as "input → the robot says X → the robot does
   Y". An input that's *dropped* by the single-in-flight loop (robot busy) doesn't appear and gets
   no reply — faithful to what actually happened.
2. **World state.** A live view of the current World — every entry the demo tracks, shown as raw
   values (key, version id, value, when it last changed). This is the robot's whole mind laid bare:
   what it heard, who's nearby, how it feels, who it's tracking, and any command currently running.
   It updates in real time as inputs arrive and the robot acts.
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
- **Speech out.** The robot's reply is one complete utterance per turn. (No token streaming in v1.)
- **Acting.** Between hearing and replying, the robot may perform robot actions (Commands, below).
  Their effects show up in the World state view, and a longer action is visible while it runs.

### Single-track attention (a visible v1 trait)

The robot handles **one thought at a time**: while it's thinking, replying, or in the middle of a
long action, a new input that arrives is **dropped rather than queued**. This is a real v1 Agent
limitation, and the demo makes it observable on purpose — e.g. talking to the robot mid-dance does
nothing. The UI should hint at this so it reads as designed behaviour, not a bug. (A later Agent
version adds concurrent, interruptible attention; this demo is not that.)

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
| **Dance** | Performs a ~10-second dance. | Long-running: visibly "in progress" in the World state for its whole duration, and (per single-track attention) the robot ignores new input until it finishes. Best illustration of a long physical action. |
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
  `WicaConfig.from_json` followed by `Agent.from_config` (see [config.md](config.md)) — provider,
  model, and persona (`system_prompt_file`, pointing at `examples/prompts/wica.md`) all live
  there. Switching providers/models, or editing the persona, is a file edit, not a code change.
- **The API key is referenced, not stored.** The committed config uses
  `"api_key_env": "WICA_ANTHROPIC_API_KEY"` — the WICA-namespaced env var (so it never collides
  with a provider key another tool in the environment already uses) that `api_key_env` reads at
  load time. No secret lives in the committed file. A literal key (e.g. to try a different
  provider without exporting an env var) goes in a local `*.local.json` copy instead, which is
  git-ignored (see config.md, "API key: literal or env reference").
- **No key, still usable.** With `WICA_ANTHROPIC_API_KEY` unset, loading the config raises
  `MissingEnvError`; the demo catches it, still opens, and clearly says what to set — it just can't
  run the robot's reasoning until the variable is present. Nothing contacts a model without
  credentials.

## Out of scope (for this demo)

- Real audio/vision — inputs are simulated via UI, not a mic/camera.
- Concurrent or interruptible reasoning, streamed speech, barge-in — all post-v1 Agent features.
- Persistence across restarts — the World starts empty each run.
