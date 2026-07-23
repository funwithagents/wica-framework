# Implementation plans

Implementation plans for WICA — each plan turns a settled part of a spec (see [specs/_index.md](../specs/_index.md)) into concrete, buildable steps. Plans are ordered by their date-time filename prefix (`YYYYMMDDHHmm_`).

## Plans

| Plan | Description | Status |
|---|---|---|
| [202607201819_world-registry-implementation.md](202607201819_world-registry-implementation.md) | World registry data model + API (`register`/`update`/`get`/listeners/triggers), `get_world()` singleton | Done |
| [202607231408_world-content-model.md](202607231408_world-content-model.md) | Neutral `Content` model (`TextPart`/`ImagePart`) and migrating World serialization/rendering from `str` to `Content`, with fresh/archival split | Done |
| [202607231754_e2e-test-framework.md](202607231754_e2e-test-framework.md) | Reusable `tests-e2e/` scaffolding: skip-without-credentials mechanics and a real-chat-model helper | Done |
| [202607231801_agent-v1-implementation.md](202607231801_agent-v1-implementation.md) | Agent v1 reasoning loop: LangChain-backed inference over World, snapshot history, event-driven async tools, output sink | Done |
| [202607232000_command-abstraction.md](202607232000_command-abstraction.md) | Command abstraction: new `commands.md` spec, reframe agent.md "Tools" → "Commands", rename code (`register_command`, `CommandExecution`, `agent:command:<id>`) — tools become the under-the-hood primitive | Done |
| [202607232308_gradio-conversation-demo.md](202607232308_gradio-conversation-demo.md) | First runnable example ([specs/conversation-demo.md](../specs/conversation-demo.md)): Gradio conversation UI over a simulated social robot, live World/prompt panels, sensor inputs + robot-action Commands; adds an `Agent.on_prompt` debug hook | Done |

## Status legend

- **Todo** — written, not yet started
- **In progress** — actively being implemented
- **Done** — implemented, verified (lint/type-check/tests pass), and merged
