# WICA

WICA is an agentic framework for agents that take multimodal inputs and return multimodal outputs. Its state is centered on a single **World** object — a store of the current context that can be serialized to text (an LLM-facing prompt).

The name breaks down as:

- **World** — the store of current state/context
- **Inputs** — external multimodal data entering the World
- **Commands** — the mechanism through which agents act upon the World
- **Agents** — the reasoning loop(s) that observe the World and issue Commands

## Specs

| Spec | Description | Status |
|---|---|---|
| [project.md](project.md) | Project structure and tooling: Python version, packaging with uv, layout conventions | Stable |
| [content.md](content.md) | Neutral, provider-agnostic multimodal content model (`TextPart`/`ImagePart`/`Content`) shared across World, Agent, Inputs, Commands | Stable |
| [world.md](world.md) | World state registry: typed entries, register/unregister/update API, rendering to LLM-facing `Content` | Stable |
| [commands.md](commands.md) | Commands: WICA's unit of agent action on the World, backed by LangChain tools; execution-as-World-entry lifecycle, generic call description | Draft |
| [agent.md](agent.md) | Agent reasoning loop: LangChain-backed inference, snapshot history, fresh/archival rendering, async cancellable Commands, output sink | Draft |

### Status legend

- **Not started** — no design decisions made yet
- **Draft** — actively being brainstormed/defined, contains open questions
- **Stable** — settled **and** fully reflected in the implementation (design and code are in sync)
- **Updated** — the design is settled, but the spec has been edited since it was last implemented, so the code no longer matches it; a new implementation plan is needed (or in progress) to catch up. Returns to **Stable** once that plan is Done.
