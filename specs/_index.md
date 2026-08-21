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
| [project.md](project.md) | Project structure and tooling: Python version, packaging with uv, layout conventions | Implemented |
| [testing.md](testing.md) | Testing strategy: two-tier `tests/`/`tests-e2e/` split, functional-test philosophy, `reset_world` isolation, provider-parametrized live tier | Implemented |
| [config.md](config.md) | Framework config loaded from JSON: provider/model/api key/system prompt, plain-dataclass `from_json`, strict validation, `api_key`/`api_key_env` | Implemented |
| [content.md](content.md) | Neutral, provider-agnostic multimodal content model (`TextPart`/`ImagePart`/`Content`) shared across World, Agent, Inputs, Commands | Implemented |
| [events.md](events.md) | Generic, project-agnostic `Event[T]` pub/sub primitive: synchronous subscribe/unsubscribe/emit, a standalone dependency leaf | Implemented |
| [world.md](world.md) | World state registry: typed entries, register/unregister/update API, rendering to LLM-facing `Content` | Implemented |
| [inputs.md](inputs.md) | Inputs: external multimodal data entering the World, modeled as externally-fed World entries (register + `update`, `triggers_llm_call`) — a role/pattern, not new machinery | Implemented |
| [commands.md](commands.md) | Commands: WICA's unit of agent action on the World, backed by LangChain tools; execution-as-World-entry lifecycle, generic call description | Implemented |
| [agent.md](agent.md) | Agent reasoning loop: LangChain-backed inference, snapshot history, fresh/archival rendering, async cancellable Commands, output sink | Implemented |
| [conversation-demo.md](conversation-demo.md) | Conversation demo product: browser UI to talk to a simulated social robot, with live World state and prompt views; simulated sensor inputs and robot-action Commands | Implemented |
| [fake-provider.md](fake-provider.md) | Deterministic `provider: "fake"` chat model — a scripted, network-free, key-less double selected via config, for always-run scripted whole-flow tests in the `tests-e2e/` full-loop tier | Implemented |

Each spec also opens with a YAML **frontmatter** block declaring the `code:` and `tests:` files it governs — the spec → code/tests mapping the spec-drift checks use to scope what they compare. Keep it current when files move, and see [AGENTS.md](../AGENTS.md) ("Spec frontmatter") for the full convention.

### Status legend

- **Not started** — no design decisions made yet
- **Draft** — actively being brainstormed/defined, contains open questions
- **Stable** — design settled, reviewed and validated (open questions are deferrals only), **ready to implement but not necessarily implemented yet**. This is the design-review gate, before code is written.
- **Implemented** — a **Stable** spec that a `Done` plan has built: the code now exists and matches the spec (design and code in sync)
- **Updated** — an **Implemented** spec since edited in a way that needs new code, so the code no longer matches it; a new implementation plan is needed (or in progress) to catch up. Returns to **Implemented** once that plan is `Done`.
