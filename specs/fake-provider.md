---
code:
  - src/wica/fake_model.py
  - src/wica/agent.py
tests:
  - tests/test_fake_model.py
  - tests-e2e/test_fake_flows.py
---

# Fake provider (deterministic fake model)

**Status:** Implemented

## Purpose

A `provider: "fake"` that builds a **deterministic, network-free, key-less** chat model, selected the same way as any other provider (a config edit — see [config.md](config.md)). It exists to write **scripted whole-flow tests** of the Agent's full step loop — observe → render prompt → model call → `tool_calls` → dispatch — where the model's every step is canned, so a flow can be asserted exactly. These tests live in the `tests-e2e/` **full-loop tier** (see [testing.md](testing.md)) as an **always-run** set, a deterministic complement to the parametrized live-provider tests in that same tier.

## Motivation

Today the Agent's reasoning loop ([agent.md](agent.md)) can only be exercised end-to-end against a **live** provider, which is both non-deterministic (you can't assert that a run took *exactly* these steps and issued *exactly* these Commands) and network+key gated. So there is no way to write a **deterministic whole-flow test** — "given these inputs, the Agent walks this precise sequence of steps and Commands" — the very assertions a flow test exists to make.

Putting a scripted fake **behind a provider value** fixes this at the framework boundary: the exact same `WicaConfig → AgentConfig → build_chat_model → Agent` path production ships is exercised, just with `provider: "fake"` and a canned script, so a flow test drives the real loop over a model whose output it fully controls. No downstream app needs test-only construction code — and this generalizes beyond WICA's own suite: a **project that integrates WICA** can select the same `provider: "fake"` in its own e2e configs to script deterministic flows of its WICA-built agent (see "Public module" below).

## Settled

### Selection & construction

- [build_chat_model](../src/wica/agent.py) recognizes `provider: "fake"` and constructs the fake **directly** — the same pattern as the existing `huggingface-hub` branch — *before* the `init_chat_model` fallthrough. No integration package/extra, no network import.
- **No API key required.** `api_key`/`api_key_env` may be omitted; config loading already resolves to `api_key: None` when neither is present (see [config.md](config.md), "API key"), so config parsing needs no change. Any key that *is* resolved is simply ignored.
- `model` is a free-text label (e.g. `"scripted"`), never used to reach a backend.
- The fake class lives in a small **public** module, `src/wica/fake_model.py` (imported as `from wica.fake_model import FakeChatModel`), so a flow test can also construct it directly (bypassing config) when convenient — the same construction seam `Agent(model=…)` already supports.

### Public module — usable by integrating projects

The fake ships as a **public** `src/wica/fake_model.py` — not a private `_fake_model.py` — but is **not** re-exported into the top-level `wica` namespace. That namespace is WICA's *runtime* API (`World`/`Agent`/`Content`/config); the fake is a **test double**, so it stays reachable at its own submodule path (`from wica.fake_model import FakeChatModel`), where the location itself signals "test tooling." This is still a first-class use case, not just an internal convenience: a **project that integrates WICA** can select `provider: "fake"` in its own config (which imports nothing) — or import the class directly and pass it to `Agent(model=…)` — to write **its own deterministic e2e tests** of flows built on WICA, with no live provider, no key, and no bespoke test-only construction code. Adding a top-level `src/wica/` module means the **Project map in AGENTS.md** and its drift-guard test (`tests/test_project_map.py`) get a row in the same change that builds it (independent of the re-export decision).

### Placement: the always-run flow set in `tests-e2e/`

The scripted-flow tests belong in `tests-e2e/`, not `tests/` — they exercise the **whole loop** (trigger coalescing, async Command dispatch, history rendering), the same integration surface the live-provider e2e tests cover, just deterministically. This broadens the tier's framing: `tests-e2e/` is the **full-loop tier**, of which the **live-provider** tests are one kind and the **scripted-fake** flows another.

- **Outside the `PROVIDER_CONFIGS` parametrization.** The live tests are parametrized over one config per provider and **skip** when their key env var is unset (see [testing.md](testing.md), "Live/e2e tests"). The fake flows are neither parametrized over providers nor keyed, so they sit apart from that machinery.
- **Always run.** Being network-free and key-less, a fake flow test **never skips** — it runs on every `uv run pytest tests-e2e` regardless of which provider keys are present. It is the one part of the e2e tier that gives deterministic signal with no credentials.
- **Still excluded from the default `tests/` run.** `tests-e2e/` remains uncollected by a bare `uv run pytest` (see [testing.md](testing.md)); the fake flows are full-loop tests, so they stay opt-in with the rest of the tier — this is not a route back into the fast tier.
- On implementation, [testing.md](testing.md) and AGENTS.md's e2e description are updated to reflect this broadened tier (live-provider *and* always-run scripted-fake), in the same change.

### Behavior: a scripted response queue

The fake is **scripted via `model_kwargs`** — JSON-expressible, so it flows through the ordinary config path with no special casing:

- `model_kwargs.script`: an ordered list of step responses. Each `ainvoke(messages)` **consumes the next one** and returns the corresponding `AIMessage`.
- Each step: `{"text": str?, "tool_calls": [{"name": str, "args": object}]?}` — text, tool calls, or both. Each tool call becomes a **native LangChain tool call** (`AIMessage.tool_calls`), with a stable generated `id` when none is supplied, so the Agent dispatches it exactly like a real model's call (Commands *are* native tool calls — see [commands.md](commands.md)).
- **Exhaustion:** once the script is spent, return `model_kwargs.default` (default: empty text, no tool calls) so an extra reasoning step never crashes a test. Optional `model_kwargs.loop: true` cycles the script instead.

### `bind_tools` & name checking

- `bind_tools(tools)` is accepted and returns a bound runnable (the same fake, tools recorded). It has no effect on the canned output — but it **validates** that every scripted `tool_calls[].name` is among the bound tool names and raises a clear error otherwise, catching a typo'd Command name in a flow test instead of silently issuing an unknown tool call.

### Introspection for assertions

- The fake records the messages it received per call (e.g. `.calls: list[list[BaseMessage]]`), so a flow test can assert what the prompt looked like at each step (World rendering, prior tool results, …). Purely additive.

### Two test homes: fast unit tests + e2e flows

The fake is exercised at two levels, mirroring the rest of the repo (a module's mechanics are unit-tested in `tests/`; its behavior wired into the whole loop is an e2e concern):

- **`tests/test_fake_model.py` (fast tier)** — unit tests of `FakeChatModel` *in isolation*, driving the model directly with no Agent and no network: script consumption order, text/tool-call mapping and deterministic `id`s, exhaustion → `default`, `loop` cycling, `delay_s`/cancellation, and the `bind_tools` name-check. These pin the component's mechanics deterministically and run in the default `uv run pytest`.
- **`tests-e2e/test_fake_flows.py` (full-loop tier)** — the scripted whole-flow tests that drive the fake *through the Agent loop* and the config path (see "Placement in the e2e tier" above). Always-run, no key.

### Async & cancellation

- `ainvoke` is a genuine coroutine (no blocking call), so it runs on the shared loop like any provider and doesn't interfere with `task.cancel()` (see [agent.md](agent.md), "The shared event loop").
- **Simulated latency, on by default.** Each `ainvoke` `await`s `asyncio.sleep(model_kwargs.delay_s)` before returning its scripted step. `delay_s` **defaults to `0.2` (200 ms)** and is overridable (set `0` for instant responses). A non-zero default is deliberate: an instant fake collapses the loop's timing to zero and masks exactly the timing-dependent behavior a flow test needs to exercise — the coalescing window (also 200 ms by default — see [agent.md](agent.md), "Trigger coalescing"), barge-in, and mid-step cancellation. Because the delay is a real `await`, a `task.cancel()` landing during it unwinds cleanly, giving cancellation a realistic point to fire.

### Example config

The command names below are illustrative (they match the `walk_to`-style examples used elsewhere in the specs); the fake never interprets them beyond the `bind_tools` name check.

```json
{
  "agent": {
    "provider": "fake",
    "model": "scripted",
    "system_prompt": "You are a test double.",
    "model_kwargs": {
      "delay_s": 0.2,
      "script": [
        {"text": "Hello! How can I help?"},
        {"tool_calls": [{"name": "set_expression", "args": {"expression": "happy"}}]},
        {"text": "On my way.",
         "tool_calls": [
           {"name": "walk_to", "args": {"place": "kitchen"}}
         ]}
      ],
      "default": {"text": ""}
    }
  }
}
```

## Open questions

1. **Programmable (reactive) fake — a convenience, not a requirement.** A fixed script already covers **whole rounds** when the flow test controls the trigger sequence: the test sends the utterances and the remaining steps are woken by command completions in a known order, so a per-step script aligns 1:1 with it. A reactive fake (a Python callable `messages -> response`) only matters for flows that *don't* control that sequence. Callables can't live in JSON config, so this would need either passing a constructed model instance to `Agent(model=…)` directly (already supported) or a small in-process registry keyed by `model_kwargs.script_id`. Deferred until a canned script proves insufficient.
2. **Alignment aids.** Because a step-per-`ainvoke` script misaligns if triggers coalesce or a trigger is dropped while a step is in flight (see [agent.md](agent.md), "Trigger coalescing" and "Future improvements"), a flow test must poll-then-act. WICA could optionally help — `coalesce_window` is already settable on `Agent`, and the `.calls` introspection above lets a test assert alignment — but no new mechanism is required.
3. **Docs to update on implementation.** Deferred to the implementation change so the docs keep describing only what's built:
   - [config.md](config.md) — add a `fake` row to the Providers table (extra: "none — built in") and note it's for testing only.
   - [INTEGRATING.md](../INTEGRATING.md) — the consumer entry point, so an integrating developer discovers the fake for their own deterministic e2e tests (see "Public module" above): a short "testing flows deterministically" recipe (a `provider: "fake"` config with a `script`), a `fake` mention in the Config-schema provider row, and a read-deeper row pointing here.
