---
code:
  - pyproject.toml
  - tests-e2e/support.py
  - tests-e2e/e2e.anthropic.config.json
  - tests-e2e/e2e.openai.config.json
  - tests-e2e/e2e.huggingface-hub.config.json
tests:
  - tests-e2e/test_smoke.py
  - tests-e2e/test_wica.py
  - tests-e2e/test_fake_flows.py
---

# Testing

**Status:** Implemented

## Purpose

WICA's testing strategy — the two-tier structure, what a good WICA test looks like, how deterministic full-loop flows use the fake provider, and how the live set fans out across providers. It's a **cross-cutting practice**, not a runtime concept like [World](world.md) or [Agent](agent.md): nothing here ships in the library. It exists as a spec so the decisions have one honest home that stays in sync with the setup, rather than living half in [project.md](project.md) (the tooling choices) and half in [AGENTS.md](../AGENTS.md) (the operational how-to).

This spec is the **design/strategy** view. The exact shell invocations to run each tier — including the `zsh -ic 'source ~/.zshrc …'` dance needed to see the API keys — live in [AGENTS.md](../AGENTS.md) "Testing" and are not duplicated here; this spec links to them, the same way [conversation-demo.md](conversation-demo.md) is a spec while its run details live with the example.

## Two tiers, physically separated

Tests split into two directories, and the split is structural — a directory boundary, not a marker or an opt-out flag:

| Tier / set | Directory | Network | Deterministic | Runs by default |
|---|---|---|---|---|
| Unit / integration | `tests/` | never | yes | **yes** |
| Full-loop / scripted fake | `tests-e2e/` | never | yes | **no** |
| Full-loop / live provider | `tests-e2e/` | yes | no | **no** |

- **`tests/` is the normal dev loop.** Fast, deterministic, no real network, no API key. `pyproject.toml`'s `testpaths = ["tests"]` points the default `uv run pytest` here, so this is what runs on every change and what any contributor or CI can run with zero credentials.
- **`tests-e2e/` is opt-in.** It is the **full-loop tier** — tests that exercise the Agent's whole step loop. Its *live-provider* tests call a real LLM (network, an API key, non-deterministic output, and it costs money), so the tier is deliberately *not* collected by the default run. Because `testpaths` already excludes it, no pytest marker or `--run-e2e` flag is needed: the physical separation is the whole mechanism. Run it explicitly (`uv run pytest tests-e2e`). One kind of test here is the exception to "network + non-deterministic": the **always-run scripted-fake flows** (below) need no key and never skip.

The two tiers mirror the structure of what they exercise: `tests/` mirrors the `src/wica/` module layout (`test_world.py`, `test_agent.py`, `test_config.py`, `test_content.py`, `test_fake_model.py`, plus the `test_project_map.py` drift-guard), while `tests-e2e/` is organized around whole-loop scenarios (`test_smoke.py`, `test_wica.py`, `test_fake_flows.py`) rather than modules.

## Always-run scripted-fake flows

`tests-e2e/` is the full-loop tier, of which live-provider tests are one kind and **scripted-fake** tests another. The fake set (`test_fake_flows.py`, over `provider: "fake"` — see [fake-provider.md](fake-provider.md)) drives the whole loop over a deterministic, network-free, key-less model, so it:

- **always runs** when you invoke the tier — there's no key to skip on — unlike the live tests, and
- sits **outside** `PROVIDER_CONFIGS`: it isn't parametrized over providers.

It lives in `tests-e2e/` rather than `tests/` because it's a whole-loop integration test (trigger coalescing, async Command dispatch, history rendering), not a unit test — and it stays excluded from the default `uv run pytest` (that's `testpaths = ["tests"]`) with the rest of the tier. Run just this deterministic suite, no credentials needed, with `uv run pytest tests-e2e -k fake`. The fake model's own mechanics (script consumption, exhaustion, `bind_tools` name-check) are unit-tested separately in the fast tier (`tests/test_fake_model.py`).

## What a WICA test asserts

The house style for both tiers (also stated in [AGENTS.md](../AGENTS.md)):

- **Functional, not tautological.** Exercise what a feature actually does — inputs → outputs, state changes, side effects — not that it runs or that it matches its own signature. A test that would pass against a broken implementation (asserting a constant, that an object isn't `None`, that a mock was called) isn't worth writing.
- **Drive the public API like a real caller.** Prefer registering entries and calling `update()`/`render_*` the way a consumer would over reaching into internals; assert on the observable result.
- **In the e2e tier, assert on behavior, not exact text.** Real model output varies run to run, so a live test asserts "the output sink received non-empty text" or "the registered Command was actually invoked with plausible args," never a specific string.

## Test isolation: a fresh `World` per test

There is no process-global World: each `Wica` (see [wica.md](wica.md)) owns its own `World`, so isolation is a matter of each test building its own rather than resetting a shared one. A test that exercises the reactive path constructs a `World(loop)` on an event loop it controls and drives its `start()`/`stop()` lifecycle (see [world.md](world.md), "Lifecycle") — `stop()` cancels the World's outstanding TTL timers, so no timer bleeds into the next test. The World- and Agent-focused modules each provide local loop-plus-`World` fixtures suited to their teardown needs; there is currently no shared repository-wide `conftest.py`. Tests that only touch the data model (`register`/`get`/`render_*`) need no running loop at all, since those methods are unguarded.

## Live tier: parametrized over every provider

The live e2e set is **parametrized over one committed config per provider**, not a single reference provider — so the live tests verify WICA's real provider-branching construction across the whole supported surface, not just one privileged backend.

- **One config file per provider**, checked in and named symmetrically: `tests-e2e/e2e.<provider>.config.json` for `anthropic`, `openai`, and `huggingface-hub`. They're wired together as `PROVIDER_CONFIGS` in `tests-e2e/support.py`, so **every live e2e test runs once per config**.
- **The live tests go through WICA's own entrypoint**, not a divergent `init_chat_model` call: `support.real_chat_model()` builds via `build_chat_model()` (the smoke test), and `support.real_wica()` stands up the whole system via `Wica.init(WicaConfig.from_json(...))` — the real production path, loop + World + Agent and all. This is what makes the tier meaningful: it exercises the actual per-provider construction (including `huggingface-hub`'s dedicated non-`init_chat_model` path — see [agent.md](agent.md), "Provider-agnostic model, from config") end-to-end, so a branch that builds the wrong model is caught here rather than slipping through a test that bypassed it.
- **Each config uses `api_key_env`**, so it carries no secret and is safe to commit (see [config.md](config.md), "API key"). A provider whose key env var is unset makes that config **skip itself** — never fail — via `MissingEnvError → pytest.skip` (in `support.load_agent_config`). You exercise only the providers you hold keys for; the rest skip cleanly, so a contributor with one key, or CI with none, is never broken by the others' absence.
- **Filter to one provider with `-k <provider>`.** Because the configs are named symmetrically, the provider name matches its config-filename stem, so `-k openai` runs just that one.

The committed configs and their key env vars:

| Config | Provider | Key env var |
|---|---|---|
| `tests-e2e/e2e.anthropic.config.json` | `anthropic` | `WICA_ANTHROPIC_API_KEY` |
| `tests-e2e/e2e.openai.config.json` | `openai` | `WICA_OPENAI_API_KEY` |
| `tests-e2e/e2e.huggingface-hub.config.json` | `huggingface-hub` | `WICA_HF_TOKEN` |

The provider/extra surface these configs select from is specced in [config.md](config.md) ("Providers") and packaged as install-time extras per [project.md](project.md) ("Provider integrations are optional extras").

## Tooling

- **`pytest`** is the runner; **`ruff`** lints/formats; **`pyright`** (`standard` mode) type-checks. All three are the gate after any change — lint, type check, and tests must pass before work is considered done (see [AGENTS.md](../AGENTS.md), "Verification").
- **`pyright` covers test code too:** its `include` is `src`, `tests`, and `tests-e2e`, so tests are type-checked alongside the library rather than being a blind spot.
- The concrete commands (`uv run ruff check .`, `uv run pyright`, `uv run pytest`, and the sourced-env e2e invocations) live in [AGENTS.md](../AGENTS.md).

## Open questions

1. **CI wiring.** Nothing here sets up continuous integration. The default `tests/` tier is CI-ready (deterministic, no credentials), and the e2e tier is designed to skip cleanly when keys are absent — but actually running either on a hosted runner, and scheduling the credentialed e2e tier with secrets, is unbuilt. Today all testing is a local, manual command.
2. **e2e assertion depth.** The live tests assert coarse behavior (non-empty output, a Command was invoked). Whether richer live assertions are worth their flakiness/cost — or whether a recorded-cassette approach (replaying captured provider responses in the fast tier) would cover more without the live tax — is unexplored.
3. **Provider matrix growth.** `PROVIDER_CONFIGS` lists the three providers WICA supports today. As providers are added (or the extras change), the parametrization grows with them; there's no automated check that every supported provider has a committed e2e config, so that pairing is kept honest by hand for now.
