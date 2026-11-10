---
code:
  - pyproject.toml
  - tests-e2e/support.py
  - tests-e2e/conftest.py
  - tests-e2e/e2e.anthropic.config.json
  - tests-e2e/e2e.openai.config.json
  - tests-e2e/e2e.huggingface-hub.config.json
  - tests-e2e/e2e.google-genai.config.json
  - tests-e2e/e2e.mistralai.config.json
  - tests-e2e/robot_app.py
  - tests-e2e/scenarios
tests:
  - tests-e2e/test_smoke.py
  - tests-e2e/test_wica.py
  - tests-e2e/test_robot.py
  - tests-e2e/test_fake_flows.py
  - tests-e2e/test_example_flow.py
---

# Testing

**Status:** Updated

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

**On CI** ([ci.md](ci.md)) the split carries over as jobs: the fast tier on every pull request and every supported platform, the scripted-fake flows on every pull request, the live set on pushes to `main` with the repository's keys.

The two tiers mirror the structure of what they exercise: `tests/` mirrors the `src/wica/` module layout (`test_world.py`, `test_agent.py`, `test_command.py`, `test_config.py`, `test_content.py`, `test_events.py`, `test_fake_model.py`, `test_wica.py`, plus the `test_project_map.py` drift-guard), while `tests-e2e/` is organized around whole-loop scenarios (`test_smoke.py`, `test_wica.py`, `test_robot.py`, `test_fake_flows.py`) rather than modules. The runnable **example** is tested the same way, one tier per grain: its presenter logic in the fast tier (`tests/test_conversation_demo.py`) and its whole wiring end-to-end in the full-loop tier (`tests-e2e/test_example_flow.py`, an always-run scripted-fake flow — see below). Both are governed by [conversation-demo.md](conversation-demo.md), so they sit outside the `src/wica`-module-mirroring rule; importing the `examples` package under pytest is what `pyproject.toml`'s `pythonpath = ["."]` enables.

## Always-run scripted-fake flows

`tests-e2e/` is the full-loop tier, of which live-provider tests are one kind and **scripted-fake** tests another. The fake set (`test_fake_flows.py`, over `provider: "fake"` — see [fake-provider.md](fake-provider.md)) drives the whole loop over a deterministic, network-free, key-less model, so it:

- **always runs** when you invoke the tier — there's no key to skip on — unlike the live tests, and
- sits **outside** `PROVIDER_CONFIGS`: it isn't parametrized over providers.

It lives in `tests-e2e/` rather than `tests/` because it's a whole-loop integration test (trigger coalescing, async Command dispatch, history rendering), not a unit test — and it stays excluded from the default `uv run pytest` (that's `testpaths = ["tests"]`) with the rest of the tier. Run just this deterministic suite, no credentials needed, with `uv run pytest tests-e2e -k fake`. The fake model's own mechanics (script consumption, exhaustion, `bind_tools` name-check) are unit-tested separately in the fast tier (`tests/test_fake_model.py`).

The **conversation demo's** end-to-end test (`tests-e2e/test_example_flow.py`, governed by [conversation-demo.md](conversation-demo.md)) is a second always-run scripted-fake flow of the same shape: key-less, deterministic, never skipping, and outside `PROVIDER_CONFIGS`. It differs only in what it drives — the *example's* wiring rather than the framework loop in isolation — and is selected with `-k example`. So "always-run scripted-fake" now names both — and a third, `tests-e2e/test_tester_flows.py` (`-k tester`), in which a scripted-fake *tester* `Wica` drives a scripted-fake SUT through the `wica.testing` toolkit ([wica-tester.md](wica-tester.md)) — and a fourth, `tests-e2e/test_demo_bench_fake.py`, the demo benchmark's key-less smoke through `wica.benchmark` over the matrix's opt-in `fake` entry ([benchmark.md](benchmark.md)).

## What a WICA test asserts

The house style for both tiers (also stated in [AGENTS.md](../AGENTS.md)):

- **Functional, not tautological.** Exercise what a feature actually does — inputs → outputs, state changes, side effects — not that it runs or that it matches its own signature. A test that would pass against a broken implementation (asserting a constant, that an object isn't `None`, that a mock was called) isn't worth writing.
- **Drive the public API like a real caller.** Prefer registering entries and calling `update()`/`render_*` the way a consumer would over reaching into internals; assert on the observable result.
- **In the e2e tier, assert on behavior, not exact text.** Real model output varies run to run, so a live test asserts "the output sink received non-empty text" or "the registered Command was actually invoked with plausible args," never a specific string.

## Test isolation: a fresh `World` per test

There is no process-global World: each `Wica` (see [wica.md](wica.md)) owns its own `World`, so isolation is a matter of each test building its own rather than resetting a shared one. A test that exercises the reactive path constructs a `World(loop)` on an event loop it controls and drives its `start()`/`stop()` lifecycle (see [world.md](world.md), "Lifecycle") — `stop()` cancels the World's outstanding TTL timers, so no timer bleeds into the next test. The World- and Agent-focused modules each provide local loop-plus-`World` fixtures suited to their teardown needs; there is currently no shared repository-wide `conftest.py`. Tests that only touch the data model (`register`/`get`/`render_*`) need no running loop at all, since those methods are unguarded.

## Live tier: parametrized over every provider

The live e2e set is **parametrized over one committed config per provider**, not a single reference provider — so the live tests verify WICA's real provider-branching construction across the whole supported surface, not just one privileged backend.

- **One config file per provider**, checked in and named symmetrically: `tests-e2e/e2e.<provider>.config.json` for `anthropic`, `openai`, `huggingface-hub`, `google-genai`, and `mistralai` (the file stem is the extra name — hyphenated for `google-genai`, whose `provider` value inside is LangChain's `google_genai`). They're wired together as `PROVIDER_CONFIGS` in `tests-e2e/support.py`, so **every live e2e test runs once per config**.
- **The live tests go through WICA's own entrypoint**, not a divergent `init_chat_model` call: `support.real_chat_model()` builds via `build_chat_model()` (the smoke test), and `support.real_wica()` stands up the whole system via `Wica.init(WicaConfig.from_json_file(...))` — the real production path, loop + World + Agent and all. This is what makes the tier meaningful: it exercises the actual per-provider construction (including `huggingface-hub`'s dedicated non-`init_chat_model` path — see [agent.md](agent.md), "Provider-agnostic model, from config") end-to-end, so a branch that builds the wrong model is caught here rather than slipping through a test that bypassed it.
- **Each config uses `api_key_env`**, so it carries no secret and is safe to commit (see [config.md](config.md), "API key"). A provider whose key env var is unset makes that config **skip itself** — never fail — via `MissingEnvError → pytest.skip` (in `support.load_agent_config`). You exercise only the providers you hold keys for; the rest skip cleanly, so a contributor with one key, or CI with none, is never broken by the others' absence.
- **A provider's rate limit skips, too.** A key that is set but rate-limited or out of quota says nothing about the code, yet it fails the call. A `pytest_runtest_call` wrapper in `tests-e2e/conftest.py` therefore turns a test failure into a skip when an HTTP 429 sits in the chain of the raised exception **or of any exception logged under the `wica` logger while the test ran** — the Agent catches a failed model call and logs it (`model call failed; step abandoned`), so in a `Wica`-driven test the 429 surfaces only as an expectation that timed out. The check is duck-typed over the providers' SDKs (`status_code`, `code`, `response.status_code` equal to 429; none is imported), as tts-engine's is; any other failure still fails, and the skip reason carries the provider's message, so `-rs` shows it. A test that failed for another reason while a 429 was also logged skips too — accepted, since the provider then failed a call of that test anyway.
- **Filter to one provider with `-k <provider>`.** Because the configs are named symmetrically, the provider name matches its config-filename stem, so `-k openai` runs just that one.
- **The live cases run on the `wica.testing` Harness** and can keep one JSON report per test per provider (`--wica-report-dir`) — see [wica-tester.md](wica-tester.md), "Relation to existing tests". Their assertions stay behavioral; no latency threshold is added.
- **A missing key can be made to fail.** With `WICA_E2E_FAIL_ON_MISSING=1` in the environment (CI's `live` job sets it — [ci.md](ci.md) "Secrets"), a config whose key env var is unset **fails** its tests instead of skipping, so the job cannot go green with a provider silently untested. A local run without the variable keeps skipping. The name and meaning are `wica-interaction`'s. A rate-limited key still skips (below): that is the account's state, not a missing secret.
- **The live set is not linked to the benchmark.** It is a fixed, scoped list — `PROVIDER_CONFIGS`, one committed config per provider — and nothing in it is parametrized over the benchmark's matrix (`benchmarks/matrix.json`), which lives its own life ([benchmark.md](benchmark.md) "Testing"). The benchmark's models are exercised by running the benchmark (`python -m benchmarks.bench --k N`); its code is covered key-less by `tests-e2e/test_demo_bench_fake.py`.

## The live set as a regression gate

The live set's job is to turn `main` red when a change breaks a behaviour the framework promises, on a real provider, and to say which. It is **not** the benchmark ([benchmark.md](benchmark.md)): the benchmark compares models over *k* runs with no threshold, hard and ambiguous cases included; the gate runs **once per committed model** and fails on a single bad run, so it holds only clear-cut behaviours every committed model reliably shows.

### The SUT: a test robot

The gate's steps run against a **dedicated test app**, `tests-e2e/robot_app.py`: a social robot about as rich as the conversation demo, so the model faces a realistic World and Command list, not a two-tool toy. It is not the demo and imports nothing from `examples/`, so a demo change cannot turn `main` red. No image Input.

| Kind | Name | Triggers | Role |
|---|---|---|---|
| Input | `speech_input` | yes | what someone says to the robot |
| Input | `closest_user` | yes | who stands closest (empty: no one) |
| Input | `ambient_sound` | yes | a background change addressed to no one |
| Input | `people_present`, `battery_level`, `time_of_day` | no | ordinary context |
| State | `emotion`, `tracked_user` | no | written by the Commands |
| Output Command | `say(text)` | | speech: takes time in proportion to its words, up to a cap, and is cancellable |
| Command | `set_emotion`, `switch_user_tracking`, `look_at`, `wave` | | short expressive actions |
| Command | `dance` | | a few seconds, cancellable |
| Command | `write_note(title, text)` | | saves a note; its result names the word count, marked `(long)` from 150 words |

`say` is a fake: no TTS, it waits and completes. Only the Interrupt step needs that wait (the running `say` must outlast the model's reaction to "Stop"); every other step speaks a sentence or two. Long output goes through `write_note`, not `say`: the failure it guards against — a reasoning model cut off by an output cap before its tool call — is about tool arguments, not speech. Each `Wica` is set up as a person (`u1`) walking up: the app writes its context and `closest_user`, then waits for the robot to settle **before** the step's own `TestRun` begins, so that first reaction is not judged.

### The steps

Each behaviour is one **scenario file** in `tests-e2e/scenarios/`, in the format of [wica-tester.md](wica-tester.md) "Scenarios" — the benchmark's format, so the gate and the benchmark judge "correct" with the same expects and checks. One file per behaviour, so a stopped scenario stops only that behaviour's test.

| Step | The input | Fails when | Guards against |
|---|---|---|---|
| `greeting` | "Hello!" | no `say`, or no settle | replies in plain text; reply loops |
| `act_on_request` | "Can you dance for me?" | `dance` not called or not completed | tools not bound, a broken forced binding |
| `background` | an `ambient_sound` change | `say` issued | speaking at every observation (a `noop` or primer regression) |
| `long_answer` | a request for a ~300-word story in the notebook | no `write_note` marked `(long)`, a failed or empty reaction | an output cap before the tool call, timeouts, truncated arguments |
| `interrupt` | a request for a long story; once `say` runs and the Agent is ready, "Stop, please." | no `cancel_command`, or a colliding one | cancellation, or the visibility of running Commands, breaking |
| `memory` | "My name is Alex." → settled → "What's my name?" | the reply lacks "Alex" | history rendering, caching changes |
| `goodbye` | "Goodbye!" | no `say`, or no settle | — |

Every step carries `no_failed_reactions`; the steps that address the robot carry `answers_each_input` and `no_noop_on_input`; each step ends on `wait idle within 30` — **30 s to settle**, so a runaway chain of steps fails fast instead of at a test timeout.

### Two kinds of test, written once

`tests-e2e/test_robot.py` plays the same step files two ways, parametrized over `PROVIDER_CONFIGS`:

- **Focused tests**, one per (provider, step), each on a **fresh** robot: a quick diagnostic that names the behaviour that broke.
- **A conversation per provider** (`TestConversation`): the steps in the order above as one test each, on **one** robot, run with **`history_reactions = 4`** so the history is cut several times during it. A step plays on its own `TestRun` (a scenario ends its run), attached to the shared `Wica`; before it, the previous step's run waits for the robot to settle and anything a failed step left running is cancelled, so one failure does not take the next step with it. No step relies on an earlier one: `-k` on a single step runs it alone. After the last step the conversation asserts that **a history cut happened** — some prompt held fewer messages than the one before it — which checks the cut against a real provider without asking the model to show it. Recall across a cut is **not** tested: state that must outlive the window belongs in a World entry ([agent.md](agent.md), "History window").

Each step's run writes its own report under `--wica-report-dir`.

### What fails

**One run per model; a missed `expect` (or a timed-out `wait`) or a failed `check` fails the step.** There is no "record and pass" tier. What keeps the gate from going red on noise is the choice of steps and checks, not leniency: a step lists only the checks that are right for it, and a step joins the gate only after it passed **at least three runs by hand on every committed model**. A (provider, step) pair that does not pass reliably is marked `xfail(strict=False)` with the reason, not dropped — `KNOWN_FLAKY` in `tests-e2e/test_robot.py`, which holds today (huggingface-hub, background) and (mistralai, background): both models, woken by a background sound, sometimes check in or repeat the previous request instead of calling `noop`.

### Alongside the steps

`test_wica.py` keeps the cases the robot does not cover: `test_plain_text_round_trip` (a reply in free text, no output Command) and `test_noop_empty_reaction_live` (`noop` with **no** output Command, so not forced). `test_smoke.py` checks the model answers at all. The two cases it held before are steps now: the voice-persona output-Command case is `greeting`, the tool-calling round trip is `act_on_request`.

The committed configs and their key env vars:

| Config | Provider | Model | Key env var |
|---|---|---|---|
| `tests-e2e/e2e.anthropic.config.json` | `anthropic` | `claude-haiku-4-5` | `WICA_ANTHROPIC_API_KEY` |
| `tests-e2e/e2e.openai.config.json` | `openai` | `gpt-4o` | `WICA_OPENAI_API_KEY` |
| `tests-e2e/e2e.huggingface-hub.config.json` | `huggingface-hub` | `openai/gpt-oss-120b` on `cerebras` | `WICA_HF_TOKEN` |
| `tests-e2e/e2e.google-genai.config.json` | `google_genai` | `gemini-3.5-flash-lite` | `WICA_GEMINI_API_KEY` |
| `tests-e2e/e2e.mistralai.config.json` | `mistralai` | `ministral-14b-latest` | `WICA_MISTRAL_API_KEY` |

**The models are chosen for the gate**, by running its steps at least three times on each candidate: a gate model must show the promised behaviours reliably, or the gate goes red on noise. `gemini-3.8-flash` was replaced by `gemini-3.5-flash-lite` (the former hit the 10 s `model_timeout` twice on long texts; the latter passed 45/45 at about 1 s a call), `ministral-8b` by `ministral-14b-latest` (the former ran away in the conversation in every run; the latter passed 44/45). Which models an application should use is the benchmark's question, not this table's.

The provider/extra surface these configs select from is specced in [config.md](config.md) ("Providers") and packaged as install-time extras per [project.md](project.md) ("Provider integrations are optional extras").

## Tooling

- **`pytest`** is the runner, with **`pytest-xdist`** running the fast tier in parallel: `addopts = ["-n", "logical", "--maxprocesses", "8"]` in `pyproject.toml`, as reachy-mini-bridge and naoqi-sim do. The tier is sleep-bound (coalescing windows, step loops and `wait_until` deadlines run in real time), so spreading it over workers cuts a laptop run from about 43 s to about 17 s; the floor is the slowest single test. `-n 0` runs it serially. Each worker is its own interpreter, so per-test isolation (a fresh `World` and loop per test, the OpenTelemetry in-memory exporter) holds per worker. A test that sends a second trigger must wait for the first reaction to **end** (`not agent._busy`), not just for the model call — under parallel load a trigger arriving while the step is in flight is dropped by design ([agent.md](agent.md)). **`tests-e2e/` stays serial:** its `conftest.py` sets `numprocesses = 0` in `pytest_cmdline_main`, since parallel live cases would only hit provider rate limits and the scripted-fake flows take seconds; it loads only when that tier is collected, so the fast tier keeps its workers.
- **`ruff`** lints/formats; **`pyright`** (`standard` mode) type-checks. All three are the gate after any change — lint, type check, and tests must pass before work is considered done (see [AGENTS.md](../AGENTS.md), "Verification").
- **`pyright` covers test code too:** its `include` is `src`, `tests`, and `tests-e2e`, so tests are type-checked alongside the library rather than being a blind spot.
- The concrete commands (`uv run ruff check .`, `uv run pyright`, `uv run pytest`, and the sourced-env e2e invocations) live in [AGENTS.md](../AGENTS.md).

## Open questions

1. **e2e assertion depth.** ~~The live tests assert coarse behavior.~~ The gate's steps (above) assert the framework's promised behaviours with the benchmark's expects and checks. Whether a recorded-cassette approach (replaying captured provider responses in the fast tier) would cover more without the live tax is unexplored.
2. **Provider matrix growth.** `PROVIDER_CONFIGS` lists the three providers WICA supports today. As providers are added (or the extras change), the parametrization grows with them; there's no automated check that every supported provider has a committed e2e config — nor that CI's `live` job maps a secret for each ([ci.md](ci.md) "Secrets") — so that pairing is kept honest by hand for now.
