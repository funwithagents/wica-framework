# Live tier: a rate-limit preflight per provider

**Status:** Done

Implements the "A rate limit is also checked once per provider" rule of [specs/testing.md](../specs/testing.md) ("Live tier") and its line in [specs/ci.md](../specs/ci.md).

**Why.** The live tier already skips a test when an HTTP 429 surfaces during it. On a spent quota it often does not surface: Gemini's client retries the 429 with backoff for longer than `model_timeout`, so every reaction ends `model_timeout`, no 429 is raised or logged, and the test fails on a missed expectation. A CI run on a key over its spending cap failed 17 Gemini tests in ten minutes, with one skip (the smoke test, which calls the model directly).

## Scope

- **`tests-e2e/support.py`**: `rate_limited` (moved from `conftest.py`); `skip_if_rate_limited(config_path)` — one direct call per provider per session, cached, a 429 skipping the calling test with the provider's message; called by `real_chat_model` and `real_wica` before they build. Another failure, or an answer, lets the tests run; a missing key propagates to the callers' own rule; the `fake` provider is never called.
- **`tests-e2e/conftest.py`**: the per-test check imports `rate_limited`; unchanged otherwise.
- **`tests-e2e/test_preflight_fake.py`**: the preflight over a stubbed provider, key-less, added to CI's scripted-fake job.
- **AGENTS.md**: the rule and the new always-run test.

Not in scope: a quota that runs out mid-run behind a client's retries — still a failure; the per-test check covers the 429s that surface.

## Tests

- A stub raising a 429-shaped error: `real_chat_model` skips with the provider's message, `real_wica` skips before building a `Wica`, and the provider was called once.
- A 429 as the cause of another error is found.
- A provider that answers, raises a 500 or raises something else does not skip, and is called once over two checks.
- A missing key raises `MissingEnvError` and caches nothing.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` and `uv run pytest tests-e2e -k "fake or example or tester"`.
- The Gemini live set against the key over its cap; a healthy provider's smoke test.

## Result

All clean; the fast tier and the key-less e2e set pass. Against the real Gemini key over its spending cap, the provider's live set gave 18 skipped in 36 s, each with the provider's message, where CI had 17 failed in ten minutes. Healthy providers still run: the smoke and `Wica` tests passed on Anthropic, Mistral and the Hugging Face Hub (9 passed). OpenAI's key turned out to have no credits left during this check and skipped the same way, in 3 s.
