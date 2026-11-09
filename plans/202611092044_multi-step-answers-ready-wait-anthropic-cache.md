# Multi-step answers, a `ready` wait, and Anthropic prompt caching

**Status:** Done

Revises [specs/wica-tester.md](../specs/wica-tester.md) ("Scenarios"), [specs/benchmark.md](../specs/benchmark.md) ("Measures", "The summary", the workload) and [specs/agent.md](../specs/agent.md) (prompt caching), after the `checks-baseline` result (`benchmarks/results/20261109-1849_checks-baseline`).

**Why.** Three findings from that batch:

- **The scenarios raced the Agent.** An input written while a reaction is in flight is dropped (agent.md, "Concurrency and long-running Commands in v1"). "Can you hear me?", sent 0.5 s after "Hi, I'm Alex.", was dropped in 17 of 18 conversation runs; "Stop!", sent two seconds into the dance, landed during the step the opening utterance's completion woke and was dropped for four models — exactly the four `cancels_when_asked` misses. The benchmark measured the framework's drop rule, not the models.
- **A multi-step answer read as chatter.** A model may issue a side Command first (an emotion change) and speak on the step its completion wakes. The checks already judge a whole turn, so they accept it; but `unprompted_output` counted that utterance as unprompted (25 of 59), and reaction latency stopped at the first Command, not at the first word.
- **Anthropic never cached.** OpenAI read 78–89% of its input from cache and Gemini 28–43%; Anthropic 0%, because it caches only at a `cache_control` breakpoint and the Agent sends none.

## Scope

- `wica.testing`:
  - a `"ready"` wait condition — the Agent can take an input now: no write unseen, no trigger unconsumed, no completion whose step has not woken; Commands may still run. `"idle"` is `"ready"` with every Command terminal.
  - measures: `extra_output` (output issued in a turn after its first one) replaces `unprompted_output`; `answer_latency` samples — a turn's opening write → its first output Command — the time to the first word, whatever the number of steps.
- `wica.benchmark`: the tables show **answer p50 / p95** before reaction latency, and **extra replies** in place of unprompted.
- `benchmarks/scenarios`: a `{"wait": "ready"}` before every input written over a running Command; the 0.5 s gap becomes a `ready` wait.
- `wica.agent`: with `provider: "anthropic"`, a `cache_control` breakpoint on the last block before the newest observation — the deepest block the next request renders byte-identically (the newest observation is re-rendered archival on the next step). On the first step that block is the system prompt.
- Specs, docs, `benchmarks/README.md` and the tests follow.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- The e2e-fake set — key-less, passes.
- A live Anthropic smoke run of the benchmark shows a non-zero cache read share: the four Anthropic rows, *k* = 1 (a batch, not kept), read 44–86% of their input from cache, against 0% in `checks-baseline`. No user input was dropped; the only drops left are `say` completions landing during the step "Stop!" opened, which the summary's `dropped` column also counts.
