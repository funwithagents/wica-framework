# Testing your agent

Two tools, both network-free and key-less, both in the package: the **`fake` provider**, a scripted model that lets a test drive the whole loop and assert the exact Commands and utterances a live model could never pin, and **`wica.testing`**, a harness that probes a running `Wica`, records a timeline with the framework's own timing records, waits for what you expect and writes a report. The design is [specs/fake-provider.md](../../specs/fake-provider.md) and [specs/wica-tester.md](../../specs/wica-tester.md).

## The `fake` provider

Select `provider: "fake"` and the model replays `model_kwargs.script` step by step: each reasoning step consumes the next entry and returns it as the model's response. A step is `{"text": …}`, `{"tool_calls": [{"name": …, "args": {…}}]}` or both; each tool call is dispatched exactly like a real provider's. Once the script is spent, every step returns `model_kwargs.default` (`{"text": ""}` unless set), or `"loop": true` cycles the script. `delay_s` (200 ms by default) is the simulated model latency; set it to `0` in a test.

```python
import threading

from wica import TextPart, Wica, WicaConfig

config = WicaConfig.from_dict(
    {
        "agent": {
            "provider": "fake",
            "model": "scripted",
            "system_prompt": "You are a test double.",
            "model_kwargs": {
                "delay_s": 0,
                "script": [
                    {"tool_calls": [{"name": "walk_to", "args": {"place": "kitchen"}}]},
                    {"text": "I am in the kitchen."},
                ],
            },
        }
    }
)

said = threading.Event()
heard: list[str] = []
visited: list[str] = []


async def sink(text: str) -> None:
    heard.append(text)
    said.set()


async def walk_to(place: str) -> str:
    """Walk to a place."""
    visited.append(place)
    return f"Arrived at the {place}."


wica = Wica.init(config, coalesce_window=0)
wica.set_output_sink(sink)
wica.register_command(walk_to)
wica.world.register(
    "speech_input",
    str,
    serialize_fn=lambda value, previous: [TextPart(f'The person said: "{value}"')],
    triggers_llm_call=True,
)
wica.start()
wica.world.update("speech_input", "Go to the kitchen")
assert said.wait(timeout=5)
wica.close()

assert visited == ["kitchen"]
assert heard == ["I am in the kitchen."]
```

- Each scripted tool call's name is validated against your registered Commands at the first model call: a typo raises instead of silently emitting an unknown call.
- Every response reports **token usage** like a real provider: an estimate of the prompt's text length by default (so a `history_budget` can be watched tripping naturally), or a per-step `"usage": {"input_tokens": N, "output_tokens": M}` to trip it at a chosen step.
- The script is JSON, so a committed `*.config.json` with `provider: "fake"` is a complete test fixture, the same loader as production.
- For a model double that cannot live in JSON (a reactive, programmatic one), build the Agent directly with the raw-model override: `from wica.fake_model import FakeChatModel`, then `Agent(AgentConfig(provider="fake", model="…", system_prompt="…"), world=World(loop), loop=loop, model=FakeChatModel(...))`.

## The harness: `TestRun`

`wica.testing.TestRun` connects to a running `Wica` through a **Probe** (a mirror of its Inputs and Commands over the eight Events) and records a **timeline**: every World write, trigger, prompt, Command issuance and state change, text and reaction record, stamped with the framework's own instrumentation. On it you call the shared expectation set from your test's thread, as blocking calls with a timeout:

```python
from wica.testing import TestRun

wica = make_my_agent()  # your application's build + wire, on any provider
wica.start()
run = TestRun(wica, scenario="greeting", sut_label="fake")
try:
    run.set("speech_input", "hello")
    run.expect_action("say", within=5)  # the Command was issued
    run.expect_action("say", state="complete", within=10)  # ...and finished
    run.expect_no_action("dance", for_=0.5)  # nothing else happened
    run.finish("pass", "greeted once")
    report = run.report()
finally:
    run.close()
    wica.close()

assert report.ok, [f for f in report.findings if not f.ok]
print(report.metrics["reaction_latency"].median)
```

- **The expectation family**: `set(key, value)` (through the Probe, moving the watermark), `wait(seconds)`, `expect_action(name, where=…, state=…, result_contains=…, within=…)`, `expect_output(contains=…, within=…)` (the output Command, whatever its name), `expect_text(contains=…, within=…)` (the free text), `expect_no_action(name, for_=…)`, `expect_world(key, equals=…, within=…)`, `expect_latency(metric, p95_below=…)`, plus `note`, `report_issue` and `finish(verdict, summary)`. Each returns a pass/fail **finding**. Expectations scan forward from a watermark, so a sequence never re-matches an earlier event and repeating one means "the next one".
- **The report** (`run.report()`) is a dataclass with the timeline, the metrics (`reaction_latency`, `model_latency`, `coalescing_wait`, `busy_time`, `command_duration`, `reactions_per_input`, `input_tokens`, `output_tokens`, each a min/median/p95 summary), the dropped-input count, the findings, the issues and the verdict; `report.ok` is what a test asserts; `report.write(path)` saves it as JSON, or as Markdown for a `.md` path; `report.to_markdown()` renders it. Latency is a measure, never a built-in threshold: `expect_latency` is the one way to make it fail a run.
- The same harness drives a **live** model: the repo's own live e2e set is one `TestRun` per provider ([specs/testing.md](../../specs/testing.md)), each a smoke test with a report, not a benchmark.

## A scripted tester

The expectation set also exists as **Commands on a second `Wica`**, a tester over the `fake` provider whose script is the scenario: `run_script(run, script, timeout=…)` builds that tester, fires its steps one after the other (each step runs when the previous Command completes, so the chain is linear whatever the agent under test does), and returns the report when the script calls `finish`:

```python
from wica.testing import TestRun, run_script

script = [
    {"tool_calls": [{"name": "set_speech_input", "args": {"value": "hello"}}]},
    {"tool_calls": [{"name": "expect_output", "args": {"contains": "hi", "within": 5}}]},
    {"tool_calls": [{"name": "expect_no_action", "args": {"name": "dance", "for_": 0.5}}]},
    {"tool_calls": [{"name": "finish", "args": {"verdict": "pass"}}]},
]
run = TestRun(wica, scenario="greeting", sut_label="fake")
report = run_script(run, script, timeout=15)
```

The tester's Commands are `set_<key>` for each Input the Probe found on your agent, and the expectation family above with the same arguments. This JSON shape is what a **scenario file** holds; the benchmark runs a directory of them against a matrix of models ([benchmarking.md](benchmarking.md)). A script that never calls `finish` within the timeout yields a failed finding.

## Where the repo's own tests sit

The fast tier (`tests/`) exercises every feature against the `fake` provider; `tests-e2e/` holds the always-run scripted flows (`-k fake`, `-k tester`, `-k example`) and the live set, one config per provider, each skipping when its key is unset ([specs/testing.md](../../specs/testing.md); the commands are in [AGENTS.md](../../AGENTS.md)).
