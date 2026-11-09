# Benchmarking your application

A benchmark answers "how well does model X drive my agent": the same scripted sessions run k times per model, read as one row per model with its pass rate, its latencies and its token costs. `wica.benchmark` is the runner an application's benchmark imports: a **matrix** of configurations, a directory of **scenarios**, a **`Target`** you implement to stand your application up, and one CLI. The repo's own benchmark, the conversation demo over nineteen models, is the worked example ([benchmarks/README.md](../../benchmarks/README.md)); the design is [specs/benchmark.md](../../specs/benchmark.md).

## The target

A `Target` stands your application up for one run and tears it down. `open` is called per run, so every repetition is a fresh system; it returns the started `Wica`, or raises `Skip` with a reason when this configuration cannot run here (its key is unset, its extra is absent), which the summary records without failing:

```python
from pathlib import Path

from wica import Wica, WicaConfig
from wica.benchmark import Entry, Skip, cli
from wica.testing import Scenario

BENCH_DIR = Path(__file__).parent


class MyTarget:
    def __init__(self, entry: Entry) -> None:
        self.entry = entry
        self.label = entry.id  # the row's label in every report and summary
        self.agent_config = dict(entry.agent)  # recorded in the summary header
        self._wica: Wica | None = None

    def open(self, scenario: Scenario) -> Wica:
        reason = self.entry.skip_reason()  # the key variable, or a requires_env, unset
        if reason is not None:
            raise Skip(reason)
        config = WicaConfig.from_dict({"agent": self.entry.agent}, base_dir=self.entry.base_dir)
        self._wica = build_my_agent(config)  # your build + wire, as your main() does
        self._wica.start()
        return self._wica

    def close(self) -> None:
        if self._wica is not None:
            self._wica.close()
            self._wica = None


def main(argv: list[str] | None = None) -> int:
    return cli(
        argv,
        target_factory=MyTarget,
        default_matrix=BENCH_DIR / "matrix.json",
        default_scenarios=BENCH_DIR / "scenarios",
        default_runs=BENCH_DIR / "runs",
        default_results=BENCH_DIR / "results",
        prog="my-bench",
    )
```

`scenario.params` is an opaque object from the scenario file your target may read (an initial state to set before the run); the runner never does.

## The matrix

One JSON file lists every configuration, so adding a model is adding a row:

```json
{
  "agent_defaults": {"system_prompt_file": "../prompts/robot.md"},
  "entries": [
    {"id": "anthropic/claude-haiku-4-5",
     "agent": {"provider": "anthropic", "model": "claude-haiku-4-5", "api_key_env": "WICA_ANTHROPIC_API_KEY"},
     "tags": ["api", "default"], "notes": "the committed model"},
    {"id": "fake",
     "agent": {"provider": "fake", "model": "scripted", "system_prompt": "…", "model_kwargs": {"script": ["…"]}},
     "enabled": false, "tags": ["fixture"]}
  ]
}
```

- **`id`** is the label everywhere, by convention `<provider>/<model>`. **`agent`** is the config's agent block, merged over `agent_defaults` (entry keys win; `system_prompt` in an entry replaces a default `system_prompt_file`). A relative `system_prompt_file` resolves against the matrix file's directory through `entry.base_dir`.
- **`tags`** are the selection vocabulary (`--tags`, `--exclude-tags`); **`enabled: false`** marks an opt-in row (`--include-disabled`); **`requires_env`** names extra variables that gate the row; **`notes`** is for the reader. Any other key lands in `entry.extra` for your target.

## The scenarios

A scenario is a fixed user written as data: one JSON file, `name` equal to the file stem, holding a flat list of steps. Each step is one of four kinds ([specs/wica-tester.md](../../specs/wica-tester.md), "Scenarios"):

```json
{
  "name": "greeting",
  "description": "A greeting is answered out loud, then interrupted.",
  "checks": ["answers_each_input", "no_failed_reactions"],
  "steps": [
    {"input": {"speech_input": "Tell me about your day."}},
    {"wait": "speaking"},
    {"wait": "ready"},
    {"input": {"speech_input": "Stop!"}},
    {"check": "cancels_when_asked"},
    {"wait": "idle"}
  ]
}
```

- **`input`** writes one Input of your agent, as the user would.
- **`wait`** paces the session, and nothing is implicit. `"idle"` waits until the agent has finished, its Commands included. `"ready"` waits until the agent can take an input, while a speech or an action may still run. `"speaking"` and `{"running": "<Command>"}` wait until that Command has started since the last input. `{"seconds": n}` is the only timer. A wait that times out fails the run.
- **`expect`** asserts something must happen (an output, a Command, a World value); a miss fails the run. A benchmark uses few of them, because a failed run is not scored.
- **`check`** is a rule judged after a completed run: on a step, over the window from the input it follows to the next input; in the top-level `checks`, over the whole run. The built-ins are `answers_each_input`, `no_noop_on_input`, `no_failed_reactions`, `no_colliding_cancel`, `cancels_when_asked`, `acts_when_asked` and `quiet_on_background_change`. Add your own through the CLI's `extra_checks` (`wica.testing.Check`).
- **Write `{"wait": "ready"}` before any input that talks over the agent.** The agent runs one reaction at a time and drops an input that arrives during one; a scenario that writes into that window measures the drop, not the model.
- **What the target must expose is derived from the steps** (the Inputs written, the Commands named, whether an output Command is needed); a pair whose target lacks one is skipped and said so, so one directory can mix generic and application-specific scenarios.
- **Matchers stay loose**: one script runs against every model, so a step asserts that the right kind of thing happened (an output, a Command by name, a silence), not its wording.

## Running

```
uv run python -m my_bench --k 5                                   # the whole matrix, five runs per cell
uv run python -m my_bench --only 'openai/*' --scenario greeting   # one provider's rows, one scenario
uv run python -m my_bench --tags default --exclude-tags reasoning
uv run python -m my_bench --only fake --include-disabled           # the key-less path
```

Runs are sequential (parallel runs would confound the latencies); a row whose key is unset skips. Each run writes its report (`<out>/<id>/<scenario>/<i>.json` and `.md`) into a **batch** folder, by default `<runs>/<YYYYMMDD-HHMM>/`, with `summary.json` and `summary.md`: one row per configuration, with the pass rate and pass^k per scenario, and the pooled distributions of the measures (answer latency, the time from an input to the first word whatever the number of steps; reaction latency; model latency; tokens) with no thresholds. The exit code is 0 whatever the results: a benchmark compares, it does not gate.

**`--result <name>`** keeps the batch under `<results>/<stamp>_<name>/` with a `description.md` (the machine, the network, what was run) and a `findings.md` (what you concluded) to finish by hand before committing. Batches are working data, git-ignored; results are what the repo keeps. The repo's own [benchmarks/README.md](../../benchmarks/README.md) shows the shape, and the two smoke tests in `tests-e2e/` the way to keep a benchmark honest with a `fake` row.
