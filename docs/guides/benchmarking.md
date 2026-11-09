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

A scenario is a fixed user as a scripted-tester script ([testing.md](testing.md#a-scripted-tester)) plus the checks to run over the whole session, one JSON file each, `name` equal to the file stem:

```json
{
  "name": "greeting",
  "description": "A greeting is answered out loud.",
  "requires": {"inputs": ["speech_input"], "output_command": true},
  "timeout_s": 60,
  "checks": ["answers_each_input", "no_failed_reactions"],
  "script": [
    {"tool_calls": [{"name": "set_speech_input", "args": {"value": "Hello there!"}}]},
    {"tool_calls": [{"name": "expect_output", "args": {"within": 20}}]},
    {"tool_calls": [{"name": "finish", "args": {"verdict": "pass"}}]}
  ]
}
```

- **`requires`** says what the target must expose (input keys, Command names, an output Command); a pair whose target lacks one is skipped and said so, so one directory can mix generic and application-specific scenarios.
- **`checks`** are primer-level checks over the finished timeline, resolved against a registry: `answers_each_input` (every input got an output), `no_noop_on_input` (no input was answered with `noop`), `no_failed_reactions`, `no_colliding_cancel` (no reaction issues `cancel_command` together with another Command). Absent, every check whose own requirements the target satisfies runs. Add your own through the CLI's `extra_checks` (`wica.testing.Check`).
- **Matchers stay loose**: one script runs against every model, so a step asserts that the right kind of thing happened (an output, a Command by name, a silence), not its wording.

## Running

```
uv run python -m my_bench --k 5                                   # the whole matrix, five runs per cell
uv run python -m my_bench --only 'openai/*' --scenario greeting   # one provider's rows, one scenario
uv run python -m my_bench --tags default --exclude-tags reasoning
uv run python -m my_bench --only fake --include-disabled           # the key-less path
```

Runs are sequential (parallel runs would confound the latencies); a row whose key is unset skips. Each run writes its report (`<out>/<id>/<scenario>/<i>.json` and `.md`) into a **batch** folder, by default `<runs>/<YYYYMMDD-HHMM>/`, with `summary.json` and `summary.md`: one row per configuration, with the pass rate and pass^k per scenario, and the pooled distributions of the measures (reaction latency, model latency, tokens) with no thresholds. The exit code is 0 whatever the results: a benchmark compares, it does not gate.

**`--result <name>`** keeps the batch under `<results>/<stamp>_<name>/` with a `description.md` (the machine, the network, what was run) and a `findings.md` (what you concluded) to finish by hand before committing. Batches are working data, git-ignored; results are what the repo keeps. The repo's own [benchmarks/README.md](../../benchmarks/README.md) shows the shape, and the two smoke tests in `tests-e2e/` the way to keep a benchmark honest with a `fake` row.
