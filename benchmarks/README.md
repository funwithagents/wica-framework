# WICA benchmark

The framework's own benchmark ([specs/benchmark.md](../specs/benchmark.md)): the conversation demo
([specs/conversation-demo.md](../specs/conversation-demo.md)) as the system under test, one config per
provider as the matrix, and a scenario set written against the demo's `speech_input` + `say` shape.
It answers "how well does model X drive a WICA agent at all" with one or two **session scenarios run k
times per model**, read as one row per model; nothing here is installed — the runner
it uses is `wica.benchmark`, the scenario format and checks are `wica.testing`'s.

## Layout

| Path | What |
|---|---|
| `bench.py` | `DemoTarget` and the `wica-demo-bench` entry point |
| `matrix.json` | the matrix: one entry per model under test (`id`, `agent` block, `tags`, `enabled`, `notes`), the demo's persona shared through `agent_defaults`; the `fake` fixture is its opt-in key-less row |
| `scenarios/*.json` | **the workload**: `conversation` (a generic session — greeting, question, a story interrupted by "Stop!", two quick utterances, goodbye) and `commands` (the demo's own actions); each a fixed user as a scripted-tester script + the checks to run over the whole session |
| `pins/*.json` | seven rule-by-rule micro scenarios (one primer clause each), for debugging a specific rule with `--scenarios benchmarks/pins`; not the workload, not run by CI beyond the `greeting` smoke |
| `runs/<stamp>/` | **batches** — per-run reports and summaries, git-ignored working data |
| `results/<stamp>_<name>/` | **results** — kept batches: `summary.md`, `summary.json`, `runs.json`, hand-written `description.md` and `findings.md` |

## Running

From the repo root, with the `demo` group (the demo's presenters):

```bash
# the whole matrix, five repetitions per (configuration, scenario); a provider whose key is unset skips
zsh -ic 'source ~/.zshrc >/dev/null 2>&1; uv run --group demo python -m benchmarks.bench --k 5'

# one provider's rows, the generic session only
uv run --group demo python -m benchmarks.bench --only 'openai/*' --scenario conversation --k 3

# the rule pins, one model, to see which clause a model breaks
uv run --group demo python -m benchmarks.bench --scenarios benchmarks/pins --only 'openai/*' --k 3

# only the providers' committed defaults, every scenario
uv run --group demo python -m benchmarks.bench --tags default --k 5

# the key-less path: the fake fixture plays the whole `conversation` session, no network
uv run --group demo python -m benchmarks.bench --only fake --include-disabled --scenario conversation
```

Entries are selected by id glob (`--only`, `--exclude`) and tag (`--tags`, `--exclude-tags`); a row
marked `enabled: false` runs only with `--include-disabled`; a row whose key variable is unset skips
with the reason. A batch prints its progress and its summary, and lands in `runs/<YYYYMMDD-HHMM>/`
(`--out` to choose). The summary opens with **one row per model** pooled over every run — pass^k,
rule compliance, turns answered, reaction and model latency, tokens and words per turn, chained
replies — then one table per scenario. Runs are sequential; a `conversation` run is seven user turns
and about a minute, so five models × two scenarios × five repetitions is on the order of an hour.

## Model catalog

`matrix.json` lists one `default`-tagged row per provider — the model the demo's own
`agent.<provider>.config.json` commits, verified by the live e2e tier — and is where more models go:
one row per model to compare, the id as `<provider>/<model>`, a tag per family you want to select on.
Add `enabled: false` for a row you do not want in every batch (an older snapshot, an expensive
model), `requires_env` for a gate beyond the key variable.

## Keeping a result

A batch that marks a decision — a model chosen, a primer change measured — is kept with
`--result <name>`: its summaries and a `runs.json` manifest are copied to
`results/<stamp>_<name>/`, and two files are created once for you to finish by hand before
committing: `description.md` (the machine, the network, why the batch was run) and `findings.md`
(what the numbers say, which decision they fed, what remains). A spec or plan that quotes a number
quotes the result id (the folder name), never a batch.

## Results

None yet.
