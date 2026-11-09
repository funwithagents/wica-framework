# Documentation

The documentation of WICA, by what you want to do. The [README](../README.md) is the landing page: what WICA is, how to install it and a first quick start; [getting-started.md](getting-started.md) continues from there.

## By task

| I want to… | Read |
|---|---|
| write my first agent | [getting-started.md](getting-started.md): a complete program on the scripted `fake` provider, then the config edit that moves it to a real model |
| feed the agent a perception (speech, a sensor, a camera frame) | [guides/world-and-inputs.md](guides/world-and-inputs.md): registering an entry, serializing it to the prompt, what wakes the Agent, TTLs, bursts |
| give the agent an action it can take | [guides/commands.md](guides/commands.md): functions as Commands, cancellation, the output Command, `noop`, what runs concurrently |
| configure the model, the key, the prompt, the history | [reference/configuration.md](reference/configuration.md): every field, the loaders, the providers |
| watch the World, the prompts and the transcript in a browser | [guides/gradio-panels.md](guides/gradio-panels.md): the three `wica[gradio]` panels in your own `gr.Blocks` |
| test my agent without a key or a network | [guides/testing.md](guides/testing.md): the scripted `fake` provider, the `TestRun` harness and its report, a scripted tester |
| measure how fast it reacts, or see one trace end to end | [guides/latency-and-traces.md](guides/latency-and-traces.md): the reaction record, the measures, OpenTelemetry |
| compare models on my application | [guides/benchmarking.md](guides/benchmarking.md): a `Target`, a matrix, scenarios, `wica.benchmark`'s runner and CLI |
| know what v1 does not do yet | [guides/limits.md](guides/limits.md): one reaction at a time, dropped triggers, complete-text output, lossy history bounds |
| find out why nothing happens, or why a call raises | [guides/troubleshooting.md](guides/troubleshooting.md): the checks for each symptom, routed to the page that explains it |
| look up a name, a signature, an Event or an error | [reference/api.md](reference/api.md) |
| look up a config field | [reference/configuration.md](reference/configuration.md) |
| try it without writing code | the [conversation demo](../README.md#the-conversation-demo): a browser UI over a simulated social robot |

## The references

- [reference/api.md](reference/api.md): what you import, construction, the `Wica` surface, the World API, `Command`, the eight Events, the errors, and the contracts that hold across them (threads and the loop, lifecycle, keys and names).
- [reference/configuration.md](reference/configuration.md): the `agent` block field by field, the three loaders and direct construction, the providers and their extras.

## Beyond the user documentation

- [specs/](../specs/_index.md) is the normative design: every guarantee with its rationale, one spec per concept, each with a status.
- [examples/](../examples/) holds the conversation demo, the complete application the docs' snippets are cut from.
- [benchmarks/](../benchmarks/README.md) is the framework's own benchmark: the demo driven by many models over a scenario set, with its kept results.
- [AGENTS.md](../AGENTS.md) is the operating manual for working in the repository.
