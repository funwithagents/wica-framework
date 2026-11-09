# Configuration

`Wica.init()` consumes a `WicaConfig`, whatever its values' origin. Construct the dataclasses directly when Python owns the settings, or parse the same shape from a dictionary, a JSON string or a JSON file. The dictionary and JSON shape is one `agent` block:

```json
{
  "agent": {
    "provider": "anthropic",
    "model": "claude-haiku-4-5",
    "api_key_env": "WICA_ANTHROPIC_API_KEY",
    "system_prompt_file": "prompts/robot.md",
    "model_kwargs": { "temperature": 0.7 }
  }
}
```

On this page: [The loaders](#the-loaders) · [The `agent` block](#the-agent-block) · [API key](#api-key) · [System prompt](#system-prompt) · [History bounds](#history-bounds) · [Providers](#providers) · [What is not configuration](#what-is-not-configuration).

## The loaders

| Source | Constructor | Notes |
|---|---|---|
| Values owned by Python code | `WicaConfig(agent=AgentConfig(...))` | Static type checking; the structural invariants below are enforced on construction, the field types are not |
| A dictionary, including one cut from a larger application config | `WicaConfig.from_dict(data, base_dir=None)` | Pass the WICA-shaped subsection, not the application's other keys |
| A JSON string | `WicaConfig.from_json(text, base_dir=None)` | Same as `from_dict` |
| A dedicated JSON file | `WicaConfig.from_json_file(path)` | A relative `system_prompt_file` is located relative to the file's directory, so a config-plus-prompts folder stays relocatable |

```python
from wica import AgentConfig, WicaConfig

# Values owned by this Python code.
config = WicaConfig(
    agent=AgentConfig(provider="openai", model="gpt-4o", system_prompt="Be concise.")
)

# A validated block embedded in a larger application config.
config = WicaConfig.from_dict(app_settings["wica"], base_dir=app_settings_path.parent)

# A dedicated file.
config = WicaConfig.from_json_file("agent.config.json")
```

**The loaders are strict.** A missing required key, an unknown key (a typo included), an invalid field combination or a wrong type raises `ConfigError`. **`base_dir`** is only there to give a relative `system_prompt_file` a stable origin: without it, `from_dict` and `from_json` keep a relative path as written, and it is read relative to the process working directory when the Agent is built.

**The dataclasses are frozen** and enforce their structural invariants in `__post_init__` (exactly one of `system_prompt`/`system_prompt_file`, at most one of `api_key`/`api_key_env`), so direct construction fails with the same `ConfigError` the loaders raise. Derive a variant with `dataclasses.replace(...)` rather than assigning.

**Creating a config has no side effects.** No loader reads the key variable or the prompt file; both are resolved when `Wica.init()` builds the Agent. That is where a referenced but unset `api_key_env` raises `MissingEnvError`, an unreadable prompt file raises `ConfigError`, a provider whose extra is not installed raises `ImportError`, and a `history_budget` with no known context window raises `ConfigError`. A caller that wants to degrade on any of these wraps `Wica.init()`, not config creation.

## The `agent` block

| Field | Required | Notes |
|---|---|---|
| `agent.provider` | yes | One of the [providers](#providers) below, or any other value LangChain's `init_chat_model` accepts as a provider |
| `agent.model` | yes | The model id, or the Hub `repo_id` for `huggingface-hub`; a free label for `fake` |
| `agent.system_prompt` / `agent.system_prompt_file` | exactly one | The persona, inline or in a file; see [System prompt](#system-prompt) |
| `agent.api_key` / `agent.api_key_env` | at most one | A literal key, or the name of the environment variable to read at Agent build; neither uses the provider's own default variable. See [API key](#api-key) |
| `agent.model_kwargs` | no | Forwarded to the provider's model constructor as given: `temperature`, Gemini's `thinking_level`, the `fake` provider's `script`. Default empty |
| `agent.hf_provider` | no | `huggingface-hub` only: the Hub backend that serves the model (`auto`, `fireworks-ai`, …). Default `auto` |
| `agent.history_reactions` | no | A positive integer: keep the last N reactions in the prompt. Default unbounded. See [History bounds](#history-bounds) |
| `agent.history_budget` | no | A number in `(0, 1]`: keep the prompt under this fraction of the model's context window. Default off. See [History bounds](#history-bounds) |
| `agent.model_timeout` | no | Seconds one model call may take, the provider's retries included; past it the step is abandoned with the `model_timeout` outcome and the Agent is free for the next input. Default `10`; `null` removes the deadline. Raise it for a slow reasoning model. See [specs/agent.md](../../specs/agent.md), "Model-call deadline" |
| `agent.context_window` | no | The context window in tokens, for `history_budget`; absent, the model's LangChain profile supplies it. Unused without a budget |

Runtime values that are callables or runtime objects are not configuration and stay on `Wica.init` and the wiring calls: the coalescing window (`Wica.init(config, coalesce_window=0.2)`), an injected event loop (`loop=`), the output sink and output Command (`set_output_sink`, `set_output_command`).

## API key

Give a literal `api_key`, or an `api_key_env` naming the environment variable to read, at most one. With neither, the provider's own package reads its standard variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` then `GEMINI_API_KEY`, `MISTRAL_API_KEY`, the Hub token). Prefer `api_key_env`: the config then carries no secret and can be committed, which is what the repo's own configs do (`WICA_<PROVIDER>_API_KEY`). A config with a literal key belongs in a git-ignored file. The variable is read by `Wica.init`, which raises `MissingEnvError` when it is unset; the `fake` provider ignores both fields.

## System prompt

Exactly one of `system_prompt` (inline) or `system_prompt_file` (a path, read when the Agent is built; relative handling per loader as above). This is the **persona**, kept verbatim. WICA appends a short **runtime primer** after it: how to read the observations (the user messages are the current World, not a chat transcript), that Commands run asynchronously and all the calls of one response run at once, how to reply (as text, or through the output Command when one is set), that a bounded history holds only the latest reactions (only when a bound is set), and that `noop` may be called to not react. The primer is recomposed only when the output Command is set or cleared, so in the intended flow (wire, then `start()`) the system message is fixed before the first step and stays in the model's cached prefix. It cannot be turned off in v1 ([specs/agent.md](../../specs/agent.md), "System prompt composition").

## History bounds

History is the Agent's only memory and is unbounded by default. Two optional fields bound it, independently or together; each adds the history-window clause to the primer:

- **`history_reactions: N`** keeps the last N reactions. History grows to 2N, then is cut back to N in one go, so the cached prompt prefix stays stable between cuts.
- **`history_budget: 0.5`** keeps the prompt under half the model's context window, using the input-token count each model call reports: when a call reports more than the budget, the oldest reactions are cut down to about half of it. It needs a known window: `context_window`, else the model's LangChain profile (`max_input_tokens`, known for most Anthropic, OpenAI, Gemini and Mistral models, not for Hub models or models newer than the installed profile data); with a budget and neither, `Wica.init` raises `ConfigError`.

Bounding is lossy: anything older than the bound is gone for the model, an earlier instruction, a name, its own replies. State that must outlive it belongs in a World entry. The budget's signal lags one step (it is the previous call's usage), so keep the fraction well below 1 (0.5 to 0.7); a `history_reactions` cap is the backstop when a prompt overflows the window outright, since a failed call reports no usage.

## Providers

Core `wica` bundles no provider; install the extra for the one you use. Selecting a provider whose extra is not installed fails at `Wica.init` with an `ImportError` naming it.

| `provider` | Install extra | Backed by | Notes |
|---|---|---|---|
| `anthropic` | `wica[anthropic]` | `langchain-anthropic` | |
| `openai` | `wica[openai]` | `langchain-openai` | |
| `google_genai` | `wica[google-genai]` | `langchain-google-genai` | Gemini through the Gemini Developer API (an AI Studio key, not Vertex). Gemini thinks by default, which adds latency to every reaction: set `model_kwargs: {"thinking_level": "low"}` on Gemini 3+ (`{"thinking_budget": 0}` on 2.5) for reactive use |
| `mistralai` | `wica[mistralai]` | `langchain-mistralai` | Mistral through La Plateforme. On a workspace without a paid tier the Small/Medium families answer every call with a 429; the Ministral family (the committed `ministral-3b-latest`) is served with full quota |
| `huggingface-hub` | `wica[huggingface-hub]` | `langchain-huggingface` | The Hub's serverless Inference Providers; `model` is the `repo_id`, `hf_provider` the backend. No model profile, so `history_budget` needs `context_window` |
| `fake` | none | built in | A deterministic scripted model for tests and demos: no key, no network. `model_kwargs` carries `script`, `default`, `loop`, `delay_s` ([guides/testing.md](../guides/testing.md)) |

Switching provider is a config edit. The four hosted providers above are plain `init_chat_model` pass-throughs, so another LangChain-supported provider value works the same way once its package is installed; `huggingface-hub` is built on WICA's own path ([specs/config.md](../../specs/config.md), "Providers").

## What is not configuration

**Logging.** WICA is a library: it emits records under the `wica.*` loggers and installs only a `NullHandler`. Configure logging in your application (`logging.basicConfig(...)`, `logging.getLogger("wica").setLevel(logging.DEBUG)` for a full World and Agent lifecycle trace).

**Tracing.** WICA opens OpenTelemetry spans through `opentelemetry-api` and never imports an SDK; install one and set a `TracerProvider` before `Wica.init` to export them ([guides/latency-and-traces.md](../guides/latency-and-traces.md)).
