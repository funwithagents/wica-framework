# Troubleshooting

The checks for each symptom, in the order to make them, each pointing at the page that owns the explanation. The errors themselves are listed in [../reference/api.md](../reference/api.md#errors). Turn on WICA's logs first: it is a library and configures none, so `logging.basicConfig(level=logging.WARNING)` and `logging.getLogger("wica").setLevel(logging.DEBUG)` in your application show every trigger, step, Command and cancellation.

## `Wica.init` raises

- **`ImportError` naming a package**: the provider's extra is not installed. `uv add "wica[<provider>] @ git+…"`; the extras are in [../reference/configuration.md](../reference/configuration.md#providers).
- **`MissingEnvError`**: `api_key_env` names a variable that is unset in the process. The shell that runs your program must export it; a non-interactive shell does not read your `~/.zshrc`. The variable is read at `init`, not at config creation ([../reference/configuration.md](../reference/configuration.md#api-key)).
- **`ConfigError` about an unknown key**: the loaders are strict; a typo in the JSON is rejected, and so is a key WICA does not define. Both `system_prompt` and `system_prompt_file`, or both `api_key` and `api_key_env`, are rejected too ([../reference/configuration.md](../reference/configuration.md#the-loaders)).
- **`ConfigError` about the context window**: a `history_budget` is set and neither `context_window` nor the model's LangChain profile gives the window: Hub models, and models newer than the installed profile data. Set `context_window` ([../reference/configuration.md](../reference/configuration.md#history-bounds)).

## Nothing happens after `update()`

- **Is the entry an Input?** Only an entry registered `triggers_llm_call=True` wakes the Agent; others are context the next step sees ([world-and-inputs.md](world-and-inputs.md#what-wakes-the-agent)). A `trigger_condition_fn` may have declined this update.
- **Is the system running?** `update()` before `start()` or after `stop()` raises `RuntimeError("World is not running")` rather than doing nothing; a UI callback that fires after `stop()` fails the same way.
- **Was the trigger dropped?** A trigger arriving while a step is in flight is dropped, not queued. Subscribe to `wica.on_agent_trigger_dropped`, or look for the `dropped` line in the logs ([limits.md](limits.md)).
- **Is it the window?** The first 200 ms after a trigger are the coalescing window; a reaction shorter than that is not possible by design ([world-and-inputs.md](world-and-inputs.md#bursts-are-coalesced)).
- **Did the model answer with nothing?** A reaction whose `outcome` is `empty` (no text, no Command) or `noop` reaches no sink. Subscribe to `wica.on_agent_reaction_ended`, or open the reaction panel ([latency-and-traces.md](latency-and-traces.md)).
- **Did the model call time out?** A call longer than `model_timeout` (10 s by default) is abandoned: the reaction's `outcome` is `model_timeout` and an ERROR log line names the deadline. A slow reasoning model needs a higher value ([../reference/configuration.md](../reference/configuration.md#the-agent-block)).
- **Is the process still alive?** The system runs on a daemon thread: a script that reaches its end exits with it. Wait on an event the sink sets, or serve something, as in [../getting-started.md](../getting-started.md#your-first-agent).

## The model does not do what I expect

- **It never calls my Command.** The model reads the function's name, docstring and signature: say what the Command does and when to use it. Check what was sent with `wica.on_agent_prompt` or the reaction panel's Prompt tab ([gradio-panels.md](gradio-panels.md)).
- **It calls `noop` on a plain request.** The primer tells it to answer requests; a persona that frames inputs as background observations can override that. With an output Command set, check that the model's text is not going to the sink as private reasoning while nothing is said ([commands.md](commands.md#the-output-command)).
- **It answers in plain text instead of calling the output Command.** The reply reaches the sink as private reasoning and nobody hears it. Check that `force_tool_call` is not `false`: forced (the default), every model call must call a tool. A small model can then pick `noop` over the output Command; with only the output Command and WICA's own two bound, `ministral-3b` did so on most greetings ([commands.md](commands.md#the-output-command)).
- **A warning that `tool_choice` is forced but thinking is enabled.** Anthropic cannot force a tool call with extended thinking, so `langchain-anthropic` drops the forced choice and the model may answer in plain text again. Set `force_tool_call: false` to silence the warning, or turn thinking off to keep the forcing ([../reference/configuration.md](../reference/configuration.md#the-agent-block)).
- **It acts twice, or on stale state.** Every Command completion wakes a step by default (`reactions_per_input` above one), and a bounded history forgets older turns. See `triggers_on_completion` ([commands.md](commands.md)) and [limits.md](limits.md).
- **It issues conflicting Commands in one response.** They run concurrently; guard the shared resource inside the Commands, and tell the model in the persona which actions exclude each other ([commands.md](commands.md#what-happens-when-the-model-calls-one)).
- **Gemini is slow on every reaction.** It thinks by default: `model_kwargs: {"thinking_level": "low"}` ([../reference/configuration.md](../reference/configuration.md#providers)).
- **Mistral answers 429 on every call.** On a workspace without a paid tier the Small/Medium families have a zero request quota; use the Ministral family ([../reference/configuration.md](../reference/configuration.md#providers)).

## A call raises at runtime

- **`ValueError` from `register_command`**: a duplicate name, or `noop`/`cancel_command`, which WICA reserves. From `world.register`: a key with whitespace or `" < > &`, or one already registered ([../reference/api.md](../reference/api.md#contracts)).
- **`TypeError` from `update()`**: the value is not of the entry's declared type; `None` is always accepted.
- **`RuntimeError` from `stop()` or `close()`**: it was called from the Wica's own loop thread, inside a sink, a Command or an Event subscriber. Hand it to another thread, or call it after `launch()` or your main loop returns ([../getting-started.md](../getting-started.md#the-lifecycle-in-a-host-of-your-own)).
- **`RuntimeError` from `start()`**: the Wica is closed (a reset is a new `Wica.init`), or the injected loop is not running yet.

## A cancelled Command keeps running

The Command is a sync function: cancellation reaches the task, not the thread. Its entry reads `cancelled` and the Agent moves on while the function completes in the background. Make it `async`, or cooperative ([commands.md](commands.md#cancellation-reaches-the-task-not-always-the-work)).

## The Gradio panels

- **`ImportError` on `wica.contrib.gradio`**: install the `gradio` extra.
- **A panel raises when built**: panels create components and must be called inside a `gr.Blocks` context; presenters are plain objects and may be built anywhere after `Wica.init` ([gradio-panels.md](gradio-panels.md#the-ordering-rules)).
- **The transcript shows 💭 but the robot says nothing**: an output Command is set, so the free text is private reasoning; the 🗣️ item is what the person hears. Check that the Command does speak.
- **Typing into the page raises `World is not running`**: start the Wica before `launch()`.
