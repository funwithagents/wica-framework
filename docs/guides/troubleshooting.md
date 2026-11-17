# Troubleshooting

The checks for each symptom, in the order to make them, each pointing at the page that owns the explanation. The errors themselves are listed in [../reference/api.md](../reference/api.md#errors). Turn on WICA's logs first: it is a library and configures none, so `logging.basicConfig(level=logging.WARNING)` and `logging.getLogger("wica").setLevel(logging.DEBUG)` in your application show every trigger, step, Command and cancellation.

## `Wica.init` raises

- **`ImportError` naming a package**: the provider's extra is not installed. `uv add "wica[<provider>] @ git+…"`; the extras are in [../reference/configuration.md](../reference/configuration.md#providers).
- **`MissingEnvError`**: `api_key_env` names a variable that is unset in the process. The shell that runs your program must export it; a non-interactive shell does not read your `~/.zshrc`. The variable is read at `init`, not at config creation ([../reference/configuration.md](../reference/configuration.md#api-key)).
- **`ConfigError` about an unknown key**: the loaders are strict; a typo in the JSON is rejected, and so is a key WICA does not define. `force_tool_call` is such a key now: remove it, and pass `as_tool=True` to `set_output_command` if you want the forced tool call back ([../reference/api.md](../reference/api.md#migrating-from-the-output-sink)). Both `system_prompt` and `system_prompt_file`, or both `api_key` and `api_key_env`, are rejected too ([../reference/configuration.md](../reference/configuration.md#the-loaders)).
- **`ConfigError` about the context window**: a `history_budget` is set and neither `context_window` nor the model's LangChain profile gives the window: Hub models, local models, and models newer than the installed profile data. Set `context_window` (for a local server, its own context size) ([../reference/configuration.md](../reference/configuration.md#history-bounds)).
- **`ConfigError` about `base_url`**: `openai-chat-completions-compatible` needs it, every other provider rejects it ([../reference/configuration.md](../reference/configuration.md#local-and-self-hosted-models)).
- **A deprecation warning about `google_genai` or `mistralai`**: the providers are now `gemini` and `mistral`; the old names still load. Rename them, and install `wica[gemini]` or `wica[mistral]` ([../reference/configuration.md](../reference/configuration.md#providers)).

## Nothing happens after `update()`

- **Is the entry an Input?** Only an entry registered `triggers_llm_call=True` wakes the Agent; others are context the next step sees ([world-and-inputs.md](world-and-inputs.md#what-wakes-the-agent)). A `trigger_condition_fn` may have declined this update.
- **Is the system running?** `update()` before `start()` or after `stop()` raises `RuntimeError("World is not running")` rather than doing nothing; a UI callback that fires after `stop()` fails the same way.
- **Was the trigger dropped?** A trigger arriving while a step is in flight is dropped, not queued. Subscribe to `wica.on_agent_trigger_dropped`, or look for the `dropped` line in the logs ([limits.md](limits.md)).
- **Is it the window?** The first 200 ms after a trigger are the coalescing window; a reaction shorter than that is not possible by design ([world-and-inputs.md](world-and-inputs.md#bursts-are-coalesced)).
- **Did the model decline?** A reaction whose record has `noop=True` (the model wrote `<noop/>` or called the `noop` tool, or wrote nothing at all: `outcome` is then `empty`) says and does nothing. Subscribe to `wica.on_agent_reaction_ended`, or open the reaction panel ([latency-and-traces.md](latency-and-traces.md)).
- **Is there an output Command?** Without one, all the model's text is private and reaches only `wica.on_agent_private_text` ([commands.md](commands.md#make-your-agent-speak)).
- **Did the model call time out?** A call longer than `model_timeout` (10 s by default) is abandoned: the reaction's `outcome` is `model_timeout` and an ERROR log line names the deadline. A slow reasoning model needs a higher value ([../reference/configuration.md](../reference/configuration.md#the-agent-block)).
- **Is the process still alive?** The system runs on a daemon thread: a script that reaches its end exits with it. Wait on an event the output Command sets, or serve something, as in [../getting-started.md](../getting-started.md#your-first-agent).

## The model does not do what I expect

- **It never calls my Command.** The model reads the function's name, docstring and signature: say what the Command does and when to use it. Check what was sent with `wica.on_agent_prompt` or the reaction panel's Prompt tab ([gradio-panels.md](gradio-panels.md)). A model that says what it is doing ("I'll wave") instead of calling the Command may do better with the output Command bound as a tool, `set_output_command(say, as_tool=True)`, which forces a tool call every step ([commands.md](commands.md#tag-or-tool-call)).
- **It writes `<noop/>` on a plain request.** The primer tells it to answer requests; a persona that frames inputs as background observations can override that ([commands.md](commands.md#staying-silent)).
- **It answers in plain text, without the tag.** That is heard: with an output Command set, a response with text and no tag is delivered whole when the response ends. It is not streamed, so the first words arrive later than with the tag. With `as_tool=True`, plain text is private: the `output_command_required` guard rejects a response with no tool call and asks again.
- **Part of what it writes is not heard.** Once a response holds an output tag, only the text inside the tags is delivered; the rest is private and reported on `wica.on_agent_private_text`. A persona that asks for the model's reasoning before the answer gets exactly this.
- **It calls the output Command as a tool.** With the tag (the default) the output Command is not a tool, so the call names an unknown Command and the `unknown_command` guard sends it back. A persona that tells the model to "call `say`" causes it: say "speak" instead, and let the primer teach the tag, or bind it as a tool with `as_tool=True` if your model prefers that ([commands.md](commands.md#tag-or-tool-call)).
- **It says it will do something after speaking, and does not.** With the tag, the end of an utterance wakes a step only when the model wrote a `<next>` note beside it, at most `max_next_steps` times in a row (default `1`; a WARNING is logged past the cap). Raise the cap for longer chains; with `0` the note is off and no utterance wakes a step ([commands.md](commands.md#continuing-after-speaking-the-next-note)).
- **Its speech is cut off.** A response that speaks cancels the speech still running from an earlier response: one voice ([commands.md](commands.md#interruption-and-what-was-heard)).
- **Reactions end `rejected`.** A [response guard](../reference/api.md#response-guards) rejected the response, and the retry too; the ERROR log line and the reaction's `rejections` say why. An unknown Command usually means the persona names one that is not registered. A response that has started speaking is rejected only when it turns out garbled (below). Raise `max_retries`, or remove the guard by name ([../reference/configuration.md](../reference/configuration.md#the-agent-block)).
- **It says nothing, and the log shows "garbled response".** The model degenerated (dots, ellipses and blank space, or a phrase repeated until its output cap) and its answer was discarded, then the retry too; `garbled_attempts` on the reaction counts it. A model that does it often needs other sampling settings or a smaller `max_tokens`. Out of retries the reaction ends `rejected` in silence: say a fallback line from `on_agent_reaction_ended` if you want one ([../reference/api.md](../reference/api.md#response-guards)).
- **Speech stops part-way and is said again differently.** The output turned garbled after it had started: it was cut off, kept out of the history, and the model asked again ([../reference/api.md](../reference/api.md#response-guards)).
- **The first words come later than the model's first tokens.** An output is held until its first 80 characters are checked for garbled text, and with `stream_output: false` or the output Command as a tool call until the whole response passed the guards ([latency-and-traces.md](latency-and-traces.md#reading-the-numbers)).
- **It acts twice, or on stale state.** Every Command completion wakes a step by default (`reactions_per_input` above one), and a bounded history forgets older turns. See `triggers_on_completion` ([commands.md](commands.md)) and [limits.md](limits.md).
- **It issues conflicting Commands in one response.** They run concurrently; guard the shared resource inside the Commands, and tell the model in the persona which actions exclude each other ([commands.md](commands.md#what-happens-when-the-model-calls-one)).
- **Gemini is slow on every reaction.** It thinks by default: `model_kwargs: {"thinking_level": "low"}` ([../reference/configuration.md](../reference/configuration.md#providers)).
- **A local model: every reaction ends `model_error` with "Connection error".** The server is not running, or `base_url` names the wrong port or misses `/v1` ([local-models.md](local-models.md#what-every-server-has-in-common)).
- **A local model: the first reaction ends `model_timeout`.** The server loads the model on the first request. Raise `model_timeout`, or warm the server before `start()` ([local-models.md](local-models.md#what-every-server-has-in-common)).
- **A local model: reactions end `model_timeout`, or long replies stop mid-sentence.** The deadline covers the whole reply: raise `model_timeout` to cover your longest one. A model that writes without end is bounded by `model_kwargs.max_tokens`, or `model_kwargs.extra_body.max_tokens` on Ollama, which ignores the first ([local-models.md](local-models.md#what-every-server-has-in-common)).
- **A local model on Ollama forgets its persona once the conversation grows.** Ollama's default context is 4096 tokens and it cuts a longer prompt from the start, without an error. Start it with `OLLAMA_CONTEXT_LENGTH` set to your `context_window` ([local-models.md](local-models.md#ollama)).
- **A local model speaks but never calls a Command.** A small model, or `llama-server` started without `--jinja` ([local-models.md](local-models.md#llama-server-llamacpp)).
- **Mistral answers 429 on every call.** On a workspace without a paid tier the Small/Medium families have a zero request quota; use the Ministral family ([../reference/configuration.md](../reference/configuration.md#providers)).

## A call raises at runtime

- **`ValueError` from `register_command`**: a duplicate name, or `noop`, `next` or `cancel_command`, which WICA reserves. From `world.register`: a key with whitespace or `" < > &`, or one already registered ([../reference/api.md](../reference/api.md#contracts)).
- **`ValueError` from `set_output_command`**: the function cannot be an output Command. Its first parameter must be annotated `OutputStream` (and the function `async def`) or `str`; every other parameter must be `str` or a `Literal` of strings with a string default; its name must be a valid tag name ([../reference/api.md](../reference/api.md#the-output-command)).
- **`AttributeError` on `set_output_sink` or `on_agent_text`**: both are removed ([../reference/api.md](../reference/api.md#migrating-from-the-output-sink)).
- **`TypeError` from `update()`**: the value is not of the entry's declared type; `None` is always accepted.
- **`RuntimeError` from `stop()` or `close()`**: it was called from the Wica's own loop thread, inside a Command, the output Command or an Event subscriber. Hand it to another thread, or call it after `launch()` or your main loop returns ([../getting-started.md](../getting-started.md#the-lifecycle-in-a-host-of-your-own)).
- **`RuntimeError` from `start()`**: the Wica is closed (a reset is a new `Wica.init`), or the injected loop is not running yet.

## A cancelled Command keeps running

The Command is a sync function: cancellation reaches the task, not the thread. Its entry reads `cancelled` and the Agent moves on while the function completes in the background. Make it `async`, or cooperative ([commands.md](commands.md#cancellation-reaches-the-task-not-always-the-work)).

## The Gradio panels

- **`ImportError` on `wica.contrib.gradio`**: install the `gradio` extra.
- **A panel raises when built**: panels create components and must be called inside a `gr.Blocks` context; presenters are plain objects and may be built anywhere after `Wica.init` ([gradio-panels.md](gradio-panels.md#the-ordering-rules)).
- **The transcript shows 💭 but the robot says nothing**: the 💭 item is the model's private text, what it wrote outside its output tags; the 🗣️ item is what the person hears. With no output Command set, all the model's text is private.
- **Typing into the page raises `World is not running`**: start the Wica before `launch()`.
