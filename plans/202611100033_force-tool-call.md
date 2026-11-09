# Force a tool call with an output Command

**Status:** Todo

Implements [specs/agent.md](../specs/agent.md) ("Forcing a tool call") and the `force_tool_call` field of [specs/config.md](../specs/config.md).

**Why.** With an output Command set, the only thing steering the model to it is one primer line, and weak models ignore it. In a client voice assistant, `ministral-3b` answered in plain text on every run (0/10): the reply went to the sink, which in that mode is private reasoning, so the user heard nothing — and once one plain-text reply is in history, the model keeps copying it. Rewording the clause (1/6), moving it last (0/6) and restating it in the Command's description (0/6) did not help; forcing a tool call did (6/6), with `noop` as the "nothing to say" choice. Strong models behaved the same forced or not. The existing live e2e set never caught it: no live test used an output Command.

## Scope

- `wica.config`: `AgentConfig.force_tool_call: bool = True`; the loaders default an absent key to `true` and reject a non-bool with the key named (direct construction too).
- `wica.agent`: `_rebind_model` forces a tool call while the output Command is **bound** and `force_tool_call` is set — `bind_tools(tools, tool_choice="any")` for every provider except `ChatHuggingFace`, which is bound with `bind(tools=<OpenAI-format tools>, tool_choice="required")` (its `bind_tools` refuses a `tool_choice` with more than one tool). Without an output Command, or with `force_tool_call` false, the binding is unchanged. Every attach/detach already rebinds through `_rebind_model`, so `set_output_command` (set, replace, clear, before or after `start()`) needs no other change.
- Docs: the configuration reference's field table, the output Command section of the Commands guide, a troubleshooting entry (a reply in plain text; Anthropic extended thinking).

## Tests

- Config: default `True`; `false` accepted; `"yes"`, `1`, `null` raise `ConfigError` naming the key — direct construction and `from_dict`.
- Agent, over a model double that records how it is bound: no output Command → no `tool_choice`; output Command set before `start()` → forced from `start()`; set while running → forced from that call; cleared → unforced; `force_tool_call=False` → never forced. The forced binding carries every Command (`noop` included).
- The `huggingface-hub` branch: a `ChatHuggingFace` (built offline) bound forced carries `tool_choice="required"` and every tool, without the single-tool `ValueError`.
- Live (`tests-e2e/test_wica.py`, every provider): `test_output_command_live` with a voice-assistant persona — the model must speak through `say` exactly once on a greeting, then end the follow-up step with `noop`; the failure message shows any text that went to the sink instead.

## Verification

- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` — clean.
- `uv run pytest` — the fast tier.
- `uv run pytest tests-e2e -k "fake or example or tester"` — the key-less e2e set.
- The live e2e set on the five providers.
- CI on the pull request.
